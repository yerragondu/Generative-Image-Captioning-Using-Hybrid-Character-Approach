# Generative Image Captioning Using Hybrid Character Approach

This project presents a **Generative Image Captioning system** based on a custom **Encoder–Bridge–Decoder architecture**. It utilizes convolutional and recurrent neural networks to generate accurate and descriptive captions from images. A hybrid character-level modeling approach is employed to enhance robustness in low-resource or noisy scenarios.

---

## 🧠 Model Architecture Overview

The pipeline follows an **Encoder → Bridge → Decoder** structure:

- **Encoder:** A pretrained CNN (e.g., ResNet) extracts deep features from input images.
- **Bridge:** Transforms visual features into a compact latent representation compatible with sequential decoding.
- **Decoder:** A character-level RNN (GRU/LSTM) generates captions one character at a time, allowing for finer granularity.

---

## 🗂️ Project Structure

```
.
├── png/                 # Contains images for demonstration or results
├── build_vocab.py       # Script to generate vocabulary from captions
├── data_loader.py       # Custom PyTorch DataLoader for COCO dataset
├── download.sh          # Shell script to download MS COCO dataset
├── main.ipynb           # Main notebook for interactive experimentation
├── model.py             # Encoder-Bridge-Decoder model definition
├── requirements.txt     # Required Python packages
├── resize.py            # Resizing script for COCO images
├── sample.py            # Generate captions using trained model
├── train.py             # Model training script
├── vocab.pkl            # Pickled vocabulary object
```

---

## 📚 Dataset

This project uses the **MS COCO 2014** dataset, a standard benchmark for image captioning:

- **Training images:** ~82,000
- **Validation images:** ~40,000
- Captions per image: 5 human-generated descriptions

To prepare the dataset:

```bash
bash download.sh
```

You may need to manually organize image folders and annotation files as expected by the data loader.

---

## ▶️ How to Run

### 1. Install Dependencies

Set up a Python virtual environment and install required packages:

```bash
pip install -r requirements.txt
```

Typical packages include:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `scikit-learn`
- `tqdm`
- `nltk`

### 2. Build Vocabulary

```bash
python build_vocab.py
```

### 3. Train the Model

```bash
python train.py --model_path ./models --crop_size 224 --vocab_path ./vocab.pkl
```

### 4. Generate Captions

```bash
python sample.py --image_path ./png/example.jpg --model_path ./models/model.ckpt
```

Or use `main.ipynb` for step-by-step demo in notebook format.

---

## 💡 Key Features

- Character-level generation for high-resolution caption control
- Modular architecture with customizable components
- PyTorch-based and fully reproducible
- Easily extendable for multilingual captioning

---

## 📄 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute the code with attribution.

---

> 📩 For questions or dataset setup assistance, feel free to open an issue or reach out.
