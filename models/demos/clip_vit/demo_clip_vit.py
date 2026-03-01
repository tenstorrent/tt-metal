"""
Demo script for CLIP-ViT on Tenstorrent hardware.
Loads the model, encodes an image and text, and computes cosine similarity.
"""

import ttnn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import urllib.request
import os

from models.demos.clip_vit.tt.clip_vit_encoder import CLIPVisionTransformer
from models.demos.clip_vit.tt.clip_text_encoder import CLIPTextTransformer

def preprocess_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for CLIP ViT."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def tokenize_text(text: str, max_length: int = 77) -> torch.Tensor:
    """Dummy tokenizer for demonstration."""
    words = text.lower().split()
    tokens = [abs(hash(word)) % 49408 for word in words][:max_length]
    padded_tokens = tokens + [0] * (max_length - len(tokens))
    return torch.tensor([padded_tokens], dtype=torch.int32)

def main():
    device_id = 0
    device = ttnn.open_device(device_id=device_id)

    try:
        print("Initializing CLIP Vision Encoder on TTNN...")
        vision_model = CLIPVisionTransformer(device)
        
        print("Initializing CLIP Text Encoder on TTNN...")
        text_model = CLIPTextTransformer(device)

        # Download sample image
        sample_img_path = "sample_cat.jpg"
        if not os.path.exists(sample_img_path):
            urllib.request.urlretrieve("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", sample_img_path)

        # Inputs
        image_input = preprocess_image(sample_img_path)
        text_input = tokenize_text("a photo of a cat")

        print("Encoding image...")
        image_embedding = vision_model(image_input)
        
        print("Encoding text...")
        text_embedding = text_model(text_input)

        # Convert back to torch for similarity computation
        image_emb_torch = ttnn.to_torch(image_embedding)
        text_emb_torch = ttnn.to_torch(text_embedding)

        # Normalize and compute similarity
        image_emb_torch = torch.nn.functional.normalize(image_emb_torch, p=2, dim=-1)
        text_emb_torch = torch.nn.functional.normalize(text_emb_torch, p=2, dim=-1)
        similarity = torch.matmul(image_emb_torch, text_emb_torch.T)

        print(f"Cosine Similarity between image and text: {similarity.item():.4f}")
        print("Successfully ran CLIP-ViT bring-up on TTNN!")

    finally:
        ttnn.close_device(device)

if __name__ == "__main__":
    main()
