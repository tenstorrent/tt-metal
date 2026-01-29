# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""PyTorch implementation of CLIP Vision Encoder + IP-Adapter Resampler."""

import os

import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from transformers.image_utils import load_image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
IMAGE_ENCODER_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
IP_ADAPTER_WEIGHTS_NAME = "ip-adapter-plus_sdxl_vit-h.bin"
MODEL_CACHE_PATH = "clip_resampler_sdxl.pt"
DTYPE = torch.bfloat16


class CLIPVisionEncoderAndResamplerPT(nn.Module):
    """Combined CLIP Vision Encoder + IP-Adapter Resampler module."""

    def __init__(self, cache_path=MODEL_CACHE_PATH):
        """
        Initialize the model, loading from cache if available.

        Args:
            cache_path: Path to cached model file. If exists, loads from cache.
                       If None, always loads fresh from HuggingFace.
        """
        super().__init__()

        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            self._load_from_huggingface()
            if cache_path:
                self._save_to_cache(cache_path)

        self.eval()

    def _load_from_cache(self, path):
        """Load model components from cached file."""
        print(f"Loading model from {path}...")
        cached_model = torch.load(path, weights_only=False)
        self.image_encoder = cached_model.image_encoder
        self.resampler = cached_model.resampler
        print(f"Model loaded from {path}")

    def _load_from_huggingface(self):
        """Load model components from HuggingFace."""
        # Load CLIP Vision Encoder
        print(f"Loading CLIP Vision Encoder in {DTYPE}...")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(IMAGE_ENCODER_ID, torch_dtype=DTYPE)

        # Load SDXL Pipeline to extract Resampler
        print(f"Loading SDXL Pipeline to extract Resampler...")
        pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, image_encoder=self.image_encoder, torch_dtype=DTYPE)

        # Attach IP-Adapter & Extract Resampler
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name=IP_ADAPTER_WEIGHTS_NAME,
        )

        # Isolate the exact Resampler module used by the SDXL pipeline
        self.resampler = pipe.unet.encoder_hid_proj.image_projection_layers[0]

        self.to(DTYPE)

    def _save_to_cache(self, path):
        """Save the model to cache for future use."""
        print(f"Saving model to {path}...")
        torch.save(self, path)
        print(f"Model saved to {path}")

    def forward(self, pixel_values):
        # Get CLIP hidden states
        clip_outputs = self.image_encoder(pixel_values=pixel_values, output_hidden_states=True)
        # Extract penultimate layer (Standard for IP-Adapter Plus)
        # Shape: [batch, 257, 1280]
        patches = clip_outputs.hidden_states[-2]
        # Run through the Resampler
        # Shape: [batch, 16, 2048]
        output_tokens = self.resampler(patches)
        return output_tokens


def get_input():
    raw_image = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")
    processor = CLIPImageProcessor.from_pretrained(IMAGE_ENCODER_ID)
    input_data = processor(images=raw_image, return_tensors="pt")
    input_data["pixel_values"] = input_data["pixel_values"].to(DTYPE)
    return input_data
