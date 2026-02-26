# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Thin wrapper around HuggingFace Molmo2ForConditionalGeneration that exposes
each sub-module for isolated testing during TTNN implementation.
"""

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


class Molmo2Reference:
    """
    Reference model wrapper for Molmo2-8B that provides access to individual
    submodules for layer-by-layer verification against TTNN implementations.
    """

    def __init__(self, ckpt_dir: str, torch_dtype=torch.float32):
        """
        Load the Molmo2 model and processor from HuggingFace.

        Args:
            ckpt_dir: Path or HuggingFace model identifier (e.g., "allenai/Molmo2-8B")
            torch_dtype: Data type for model weights (default: torch.float32 for reference)
        """
        self.ckpt_dir = ckpt_dir
        self.model = AutoModelForImageTextToText.from_pretrained(
            ckpt_dir,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(ckpt_dir, trust_remote_code=True)
        self.model.eval()

    @property
    def config(self):
        """Access the model's configuration."""
        return self.model.config

    @property
    def vision_backbone(self):
        """Access the full vision backbone (ViT + pooling + projector)."""
        return self.model.model.vision_backbone

    @property
    def image_vit(self):
        """Access the Vision Transformer encoder (25 layers)."""
        return self.model.model.vision_backbone.image_vit

    @property
    def image_pooling_2d(self):
        """Access the 2D image pooling (cross-attention based)."""
        return self.model.model.vision_backbone.image_pooling_2d

    @property
    def image_projector(self):
        """Access the SwiGLU image projector (1152 -> 12288 -> 4096)."""
        return self.model.model.vision_backbone.image_projector

    @property
    def text_model(self):
        """Access the language model backbone (36-layer transformer)."""
        return self.model.model.model

    @property
    def lm_head(self):
        """Access the language model head."""
        return self.model.lm_head

    @property
    def embed_tokens(self):
        """Access the token embedding layer."""
        return self.model.model.model.wte

    def get_vit_block(self, layer_num: int):
        """Get a specific ViT transformer block."""
        return self.image_vit.transformer.resblocks[layer_num]

    def get_text_block(self, layer_num: int):
        """Get a specific text model transformer block."""
        return self.text_model.blocks[layer_num]

    def preprocess_image(self, images, prompt: str):
        """
        Preprocess images and prompt using the Molmo2 processor.

        Args:
            images: PIL Image or list of PIL Images
            prompt: Text prompt

        Returns:
            Processor outputs including pixel_values, input_ids, etc.
        """
        return self.processor(text=prompt, images=images, return_tensors="pt")

    @torch.no_grad()
    def forward(self, **inputs):
        """Run full model forward pass."""
        return self.model(**inputs)

    @torch.no_grad()
    def generate(self, **inputs):
        """Run autoregressive generation."""
        return self.model.generate(**inputs)

    @torch.no_grad()
    def get_vit_hidden_states(self, pixel_values, return_layers=None):
        """
        Run ViT encoder and return hidden states from specified layers.

        Args:
            pixel_values: Preprocessed image tensor
            return_layers: List of layer indices to return (default: all layers)

        Returns:
            List of hidden states tensors from requested layers
        """
        # This may need adjustment based on actual Molmo2 HF implementation
        # The exact API depends on how HuggingFace exposes intermediate states
        hidden_states = []
        x = self.image_vit.patch_embedding(pixel_values)
        x = x + self.image_vit.positional_embedding

        for i, block in enumerate(self.image_vit.transformer.resblocks):
            x = block(x)
            if return_layers is None or i in return_layers:
                hidden_states.append(x.clone())

        return hidden_states
