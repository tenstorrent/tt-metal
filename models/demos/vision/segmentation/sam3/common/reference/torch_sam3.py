# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference pipeline for SAM3 inference.
Used to generate reference outputs for comparison with ttnn implementation.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


def preprocess_image(image: torch.Tensor, target_size: int = 1008) -> torch.Tensor:
    """
    Preprocess image for SAM3 inference.

    Args:
        image: (B, C, H, W) or (C, H, W) tensor in [0, 255] or [0, 1] range
        target_size: target image size (SAM3 uses 1008)

    Returns:
        Preprocessed image tensor (B, 3, target_size, target_size) normalized to [-1, 1]
    """
    if image.ndim == 3:
        image = image.unsqueeze(0)

    # Normalize to [0, 1] if needed
    if image.max() > 1.0:
        image = image.float() / 255.0

    # Resize
    if image.shape[-2:] != (target_size, target_size):
        image = F.interpolate(
            image, size=(target_size, target_size), mode="bilinear", align_corners=False
        )

    # Normalize to [-1, 1] (SAM3 uses mean=0.5, std=0.5)
    image = (image - 0.5) / 0.5

    return image


def run_vit_backbone(model, pixel_values: torch.Tensor) -> torch.Tensor:
    """
    Run just the ViT backbone portion of SAM3.

    Args:
        model: SAM3 ViT backbone (sam3_model.backbone.visual.trunk)
        pixel_values: (B, 3, 1008, 1008) preprocessed image

    Returns:
        ViT features (B, 72, 72, 1024)
    """
    with torch.no_grad():
        features = model(pixel_values)
    return features


def run_vit_single_block(block, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Run a single ViT block.

    Args:
        block: ViT Block module
        hidden_states: (B, H, W, C) tensor

    Returns:
        (B, H, W, C) tensor
    """
    with torch.no_grad():
        output = block(hidden_states)
    return output


def run_neck(neck_model, pixel_values: torch.Tensor) -> Dict:
    """
    Run the full visual backbone (ViT + FPN neck).

    Args:
        neck_model: SAM3 visual backbone (sam3_model.backbone.visual)
        pixel_values: (B, 3, 1008, 1008) preprocessed image

    Returns:
        Dict with 'backbone_fpn' (list of feature maps) and 'vision_pos_enc' (position encodings)
    """
    with torch.no_grad():
        output = neck_model(pixel_values)
    return output


def run_text_encoder(text_encoder, text_prompts: List[str]) -> torch.Tensor:
    """
    Run the text encoder on text prompts.

    Args:
        text_encoder: SAM3 text encoder
        text_prompts: list of text strings

    Returns:
        Text embeddings tensor
    """
    with torch.no_grad():
        text_features = text_encoder(text_prompts)
    return text_features


def extract_vit_block_params(block) -> Dict[str, torch.Tensor]:
    """
    Extract all parameters from a ViT Block for ttnn preprocessing.

    Args:
        block: ViT Block module (sam3.model.vitdet.Block)

    Returns:
        Dict mapping parameter names to tensors
    """
    params = {}

    # LayerNorm 1
    params["norm1_weight"] = block.norm1.weight.data.clone()
    params["norm1_bias"] = block.norm1.bias.data.clone()

    # Attention QKV (fused)
    params["attn_qkv_weight"] = block.attn.qkv.weight.data.clone()
    params["attn_qkv_bias"] = block.attn.qkv.bias.data.clone()

    # Attention output projection
    params["attn_proj_weight"] = block.attn.proj.weight.data.clone()
    params["attn_proj_bias"] = block.attn.proj.bias.data.clone()

    # RoPE frequencies (if present)
    if hasattr(block.attn, "freqs_cis") and block.attn.freqs_cis is not None:
        params["attn_freqs_cis"] = block.attn.freqs_cis.clone()

    # Relative position biases (if present)
    if hasattr(block.attn, "rel_pos_h") and block.attn.rel_pos_h is not None:
        params["attn_rel_pos_h"] = block.attn.rel_pos_h.data.clone()
        params["attn_rel_pos_w"] = block.attn.rel_pos_w.data.clone()

    # LayerNorm 2
    params["norm2_weight"] = block.norm2.weight.data.clone()
    params["norm2_bias"] = block.norm2.bias.data.clone()

    # MLP
    params["mlp_fc1_weight"] = block.mlp.fc1.weight.data.clone()
    params["mlp_fc1_bias"] = block.mlp.fc1.bias.data.clone()
    params["mlp_fc2_weight"] = block.mlp.fc2.weight.data.clone()
    params["mlp_fc2_bias"] = block.mlp.fc2.bias.data.clone()

    # Window size
    params["window_size"] = block.window_size

    return params


def extract_vit_backbone_params(vit_model) -> Dict:
    """
    Extract all parameters from the ViT backbone.

    Args:
        vit_model: SAM3 ViT model (sam3.model.vitdet.ViT)

    Returns:
        Dict with all model parameters organized by component
    """
    params = {}

    # Patch embedding
    params["patch_embed"] = {
        "weight": vit_model.patch_embed.proj.weight.data.clone(),
        "bias": (
            vit_model.patch_embed.proj.bias.data.clone()
            if vit_model.patch_embed.proj.bias is not None
            else None
        ),
    }

    # Position embeddings
    if hasattr(vit_model, "pos_embed") and vit_model.pos_embed is not None:
        params["pos_embed"] = vit_model.pos_embed.data.clone()

    # Pre-LayerNorm (if present)
    if hasattr(vit_model, "ln_pre") and vit_model.ln_pre is not None:
        params["ln_pre_weight"] = vit_model.ln_pre.weight.data.clone()
        params["ln_pre_bias"] = vit_model.ln_pre.bias.data.clone()

    # Blocks
    params["blocks"] = []
    for block in vit_model.blocks:
        params["blocks"].append(extract_vit_block_params(block))

    return params
