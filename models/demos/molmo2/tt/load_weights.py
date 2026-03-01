# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Weight loading utilities for Molmo2 model.

The Molmo2 safetensors checkpoint uses the following key structure:

Vision Transformer (ViT):
- model.vision_backbone.image_vit.patch_embedding.{weight,bias}
- model.vision_backbone.image_vit.positional_embedding
- model.vision_backbone.image_vit.transformer.resblocks.{N}.attention.{wq,wk,wv,wo}.{weight,bias}
- model.vision_backbone.image_vit.transformer.resblocks.{N}.attention_norm.{weight,bias}
- model.vision_backbone.image_vit.transformer.resblocks.{N}.feed_forward.{w1,w2}.{weight,bias}
- model.vision_backbone.image_vit.transformer.resblocks.{N}.ffn_norm.{weight,bias}

Vision Adapter:
- model.vision_backbone.image_pooling_2d.{wq,wk,wv,wo}.{weight,bias}
- model.vision_backbone.image_projector.{w1,w2,w3}.weight (no bias)

Text Model:
- model.transformer.wte.embedding (base vocabulary, 151936 tokens)
- model.transformer.wte.new_embedding (extended vocabulary, 128 tokens)
- model.transformer.ln_f.weight
- model.transformer.blocks.{N}.attn_norm.weight
- model.transformer.blocks.{N}.self_attn.att_proj.weight (fused QKV)
- model.transformer.blocks.{N}.self_attn.attn_out.weight
- model.transformer.blocks.{N}.self_attn.{q_norm,k_norm}.weight
- model.transformer.blocks.{N}.ff_norm.weight
- model.transformer.blocks.{N}.mlp.ff_proj.weight (fused gate+up)
- model.transformer.blocks.{N}.mlp.ff_out.weight
- lm_head.weight
"""

from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open


def load_state_dict_from_safetensors(
    model_id: str,
    weight_keys: Optional[list] = None,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load weights from HuggingFace safetensors files.

    Args:
        model_id: HuggingFace model ID (e.g., "allenai/Molmo2-8B")
        weight_keys: Optional list of specific keys to load (loads all if None)
        device: Device to load weights to

    Returns:
        State dict with loaded weights
    """
    import json

    # Download index file
    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Determine which files we need
    if weight_keys is None:
        weight_keys = list(weight_map.keys())

    files_needed = set()
    for key in weight_keys:
        if key in weight_map:
            files_needed.add(weight_map[key])

    # Download and load needed files
    state_dict = {}
    for filename in files_needed:
        filepath = hf_hub_download(model_id, filename)
        with safe_open(filepath, framework="pt", device=device) as f:
            for key in f.keys():
                if weight_keys is None or key in weight_keys:
                    state_dict[key] = f.get_tensor(key)

    return state_dict


def load_vision_weights(model_id: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load only vision-related weights (ViT, pooling, projector).

    Args:
        model_id: HuggingFace model ID
        device: Device to load weights to

    Returns:
        State dict with vision weights
    """
    import json

    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    # Filter to vision keys
    vision_keys = [k for k in index["weight_map"].keys() if "vision" in k.lower()]

    return load_state_dict_from_safetensors(model_id, vision_keys, device)


def load_vit_block_weights(
    model_id: str,
    layer_num: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    """
    Load weights for a specific ViT block.

    Args:
        model_id: HuggingFace model ID
        layer_num: Block number (0-indexed)
        device: Device to load weights to

    Returns:
        State dict with block weights
    """
    import json

    index_path = hf_hub_download(model_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}."
    block_keys = [k for k in index["weight_map"].keys() if k.startswith(prefix)]

    return load_state_dict_from_safetensors(model_id, block_keys, device)


def get_vit_block_keys(layer_num: int) -> list:
    """
    Get the expected state dict keys for a ViT block.

    Args:
        layer_num: Block number (0-indexed)

    Returns:
        List of expected key names
    """
    prefix = f"model.vision_backbone.image_vit.transformer.resblocks.{layer_num}"
    keys = []

    # Attention
    for proj in ["wq", "wk", "wv", "wo"]:
        keys.append(f"{prefix}.attention.{proj}.weight")
        keys.append(f"{prefix}.attention.{proj}.bias")

    # Norms
    for norm in ["attention_norm", "ffn_norm"]:
        keys.append(f"{prefix}.{norm}.weight")
        keys.append(f"{prefix}.{norm}.bias")

    # MLP
    for w in ["w1", "w2"]:
        keys.append(f"{prefix}.feed_forward.{w}.weight")
        keys.append(f"{prefix}.feed_forward.{w}.bias")

    return keys


def get_vit_keys() -> list:
    """
    Get the expected state dict keys for the full ViT.

    Returns:
        List of expected key names
    """
    prefix = "model.vision_backbone.image_vit"
    keys = [
        f"{prefix}.patch_embedding.weight",
        f"{prefix}.patch_embedding.bias",
        f"{prefix}.positional_embedding",
    ]

    # 27 blocks total (only 25 used)
    for layer_num in range(27):
        keys.extend(get_vit_block_keys(layer_num))

    return keys


def get_adapter_keys() -> list:
    """
    Get the expected state dict keys for the vision adapter.

    Returns:
        List of expected key names
    """
    prefix = "model.vision_backbone"
    keys = []

    # Image pooling
    for proj in ["wq", "wk", "wv", "wo"]:
        keys.append(f"{prefix}.image_pooling_2d.{proj}.weight")
        keys.append(f"{prefix}.image_pooling_2d.{proj}.bias")

    # Image projector (no bias)
    for w in ["w1", "w2", "w3"]:
        keys.append(f"{prefix}.image_projector.{w}.weight")

    return keys
