# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Reference model loader for Qwen3-Coder-Next.

Wraps HuggingFace Qwen3NextForCausalLM for PCC comparison against TT implementation.
Requires transformers >= 4.57.0.dev0 for Qwen3NextForCausalLM support.
"""

import os
from typing import Optional

import torch
from loguru import logger


def load_reference_model(
    model_name: Optional[str] = None,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the HuggingFace reference model for PCC comparison.

    Args:
        model_name: HF model name or local path. Defaults to HF_MODEL env var
            or 'Qwen/Qwen3-Coder-Next'.
        device: Device to load model on ('cpu' for reference comparison).
        dtype: Weight dtype (bfloat16 for reference).

    Returns:
        Tuple of (model, config) where model is the HF Qwen3NextForCausalLM
        and config is the HF AutoConfig.
    """
    if model_name is None:
        model_name = os.environ.get("HF_MODEL", "Qwen/Qwen3-Coder-Next")

    from transformers import AutoConfig, AutoModelForCausalLM

    logger.info(f"Loading reference model: {model_name}")

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device,
    )
    model.eval()

    logger.info(
        f"Loaded {model_name}: {config.num_hidden_layers} layers, "
        f"{config.num_experts} experts, {config.vocab_size} vocab"
    )
    return model, config


def load_reference_state_dict(model_name: Optional[str] = None):
    """Load just the state dict without instantiating the full model.

    More memory-efficient for weight comparison and conversion.

    Args:
        model_name: HF model name or local path.

    Returns:
        Tuple of (state_dict, config).
    """
    if model_name is None:
        model_name = os.environ.get("HF_MODEL", "Qwen/Qwen3-Coder-Next")

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Use safetensors for efficient loading
    from pathlib import Path

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_path = Path(model_name)
    if not model_path.is_dir():
        model_path = Path(snapshot_download(model_name))

    state_dict = {}
    for shard_file in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    logger.info(f"Loaded state dict: {len(state_dict)} tensors from {model_name}")
    return state_dict, config


def get_reference_layer(model, layer_idx: int):
    """Extract a single decoder layer from the reference model.

    Args:
        model: HF Qwen3NextForCausalLM instance.
        layer_idx: Layer index (0-47).

    Returns:
        The decoder layer module.
    """
    return model.model.layers[layer_idx]


def get_reference_embedding(model):
    """Extract the embedding layer from the reference model."""
    return model.model.embed_tokens


def get_reference_lm_head(model):
    """Extract the LM head from the reference model."""
    return model.lm_head


def get_reference_norm(model):
    """Extract the final RMS norm from the reference model."""
    return model.model.norm
