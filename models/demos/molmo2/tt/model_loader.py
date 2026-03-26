# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Model loading utilities for Molmo2.

Contains functions for loading the tokenizer/processor and model weights,
used by both the standalone demo and vLLM integration.
"""

import os
from typing import Optional

from loguru import logger

import ttnn

# Default model ID
MODEL_ID = "allenai/Molmo2-8B"


def load_processor():
    """Load the Molmo2 tokenizer from HuggingFace."""
    from transformers import AutoTokenizer

    logger.info(f"Loading tokenizer from {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        local_files_only=os.getenv("CI") == "true",
    )
    return tokenizer


def load_model_weights():
    """Load all model weights from HuggingFace."""
    from models.demos.molmo2.tt.load_weights import load_state_dict_from_safetensors

    logger.info(f"Loading model weights from {MODEL_ID}")
    state_dict = load_state_dict_from_safetensors(MODEL_ID)
    logger.info(f"Loaded {len(state_dict)} weight tensors")
    return state_dict


def create_model(mesh_device, state_dict, num_layers: Optional[int] = None):
    """
    Create the Molmo2 TTNN model.

    Args:
        mesh_device: TTNN device or mesh device
        state_dict: Model state dict
        num_layers: Optional number of text layers (default: 36)

    Returns:
        Molmo2Model instance
    """
    from models.demos.molmo2.tt.molmo2_model import Molmo2Model

    logger.info("Creating Molmo2 TTNN model")

    text_num_layers = num_layers if num_layers is not None else 36

    model = Molmo2Model(
        mesh_device=mesh_device,
        state_dict=state_dict,
        # Vision config
        vit_num_layers=25,
        vit_hidden_dim=1152,
        vit_intermediate_dim=4304,
        vit_num_heads=16,
        vit_head_dim=72,
        patch_size=14,
        image_size=378,
        feature_layers=(24, 18),  # HF order: vit_layers=[-3, -9] -> layers [24, 18]
        # Adapter config
        adapter_hidden_dim=1152,
        adapter_intermediate_dim=12288,
        adapter_num_heads=16,
        adapter_head_dim=72,
        # Text config
        text_num_layers=text_num_layers,
        text_hidden_dim=4096,
        text_intermediate_dim=12288,
        text_num_heads=32,
        text_num_kv_heads=8,
        text_head_dim=128,
        vocab_size=152064,
        max_seq_len=8192,
        rope_theta=1000000.0,
        rms_norm_eps=1e-5,
        dtype=ttnn.bfloat8_b,
    )

    logger.info("Model created successfully")
    return model
