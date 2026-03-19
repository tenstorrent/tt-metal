# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""VETextEncoder for SAM3 on ttnn.

The text encoder runs entirely on CPU via PyTorch (full CPU fallback).
Text encoding happens once per prompt and is not the inference bottleneck.
Results are converted to ttnn tensors for downstream on-device processing.
"""

from typing import Dict, List

import torch
import ttnn


def tt_text_encoder(
    text_prompts: List[str],
    text_encoder_module,
    device,
) -> Dict:
    """Run text encoder on CPU, return results as ttnn tensors.

    Args:
        text_prompts: list of text strings
        text_encoder_module: PyTorch VETextEncoder module
        device: ttnn device

    Returns:
        dict with 'text_features' and 'text_mask' as ttnn tensors on device
    """
    text_encoder_module.eval()

    with torch.no_grad():
        # VETextEncoder.forward(text, input_boxes=None, device=None)
        # Returns: (text_attention_mask, text_memory_resized, inputs_embeds_transposed)
        #   text_attention_mask: [batch, seq_len] bool - True where padding (inverted)
        #   text_memory_resized: [seq_len, batch, d_model] - encoder hidden states resized
        #   inputs_embeds_transposed: [seq_len, batch, width] - token embeddings
        text_attention_mask, text_memory_resized, inputs_embeds = (
            text_encoder_module(text_prompts, input_boxes=None, device=None)
        )

    # text_memory_resized: [seq_len, batch, d_model] -> convert to ttnn
    # text_attention_mask: [batch, seq_len] bool -> convert to ttnn
    text_features_torch = text_memory_resized.float().contiguous()
    text_mask_torch = text_attention_mask.float().contiguous()

    text_features_tt = ttnn.from_torch(
        text_features_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    text_mask_tt = ttnn.from_torch(
        text_mask_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return {
        "text_features": text_features_tt,
        "text_mask": text_mask_tt,
    }
