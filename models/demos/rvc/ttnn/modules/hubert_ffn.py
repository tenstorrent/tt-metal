# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN Hubert FFN (Feed-Forward Network) block.

The FFN is the simplest submodule of TransformerSentenceEncoderLayer:
    x → fc1 → activation(gelu) → fc2 → output

Torch reference (hubert.py L152-153):
    x = self.activation_fn(self.fc1(x))
    x = self.fc2(x)

Shapes (for embed_dim=768, ffn_embed_dim=3072):
    fc1: [B, S, 768] → [B, S, 3072]
    gelu: element-wise
    fc2: [B, S, 3072] → [B, S, 768]

Design decisions:
    - Weights stored in DRAM, compute in L1 (Whisper FFN pattern)
    - Fused GELU via ttnn.linear(..., activation="gelu") for fc1
    - DRAM_MEMORY_CONFIG for all weight storage
    - No explicit core_grid (Stage 1 default)
"""

import torch
import ttnn

from models.demos.rvc.ttnn.utils import preprocess_linear_weight, preprocess_linear_bias


def ttnn_hubert_ffn(
    hidden_states: ttnn.Tensor,
    fc1_weight: ttnn.Tensor,
    fc1_bias: ttnn.Tensor,
    fc2_weight: ttnn.Tensor,
    fc2_bias: ttnn.Tensor,
) -> ttnn.Tensor:
    """
    Hubert FFN forward pass.

    Follows Whisper's ffn_forward pattern (L170-194):
    fc1 with fused GELU, then fc2, all in DRAM.

    Args:
        hidden_states: [B, S, embed_dim] on device, TILE_LAYOUT.
        fc1_weight: [embed_dim, ffn_embed_dim] on device (pre-transposed).
        fc1_bias: [1, 1, ffn_embed_dim] on device.
        fc2_weight: [ffn_embed_dim, embed_dim] on device (pre-transposed).
        fc2_bias: [1, 1, embed_dim] on device.

    Returns:
        [B, S, embed_dim] on device.
    """
    # fc1 + fused GELU (single kernel)
    h = ttnn.linear(
        hidden_states,
        fc1_weight,
        bias=fc1_bias,
        activation="gelu",
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # fc2
    h = ttnn.linear(
        h,
        fc2_weight,
        bias=fc2_bias,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return h


def preprocess_ffn_weights(
    fc1: torch.nn.Linear,
    fc2: torch.nn.Linear,
    device,
) -> dict:
    """
    Preprocess FFN weights for TTNN.

    Transposes weights and places on device in DRAM.

    Args:
        fc1: PyTorch fc1 Linear module.
        fc2: PyTorch fc2 Linear module.
        device: TTNN device.

    Returns:
        Dict with fc1_weight, fc1_bias, fc2_weight, fc2_bias.
    """
    return {
        "fc1_weight": preprocess_linear_weight(fc1.weight, device),
        "fc1_bias": preprocess_linear_bias(fc1.bias, device),
        "fc2_weight": preprocess_linear_weight(fc2.weight, device),
        "fc2_bias": preprocess_linear_bias(fc2.bias, device),
    }
