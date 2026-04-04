# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized DiT attention for GR00T N1.6 — fully on-device.

head_dim=48 is padded to 64 (tile-aligned) in weight preprocessing.
This eliminates CPU-device transfers during attention computation.

Strategy: Pad Q/K/V projection weights so they output padded head_dim=64.
Then use SDPA with the correct scale factor (1/sqrt(48), not 1/sqrt(64)).
"""

import math
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.groot_n16.tt.ttnn_common import (
    CORE_GRID_BH,
    preprocess_linear_bias,
)


PADDED_HEAD_DIM = 64  # 48 padded to nearest multiple of 32


def _pad_attention_weight_for_heads(
    weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    device: Any,
    is_output: bool = False,
) -> ttnn.Tensor:
    """Pad attention weight to tile-aligned head dimensions.

    For Q/K/V weights [out_features, in_features]:
        out_features = num_heads * head_dim -> num_heads * padded_head_dim

    For output weight [out_features, in_features]:
        in_features = num_heads * head_dim -> num_heads * padded_head_dim
    """
    if is_output:
        # Output proj: [inner_dim, inner_dim] -> [inner_dim, num_heads * padded_head_dim]
        # Pad input dimension
        out_dim, in_dim = weight.shape
        w = weight.reshape(out_dim, num_heads, head_dim)
        w = F.pad(w, (0, padded_head_dim - head_dim))  # pad last dim
        w = w.reshape(out_dim, num_heads * padded_head_dim)
        # Transpose for ttnn: [in, out]
        return ttnn.from_torch(
            w.t().contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )
    else:
        # Q/K/V proj: [num_heads * head_dim, in_dim] -> [num_heads * padded_head_dim, in_dim]
        out_dim, in_dim = weight.shape
        w = weight.reshape(num_heads, head_dim, in_dim)
        w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # pad head_dim
        w = w.reshape(num_heads * padded_head_dim, in_dim)
        # Transpose for ttnn: [in, out]
        return ttnn.from_torch(
            w.t().contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )


def _pad_attention_bias(
    bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
    device: Any,
) -> ttnn.Tensor:
    """Pad attention bias from num_heads*head_dim to num_heads*padded_head_dim."""
    b = bias.reshape(num_heads, head_dim)
    b = F.pad(b, (0, padded_head_dim - head_dim))
    b = b.reshape(num_heads * padded_head_dim)
    return ttnn.from_torch(
        b.unsqueeze(0).to(torch.bfloat16),
        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
    )


class DiTAttentionOptimizedTTNN:
    """
    Fully on-device DiT attention with padded head dimensions.

    Weights are preprocessed to output padded_head_dim=64 per head.
    Uses nlp_create_qkv_heads and SDPA for efficient on-device attention.
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str,
                 num_heads: int, head_dim: int, device: Any):
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.padded_head_dim = PADDED_HEAD_DIM
        self.inner_dim = num_heads * head_dim  # 1536

        # Detect cross-attention from K weight input dim
        k_w = weights.get(f"{prefix}to_k.weight")
        self.is_cross_attention = (k_w is not None and k_w.shape[1] != k_w.shape[0])

        # Preprocess weights with padded head dimensions
        for proj in ["to_q", "to_k", "to_v"]:
            w = weights.get(f"{prefix}{proj}.weight")
            b = weights.get(f"{prefix}{proj}.bias")
            if w is not None:
                setattr(self, f"{proj.replace('.', '_')}_weight",
                        _pad_attention_weight_for_heads(w, num_heads, head_dim, PADDED_HEAD_DIM, device))
                if b is not None:
                    setattr(self, f"{proj.replace('.', '_')}_bias",
                            _pad_attention_bias(b, num_heads, head_dim, PADDED_HEAD_DIM, device))
                else:
                    setattr(self, f"{proj.replace('.', '_')}_bias", None)

        # Output projection: needs padded input dim
        out_w = weights.get(f"{prefix}to_out.0.weight")
        out_b = weights.get(f"{prefix}to_out.0.bias")
        if out_w is not None:
            self.to_out_0_weight = _pad_attention_weight_for_heads(
                out_w, num_heads, head_dim, PADDED_HEAD_DIM, device, is_output=True)
            self.to_out_0_bias = preprocess_linear_bias(out_b, device) if out_b is not None else None

    def __call__(self, hidden_states: ttnn.Tensor,
                 encoder_hidden_states: Optional[ttnn.Tensor] = None) -> ttnn.Tensor:
        """Fully on-device attention with padded heads."""
        batch_size = hidden_states.shape[0]
        q_seq = hidden_states.shape[1]

        # Q projection: [B, q_seq, inner_dim] -> [B, q_seq, num_heads * padded_head_dim]
        q = ttnn.linear(hidden_states, self.to_q_weight, bias=self.to_q_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # K/V source
        kv_source = encoder_hidden_states if (self.is_cross_attention and encoder_hidden_states is not None) else hidden_states
        kv_seq = kv_source.shape[1]

        k = ttnn.linear(kv_source, self.to_k_weight, bias=self.to_k_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        v = ttnn.linear(kv_source, self.to_v_weight, bias=self.to_v_bias,
                        memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)

        # Reshape to multi-head: [B, seq, num_heads*padded_hd] -> [B, num_heads, seq, padded_hd]
        # padded_head_dim=64 is tile-aligned, so reshape+permute+matmul all work
        q = ttnn.reshape(q, (batch_size, q_seq, self.num_heads, self.padded_head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch_size, kv_seq, self.num_heads, self.padded_head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch_size, kv_seq, self.num_heads, self.padded_head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Scaled dot-product attention: Q @ K^T / sqrt(head_dim)
        scale = 1.0 / math.sqrt(self.head_dim)
        q = ttnn.mul(q, scale)

        # Transpose K: [B, heads, kv_seq, padded_hd] -> [B, heads, padded_hd, kv_seq]
        k = ttnn.permute(k, (0, 1, 3, 2))

        attn = ttnn.matmul(q, k, memory_config=ttnn.L1_MEMORY_CONFIG,
                           dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        attn = ttnn.softmax_in_place(attn, numeric_stable=True)

        context = ttnn.matmul(attn, v, memory_config=ttnn.L1_MEMORY_CONFIG,
                              dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(attn)
        ttnn.deallocate(v)

        # Reshape back: [B, heads, q_seq, padded_hd] -> [B, q_seq, heads*padded_hd]
        context = ttnn.permute(context, (0, 2, 1, 3))
        attn_3d = ttnn.reshape(context, (batch_size, q_seq, self.num_heads * self.padded_head_dim))
        ttnn.deallocate(context)

        # Output projection (weight already padded for input)
        output = ttnn.linear(attn_3d, self.to_out_0_weight, bias=self.to_out_0_bias,
                             memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16, core_grid=CORE_GRID_BH)
        ttnn.deallocate(attn_3d)
        return output
