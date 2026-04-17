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
from models.experimental.gr00t_n1_6.tt.ttnn_common import (
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
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
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
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
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
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


class DiTAttentionOptimizedTTNN:
    """
    Fully on-device DiT attention with padded head dimensions.

    Weights are preprocessed to output padded_head_dim=64 per head.
    Uses nlp_create_qkv_heads and SDPA for efficient on-device attention.
    """

    def __init__(self, weights: Dict[str, torch.Tensor], prefix: str, num_heads: int, head_dim: int, device: Any):
        self.device = device
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.padded_head_dim = PADDED_HEAD_DIM
        self.inner_dim = num_heads * head_dim  # 1536
        self.padded_inner = num_heads * PADDED_HEAD_DIM  # 2048

        # Detect cross-attention from K weight input dim
        k_w = weights.get(f"{prefix}to_k.weight")
        self.is_cross_attention = k_w is not None and k_w.shape[1] != k_w.shape[0]

        # ---- Fused projection weights ----
        # Self-attn: fuse Q/K/V into one matmul (Q/K/V all come from hidden_states).
        # Cross-attn: Q from hidden, K/V from encoder — Q stays separate but K/V fuse.
        q_w = weights.get(f"{prefix}to_q.weight")
        q_b = weights.get(f"{prefix}to_q.bias")
        v_w = weights.get(f"{prefix}to_v.weight")
        v_b = weights.get(f"{prefix}to_v.bias")

        def _pad_qkv_weight(w: torch.Tensor) -> torch.Tensor:
            out_dim, in_dim = w.shape
            return (
                F.pad(w.reshape(num_heads, head_dim, in_dim), (0, 0, 0, PADDED_HEAD_DIM - head_dim))
                .reshape(num_heads * PADDED_HEAD_DIM, in_dim)
            )

        def _pad_qkv_bias(b: torch.Tensor) -> torch.Tensor:
            return F.pad(b.reshape(num_heads, head_dim), (0, PADDED_HEAD_DIM - head_dim)).reshape(
                num_heads * PADDED_HEAD_DIM
            )

        def _to_ttnn_weight(w_cat: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                w_cat.t().contiguous().to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        def _to_ttnn_bias(b_cat: torch.Tensor) -> ttnn.Tensor:
            return ttnn.from_torch(
                b_cat.unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

        if self.is_cross_attention:
            # Q alone (in_dim = inner_dim = 1536)
            self.to_q_weight = _to_ttnn_weight(_pad_qkv_weight(q_w))
            self.to_q_bias = _to_ttnn_bias(_pad_qkv_bias(q_b)) if q_b is not None else None
            # K/V fused (in_dim = cross_attention_dim = 2048)
            kv_w = torch.cat([_pad_qkv_weight(k_w), _pad_qkv_weight(v_w)], dim=0)
            self.to_kv_weight = _to_ttnn_weight(kv_w)
            k_b = weights.get(f"{prefix}to_k.bias")
            if k_b is not None:
                kv_b = torch.cat([_pad_qkv_bias(k_b), _pad_qkv_bias(v_b)], dim=0)
                self.to_kv_bias = _to_ttnn_bias(kv_b)
            else:
                self.to_kv_bias = None
        else:
            # Fused QKV (in_dim = inner_dim = 1536)
            qkv_w = torch.cat(
                [_pad_qkv_weight(q_w), _pad_qkv_weight(k_w), _pad_qkv_weight(v_w)], dim=0
            )
            self.to_qkv_weight = _to_ttnn_weight(qkv_w)
            if q_b is not None:
                k_b = weights.get(f"{prefix}to_k.bias")
                qkv_b = torch.cat([_pad_qkv_bias(q_b), _pad_qkv_bias(k_b), _pad_qkv_bias(v_b)], dim=0)
                self.to_qkv_bias = _to_ttnn_bias(qkv_b)
            else:
                self.to_qkv_bias = None

        # Output projection: needs padded input dim
        out_w = weights.get(f"{prefix}to_out.0.weight")
        out_b = weights.get(f"{prefix}to_out.0.bias")
        if out_w is not None:
            self.to_out_0_weight = _pad_attention_weight_for_heads(
                out_w, num_heads, head_dim, PADDED_HEAD_DIM, device, is_output=True
            )
            self.to_out_0_bias = preprocess_linear_bias(out_b, device) if out_b is not None else None

    def precompute_kv(self, encoder_hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        """Precompute fused KV for cross-attn. Backbone is constant across Euler steps."""
        assert self.is_cross_attention, "precompute_kv is only for cross-attn blocks"
        return ttnn.linear(
            encoder_hidden_states,
            self.to_kv_weight,
            bias=self.to_kv_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        precomputed_kv: Optional[ttnn.Tensor] = None,
    ) -> ttnn.Tensor:
        """Fully on-device attention with padded heads + fused QKV matmul."""
        batch_size = hidden_states.shape[0]
        q_seq = hidden_states.shape[1]
        padded_inner = self.padded_inner

        if self.is_cross_attention and encoder_hidden_states is not None:
            # Q: single matmul; K/V: use precomputed if provided (constant across Euler steps)
            q = ttnn.linear(
                hidden_states,
                self.to_q_weight,
                bias=self.to_q_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=CORE_GRID_BH,
            )
            if precomputed_kv is not None:
                kv = precomputed_kv
                own_kv = False
            else:
                kv = ttnn.linear(
                    encoder_hidden_states,
                    self.to_kv_weight,
                    bias=self.to_kv_bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    dtype=ttnn.bfloat16,
                    core_grid=CORE_GRID_BH,
                )
                own_kv = True
            kv_seq = encoder_hidden_states.shape[1]
            k = ttnn.slice(kv, [0, 0, 0], [batch_size, kv_seq, padded_inner])
            v = ttnn.slice(kv, [0, 0, padded_inner], [batch_size, kv_seq, 2 * padded_inner])
            if own_kv:
                ttnn.deallocate(kv)
        else:
            # Self-attn: one fused QKV matmul (1 instead of 3)
            qkv = ttnn.linear(
                hidden_states,
                self.to_qkv_weight,
                bias=self.to_qkv_bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
                core_grid=CORE_GRID_BH,
            )
            kv_seq = q_seq
            q = ttnn.slice(qkv, [0, 0, 0], [batch_size, q_seq, padded_inner])
            k = ttnn.slice(qkv, [0, 0, padded_inner], [batch_size, q_seq, 2 * padded_inner])
            v = ttnn.slice(qkv, [0, 0, 2 * padded_inner], [batch_size, q_seq, 3 * padded_inner])
            ttnn.deallocate(qkv)

        # Reshape to multi-head: [B, seq, num_heads*padded_hd] -> [B, num_heads, seq, padded_hd]
        q = ttnn.reshape(q, (batch_size, q_seq, self.num_heads, self.padded_head_dim))
        q = ttnn.permute(q, (0, 2, 1, 3))
        k = ttnn.reshape(k, (batch_size, kv_seq, self.num_heads, self.padded_head_dim))
        k = ttnn.permute(k, (0, 2, 1, 3))
        v = ttnn.reshape(v, (batch_size, kv_seq, self.num_heads, self.padded_head_dim))
        v = ttnn.permute(v, (0, 2, 1, 3))

        # Fused SDPA (FlashAttention-2) — replaces manual matmul+softmax+matmul.
        # Pass the true head_dim scale (1/sqrt(48)), not the padded 64, to preserve PCC.
        context = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Reshape back: [B, heads, q_seq, padded_hd] -> [B, q_seq, heads*padded_hd]
        context = ttnn.permute(context, (0, 2, 1, 3))
        attn_3d = ttnn.reshape(context, (batch_size, q_seq, self.num_heads * self.padded_head_dim))
        ttnn.deallocate(context)

        # Output projection (weight already padded for input)
        output = ttnn.linear(
            attn_3d,
            self.to_out_0_weight,
            bias=self.to_out_0_bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            core_grid=CORE_GRID_BH,
        )
        ttnn.deallocate(attn_3d)
        return output
