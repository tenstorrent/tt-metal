# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Pixtral vision self-attention — fully on device, no host fallback.

Architecture (per layer, hidden=1024, 16 heads × 64 dim):
    q = x @ Wq      [seq, hidden]
    k = x @ Wk
    v = x @ Wv
    reshape → [1, n_heads, seq, head_dim]
    apply 2D RoPE on q, k
    SDPA(q, k, v, is_causal=False)
    out = concat_heads @ Wo

All weights are replicated across the mesh. Total per layer at bf16:
    4 × (1024 × 1024) × 2 B  ≈  8 MB
"""

from __future__ import annotations

import math


import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    VISION_HEAD_DIM,
    VISION_HIDDEN_SIZE,
    VISION_NUM_HEADS,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import (
    _apply_rope_ttnn,
    _load_weight,
)


class TtPixtralAttention:
    """Pixtral self-attention with 2D RoPE on q/k."""

    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        state_dict: dict,
        layer_prefix: str,
        compute_kernel_config,
        dtype=ttnn.bfloat16,
    ):
        self.mesh_device = mesh_device
        self.compute_kernel_config = compute_kernel_config
        self.n_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HEAD_DIM
        self.scale = 1.0 / math.sqrt(self.head_dim)

        p = layer_prefix + "attention."
        # HF stores [out, in]; transpose for ttnn.linear.
        self.q_proj = _load_weight(state_dict, p + "q_proj.weight", True, dtype, mesh_device)
        self.k_proj = _load_weight(state_dict, p + "k_proj.weight", True, dtype, mesh_device)
        self.v_proj = _load_weight(state_dict, p + "v_proj.weight", True, dtype, mesh_device)
        self.o_proj = _load_weight(state_dict, p + "o_proj.weight", True, dtype, mesh_device)

    def forward(
        self,
        x: ttnn.Tensor,
        cos: ttnn.Tensor,
        sin: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        Args:
            x:        [1, 1, seq_len, hidden=1024] replicated on all devices
            cos/sin:  [1, 1, seq_len, head_dim=64]  (broadcast over heads)
        Returns:
            [1, 1, seq_len, hidden=1024]
        """
        seq_len = x.shape[-2]

        # ── q/k/v projections ────────────────────────────────────────────
        q = ttnn.linear(
            x,
            self.q_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        k = ttnn.linear(
            x,
            self.k_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        v = ttnn.linear(
            x,
            self.v_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ── reshape to [1, n_heads, seq, head_dim] ──────────────────────
        q = ttnn.reshape(q, [1, seq_len, self.n_heads, self.head_dim])
        q = ttnn.transpose(q, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        k = ttnn.reshape(k, [1, seq_len, self.n_heads, self.head_dim])
        k = ttnn.transpose(k, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v = ttnn.reshape(v, [1, seq_len, self.n_heads, self.head_dim])
        v = ttnn.transpose(v, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # ── apply 2D RoPE to q and k ────────────────────────────────────
        q_rot = _apply_rope_ttnn(q, cos, sin, seq_len, self.n_heads, self.head_dim)
        k_rot = _apply_rope_ttnn(k, cos, sin, seq_len, self.n_heads, self.head_dim)
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # ── SDPA (non-causal) ───────────────────────────────────────────
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # [1, n_heads, seq, head_dim]
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)

        # ── concat heads → [1, 1, seq, hidden] ──────────────────────────
        attn_t = ttnn.transpose(attn, 1, 2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)
        attn_flat = ttnn.reshape(attn_t, [1, 1, seq_len, VISION_HIDDEN_SIZE])
        ttnn.deallocate(attn_t)

        # ── output projection ───────────────────────────────────────────
        out = ttnn.linear(
            attn_flat,
            self.o_proj,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_flat)
        return out
