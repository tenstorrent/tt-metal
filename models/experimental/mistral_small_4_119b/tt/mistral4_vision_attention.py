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

import torch
import ttnn
from models.experimental.mistral_small_4_119b.constants import (
    VISION_HEAD_DIM,
    VISION_NUM_HEADS,
)
from models.experimental.mistral_small_4_119b.tt.mistral4_self_attention import _load_weight
from models.experimental.mistral_small_4_119b.tt.vision_matmul_config import (
    build_o_proj_preset,
    build_qkv_preset,
    vision_linear,
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

        grid = mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        p = layer_prefix + "attention."

        # Fused QKV: concat [q, k, v] along output dim then transpose for ttnn.linear.
        # HF weights are [out_features, in_features]; concat on dim=0 → [3*H, H], then .T → [H, 3*H].
        q_w = state_dict[p + "q_proj.weight"].to(torch.bfloat16)
        k_w = state_dict[p + "k_proj.weight"].to(torch.bfloat16)
        v_w = state_dict[p + "v_proj.weight"].to(torch.bfloat16)
        wqkv = torch.cat([q_w, k_w, v_w], dim=0).T.contiguous()  # [H, 3*H]
        self.wqkv = ttnn.as_tensor(
            wqkv,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )  # [VISION_HIDDEN_SIZE, 3 * VISION_HIDDEN_SIZE]

        self.o_proj = _load_weight(state_dict, p + "o_proj.weight", True, dtype, mesh_device)

        # Presets depend on the actual sequence length (num_patches), which
        # isn't known until the first forward. Build them lazily and cache.
        self.qkv_preset = None
        self.o_proj_preset = None
        self._preset_m: int | None = None

    def _ensure_presets(self, m: int) -> None:
        if self._preset_m == m:
            return
        self.qkv_preset = build_qkv_preset(self.mesh_device, m)
        self.o_proj_preset = build_o_proj_preset(self.mesh_device, m)
        self._preset_m = m

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
        self._ensure_presets(seq_len)

        # ── Fused QKV projection → [1, 1, seq, 3*hidden] ────────────────
        # Sweep-tuned 1D l1/dram/ws on 8×6 grid (49 TFLOPs vs 25 with the default).
        # Output stays in L1 so the rest of the attention chain (qkv heads,
        # RoPE, SDPA, concat) doesn't have to bounce through DRAM.
        qkv = vision_linear(
            x,
            self.wqkv,
            self.qkv_preset,
            compute_kernel_config=self.compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1, 1, seq, 3 * VISION_HIDDEN_SIZE]

        # ── Split into heads: each [1, n_heads, seq, head_dim] ──────────
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            qkv,
            num_heads=self.n_heads,
            num_kv_heads=self.n_heads,
            transpose_k_heads=False,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(qkv)

        # ── Apply 2D RoPE to q and k ─────────────────────────────────────
        q_rot = ttnn.experimental.rotary_embedding_hf(
            q,
            cos,
            sin,
            is_decode_mode=False,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k_rot = ttnn.experimental.rotary_embedding_hf(
            k,
            cos,
            sin,
            is_decode_mode=False,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(q)
        ttnn.deallocate(k)

        # ── SDPA (non-causal) ────────────────────────────────────────────
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_rot,
            k_rot,
            v,
            is_causal=False,
            scale=self.scale,
            program_config=self.sdpa_program_config,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )  # [1, n_heads, seq, head_dim]
        ttnn.deallocate(q_rot)
        ttnn.deallocate(k_rot)
        ttnn.deallocate(v)

        # ── Concat heads → [1, 1, seq, hidden] ──────────────────────────
        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        # ── Output projection ────────────────────────────────────────────
        # Sweep-tuned 1D l1/dram/ws on 8×4 grid, in0_block_w=16 (24 TFLOPs).
        # Output stays L1 — the block's residual add now runs L1-in/L1-out.
        out = vision_linear(
            attn_flat,
            self.o_proj,
            self.o_proj_preset,
            compute_kernel_config=self.compute_kernel_config,
            output_memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_flat)
        return out
