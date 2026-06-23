# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3-VL vision attention.

Per `MiniMaxM3VLVisionAttention` (transformers 5.12):
  q = q_proj(x).view(L, num_heads, head_dim)   # separate q/k/v projections, WITH bias
  k = k_proj(x).view(L, num_heads, head_dim)
  v = v_proj(x).view(L, num_heads, head_dim)
  q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)   # rotate_half, partial (78 of 80)
  attn = sdpa(q, k, v, attn_mask=None, scale=1/sqrt(head_dim))   # FULL attention, no window
  out  = out_proj(attn.flatten(-2))

Differences from MoonViT attention: separate q/k/v (packed host-side into a
single wqkv here), rotate_half RoPE (not interleaved), and plain full
bidirectional SDPA (the tower passes attention_mask=None — no cu_seqlens).

head_dim=80 is not tile-aligned, so q/k/v are padded to 96 across wqkv,
the heads, and wo input. Padded slots hold zeros: they add nothing to
QK^T, multiply zero-rows of wo, and (with cos=1/sin=0) are untouched by
RoPE. Scale stays 1/sqrt(80) — padded dims contribute 0 to the dot product.

RoPE application avoids the rotary_embedding_llama convention ambiguity:
we apply it explicitly as `x*cos + (x @ R)*sin` with the sign-permutation
matrix `R` from `rope.rope_rotate_matrix`. R operates in the natural padded
head layout, so q/k/v share one padding scheme and SDPA (a dot product) is
invariant to it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.minimax_m3_vl.tt.common import hifi4_compute_config, mesh_mapper, next_tile_multiple
from models.demos.minimax_m3_vl.tt.rope import rope_rotate_matrix


def _pack_wqkv_weight(
    q_w: torch.Tensor,  # [hidden, hidden] each (out = num_heads*head_dim)
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pack separate q/k/v Linear weights into one head_dim-padded wqkv.

    Returns ttnn-linear convention: [hidden_in, 3*num_heads*padded_head_dim].
    """
    in_dim = q_w.shape[1]
    w = torch.stack([q_w, k_w, v_w], dim=0).to(torch.bfloat16)  # (3, num_heads*head_dim, in)
    w = w.view(3, num_heads, head_dim, in_dim)
    w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # (3, num_heads, padded, in)
    w = w.reshape(3 * num_heads * padded_head_dim, in_dim)
    return w.transpose(0, 1).contiguous()  # [in, 3*H*padded]


def _pack_wqkv_bias(
    q_b: torch.Tensor,
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    b = torch.stack([q_b, k_b, v_b], dim=0).to(torch.bfloat16)  # (3, num_heads*head_dim)
    b = b.view(3, num_heads, head_dim)
    b = F.pad(b, (0, padded_head_dim - head_dim))
    return b.reshape(3 * num_heads * padded_head_dim).contiguous()


def _pad_wo_weight(
    weight: torch.Tensor,  # HF nn.Linear weight: [hidden_out, num_heads*head_dim]
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pad wo input dim from num_heads*head_dim to num_heads*padded_head_dim. -> [in, out]."""
    out_dim, in_dim = weight.shape
    assert in_dim == num_heads * head_dim, f"wo weight in dim {in_dim} != {num_heads} * {head_dim}"
    w = weight.detach().to(torch.bfloat16).transpose(0, 1).contiguous()  # (in=H*hd, out)
    w = w.view(num_heads, head_dim, out_dim)
    w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # (H, padded, out)
    return w.reshape(num_heads * padded_head_dim, out_dim).contiguous()


class M3VLAttention(LightweightModule):
    """Vision attention: packed QKV, rotate_half 3D RoPE, full (non-windowed) SDPA."""

    def __init__(
        self,
        mesh_device,
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        wqkv_weight: torch.Tensor,
        wo_weight: torch.Tensor,
        wqkv_bias: Optional[torch.Tensor] = None,
        wo_bias: Optional[torch.Tensor] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        assert (
            num_heads * head_dim == hidden_size
        ), f"num_heads*head_dim ({num_heads}*{head_dim}) != hidden_size {hidden_size}"
        self.device = mesh_device
        self.dtype = dtype
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.padded_head_dim = next_tile_multiple(self.head_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        mm = mesh_mapper(mesh_device)

        self.wqkv = ttnn.as_tensor(
            wqkv_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        self.wqkv_bias = (
            ttnn.as_tensor(
                wqkv_bias.view(1, 1, 1, -1).contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mm,
            )
            if wqkv_bias is not None
            else None
        )

        self.wo = ttnn.as_tensor(
            wo_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        self.wo_bias = (
            ttnn.as_tensor(
                wo_bias.view(1, 1, 1, -1).contiguous(),
                device=mesh_device,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mm,
            )
            if wo_bias is not None
            else None
        )

        # Sign-permutation matrix for the rotate_half RoPE application.
        R = rope_rotate_matrix(self.head_dim, self.padded_head_dim)
        self.rotate_mat = ttnn.as_tensor(
            R.view(1, 1, self.padded_head_dim, self.padded_head_dim).to(torch.bfloat16).contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )

        self.compute_kernel_config = hifi4_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_attn: torch.nn.Module,  # _Attn with q_proj/k_proj/v_proj/out_proj
        hidden_size: int,
        num_heads: int,
        head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLAttention":
        for n in ("q_proj", "k_proj", "v_proj", "out_proj"):
            assert hasattr(ref_attn, n), f"expected {n} on {type(ref_attn).__name__}"
        padded = next_tile_multiple(head_dim)
        wqkv = _pack_wqkv_weight(
            ref_attn.q_proj.weight.data,
            ref_attn.k_proj.weight.data,
            ref_attn.v_proj.weight.data,
            num_heads,
            head_dim,
            padded,
        )
        has_bias = ref_attn.q_proj.bias is not None
        wqkv_bias = (
            _pack_wqkv_bias(
                ref_attn.q_proj.bias.data,
                ref_attn.k_proj.bias.data,
                ref_attn.v_proj.bias.data,
                num_heads,
                head_dim,
                padded,
            )
            if has_bias
            else None
        )
        wo = _pad_wo_weight(ref_attn.out_proj.weight.data, num_heads, head_dim, padded)
        wo_bias = ref_attn.out_proj.bias.data if ref_attn.out_proj.bias is not None else None
        return cls(
            mesh_device=mesh_device,
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            wqkv_weight=wqkv,
            wo_weight=wo,
            wqkv_bias=wqkv_bias,
            wo_bias=wo_bias,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    def stage_cos_sin(self, cos_pt: torch.Tensor, sin_pt: torch.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Push (L, padded_head_dim) cos/sin (already padded by rope_cos_sin_padded) to device.

        Shaped (1, 1, L, padded_head_dim) so they broadcast over the head axis.
        """
        assert (
            cos_pt.shape[-1] == self.padded_head_dim
        ), f"cos head_dim {cos_pt.shape[-1]} != padded {self.padded_head_dim} (use rope_cos_sin_padded)"
        mm = mesh_mapper(self.device)
        cos_tt = ttnn.from_torch(
            cos_pt.view(1, 1, *cos_pt.shape).to(torch.bfloat16).contiguous(),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        sin_tt = ttnn.from_torch(
            sin_pt.view(1, 1, *sin_pt.shape).to(torch.bfloat16).contiguous(),
            device=self.device,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mm,
        )
        return cos_tt, sin_tt

    def _apply_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """x*cos + (x @ R)*sin, broadcasting cos/sin over the head axis."""
        x_rot = ttnn.matmul(x, self.rotate_mat, compute_kernel_config=self.compute_kernel_config)
        out = ttnn.add(ttnn.mul(x, cos), ttnn.mul(x_rot, sin))
        ttnn.deallocate(x_rot)
        return out

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,  # (1, 1, L, hidden_size)
        cos: ttnn.Tensor,  # (1, 1, L, padded_head_dim)
        sin: ttnn.Tensor,
        memory_config: Optional["ttnn.MemoryConfig"] = None,
    ) -> ttnn.Tensor:
        xqkv = ttnn.linear(
            x,
            self.wqkv,
            bias=self.wqkv_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(xqkv)

        q_heads = self._apply_rope(q_heads, cos, sin)
        k_heads = self._apply_rope(k_heads, cos, sin)

        # Full (non-causal, unmasked) attention.
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        attn_flat = ttnn.experimental.nlp_concat_heads(attn, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn)

        out = ttnn.linear(
            attn_flat,
            self.wo,
            bias=self.wo_bias,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_flat)
        return out
