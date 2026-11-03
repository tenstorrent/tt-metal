# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
MiniMax-M3 vision attention, as Hugging Face's MiniMaxM3VLVisionAttention computes it.

It runs the q, k and v projections with bias, 3D RoPE on q and k, full attention with no mask, and the output
projection. The q, k and v weights are packed into one wqkv. A lane is one of the positions along a head on the chip.
Each head is padded from 80 to 96 lanes, a whole number of tiles. The padding is zeros: it adds nothing to q k^T, and
the scale stays 1 / sqrt(80).

The RoPE op rotates lane pairs (2i, 2i + 1), while Hugging Face's rotate_half pairs dims i and i + 39. So the q and k
rows of wqkv are reordered at load: lane 2i holds dim i and lane 2i + 1 holds dim i + 39. The 2 dims that do not
rotate stay at lanes 78 and 79. q and k share the order, so q k^T does not change. The cos and sin tables (rope.py)
follow the same order.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.minimax_m3.tt.vision.common import (
    DRAM_PLACEMENT,
    BlockPlacement,
    as_linear_bias,
    hifi4_compute_config,
    linear,
    linear_compute_config,
    sdpa_program_config,
    to_device,
)
from models.demos.minimax_m3.tt.vision.rope import rope_axis_dim
from models.demos.minimax_m3.utils.weight_conversion import meta_partial_permute


def _rope_head_order(w: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder the q or k rows of a weight or bias into the RoPE op's lane order (see the module docstring)."""
    return meta_partial_permute(w, num_heads, head_dim, 3 * rope_axis_dim(head_dim))


def _pack_wqkv_weight(
    q_w: torch.Tensor,  # [hidden, hidden] each (out = num_heads*head_dim)
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pack the q, k and v weights ([out, in] each) into one wqkv of [in, 3 * num_heads * padded_head_dim].

    q and k rows go in the RoPE op's lane order (_rope_head_order); v keeps its order. Each head is zero-padded to
    padded_head_dim.
    """
    in_dim = q_w.shape[1]
    q_w, k_w = _rope_head_order(q_w, num_heads, head_dim), _rope_head_order(k_w, num_heads, head_dim)
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
    """Pack the q, k and v biases as _pack_wqkv_weight packs the weights, into [3 * num_heads * padded_head_dim]."""
    packed = _pack_wqkv_weight(q_b[:, None], k_b[:, None], v_b[:, None], num_heads, head_dim, padded_head_dim)
    return packed.reshape(-1)


def _pad_wo_weight(
    weight: torch.Tensor,  # HF nn.Linear weight: [hidden_out, num_heads*head_dim]
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Transpose the output projection's weight to [in, out] and zero-pad each head's input rows to padded_head_dim."""
    out_dim, in_dim = weight.shape
    assert in_dim == num_heads * head_dim, f"wo weight in dim {in_dim} != {num_heads} * {head_dim}"
    w = weight.detach().to(torch.bfloat16).transpose(0, 1).contiguous()  # (in=H*hd, out)
    w = w.view(num_heads, head_dim, out_dim)
    w = F.pad(w, (0, 0, 0, padded_head_dim - head_dim))  # (H, padded, out)
    return w.reshape(num_heads * padded_head_dim, out_dim).contiguous()


class M3VLAttention(LightweightModule):
    """Vision attention on the chip: packed QKV, 3D RoPE and full attention."""

    def __init__(
        self,
        mesh_device,
        num_heads: int,
        head_dim: int,
        wqkv_weight: torch.Tensor,  # [hidden, 3*num_heads*padded_head_dim], _pack_wqkv_weight
        wo_weight: torch.Tensor,  # [num_heads*padded_head_dim, hidden], _pad_wo_weight
        wqkv_bias: Optional[torch.Tensor] = None,
        wo_bias: Optional[torch.Tensor] = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.scale = 1.0 / math.sqrt(head_dim)

        self.wqkv = to_device(wqkv_weight, mesh_device, dtype)
        self.wqkv_bias = as_linear_bias(mesh_device, wqkv_bias, dtype) if wqkv_bias is not None else None
        self.wo = to_device(wo_weight, mesh_device, dtype)
        self.wo_bias = as_linear_bias(mesh_device, wo_bias, dtype) if wo_bias is not None else None

        # rotary_embedding_llama's transformation matrix, one tile: x @ trans_mat turns each lane pair (a, b) of a tile
        # into (-b, a). Like cos and sin, it must be bfloat16 whatever the weights' dtype.
        self.trans_mat = to_device(get_rot_transformation_mat(), mesh_device)

        # QKV and the output projection accumulate in L1; RoPE and SDPA keep the plain HiFi4 config.
        self.compute_kernel_config = linear_compute_config()
        self.rope_compute_kernel_config = hifi4_compute_config()
        self.sdpa_compute_kernel_config = hifi4_compute_config()

    @classmethod
    def from_torch(
        cls,
        mesh_device,
        ref_attn: torch.nn.Module,  # Hugging Face's MiniMaxM3VLVisionAttention: q_proj/k_proj/v_proj/out_proj, head_dim
        padded_head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLAttention":
        """Build from Hugging Face's MiniMaxM3VLVisionAttention."""
        projections = [
            (proj.weight.data, None if proj.bias is None else proj.bias.data)
            for proj in (ref_attn.q_proj, ref_attn.k_proj, ref_attn.v_proj, ref_attn.out_proj)
        ]
        return cls.from_projections(mesh_device, projections, ref_attn.head_dim, padded_head_dim, dtype=dtype)

    @classmethod
    def from_projections(
        cls,
        mesh_device,
        projections: Sequence[Tuple[torch.Tensor, Optional[torch.Tensor]]],
        head_dim: int,
        padded_head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLAttention":
        """Build from the (weight, bias) of the q, k, v and output projections, in that order.

        Weights are [out, in] and biases [out] or None, as nn.Linear holds them. The number of heads is q's output size
        divided by head_dim.
        """
        (q_weight, q_bias), (k_weight, k_bias), (v_weight, v_bias), (out_weight, out_bias) = projections
        num_heads = q_weight.shape[0] // head_dim
        wqkv = _pack_wqkv_weight(q_weight, k_weight, v_weight, num_heads, head_dim, padded_head_dim)
        wqkv_bias = None
        if q_bias is not None:
            wqkv_bias = _pack_wqkv_bias(q_bias, k_bias, v_bias, num_heads, head_dim, padded_head_dim)
        wo = _pad_wo_weight(out_weight, num_heads, head_dim, padded_head_dim)
        return cls(
            mesh_device=mesh_device,
            num_heads=num_heads,
            head_dim=head_dim,
            wqkv_weight=wqkv,
            wo_weight=wo,
            wqkv_bias=wqkv_bias,
            wo_bias=out_bias,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    def _apply_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """Rotate x with the RoPE op in prefill mode: x * cos + (x @ trans_mat) * sin, with trans_mat on each tile."""
        return ttnn.experimental.rotary_embedding_llama(
            x,
            cos,
            sin,
            self.trans_mat,
            is_decode_mode=False,
            compute_kernel_config=self.rope_compute_kernel_config,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        x: ttnn.Tensor,  # (1, 1, L, hidden_size)
        cos: ttnn.Tensor,  # (1, 1, L, padded_head_dim)
        sin: ttnn.Tensor,
        placement: BlockPlacement = DRAM_PLACEMENT,
    ) -> ttnn.Tensor:
        """Self-attention of x. QKV and the heads split from it write placement.qkv, RoPE where its input is, SDPA to
        DRAM and the output projection placement.residual."""
        xqkv = linear(x, self.wqkv, self.wqkv_bias, self.compute_kernel_config, memory_config=placement.qkv)

        q_heads, k_heads, v_heads = ttnn.experimental.nlp_create_qkv_heads(
            xqkv,
            num_heads=self.num_heads,
            num_kv_heads=self.num_heads,
            transpose_k_heads=False,
            memory_config=placement.qkv,
        )
        ttnn.deallocate(xqkv)

        q_heads = self._apply_rope(q_heads, cos, sin)
        k_heads = self._apply_rope(k_heads, cos, sin)

        # Full (non-causal, unmasked) attention, with the heads written side by side as the output projection takes
        # them: (1, 1, L, num_heads * padded_head_dim).
        attn = ttnn.transformer.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_program_config(q_heads.device(), q_heads.shape[-2], self.num_heads),
            compute_kernel_config=self.sdpa_compute_kernel_config,
            output_concat_heads=True,
        )
        ttnn.deallocate(q_heads)
        ttnn.deallocate(k_heads)
        ttnn.deallocate(v_heads)

        out = linear(attn, self.wo, self.wo_bias, self.compute_kernel_config, memory_config=placement.residual)
        ttnn.deallocate(attn)
        return out
