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

The separate q/k/v projections are packed host-side into a single wqkv
here, and SDPA is plain full bidirectional attention (the tower passes
attention_mask=None, no cu_seqlens).

head_dim=80 is not tile-aligned, so q/k/v are padded to 96 across wqkv,
the heads, and wo input. Padded slots hold zeros: they add nothing to
QK^T, multiply zero-rows of wo, and (with cos=1/sin=0) are untouched by
RoPE. Scale stays 1/sqrt(80): padded dims contribute 0 to the dot product.

RoPE is one ttnn.experimental.rotary_embedding_llama call for q and one for
k. That op rotates each pair of lanes (2i, 2i+1) of a head, while rotate_half
pairs dims i and i + 39, so the q and k rows of wqkv (and of its bias) are
reordered at load with utils/weight_conversion.meta_partial_permute, as the
text model's are: lane 2i holds dim i, lane 2i+1 dim i + 39, and the 2
pass-through dims follow at lanes 78-79. q and k share the order, so QK^T is
unchanged; v and wo keep the reference order. The cos/sin of
`rope.rope_cos_sin_padded` follow the same order, with cos=1/sin=0 on the
pass-through and pad lanes.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.tensor_utils import get_rot_transformation_mat
from models.demos.minimax_m3.reference.vision import rope_axis_dim
from models.demos.minimax_m3.tt.vision.common import (
    as_linear_bias,
    hifi4_compute_config,
    linear,
    linear_compute_config,
    sdpa_program_config,
    to_device,
)
from models.demos.minimax_m3.utils.weight_conversion import meta_partial_permute


def _rope_head_order(w: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """q or k rows of a weight or bias in the head order that rotary_embedding_llama rotates (module docstring):
    meta_partial_permute over the 3 * rope_axis_dim(head_dim) = 78 dims that rotate."""
    return meta_partial_permute(w, num_heads, head_dim, 3 * rope_axis_dim(head_dim))


def _pack_wqkv_weight(
    q_w: torch.Tensor,  # [hidden, hidden] each (out = num_heads*head_dim)
    k_w: torch.Tensor,
    v_w: torch.Tensor,
    num_heads: int,
    head_dim: int,
    padded_head_dim: int,
) -> torch.Tensor:
    """Pack separate q/k/v Linear weights into one head_dim-padded wqkv.

    The q and k rows of each head are in the order of _rope_head_order, which rotary_embedding_llama rotates; v keeps
    the reference order. Returns ttnn-linear convention: [hidden_in, 3*num_heads*padded_head_dim].
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
    """Pack separate q/k/v Linear biases into one bias of the packed wqkv: _pack_wqkv_weight on one-column tensors.
    Returns [3*num_heads*padded_head_dim]."""
    packed = _pack_wqkv_weight(q_b[:, None], k_b[:, None], v_b[:, None], num_heads, head_dim, padded_head_dim)
    return packed.reshape(-1)


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
        ref_attn: torch.nn.Module,  # VisionAttention (reference/vision.py): q_proj/k_proj/v_proj/out_proj, head_dim
        padded_head_dim: int,
        dtype=ttnn.bfloat16,
    ) -> "M3VLAttention":
        """head_dim is ref_attn's, the head count follows from q_proj, and q, k and v heads are padded to
        padded_head_dim lanes (module docstring)."""
        head_dim = ref_attn.head_dim
        num_heads = ref_attn.q_proj.weight.shape[0] // head_dim
        wqkv = _pack_wqkv_weight(
            ref_attn.q_proj.weight.data,
            ref_attn.k_proj.weight.data,
            ref_attn.v_proj.weight.data,
            num_heads,
            head_dim,
            padded_head_dim,
        )
        has_bias = ref_attn.q_proj.bias is not None
        wqkv_bias = (
            _pack_wqkv_bias(
                ref_attn.q_proj.bias.data,
                ref_attn.k_proj.bias.data,
                ref_attn.v_proj.bias.data,
                num_heads,
                head_dim,
                padded_head_dim,
            )
            if has_bias
            else None
        )
        wo = _pad_wo_weight(ref_attn.out_proj.weight.data, num_heads, head_dim, padded_head_dim)
        wo_bias = ref_attn.out_proj.bias.data if ref_attn.out_proj.bias is not None else None
        return cls(
            mesh_device=mesh_device,
            num_heads=num_heads,
            head_dim=head_dim,
            wqkv_weight=wqkv,
            wo_weight=wo,
            wqkv_bias=wqkv_bias,
            wo_bias=wo_bias,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    def _apply_rope(self, x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
        """x*cos + (x @ trans_mat)*sin in one op, trans_mat applied to each tile of the head dim (prefill mode)."""
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
    ) -> ttnn.Tensor:
        xqkv = linear(x, self.wqkv, self.wqkv_bias, self.compute_kernel_config)

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

        out = linear(attn, self.wo, self.wo_bias, self.compute_kernel_config)
        ttnn.deallocate(attn)
        return out
