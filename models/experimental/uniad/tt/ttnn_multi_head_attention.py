# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional
import torch
from torch import Tensor

import ttnn


def _canonical_mask(
    mask: Optional[Tensor],
    mask_name: str,
    other_type,
    other_name: str,
    target_type,
    check_other: bool = True,
) -> Optional[Tensor]:
    if mask is not None:  #  Condition not satisfied
        _mask_is_float = torch.is_floating_point(mask)
        if not _mask_is_float:
            mask = torch.zeros_like(mask, dtype=target_type)
            mask = mask.masked_fill_(mask, float("-inf"))

    return mask


def _in_projection_packed(
    q: Tensor, k: Tensor, v: Tensor, w: Tensor, b: Optional[Tensor] = None, embed_dims=256
) -> list[Tensor]:
    w_q, w_k, w_v = ttnn.chunk(w, 3, dim=0)
    b_q, b_k, b_v = ttnn.chunk(b, 3, dim=0)

    w_q, b_q = ttnn.permute(w_q, (1, 0)), ttnn.reshape(b_q, (1, -1))
    w_k, b_k = ttnn.permute(w_k, (1, 0)), ttnn.reshape(b_k, (1, -1))
    w_v, b_v = ttnn.permute(w_v, (1, 0)), ttnn.reshape(b_v, (1, -1))

    a = ttnn.linear(q, w_q, bias=b_q)
    b = ttnn.linear(k, w_k, bias=b_k)
    c = ttnn.linear(v, w_v, bias=b_v)
    return a, b, c


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    device=None,
) -> tuple[Tensor, Optional[Tensor]]:
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=None,
        other_name="attn_mask",
        target_type=torch.float32,
    )

    if is_causal and key_padding_mask is None and not need_weights:
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            is_causal = False

    head_dim = embed_dim // num_heads

    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias, embed_dims=embed_dim)
    else:
        raise ValueError("This implementation won't work")

    # q, k, v PC: 0.9999 ATOL: 0.04

    q = ttnn.transpose(ttnn.reshape(q, (tgt_len, bsz * num_heads, head_dim)), 0, 1)
    k = ttnn.transpose(ttnn.reshape(k, (k.shape[0], bsz * num_heads, head_dim)), 0, 1)
    v = ttnn.transpose(ttnn.reshape(v, (v.shape[0], bsz * num_heads, head_dim)), 0, 1)

    # update source sequence length after adjustments
    src_len = k.shape[1]

    if key_padding_mask is not None:
        key_padding_mask = ttnn.from_torch(
            key_padding_mask, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
        )

        key_padding_mask = ttnn.reshape(key_padding_mask, (bsz, 1, 1, src_len))
        key_padding_mask = ttnn.expand(key_padding_mask, (-1, num_heads, -1, -1))
        key_padding_mask = ttnn.reshape(key_padding_mask, (bsz * num_heads, 1, src_len))
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    if need_weights:  # True
        _B, _Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))

        assert not (is_causal and attn_mask is None), "FIXME: is_causal not implemented for need_weights"

        if attn_mask is not None:
            qk = ttnn.matmul(q_scaled, ttnn.transpose(k, -2, -1))
            attn_output_weights = qk + attn_mask  # Ensure attn_mask is of shape [B, N, M] or broadcastable
        else:
            attn_output_weights = ttnn.matmul(q_scaled, ttnn.transpose(k, -2, -1))

        attn_output_weights = ttnn.to_layout(attn_output_weights, layout=ttnn.TILE_LAYOUT)

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)
        attn_output = ttnn.matmul(attn_output_weights, v)  # [B, N, D]

        attn_output = ttnn.transpose(attn_output, 0, 1)
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))
        attn_output = ttnn.linear(attn_output, out_proj_weight, bias=out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))

        # optionally average attention weights over heads
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, num_heads, tgt_len, src_len))
        if average_attn_weights:
            attn_output_weights = ttnn.mean(attn_output_weights, dim=1)

        return attn_output, attn_output_weights


class TtMultiheadAttention:
    def __init__(
        self,
        device,
        parameters,
        embed_dim,
        num_heads,
        need_weights=True,
        attn_mask=None,
        is_causal=False,
        average_attn_weights=True,
        batch_first=False,
    ):
        self.device = device
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.in_proj_weight = ttnn.to_layout(parameters.in_proj_weight, ttnn.TILE_LAYOUT)
        self.in_proj_bias = ttnn.to_layout(parameters.in_proj_bias, ttnn.TILE_LAYOUT)
        self.out_proj_weight = parameters.out_proj.weight
        self.out_proj_bias = parameters.out_proj.bias

        # base_config
        self.need_weights = need_weights
        self.attn_mask = attn_mask
        self.is_causal = is_causal
        self.average_attn_weights = average_attn_weights
        self.batch_first = batch_first

    def __call__(self, query, key, value, key_padding_mask=None):
        if self.batch_first:
            query = ttnn.transpose(query, 0, 1)
            key = ttnn.transpose(key, 0, 1)
            value = ttnn.transpose(value, 0, 1)
        attn_output, _ = multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=self.in_proj_weight,
            in_proj_bias=self.in_proj_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            out_proj_weight=self.out_proj_weight,
            out_proj_bias=self.out_proj_bias,
            training=False,
            key_padding_mask=key_padding_mask,
            need_weights=self.need_weights,
            attn_mask=self.attn_mask,
            use_separate_proj_weight=False,
            q_proj_weight=None,
            k_proj_weight=None,
            v_proj_weight=None,
            static_k=None,
            static_v=None,
            average_attn_weights=self.average_attn_weights,
            is_causal=self.is_causal,
            device=self.device,
        )

        if self.batch_first:
            attn_output = ttnn.transpose(attn_output, 0, 1)
        return attn_output, None
