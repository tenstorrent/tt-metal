# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from typing import Optional


def to_causal_4d(
    batch_size: int,
    query_length: int,
    key_value_length: int,
    dtype,
    device="cpu",
    sliding_window: Optional[int] = None,
):
    input_shape = (batch_size, query_length)
    past_key_values_length = key_value_length - query_length
    causal_4d_mask = None
    if input_shape[-1] > 1 or sliding_window is not None:
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            device=device,
            past_key_values_length=past_key_values_length,
        )
    return causal_4d_mask


def to_4d(
    attention_mask_2d,
    query_length: int,
    dtype,
    key_value_length: Optional[int] = None,
    sliding_window: Optional[int] = None,
    is_causal: Optional[int] = None,
):
    input_shape = (attention_mask_2d.shape[0], query_length)
    causal_4d_mask = None
    if (input_shape[-1] > 1 or sliding_window is not None) and is_causal:
        past_key_values_length = key_value_length - query_length
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            device=attention_mask_2d.device,
            past_key_values_length=past_key_values_length,
        )

    expanded_attn_mask = _expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1]).to(attention_mask_2d.device)

    if causal_4d_mask is not None:
        expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), torch.finfo(dtype).min)

    expanded_4d_mask = expanded_attn_mask

    return expanded_4d_mask


def _make_causal_mask(
    input_ids_shape,
    dtype,
    device,
    past_key_values_length: int = 0,
):
    bsz, tgt_len = input_ids_shape
    mask = ttnn.full((tgt_len, tgt_len), torch.finfo(dtype).min)
    mask_cond = torch.arange(mask.shape[-1], device=device)
    mask = ttnn.to_torch(mask)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


def _expand_mask(mask, dtype, tgt_len: Optional[int] = None):
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _prepare_4d_causal_attention_mask(
    attention_mask,
    input_shape,
    inputs_embeds,
    past_key_values_length: int,
):
    is_causal = True
    key_value_length = input_shape[-1] + past_key_values_length
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            is_causal=is_causal,
        )
    else:
        attention_mask = to_causal_4d(
            input_shape[0],
            input_shape[-1],
            key_value_length,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

    return attention_mask
