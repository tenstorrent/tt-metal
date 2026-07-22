# tt/attention/masks.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""Causal attention mask construction."""

import torch

import ttnn

from ..tst_config import NEG_INF


def build_causal_mask(device, seq_len, batch_size=1):
    """
    Additive causal mask: [batch_size, 1, seq_len, seq_len], 0 where j <= i,
    NEG_INF where j > i.

    batch_size must exactly match the batch dim of the tensor this mask is
    used with — ttnn.transformer.attention_softmax requires an exact match
    and does not broadcast. Build once per sequence length, reuse across
    layers/steps.
    """
    mask = torch.zeros(seq_len, seq_len)
    mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool(), NEG_INF)
    mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    return ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
