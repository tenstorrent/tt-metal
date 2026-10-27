# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Additive attention masks for the bidirectional encoder.

HF semantics (`masking_utils`):
    full layers     keep (q, k)  iff  key k is a valid (non-padding) token
    sliding layers  keep (q, k)  iff  key k is valid  and  |q - k| <= sliding_window (512, inclusive)

TT SDPA cannot combine its `sliding_window_size` argument with an `attn_mask`
(`sdpa_device_operation.cpp:141`), so both rules are baked into one dense additive mask:

    key_bias [B, 1, 1, S] = (keep - 1) * 1e9              built per forward from the `keep` input
    band     [1, 1, S, S] = 0 if |q - k| <= W else -1e9   constant per bucket, built at setup
    sliding  [B, 1, S, S] = band + key_bias
    full     [B, 1, S, S] = repeat(key_bias, S rows)

Masked entries hold -1e9 or -2e9 (finite; a fully masked padded query row averages padded values
instead of producing NaN, and padded rows are never read back). Valid query rows always keep
themselves, so their softmax is well defined.

Contract
    prepare(S)        setup-time, host -> device upload of the band for bucket S (call before forward/trace)
    forward(keep)     keep [B, 1, 1, S] bf16 TILE (1 = valid token, 0 = padding), device-only ops
                      returns {"sliding_attention": [B,1,S,S], "full_attention": [B,1,S,S]} bf16 TILE DRAM
"""

from __future__ import annotations

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.embedding_gemma2.tt.config import FULL, MASK_NEG, SLIDING


def band_mask(seq_len: int, radius: int) -> torch.Tensor:
    idx = torch.arange(seq_len)
    dist = (idx[:, None] - idx[None, :]).abs()
    band = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    band[dist > radius] = MASK_NEG
    return band.reshape(1, 1, seq_len, seq_len)


class AttentionMasks(LightweightModule):
    def __init__(self, device, sliding_window: int):
        super().__init__()
        self.device = device
        self.sliding_window = sliding_window
        self.bands: dict[int, ttnn.Tensor | None] = {}

    def prepare(self, seq_len: int) -> None:
        if seq_len in self.bands:
            return
        if seq_len <= self.sliding_window + 1:
            # every |q - k| <= S - 1 <= W: the band masks nothing
            self.bands[seq_len] = None
            return
        self.bands[seq_len] = ttnn.from_torch(
            band_mask(seq_len, self.sliding_window),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def key_bias(self, keep: ttnn.Tensor) -> ttnn.Tensor:
        # (keep - 1) * 1e9 : 0 for valid keys, -1e9 for padding
        return ttnn.multiply(ttnn.subtract(keep, 1.0), -MASK_NEG, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def forward(self, keep: ttnn.Tensor) -> dict[str, ttnn.Tensor]:
        B, _, _, S = keep.shape
        if S not in self.bands:
            raise RuntimeError(f"AttentionMasks.prepare({S}) must run before forward (setup-time upload)")
        bias = self.key_bias(keep)
        full = ttnn.repeat(bias, ttnn.Shape([1, 1, S, 1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        band = self.bands[S]
        if band is None:
            sliding = full
        else:
            sliding = ttnn.add(full, band, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(bias)
        return {SLIDING: sliding, FULL: full}
