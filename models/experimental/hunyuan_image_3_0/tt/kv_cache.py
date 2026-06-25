# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Per-layer K/V cache for HunyuanImage-3.0 autoregressive decode on TTNN.
# Mirrors upstream HunyuanStaticCache (modeling_hunyuan_image_3.py) at a minimal
# subset: append-only keys/values per decoder layer, used after a prefix prefill.

from __future__ import annotations

import ttnn


class HunyuanTtKvCache:
    """Device-resident KV tensors per layer (pre-GQA-expansion layout)."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.keys: list[ttnn.Tensor | None] = [None] * num_layers
        self.values: list[ttnn.Tensor | None] = [None] * num_layers
        self.seq_len: int = 0

    def get(self, layer_idx: int):
        return self.keys[layer_idx], self.values[layer_idx]

    def replace(self, layer_idx: int, key: ttnn.Tensor, value: ttnn.Tensor) -> None:
        old_k, old_v = self.keys[layer_idx], self.values[layer_idx]
        if old_k is not None:
            ttnn.deallocate(old_k)
        if old_v is not None:
            ttnn.deallocate(old_v)
        self.keys[layer_idx] = key
        self.values[layer_idx] = value

    def clear(self) -> None:
        for i in range(self.num_layers):
            if self.keys[i] is not None:
                ttnn.deallocate(self.keys[i])
            if self.values[i] is not None:
                ttnn.deallocate(self.values[i])
            self.keys[i] = None
            self.values[i] = None
        self.seq_len = 0
