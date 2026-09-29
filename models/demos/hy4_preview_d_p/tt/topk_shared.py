# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared top-k (Hy4 moe_shared layers 2-4): a shared layer has no indexer and attends over the top-k of the latest
full layer (cfg.topk_source(i), layer 1 for layers 2-4) for the same chunk.

components.yaml tags the step NATIVE, no op: the module hands on the source layer's device top-k tensor as it is
([1, 1, S/2, 2048] uint32 ROW_MAJOR per chip, row-split over axis 0, replicated over axis 1, unsorted within a row,
0xFFFFFFFF sentinel tail), the layout TtHy4Indexer emits and TtHy4Attention / sparse_sdpa consume. Same idea as
deepseek_v3_d_p's ReuseIndexer (cross-layer top-k reuse, glm_5_2_config indexer_types). The caller keeps the source
tensor alive until the next full layer replaces it; this module never frees or copies it.
"""

from __future__ import annotations

import ttnn


class TtTopkShared:
    """Identity on the device: returns the latest full layer's top-k tensor. No op, no host transfer."""

    def __init__(self, mesh, source_layer: int):
        self.mesh, self.source_layer = mesh, source_layer

    def __call__(self, shared_topk: ttnn.Tensor) -> ttnn.Tensor:
        return shared_topk
