# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3.5-mini on-device sampling: ``Sampling1D`` with one top-k call for both vocabulary halves on one device.

On a single device ``Sampling1D`` splits the vocabulary in half and runs ``ttnn.topk`` on each half, then puts the two
halves' candidates side by side ([user, half 0's top k | half 1's top k]) for ``ttnn.sampling``. Each top-k call over
16032 columns costs about 83 us on a P150 (route prep, the multi-core top-k and its finish), almost all of it per call
rather than per row. Here the two halves are stacked as rows ([2 x users, half]) and one top-k call (86 us) serves
both; slicing the two row blocks back out and concatenating them gives exactly the candidates and order the parent
produces, so everything after top-k (index offsets, seeding, ``ttnn.sampling``) is unchanged.
"""

from __future__ import annotations

import ttnn
from models.common.modules.sampling.sampling_1d import Sampling1D


class Phi35MiniSampling(Sampling1D):
    def _topk_single_device(self, x_bf16):
        cfg = self.config
        rows = x_bf16.shape[2]
        k = cfg.max_top_k
        halves = ttnn.split(x_bf16, x_bf16.shape[-1] // 2, dim=3)
        stacked = ttnn.concat(halves, dim=2)  # [1, 1, 2 * rows, vocab / 2]: half h in rows [h * rows, (h + 1) * rows)
        for half in halves:
            half.deallocate()
        values, indices = ttnn.topk(stacked, k=k, dim=-1, sub_core_grids=cfg.sub_core_grid_topk)
        ttnn.deallocate(stacked)
        return self._side_by_side(values, rows, k), self._side_by_side(indices, rows, k)

    @staticmethod
    def _side_by_side(t, rows: int, k: int):
        """[1, 1, 2 * rows, k] (half 0 rows, then half 1 rows) -> [1, 1, rows, 2 * k] (half 0 | half 1 per row)."""
        top = ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, k])
        bottom = ttnn.slice(t, [0, 0, rows, 0], [1, 1, 2 * rows, k])
        ttnn.deallocate(t)
        out = ttnn.concat([top, bottom], dim=3)
        ttnn.deallocate(top)
        ttnn.deallocate(bottom)
        return out
