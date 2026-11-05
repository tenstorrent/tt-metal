# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phi-3.5-mini rotary setup: ``RotarySetup1D`` with row-major copies of the cos/sin tables for the lookups.

``ttnn.embedding`` reads a row-major table. ``RotarySetup1D`` keeps cos/sin tiled (prefill slices them as tiles), so
every lookup first untilized the whole 4096 x 96 table: 5.2 us per table per decode step on a P150, twice a step.
Here the tables are also kept row-major (2 x 786 KB of DRAM), converted once when the weights load, and the decode
and prefill lookups read those.
"""

from __future__ import annotations

from typing import List

import ttnn
from models.common.modules.rope.rope_1d import RotarySetup1D


class Phi35MiniRotarySetup(RotarySetup1D):
    def load_device_weights(self):
        if self._device_weights_loaded:
            return
        super().load_device_weights()
        self.cos_matrix_rm = ttnn.to_layout(self.cos_matrix, ttnn.ROW_MAJOR_LAYOUT)
        self.sin_matrix_rm = ttnn.to_layout(self.sin_matrix, ttnn.ROW_MAJOR_LAYOUT)

    def decode_forward(self, rot_idxs: ttnn.Tensor) -> List[ttnn.Tensor]:
        """``RotarySetup1D.decode_forward`` with the lookups on the row-major tables."""
        self.load_device_weights()
        cfg = self.config
        if rot_idxs.device != cfg.device:
            rot_idxs = ttnn.to_device(rot_idxs, cfg.device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = []
        for table in (self.cos_matrix_rm, self.sin_matrix_rm):
            t = ttnn.embedding(rot_idxs, table, layout=ttnn.TILE_LAYOUT)  # [1, batch, head_dim]
            t = ttnn.transpose(ttnn.unsqueeze_to_4D(t), 1, 2)  # [1, batch, 1[32], head_dim]
            t = t[:, : cfg.batch_size_per_device_group, :, :]
            out.append(ttnn.interleaved_to_sharded(t, cfg.cos_sin_shard_mem_config))
        return out
