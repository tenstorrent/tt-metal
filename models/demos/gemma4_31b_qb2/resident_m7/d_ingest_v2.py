# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The ingest program (d_ingest.py, C1, frozen) for every ring slot. C1's harness kept the lazy tap rows of slot 0
only: its lazy tensor is [5 x 32, 1344] per chip and _lazy_host places rows at slot 0, while the kernel reads row
(slot * 5 + tap) * 32 + position mod 32 (d_ingest_slice_dm.cpp), so a launch for slot s > 0 read past the tensor.
IngestProgramSlots sizes the lazy tensor for all the ring's slots and writes each launch's rows at its slot (the
target's lazy-tap layout, M6A addendum 1 / lazy_taps.py); the kernels are unchanged."""
import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7.common import HIDDEN, TILE
from models.demos.gemma4_31b_qb2.resident_m7.d_ingest import IngestProgram
from models.demos.gemma4_31b_qb2.resident_m7.drafter_common import D_TAPS, TAP_COLS


class IngestProgramSlots(IngestProgram):
    def __init__(self, mesh, weights, vectors, ring, **kw):
        self._slot, self._slots = 0, ring.slots
        super().__init__(mesh, weights, vectors, ring, **kw)

    def _lazy_host(self, taps, positions=None):
        """taps [5, rows, HIDDEN] at `positions` of slot self._slot -> per chip [slots x 5 x 32, 1344] (row
        (slot * 5 + tap) * 32 + position mod 32; chip c: its 1344 columns of each tap)"""
        lz = torch.zeros(self._slots, D_TAPS, TILE, HIDDEN, dtype=torch.bfloat16)
        if positions is not None:
            for r, p in enumerate(positions):
                lz[self._slot, :, p % TILE] = taps[:, r].bfloat16()
        shards = [ttnn.from_torch(lz[..., c * TAP_COLS : (c + 1) * TAP_COLS].reshape(-1, TAP_COLS).contiguous(),
                                  dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for c in range(self.n)]
        return ttnn.from_host_shards(shards, ttnn.MeshShape(1, self.n))

    def set_rows(self, taps, positions, slot=0):
        assert 0 <= slot < self._slots, (slot, self._slots)
        self._slot = slot
        super().set_rows(taps, positions, slot)
