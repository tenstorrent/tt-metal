# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in lazy tap rows for the target program (M6b C3; M6A_DESIGN_ADDENDUM_1 §2): the D-program's ingest reads the
target's tap rows from a row-major lazy ring (d_ingest_slice_dm.cpp) that the target's slices write after each tapped
layer (kernels/slice_dm_lazy.cpp). model.py and slice_dm.cpp stay unchanged (off is the default build): enable() switches a
built ResidentGemma's slice kernel to slice_dm_lazy.cpp and appends its runtime args, before the program's first launch
or trace capture.

The lazy ring: one DRAM ROW_MAJOR bf16 tensor [slots x 5 taps x 32, 1344] per chip at the same address on every chip;
chip c's row holds columns [1344 c, 1344 (c + 1)) of the tapped layer's x; row (slot x 5 + tap) x 32 + position mod 32
(tap i = the program's i-th tapped layer)."""
import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m7.common import HIDDEN, N_CHIPS

TAPS = 5
QUARTER = HIDDEN // N_CHIPS  # 1344
POSITIONS = 32


class LazyRing:
    def __init__(self, mesh, slots):
        self.mesh, self.slots = mesh, slots
        self.rows = slots * TAPS * POSITIONS
        self.t = ttnn.from_torch(torch.zeros(self.rows, QUARTER), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                                 device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)

    def fill(self, host):
        """every chip's ring from host [rows, 1344]"""
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(host.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
                            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), self.t)
        ttnn.synchronize_device(self.mesh)

    def read(self):
        """[chips, slots, 5 taps, 32 positions, 1344] bf16"""
        return torch.stack([ttnn.to_torch(x).reshape(self.slots, TAPS, POSITIONS, QUARTER)
                            for x in ttnn.get_device_tensors(self.t)])

    @staticmethod
    def row(slot, tap, pos):
        return (slot * TAPS + tap) * POSITIONS + pos % POSITIONS


def enable(model, ring):
    """switch a built (not yet launched / captured) ResidentGemma to the lazy-taps slice kernel writing `ring`"""
    assert len(model.taps_layers) <= TAPS, model.taps_layers
    extra = [ring.t.buffer_address(), model.serve.positions.buffer_address(), model.serve.slots.buffer_address(),
             ring.slots]
    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip, program in enumerate(model.programs):
        n = 0
        for kd in program.kernels:
            if kd.kernel_source.endswith("/slice_dm.cpp"):
                kd.kernel_source = f"{PKG_DIR}/kernels/slice_dm_lazy.cpp"
                for cr in kd.core_ranges.ranges():
                    for x in range(cr.start.x, cr.end.x + 1):
                        for y in range(cr.start.y, cr.end.y + 1):
                            kd.runtime_args[x][y].extend(extra)
                n += 1
        assert n == model.K, (chip, n)
        coord = ttnn.MeshCoordinate(0, chip)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
    model.mesh_pd = mesh_pd
    model.io = list(model.io) + [ring.t]
