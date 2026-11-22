# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy tap rows v2 (M6b-2a levers; lazy_taps.py stays the verified C3 version): kernels/slice_dm_lazy_v2.cpp with
  lever 0  a per-row mask row (DRAM [1, 1, 1, 32] int32, nonzero = the row's slot drafts), read once per launch: a launch
           with no drafting row (c > 16: nobody drafts) skips the tap work
  lever 1  deferred: each tapped layer's gather + row writes run in the next round, before its slot wait, staged in the
           post-norm weight CB's idle half
enable_v2() switches a built (not yet launched / captured) ResidentGemma to it, as lazy_taps.enable does."""
import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m7.lazy_taps import TAPS


class MaskRow:
    """the per-row drafting mask (the hub feed's rows: 32 int32)"""

    def __init__(self, mesh, values):
        self.mesh = mesh
        self.t = ttnn.from_torch(torch.tensor(values, dtype=torch.int32).reshape(1, 1, 1, 32), dtype=ttnn.int32,
                                 layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(mesh)

    def set(self, values):
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(torch.tensor(values, dtype=torch.int32).reshape(1, 1, 1, 32), dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh)), self.t)


def enable_v2(model, ring, mask=None, deferred=False):
    assert len(model.taps_layers) <= TAPS, model.taps_layers
    extra = [ring.t.buffer_address(), model.serve.positions.buffer_address(), model.serve.slots.buffer_address(),
             ring.slots, mask.t.buffer_address() if mask is not None else 0, int(deferred)]
    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip, program in enumerate(model.programs):
        n = 0
        for kd in program.kernels:
            if kd.kernel_source.endswith("/slice_dm.cpp"):
                kd.kernel_source = f"{PKG_DIR}/kernels/slice_dm_lazy_v2.cpp"
                for cr in kd.core_ranges.ranges():
                    for x in range(cr.start.x, cr.end.x + 1):
                        for y in range(cr.start.y, cr.end.y + 1):
                            kd.runtime_args[x][y].extend(extra)
                n += 1
        assert n == model.K, (chip, n)
        coord = ttnn.MeshCoordinate(0, chip)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
    model.mesh_pd = mesh_pd
    model.io = list(model.io) + [ring.t] + ([mask.t] if mask is not None else [])
