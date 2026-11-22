# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lazy tap rows, lever 2 (M6b; lazy_taps.py is the verified C3 version, lazy_levers.py levers 0 / 1): the slices'
NCRISC (kernels/slice_mc_lazy.cpp, in place of slice_mc.cpp, the second h multicaster) writes the lazy rows straight
from the x tiles after a tapped layer's h is out, so the BRISC (slice_dm.cpp, unchanged: lazy off) never waits on them.
The per-row drafting mask (lever 0) is the default: a launch with no drafting row writes nothing. enable_v3() switches
a built (not yet launched / captured) ResidentGemma; it needs the slices' second multicaster (mc_split > 0)."""
import torch

import ttnn
from models.demos.gemma4_31b_qb2.resident_m7 import PKG_DIR
from models.demos.gemma4_31b_qb2.resident_m7.lazy_taps import TAPS


def enable_v3(model, ring, mask):
    assert len(model.taps_layers) <= TAPS, model.taps_layers
    mesh = model.mesh
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in model.slices])
    scratch = ttnn.from_torch(
        torch.zeros(len(model.slices), 128, dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                                        ttnn.ShardSpec(grid, [1, 128], ttnn.ShardOrientation.ROW_MAJOR)))
    tap_mask = sum(1 << l for l in model.taps_layers)
    mesh_pd = ttnn.MeshProgramDescriptor()
    for chip, program in enumerate(model.programs):
        n = 0
        for kd in program.kernels:
            if kd.kernel_source.endswith("/slice_mc.cpp"):
                kd.kernel_source = f"{PKG_DIR}/kernels/slice_mc_lazy.cpp"
                extra = [ring.t.buffer_address(), model.serve.positions.buffer_address(),
                         model.serve.slots.buffer_address(), mask.t.buffer_address() if mask is not None else 0,
                         ring.slots, scratch.buffer_address(), chip, model.K, model.T, model.R, model.M,
                         tap_mask & 0xFFFFFFFF, tap_mask >> 32]
                for cr in kd.core_ranges.ranges():
                    for x in range(cr.start.x, cr.end.x + 1):
                        for y in range(cr.start.y, cr.end.y + 1):
                            kd.runtime_args[x][y].extend(extra)
                n += 1
        assert n == model.K, f"lever 2 needs the slices' second multicaster (mc_split > 0): chip {chip} has {n}"
        coord = ttnn.MeshCoordinate(0, chip)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = program
    model.mesh_pd = mesh_pd
    model.io = list(model.io) + [ring.t, scratch] + ([mask.t] if mask is not None else [])
    model._lazy_scratch = scratch
