# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``custom_ops/untilize_used`` vs ``ttnn.untilize`` on the regions MoE's expert-output shape (tt-blaze G53F-0203).

``[R, H]`` = ``[67840, 4096]`` bf8 TILE (REGIONS_T 8192 x top-8 + 72 x 32 on the 8x4), random rows with a per-row
scale. For each ``run`` in ``UU_RUNS`` (the device-side end of the used rows; different on every chip through a
per-chip offsets row, so the dynamic split is exercised per chip), ``UntilizeUsed(out, counts, offsets)`` must equal
``ttnn.untilize(out)`` bitwise on rows ``[0, run)`` of every chip and hold zeros on row ``R - 1``. Then both are timed
(``UU_REPS`` back-to-back calls, one synchronize).

    pytest models/demos/glm53_flash/tests/custom_ops/test_untilize_used.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.untilize_used.op import UntilizeUsed
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
R = int(os.environ.get("UU_R", "67840"))
H = 4096
E = 288
RUNS = [int(v) for v in os.environ.get("UU_RUNS", "0,32,4096,18432,24576,67776").split(",")]
REPS = int(os.environ.get("UU_REPS", "10"))


def _tables(run: int, chip: int):
    """counts / offsets rows whose end is ``run`` (+ 32 * chip, so every chip differs); odd chips put the end in the
    last expert's own region (offsets[E-1] + align32(counts[E-1])), even chips past it (counts[E-1] = 0)."""
    end = run + 32 * chip if run else 0
    counts = torch.zeros(1, E, dtype=torch.int32)
    offsets = torch.zeros(1, E, dtype=torch.int32)
    if end and chip % 2:
        counts[0, E - 1] = 17
        offsets[0, E - 1] = end - 32
    else:
        offsets[0, E - 1] = end
    return counts, offsets, end


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_untilize_used(mesh_device):
    D = MESH[0] * MESH[1]
    g = torch.Generator().manual_seed(5)
    x = torch.randn(R, H, generator=g) * torch.exp(torch.randn(R, 1, generator=g))
    out = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    op = UntilizeUsed(mesh_device)
    ref = ttnn.untilize(out)
    ref_h = ttnn.to_torch(ttnn.get_device_tensors(ref)[0]).reshape(R, H)
    ok = True
    for run in RUNS:
        if run + 32 * (D - 1) > R - 32:
            continue
        tabs = [_tables(run, c) for c in range(D)]
        up = lambda k: ttnn.from_torch(
            torch.stack([t[k] for t in tabs]),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        counts, offsets = [ttnn.reshape(up(k), [1, E]) for k in (0, 1)]
        y = op(out, counts, offsets)
        eq = 0
        for c, t in enumerate(ttnn.get_device_tensors(y)):
            end = tabs[c][2]
            got = ttnn.to_torch(t).reshape(R, H)
            rows_eq = torch.equal(got[:end].view(torch.int16), ref_h[:end].view(torch.int16))
            dummy_zero = bool((got[R - 1].view(torch.int16) == 0).all())
            eq += int(rows_eq and dummy_zero)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(op(out, counts, offsets))
        ttnn.synchronize_device(mesh_device)
        t_used = (time.perf_counter() - t0) / REPS * 1e3
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(ttnn.untilize(out))
        ttnn.synchronize_device(mesh_device)
        t_full = (time.perf_counter() - t0) / REPS * 1e3
        logger.info(
            f"[untilize-used] run {run} (+32/chip): used == untilize on rows [0, run) and dummy row zero on "
            f"{eq} / {D} chips; untilize_used {t_used:.3f} ms, ttnn.untilize {t_full:.3f} ms (R {R})"
        )
        ok = ok and eq == D
        for t in (y, counts, offsets):
            ttnn.deallocate(t)
    assert ok, "untilize_used differs from ttnn.untilize on the used rows"
