# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill's row-axis collectives under the mesh's default async topology (Linear) vs Ring (tt-blaze G53F-0200):
the MoE's ``all_gather_rows`` of the 16 K chunk's hidden ([1,1,2048,4096] bf16 per row -> 16384 rows), of its routing
lists ([1,1,2048,32] fp32) and the ``reduce_scatter_rows`` of the expert partials ([1,1,16384,4096] bf16 -> 2048 rows),
on the 8x4 mesh (2D torus fabric: the row axis is a physical ring). Times REPS back-to-back calls (one synchronize);
compares each Ring output with the Linear one bitwise on every chip (a gather moves bytes, so it must match; a
reduce-scatter's sum order may differ).

    pytest models/demos/glm53_flash/tests/test_ccl_rows_bench.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
REPS = int(os.environ.get("CCL_REPS", "20"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_ccl_rows_bench(mesh_device):
    rows = MESH[0]
    g = torch.Generator().manual_seed(9)
    meshes = {
        "linear": GLMMesh.from_device(mesh_device),
        "ring": GLMMesh.from_device(mesh_device, topology=ttnn.Topology.Ring),
    }
    base = meshes["linear"]
    x = base.upload(torch.randn(1, 1, 2048 * rows, 4096, generator=g).to(torch.bfloat16), ttnn.bfloat16, rows=2)
    lists = base.upload(torch.randn(1, 1, 2048 * rows, 32, generator=g), ttnn.float32, rows=2)
    part = base.upload(torch.randn(1, 1, 16384, 4096, generator=g).to(torch.bfloat16), ttnn.bfloat16)
    cases = {
        "ag_hidden": lambda m: m.all_gather_rows(x, dim=2),
        "ag_lists": lambda m: m.all_gather_rows(lists, dim=2),
        "rs_partials": lambda m: m.reduce_scatter_rows(part, dim=2),
    }

    def timed(fn):
        for _ in range(3):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    for name, op in cases.items():
        outs, times = {}, {}
        for topo, m in meshes.items():
            try:
                y = op(m)
                outs[topo] = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(y)]
                ttnn.deallocate(y)
                times[topo] = timed(lambda m=m: op(m))
            except Exception as e:  # noqa: BLE001 -- an unsupported topology is a result
                times[topo] = float("nan")
                logger.warning(f"[ccl-rows] {name} {topo}: {e!r}"[:400])
        eq = None
        if len(outs) == 2:
            eq = sum(torch.equal(a, b) for a, b in zip(outs["linear"], outs["ring"]))
        logger.info(
            f"[ccl-rows] {name}: linear {times.get('linear', float('nan')):.3f} ms, ring {times.get('ring', float('nan')):.3f} ms; "
            f"ring == linear on {eq} / {len(outs.get('linear', []))} chips"
        )
