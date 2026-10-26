# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""tt-blaze G53F-0223: the prefill MoE's closing TP all-reduce over the mesh COLUMNS (``GLMMesh.all_reduce_cols`` =
``ttnn.all_reduce(cluster_axis=1)``, [1,1,2048,4096] bf16 per chip, 3.4-4.0 ms in G53F-0217's device profile) vs an
explicit async reduce-scatter + all-gather on the column axis, Linear and Ring, on the 8x4 mesh. Times REPS calls (one
synchronize) and compares each variant with the composite bitwise on all 32 chips (a different sum order is NOT
bitwise; the gather half only moves bytes).

    pytest models/demos/glm53_flash/tests/test_ccl_cols_bench.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import TP_AXIS, GLMMesh, mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
REPS = int(os.environ.get("CCL_REPS", "20"))
ROWS = int(os.environ.get("CCL_COLS_ROWS", "2048"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_ccl_cols_bench(mesh_device):
    g = torch.Generator().manual_seed(11)
    lin = GLMMesh.from_device(mesh_device)
    ring = GLMMesh.from_device(mesh_device, topology=ttnn.Topology.Ring)
    # every chip its own partial: a [rows*R, cols*4096] host tensor split over rows (dim 2) and columns (dim 3)
    R, C = MESH
    full = torch.randn(1, 1, ROWS * R, 4096 * C, generator=g).to(torch.bfloat16)
    x = ttnn.from_torch(
        full,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH, dims=(2, 3)),
    )
    mc = x.memory_config()

    def rsag(m_rs, m_ag):
        def f():
            p = m_rs._reduce_scatter_async(x, 3, TP_AXIS, mc)
            y = m_ag._all_gather_async(p, 3, TP_AXIS, mc)
            ttnn.deallocate(p)
            return y

        return f

    cases = {
        "composite (all_reduce_cols)": lambda: lin.all_reduce_cols(x),
        "rs linear + ag linear": rsag(lin, lin),
        "rs linear + ag ring": rsag(lin, ring),
        "rs ring + ag ring": rsag(ring, ring),
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

    ref = None
    for name, fn in cases.items():
        try:
            y = fn()
            outs = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(y)]
            ttnn.deallocate(y)
            t = timed(fn)
        except Exception as e:  # noqa: BLE001 -- an unsupported variant is a result
            logger.warning(f"[ccl-cols] {name}: {e!r}"[:400])
            continue
        if ref is None:
            ref = outs
            want = full.reshape(1, 1, ROWS * R, C, 4096).float().sum(3)  # fp32 sum of the 4 partials, per row slab
            d = max((outs[r * C].float() - want[:, :, r * ROWS : (r + 1) * ROWS]).abs().max().item() for r in range(R))
            logger.info(f"[ccl-cols] {name}: {t:.3f} ms; max |d| vs fp32 host sum {d:.4g}")
            continue
        eq = sum(torch.equal(a, b) for a, b in zip(ref, outs))
        logger.info(f"[ccl-cols] {name}: {t:.3f} ms; == composite on {eq} / {len(outs)} chips")
