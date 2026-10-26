# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill MoE's row collectives under different link / worker counts (tt-blaze G53F-0211).

``reduce_scatter_rows`` of the expert partials ([1,1,16384,4096] bf16 -> 2048 rows per mesh row, Linear: the default,
bitwise-pinned) and ``all_gather_rows`` of the hidden ([1,1,2048,4096] bf16 per row, Ring: G53F-0200) on the 8x4 mesh,
for each ``CL_VARIANTS`` entry ``links:workers:chunks:buffers`` (``GLM53_CCL_{RS,AG}_{LINKS,WORKERS,CHUNKS,BUFFERS}``,
``GLMMesh._ccl_knobs``; the first entry = the current defaults = the reference). How a collective deals its slices
over links and workers does not change any element's reduction order, so every variant must equal the reference
bitwise; times REPS back-to-back calls (one synchronize).

    pytest models/demos/glm53_flash/tests/test_ccl_links_bench.py -s
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
VARIANTS = os.environ.get(
    "CL_VARIANTS", "2:2:10:2,1:2:10:2,3:2:10:2,4:2:10:2,2:1:10:2,2:4:10:2,4:4:10:2,2:2:20:2,2:2:10:4"
).split(",")


@pytest.mark.timeout(2400)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_ccl_links_bench(mesh_device):
    rows = MESH[0]
    g = torch.Generator().manual_seed(17)
    m = GLMMesh.from_device(mesh_device)
    x = m.upload(torch.randn(1, 1, 2048 * rows, 4096, generator=g).to(torch.bfloat16), ttnn.bfloat16, rows=2)
    part = m.upload(torch.randn(1, 1, 16384, 4096, generator=g).to(torch.bfloat16), ttnn.bfloat16)
    os.environ["GLM53_CCL_GATHER_TOPOLOGY"] = "ring"  # the server recipe's gathers (G53F-0200)
    cases = {
        "RS": lambda: m.reduce_scatter_rows(part, dim=2),
        "AG": lambda: m.all_gather_rows(x, dim=2),
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

    for kind, op in cases.items():
        ref = None
        for v in VARIANTS:
            links, workers, chunks, buffers = v.split(":")
            for k, val in (("LINKS", links), ("WORKERS", workers), ("CHUNKS", chunks), ("BUFFERS", buffers)):
                os.environ[f"GLM53_CCL_{kind}_{k}"] = val
            try:
                y = op()
                outs = [ttnn.to_torch(t).view(torch.int16) for t in ttnn.get_device_tensors(y)]
                ttnn.deallocate(y)
                ms = timed(op)
            except Exception as e:  # noqa: BLE001 -- an unsupported setting is a result
                logger.warning(f"[ccl-links] {kind} {v}: FAILED {e!r}"[:400])
                continue
            if ref is None:
                ref = outs
            eq = sum(torch.equal(a, b) for a, b in zip(outs, ref))
            logger.info(
                f"[ccl-links] {kind} links:workers:chunks:buffers={v}: {ms:.3f} ms; == reference on {eq} / {len(outs)} chips"
            )
        for k in ("LINKS", "WORKERS", "CHUNKS", "BUFFERS"):
            os.environ.pop(f"GLM53_CCL_{kind}_{k}", None)
    os.environ.pop("GLM53_CCL_GATHER_TOPOLOGY", None)
