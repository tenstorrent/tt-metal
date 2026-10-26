# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where ``custom_ops/topk_rows`` spends its time in the DSA prefill shape (tt-blaze G53F-0213): one call over B score
rows of pitch W (the scorer's bucketed width) with every row bounded to ``valid`` pools, for B in ``TKS_B`` and valid in
``TKS_VALID`` (W = valid rounded up to 16384 pools, the prefill's pools bucket), on one chip. The prefill ranks
2048 rows per DSA layer-chunk in 4 calls of 512 rows: G53F-0210's stage split has 11.7 ms per layer-chunk at 128k
depth (~32 K pools per row) and 32.3 ms at 1M (~254 K), against ~1.14 ms per 262 K-wide row on one core (README).

    TT_VISIBLE_DEVICES=0 pytest models/demos/glm53_flash/tests/custom_ops/test_topk_rows_sweep.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.custom_ops.topk_rows.op import TopkRows

K = 512
BUCKET = 16384
BS = [int(v) for v in os.environ.get("TKS_B", "1,128,512").split(",")]
VALIDS = [int(v) for v in os.environ.get("TKS_VALID", "1024,8192,16384,32768,65536,131072,262144").split(",")]
REPS = int(os.environ.get("TKS_REPS", "10"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_topk_rows_sweep(mesh_device):
    g = torch.Generator().manual_seed(5)
    op = TopkRows(mesh_device)
    grid = mesh_device.compute_with_storage_grid_size()
    logger.info(f"[topk-sweep] grid {grid.x}x{grid.y} = {grid.x * grid.y} cores")
    for B in BS:
        for valid in VALIDS:
            W = -(-valid // BUCKET) * BUCKET
            scores = torch.randn(B, 1, 1, W, generator=g).bfloat16()
            s = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            v = ttnn.from_torch(
                torch.full((B, 1), valid, dtype=torch.int64),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh_device,
            )
            o = ttnn.zeros((B, 1, 1, K), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
            op(s, v, o)
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            for _ in range(REPS):
                op(s, v, o)
            ttnn.synchronize_device(mesh_device)
            ms = (time.perf_counter() - t0) / REPS * 1e3
            rows_per_core = -(-B // (grid.x * grid.y))
            logger.info(
                f"[topk-sweep] B={B} valid={valid} W={W}: {ms:.3f} ms per call, {ms / rows_per_core:.3f} ms per row "
                f"({rows_per_core} rows per core), {valid / 512:.0f} chunks of 512"
            )
            for t in (s, v, o):
                ttnn.deallocate(t)
