# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The regions MoE's expert-output conversion (tt-blaze G53F-0192), on the 8x4 mesh at its shape ([max_buf, 4096] bf8
TILE, max_buf 67840 at REGIONS_T 8192): ``ttnn.typecast(bf16)`` + ``ttnn.to_layout(ROW_MAJOR)`` (today) vs one
``ttnn.untilize`` of the bf8 tensor (bf16 rows out). Checks bitwise equality on every chip and times both.

    UB_ROWS=67840 pytest models/demos/glm53_flash/tests/test_untilize_bf8_direct.py -s
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.tt.mesh import mesh_device_params

MESH = tuple(int(v) for v in os.environ.get("GLM53_MESH_SHAPE", "8,4").split(","))
ROWS = int(os.environ.get("UB_ROWS", "67840"))
H = int(os.environ.get("UB_H", "4096"))
REPS = int(os.environ.get("UB_REPS", "5"))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("device_params", [mesh_device_params(*MESH, trace_region_size=0)], indirect=True)
@pytest.mark.parametrize("mesh_device", [MESH], indirect=True)
def test_untilize_bf8_direct(mesh_device):
    g = torch.Generator().manual_seed(0)
    x_h = (torch.randn(ROWS, H, generator=g) * torch.exp(torch.randn(ROWS, 1, generator=g))).to(torch.bfloat16)
    x = ttnn.from_torch(
        x_h,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def two():
        t = ttnn.typecast(x, ttnn.bfloat16)
        y = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(t)
        return y

    def direct():
        return ttnn.untilize(x)

    a, b = two(), direct()
    assert b.dtype == ttnn.bfloat16 and b.layout == ttnn.ROW_MAJOR_LAYOUT, (b.dtype, b.layout)
    ad, bd = ttnn.get_device_tensors(a), ttnn.get_device_tensors(b)
    eq = [torch.equal(ttnn.to_torch(p).view(torch.int16), ttnn.to_torch(q).view(torch.int16)) for p, q in zip(ad, bd)]
    ttnn.deallocate(a)
    ttnn.deallocate(b)

    def timed(fn):
        for _ in range(2):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(REPS):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) / REPS * 1e3

    t_two, t_direct = timed(two), timed(direct)
    logger.info(
        f"[untilize-bf8] [{ROWS}, {H}] bf8: direct == typecast+to_layout bitwise on {sum(eq)}/{len(eq)} chips; "
        f"two ops {t_two:.3f} ms, direct {t_direct:.3f} ms per call"
    )
    assert all(eq), eq
