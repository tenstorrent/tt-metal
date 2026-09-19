# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""`rgb_to_yuv(..., wide_rows=True)` writes the same bytes as the (1, H, W, T) planes, laid out as (1, H, W*T) rows: one chip,
the strips path's per-device shard (3, 192, 168, 28) and a small odd shape; every plane compared with torch.equal after
viewing the wide rows back, both forms timed."""

import time

import pytest
import torch
from loguru import logger

import ttnn

SINGLE_DEVICE = [pytest.param((1, 1), {"l1_small_size": 65536}, id="single_device")]
SHAPES = [pytest.param(192, 168, 28, id="strips_shard"), pytest.param(64, 40, 12, id="small")]


def _timed(mesh_device, fn, n=10):
    out = fn()
    ttnn.synchronize_device(mesh_device)
    best = float("inf")
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        mark = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(mesh_device)
        best = min(best, time.perf_counter() - mark)
    return out, best * 1e3


@pytest.mark.timeout(600)
@pytest.mark.parametrize(("h", "w", "t"), SHAPES)
@pytest.mark.parametrize(("mesh_device", "device_params"), SINGLE_DEVICE, indirect=["mesh_device", "device_params"])
def test_wide_rows_match_planes(mesh_device, h, w, t):
    torch.manual_seed(0)
    rgb = ttnn.from_torch(
        (torch.rand(3, h, w, t) * 2 - 1).to(torch.bfloat16), device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16
    )
    coefficients = ttnn.experimental.yuv_bt601_coefficients()
    planes, ms_plain = _timed(mesh_device, lambda: ttnn.experimental.rgb_to_yuv(rgb, coefficients=coefficients))
    wide, ms_wide = _timed(mesh_device, lambda: ttnn.experimental.rgb_to_yuv(rgb, coefficients=coefficients, wide_rows=True))
    for name, plain_t, wide_t in zip(("Y", "Cb", "Cr"), planes, wide):
        ref = ttnn.to_torch(plain_t)
        got = ttnn.to_torch(wide_t)
        assert tuple(got.shape) == (1, ref.shape[1], ref.shape[2] * ref.shape[3]), f"{name}: wide shape {tuple(got.shape)}"
        got = got.reshape(ref.shape)
        n_diff = int((got != ref).sum())
        logger.info(f"YUVWIDE {name} ({h},{w},{t}): plain {ms_plain:.3f} ms, wide {ms_wide:.3f} ms; differing bytes {n_diff}")
        assert torch.equal(got, ref), f"{name}: {n_diff} bytes differ between the wide rows and the planes"
