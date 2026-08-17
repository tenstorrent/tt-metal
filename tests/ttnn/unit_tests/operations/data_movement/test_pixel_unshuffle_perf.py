# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Tracy device-profiler harness for ttnn.pixel_unshuffle.
#
# Compares, at the real BEV pixel-unshuffle shapes, the DEVICE KERNEL DURATION of:
#   - TILE-native   : TILE in  -> TILE out  (native TILE gather kernel, no round-trip)
#   - RM-native     : RM   in  -> RM   out  (row-major stick kernel)
#   - round-trip    : TILE in  -> to_layout(RM) -> RM kernel -> to_layout(TILE)
#
# Run with the Tracy device profiler (fast-runtime-mode OFF so op zones are
# captured), then read DEVICE KERNEL DURATION [ns] from the generated report:
#
#   TTNN_CONFIG_OVERRIDES='{"enable_fast_runtime_mode": false}' \
#       python -m tracy -r -p -v -m pytest \
#       tests/ttnn/unit_tests/operations/data_movement/test_pixel_unshuffle_perf.py
#
# Report CSV: generated/profiler/reports/<ts>/ops_perf_results_<ts>.csv
# (column "DEVICE KERNEL DURATION [ns]" per op; the round-trip case shows three
#  ops — untilize + pixel_unshuffle + tilize — sum them to compare.)

import pytest
import torch
import ttnn

# (name, N, C, H, W, r) — real BEV pipeline shapes.
SHAPES = [
    ("bev_uv", 1, 2, 768, 768, 2),  # -> [1, 8, 384, 384]
    ("bev_y", 1, 1, 1536, 1536, 4),  # -> [1,16, 384, 384]
]

ITERS = 8  # invocations per case (profiler aggregates/averages per op)


def _mk(device, layout, N, C, H, W):
    x = torch.randn(N, C, H, W, dtype=torch.bfloat16)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("name,N,C,H,W,r", SHAPES, ids=[s[0] for s in SHAPES])
def test_perf_tile_native(device, name, N, C, H, W, r):
    """TILE in -> TILE out via the native TILE kernel (no round-trip)."""
    t = _mk(device, ttnn.TILE_LAYOUT, N, C, H, W)
    for _ in range(ITERS):
        _ = ttnn.pixel_unshuffle(t, downscale_factor=r)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("name,N,C,H,W,r", SHAPES, ids=[s[0] for s in SHAPES])
def test_perf_rm_native(device, name, N, C, H, W, r):
    """ROW_MAJOR in -> ROW_MAJOR out via the row-major stick kernel."""
    t = _mk(device, ttnn.ROW_MAJOR_LAYOUT, N, C, H, W)
    for _ in range(ITERS):
        _ = ttnn.pixel_unshuffle(t, downscale_factor=r)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("name,N,C,H,W,r", SHAPES, ids=[s[0] for s in SHAPES])
def test_perf_roundtrip(device, name, N, C, H, W, r):
    """Old path: TILE -> untilize -> RM kernel -> tilize (3 device ops)."""
    t = _mk(device, ttnn.TILE_LAYOUT, N, C, H, W)
    for _ in range(ITERS):
        rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        o = ttnn.pixel_unshuffle(rm, downscale_factor=r)
        _ = ttnn.to_layout(o, ttnn.TILE_LAYOUT)
    ttnn.synchronize_device(device)
