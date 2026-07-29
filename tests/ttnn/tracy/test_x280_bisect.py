# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

# TEMP bisect harness for the tracy-capture SIGSEGV on multi-core X280 captures.
# Varies core-grid (context count) and replay count (volume) independently so we can tell
# whether the crash is driven by raw marker volume or by the number of Tracy contexts.
# Remove after diagnosis.

import pytest
import torch
import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 1996800, "dispatch_core_type": ttnn.DispatchCoreType.WORKER}],
    indirect=True,
)
@pytest.mark.parametrize(
    "gy, gx, capture, replay",
    [
        (1, 1, 100, 5),  # 1 ctx, low volume (baseline, expect OK)
        (8, 8, 100, 5),  # 64 ctx, high volume (== test_with_ops, expect CRASH)
        (1, 1, 100, 320),  # 1 ctx, HIGH volume (~matches 8x8 total invocations)
        (4, 4, 100, 5),  # 16 ctx, mid volume
    ],
    ids=["1x1_v-low", "8x8_v-high", "1x1_v-high", "4x4_v-mid"],
)
def test_grid_sweep(device, gy, gx, capture, replay):
    torch.manual_seed(0)
    m = k = n = 1024
    a = ttnn.to_layout(
        ttnn.to_device(
            ttnn.from_torch(torch.randn((m, k), dtype=torch.bfloat16)), device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.TILE_LAYOUT,
    )
    b = ttnn.to_layout(
        ttnn.to_device(
            ttnn.from_torch(torch.randn((k, n), dtype=torch.bfloat16)), device, memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.TILE_LAYOUT,
    )
    grid = ttnn.CoreGrid(y=gy, x=gx)
    ttnn.matmul(a, b, core_grid=grid)
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(capture):
        ttnn.matmul(a, b, core_grid=grid)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    for _ in range(replay):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    ttnn.release_trace(device, tid)
