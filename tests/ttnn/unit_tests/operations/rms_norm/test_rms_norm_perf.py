# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device kernel-time measurement for rms_norm (not a correctness gate; perf is printed, never asserted).

Run with the device profiler on:
    TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TT_METAL_PROFILER_CPP_POST_PROCESS=1 \
        scripts/run_safe_pytest.sh tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py -s
Without the profiler env the test only runs the op once per shape.
"""

import os

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm.rms_norm_program_descriptor import OpGeometry, select_blocking

SHAPES = [
    (1, 1, 32, 4096),
    (1, 1, 32, 8192),
    (1, 1, 32, 16384),
    (1, 1, 128, 8192),
    (4, 8, 32, 256),
    (1, 1, 2048, 256),
    (2, 512, 1024),
    (1024, 1024),
    (4, 1, 512, 512),
]


def _device_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    data = ttnn.get_latest_programs_perf_data()
    out = []
    for _chip, programs in (data or {}).items():
        for program in programs:
            res = program.program_analyses_results.get("DEVICE KERNEL DURATION [ns]")
            if res is not None:
                out.append((int(res.duration), int(getattr(program, "core_count", 0) or 0)))
    return out


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_rms_norm_perf(device, shape, dtype, layout):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    g = torch.randn((1, 1, 1, shape[-1]), dtype=torch.float32).to(torch.bfloat16)
    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(g, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = rms_norm(tx, gamma=tg)
    ttnn.synchronize_device(device)
    if os.environ.get("TT_METAL_DEVICE_PROFILER") != "1":
        return
    _device_kernel_ns(device)  # drain (includes compile run / from_torch programs)
    out = rms_norm(tx, gamma=tg)
    ttnn.synchronize_device(device)
    res = _device_kernel_ns(device)
    W = shape[-1]
    lead = 1
    for d in shape[:-1]:
        lead *= d
    nbytes = 2 * lead * W * 2
    ns = res[-1][0] if res else None
    print(
        f"\nPERF rms_norm {shape} {layout}: device_kernel_ns={ns} cores={res[-1][1] if res else None} "
        f"GB/s(x+y)={nbytes / ns if ns else None:.1f}"
        if ns
        else f"\nPERF rms_norm {shape}: no data {res}"
    )
