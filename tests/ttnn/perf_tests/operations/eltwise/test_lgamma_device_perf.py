# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device kernel time of ttnn.lgamma's generated program.

The generated program serves a lone op, and TT-NN runs no other call through the kernel it replaces,
so that path is timed head against merge-base on the same chip (perf-compare, the CI perf gate), not
here. A call's time is the device kernel duration summed over the programs it launches, the median
of REPEATS calls.
"""

import pandas as pd
import pytest
import torch
import ttnn
from loguru import logger

SHAPES = [(512, 512), (1024, 1024)]
REPEATS = 5
# Inputs inside the range the generated kernel is fitted on.
LOW, HIGH = 0.001, 100.0
PATH = "tests/ttnn/perf_tests/operations/eltwise/test_lgamma_device_perf.py"

# op: (this change's call, the boards the change installs on)
CASES = {
    "lgamma": (lambda x: ttnn.lgamma(x), ("blackhole",)),
}


def _board(device=None):
    return "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"


def _inputs(shape, device):
    generator = torch.Generator().manual_seed(0)
    x = LOW + (HIGH - LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))
    return ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def test_lgamma_profiled_calls(device):
    """The calls the device-perf test profiles, between signposts that name each one."""
    from tracy import signpost

    if not (ttnn.device.is_blackhole(device) or ttnn.device.is_wormhole_b0(device)):
        pytest.skip("the generated kernels exist for Blackhole and Wormhole only")
    for name, (fused, boards) in CASES.items():
        if _board(device) not in boards:
            continue
        for height, width in SHAPES:
            x = _inputs((1, 1, height, width), device)
            ttnn.deallocate(fused(x))
            for repeat in range(REPEATS):
                signpost(f"{name}-{height}x{width}-{repeat}-start")
                ttnn.deallocate(fused(x))
                signpost(f"{name}-{height}x{width}-{repeat}-end")
            ttnn.synchronize_device(device)


def _kernel_ns_per_region(csv_path):
    """Per pair of signposts, the device kernel duration summed over the programs between them, and their count."""
    regions, current = {}, None
    for _, row in pd.read_csv(csv_path).iterrows():
        if row["OP TYPE"] == "signpost":
            code = row["OP CODE"]
            current = code.removesuffix("-start") if code.endswith("-start") else None
            if current is not None:
                regions[current] = [0.0, 0]
        elif current is not None and str(row["DEVICE KERNEL DURATION [ns]"]) not in ("-", "nan"):
            regions[current][0] += float(row["DEVICE KERNEL DURATION [ns]"])
            regions[current][1] += 1
    return regions


@pytest.mark.models_device_performance_bare_metal
def test_lgamma_device_perf():
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    board = _board()
    subdir = "lgamma_device_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(
        f"pytest {PATH}::test_lgamma_profiled_calls", subdir, device_analysis_types=["device_kernel_duration"]
    )
    timed = [name for name, (_, boards) in CASES.items() if board in boards]
    if not timed:
        return
    regions = _kernel_ns_per_region(get_latest_ops_log_filename(subdir))
    for name in timed:
        for height, width in SHAPES:
            measured = [regions[f"{name}-{height}x{width}-{repeat}"] for repeat in range(REPEATS)]
            assert all(programs for _, programs in measured), f"a call of {name} launched no program"
            fused = sorted(ns for ns, _ in measured)[REPEATS // 2]
            logger.info(f"DEVICE_PERF {name} {board} {height}x{width} fused={fused:.0f}ns ({measured[0][1]} programs)")
