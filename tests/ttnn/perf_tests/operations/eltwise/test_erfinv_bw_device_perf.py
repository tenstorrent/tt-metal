# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Device kernel time of ttnn.erfinv_bw's fused program against the composite it replaces.

Both run in one profiled process on the same BF16 operands, a gradient in L1 beside an input in
DRAM: without an output placement that call keeps the composite, and with a DRAM output it runs
the fused program. A call's time is the device kernel duration summed over the programs it
launches, the median of REPEATS calls. The fused program must be faster by more than MARGIN.
"""

import pandas as pd
import pytest
import torch
import ttnn
from loguru import logger

SHAPES = [(512, 512), (1024, 1024)]
REPEATS = 5
MARGIN = 0.05
# The boards whose calls the fused program serves.
FUSED_BOARDS = (
    "blackhole",
    "wormhole_b0",
)
PATH = "tests/ttnn/perf_tests/operations/eltwise/test_erfinv_bw_device_perf.py"


def _calls(tt_g, tt_x):
    return {
        "composite": lambda: ttnn.erfinv_bw(tt_g, tt_x)[0],
        "fused": lambda: ttnn.erfinv_bw(tt_g, tt_x, memory_config=ttnn.DRAM_MEMORY_CONFIG)[0],
    }


def _runs_fused(call):
    """Whether ``call`` launches the fused program, as graph capture names it."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    return "UnaryBackwardDeviceOperation" in ttnn.graph.extract_calltrace(graph)


def test_erfinv_bw_profiled_calls(device):
    """The calls the device-perf test profiles, between signposts that name each one."""
    from tracy import signpost

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in FUSED_BOARDS or not (ttnn.device.is_blackhole(device) or ttnn.device.is_wormhole_b0(device)):
        pytest.skip("this board keeps the composite")
    generator = torch.Generator().manual_seed(0)
    for height, width in SHAPES:
        x = (4 * torch.randn((1, 1, height, width), generator=generator)).to(torch.bfloat16)
        g = torch.randn((1, 1, height, width), generator=generator).to(torch.bfloat16)
        tt_g = ttnn.from_torch(
            g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tt_x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        for path, call in _calls(tt_g, tt_x).items():
            # Graph capture runs the call once, which also compiles its programs.
            assert _runs_fused(call) == (path == "fused"), f"the {path} call ran the other path"
            for repeat in range(REPEATS):
                signpost(f"{path}-{height}x{width}-{repeat}-start")
                ttnn.deallocate(call())
                signpost(f"{path}-{height}x{width}-{repeat}-end")
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
def test_erfinv_bw_device_perf():
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    board = "blackhole" if ttnn.device.is_blackhole() else "wormhole_b0"
    if board not in FUSED_BOARDS:
        pytest.skip("this board keeps the composite")
    subdir = "erfinv_bw_device_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(
        f"pytest {PATH}::test_erfinv_bw_profiled_calls", subdir, device_analysis_types=["device_kernel_duration"]
    )
    regions = _kernel_ns_per_region(get_latest_ops_log_filename(subdir))
    slow = []
    for height, width in SHAPES:
        per_call = {}
        for path in ("composite", "fused"):
            measured = [regions[f"{path}-{height}x{width}-{repeat}"] for repeat in range(REPEATS)]
            assert all(programs for _, programs in measured), f"a {path} call at {height}x{width} launched no program"
            per_call[path] = (sorted(ns for ns, _ in measured)[REPEATS // 2], measured[0][1])
        (composite, composite_programs), (fused, fused_programs) = per_call["composite"], per_call["fused"]
        logger.info(
            f"DEVICE_PERF erfinv_bw {board} {height}x{width} composite={composite:.0f}ns ({composite_programs} programs) "
            f"fused={fused:.0f}ns ({fused_programs} programs) speedup={composite / fused:.2f}x"
        )
        if fused > composite * (1 - MARGIN):
            slow.append(f"{height}x{width}: fused {fused:.0f} ns, composite {composite:.0f} ns")
    assert not slow, f"the fused program is not {MARGIN:.0%} faster than the composite: {slow}"
