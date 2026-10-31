# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Device kernel time of ttnn.hardtanh_bw's fused program against the composite it replaces.

Both run in one profiled process on the same BF16 operands, a gradient in L1 beside an input in
DRAM: without an output placement that call keeps the composite, and with the composite's output
placement it runs the fused program, so both write the same memory. A call's time is the device
kernel duration summed over the programs it launches, the median of REPEATS calls. The fused
program must be faster by more than MARGIN.

The fused program serves the other memory layouts too, where no call of this build runs the
composite; the layout test times each one.
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
PATH = "tests/ttnn/perf_tests/operations/eltwise/test_hardtanh_bw_device_perf.py"


def _calls(tt_g, tt_x):
    composite = ttnn.hardtanh_bw(tt_g, tt_x, min=-1.0, max=1.0)[0]
    placement = composite.memory_config()
    ttnn.deallocate(composite)
    return {
        "composite": lambda: ttnn.hardtanh_bw(tt_g, tt_x, min=-1.0, max=1.0)[0],
        "fused": lambda: ttnn.hardtanh_bw(tt_g, tt_x, min=-1.0, max=1.0, memory_config=placement)[0],
    }


def _bits(tensor):
    return ttnn.to_torch(tensor).to(torch.bfloat16).view(torch.int16)


def _runs_fused(call):
    """Whether ``call`` launches the fused program, as graph capture names it."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    return "UnaryBackwardDeviceOperation" in ttnn.graph.extract_calltrace(graph)


def test_hardtanh_bw_profiled_calls(device):
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
        calls = _calls(tt_g, tt_x)
        # The mixed-placement call returns the bits of the call with both operands in DRAM.
        tt_g_dram = ttnn.from_torch(
            g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        assert torch.equal(_bits(calls["fused"]()), _bits(ttnn.hardtanh_bw(tt_g_dram, tt_x, min=-1.0, max=1.0)[0]))
        ttnn.deallocate(tt_g_dram)
        for path, call in calls.items():
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
def test_hardtanh_bw_device_perf():
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    board = "blackhole" if ttnn.device.is_blackhole() else "wormhole_b0" if ttnn.device.is_wormhole_b0() else None
    if board not in FUSED_BOARDS:
        pytest.skip("this board keeps the composite")
    subdir = "hardtanh_bw_device_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(
        f"pytest {PATH}::test_hardtanh_bw_profiled_calls", subdir, device_analysis_types=["device_kernel_duration"]
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
            f"DEVICE_PERF hardtanh_bw {board} {height}x{width} composite={composite:.0f}ns ({composite_programs} programs) "
            f"fused={fused:.0f}ns ({fused_programs} programs) speedup={composite / fused:.2f}x"
        )
        if fused > composite * (1 - MARGIN):
            slow.append(f"{height}x{width}: fused {fused:.0f} ns, composite {composite:.0f} ns")
    assert not slow, f"the fused program is not {MARGIN:.0%} faster than the composite: {slow}"


def _layouts(shape, device):
    """Per layout the fused program serves beyond interleaved DRAM, the memory config of both operands: shards
    of whole tile rows or columns over up to 64 cores, or over a power of two of the DRAM banks."""
    height, width = shape[-2], shape[-1]
    L1, DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
    layout = ttnn.TensorMemoryLayout
    banks = 1 << (device.dram_grid_size().x.bit_length() - 1)

    def cores(x, y):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})

    def sharded(strategy, buffer, grid, shard):
        return ttnn.MemoryConfig(strategy, buffer, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))

    def nd(buffer, grid):
        spec = ttnn.NdShardSpec(
            ttnn.Shape([1, 1, 128, 128]),
            grid,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        return ttnn.MemoryConfig(buffer, spec)

    rows, columns = min(height // 32, 64), min(width // 32, 64)
    return {
        "interleaved_l1": ttnn.L1_MEMORY_CONFIG,
        "height_sharded": sharded(layout.HEIGHT_SHARDED, L1, cores(8, rows // 8), (height // rows, width)),
        "width_sharded": sharded(layout.WIDTH_SHARDED, L1, cores(8, columns // 8), (height, width // columns)),
        "block_sharded": sharded(layout.BLOCK_SHARDED, L1, cores(8, 8), (height // 8, width // 8)),
        "height_sharded_dram": sharded(layout.HEIGHT_SHARDED, DRAM, cores(banks, 1), (height // banks, width)),
        "width_sharded_dram": sharded(layout.WIDTH_SHARDED, DRAM, cores(banks, 1), (height, width // banks)),
        "nd_sharded": nd(L1, cores(8, 8)),
        "nd_sharded_dram": nd(DRAM, cores(banks, 1)),
    }


LAYOUTS = (
    "interleaved_l1",
    "height_sharded",
    "width_sharded",
    "block_sharded",
    "height_sharded_dram",
    "width_sharded_dram",
    "nd_sharded",
    "nd_sharded_dram",
)


@pytest.mark.parametrize("name", LAYOUTS)
def test_hardtanh_bw_layout_profiled_calls(name, device):
    """A layout's call, both operands in it, between signposts that name it. This build runs every layout on
    the fused program."""
    from tracy import signpost

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in FUSED_BOARDS or not (ttnn.device.is_blackhole(device) or ttnn.device.is_wormhole_b0(device)):
        pytest.skip("this board keeps the composite")
    generator = torch.Generator().manual_seed(0)
    for height, width in SHAPES:
        x = (4 * torch.randn((1, 1, height, width), generator=generator)).to(torch.bfloat16)
        g = torch.randn((1, 1, height, width), generator=generator).to(torch.bfloat16)
        memory = _layouts((height, width), device)[name]
        tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory)
        call = lambda: ttnn.hardtanh_bw(tt_g, tt_x, min=-1.0, max=1.0)[0]
        ttnn.deallocate(call())
        for repeat in range(REPEATS):
            signpost(f"{name}-{height}x{width}-{repeat}-start")
            ttnn.deallocate(call())
            signpost(f"{name}-{height}x{width}-{repeat}-end")
        ttnn.deallocate(tt_g)
        ttnn.deallocate(tt_x)
        ttnn.synchronize_device(device)


@pytest.mark.models_device_performance_bare_metal
def test_hardtanh_bw_layout_device_perf():
    """Each layout's device kernel time per call; the route test checks which program each one runs."""
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, run_device_profiler

    board = "blackhole" if ttnn.device.is_blackhole() else "wormhole_b0" if ttnn.device.is_wormhole_b0() else None
    if board not in FUSED_BOARDS:
        pytest.skip("this board keeps the composite")
    subdir = "hardtanh_bw_layout_device_perf"
    clear_profiler_runtime_artifacts()
    # Every layout runs on the fused program in this build: a failed or missing call fails the test.
    run_device_profiler(
        f"pytest {PATH}::test_hardtanh_bw_layout_profiled_calls",
        subdir,
        check_test_return_code=True,
        device_analysis_types=["device_kernel_duration"],
    )
    regions = _kernel_ns_per_region(get_latest_ops_log_filename(subdir))
    for height, width in SHAPES:
        for name in LAYOUTS:
            assert f"{name}-{height}x{width}-0" in regions, f"no {name} call at {height}x{width} was timed"
            measured = [regions[f"{name}-{height}x{width}-{repeat}"] for repeat in range(REPEATS)]
            assert all(programs for _, programs in measured), f"a {name} call at {height}x{width} launched no program"
            ns = sorted(ns for ns, _ in measured)[REPEATS // 2]
            logger.info(
                f"DEVICE_PERF hardtanh_bw {board} {name} {height}x{width} call={ns:.0f}ns ({measured[0][1]} programs)"
            )
