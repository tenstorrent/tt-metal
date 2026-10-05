# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device kernel time of the packer's ReLU stage against the TT-NN path it replaces.

The reference runs that path in this build: a chain of the op and IDENTITY runs the unary program
with TT-NN's own kernel, since the generated kernel serves a lone op only, and a composite runs its
own TT-NN ops. Before anything is timed the reference must be TT-NN's own path: on FP32 input, which
the generated route refuses, it matches the public op bit for bit and launches the same device
operations, and on a board the route does not install on it matches the public op's BF16 output.
Both paths then run in one profiled process on the same BF16 operands; a call's time is the device
kernel duration summed over the programs it launches, the median of REPEATS calls, and the
generated program must be faster by more than MARGIN.

Each call is also broken down by RISC, to show which processor bounds the program: the kernel
duration of each RISC from its first start to its last end over all cores, and the mean over
cores of each RISC's own kernel zone. ttnn.identity, the copy program with no SFPU work, is timed
beside them as the floor any unary program on these operands can reach, and the calls are also
timed on height-sharded tensors, where every tile is already in its core's L1 and no NoC transfer
bounds the program.
"""

import glob
import os

import pandas as pd
import pytest
import torch
import ttnn
from loguru import logger

SHAPES = [(512, 512), (1024, 1024)]
# Tile rows per core of the height-sharded calls, SHARD_WIDTH wide; timed only to break the calls down.
SHARD_ROWS = (1, 4, 16)
SHARD_WIDTH = 256
REPEATS = 5
MARGIN = 0.05
# Inputs inside the range the generated kernel is fitted on.
LOW, HIGH = -10.0, 10.0
PATH = "tests/ttnn/perf_tests/operations/eltwise/test_pack_relu_device_perf.py"
RISCS = ("BRISC", "NCRISC", "TRISC0", "TRISC1", "TRISC2")
ANALYSES = [
    "device_kernel_duration",
    "device_kernel_duration_per_core",
    "device_kernel_first_to_last_start",
    "device_brisc_kernel_duration",
    "device_ncrisc_kernel_duration",
    "device_trisc0_kernel_duration",
    "device_trisc1_kernel_duration",
    "device_trisc2_kernel_duration",
]

# op: (this change's call, the call of the path it replaces, the boards the change installs on)
CASES = {
    "relu": (
        lambda x: ttnn.relu(x),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu_min": (
        lambda x: ttnn.relu_min(x, 0.0),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MIN, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "threshold": (
        lambda x: ttnn.threshold(x, 0.0, 0.0),
        lambda x: ttnn.unary_chain(
            x,
            [ttnn.UnaryWithParam(ttnn.UnaryOpType.THRESHOLD, 0.0, 0.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)],
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu6": (
        lambda x: ttnn.relu6(x),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
    "relu_max": (
        lambda x: ttnn.relu_max(x, 6.0),
        lambda x: ttnn.unary_chain(
            x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU_MAX, 6.0), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        ),
        ("blackhole", "wormhole_b0"),
    ),
}


def _board(device=None):
    return "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"


def _inputs(shape, dtype, device, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    generator = torch.Generator().manual_seed(0)
    x = LOW + (HIGH - LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))
    return ttnn.from_torch(
        x.to(torch.bfloat16).to(dtype),
        dtype=ttnn.bfloat16 if dtype == torch.bfloat16 else ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )


def _device_operations(call):
    """The result of ``call`` and the device operations it launches, as graph capture names them."""
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = call()
    finally:
        graph = ttnn.graph.end_graph_capture()
    return result, [name for name in ttnn.graph.extract_calltrace(graph) if "DeviceOperation" in name]


def _same_bits(a, b):
    a, b = ttnn.to_torch(a), ttnn.to_torch(b)
    width = torch.int32 if a.dtype == torch.float32 else torch.int16
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(width), b.view(width))


def _prove_reference(name, fused, reference, boards, device):
    """Fail, and time nothing, unless ``reference`` is TT-NN's own path for ``name``."""
    height, width = SHAPES[0]
    x32 = _inputs((1, 1, height, width), torch.float32, device)
    x16 = _inputs((1, 1, height, width), torch.bfloat16, device)
    own, own_operations = _device_operations(lambda: fused(x32))
    mine, my_operations = _device_operations(lambda: reference(x32))
    assert _same_bits(mine, own), f"{name}: on FP32 input the reference differs from TT-NN's own path; not timing"
    _, my_bf16_operations = _device_operations(lambda: reference(x16))
    assert (
        my_bf16_operations == own_operations
    ), f"{name}: the reference launches {my_bf16_operations}, TT-NN's own path {own_operations}; not timing"
    if _board(device) not in boards:
        assert _same_bits(
            reference(x16), fused(x16)
        ), f"{name}: on a board that keeps TT-NN's path the reference differs from it"
    logger.info(f"REFERENCE {name} {_board(device)} is TT-NN's own path: {len(own_operations)} device operations")


def _profile(name, path, call, x, memory, height, width):
    from tracy import signpost

    ttnn.deallocate(call(x))
    for repeat in range(REPEATS):
        signpost(f"{name}-{path}-{memory}-{height}x{width}-{repeat}-start")
        ttnn.deallocate(call(x))
        signpost(f"{name}-{path}-{memory}-{height}x{width}-{repeat}-end")


def _layouts(device):
    """(memory, shape, memory config) of every profiled call; the margin is checked on SHAPES in DRAM."""
    grid = device.compute_with_storage_grid_size()
    layouts = [("dram", (height, width), ttnn.DRAM_MEMORY_CONFIG) for height, width in SHAPES]
    for rows in SHARD_ROWS:
        shard = (32 * rows, SHARD_WIDTH)
        config = ttnn.create_sharded_memory_config(
            shape=shard,
            core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        layouts.append(("sharded", (shard[0] * grid.x * grid.y, SHARD_WIDTH), config))
    return layouts


def test_pack_relu_profiled_calls(device):
    """The proofs, then the calls the device-perf test profiles, between signposts that name each one."""
    if not (ttnn.device.is_blackhole(device) or ttnn.device.is_wormhole_b0(device)):
        pytest.skip("the generated kernels exist for Blackhole and Wormhole only")
    for name, (fused, reference, boards) in CASES.items():
        _prove_reference(name, fused, reference, boards, device)
    for memory, (height, width), memory_config in _layouts(device):
        x = _inputs((1, 1, height, width), torch.bfloat16, device, memory_config)
        _profile("identity", "copy", lambda x: ttnn.identity(x), x, memory, height, width)
        for name, (fused, reference, boards) in CASES.items():
            if _board(device) not in boards:
                continue
            for path, call in (("old", reference), ("fused", fused)):
                _profile(name, path, call, x, memory, height, width)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        ttnn.deallocate(x)


def _regions(csv_path):
    """Per pair of signposts, the GLOBAL CALL COUNT and analysis columns of each program between them."""
    regions, current = {}, None
    for _, row in pd.read_csv(csv_path).iterrows():
        if row["OP TYPE"] == "signpost":
            code = row["OP CODE"]
            current = code.removesuffix("-start") if code.endswith("-start") else None
            if current is not None:
                regions[current] = []
        elif current is not None and str(row["DEVICE KERNEL DURATION [ns]"]) not in ("-", "nan"):
            regions[current].append(row)
    return regions


def _calls(regions):
    """The profiled calls, each the signpost name of its repeats without the repeat index, in profiled order."""
    return list(dict.fromkeys(region.rsplit("-", 1)[0] for region in regions))


def _column(rows, column):
    values = [float(row[column]) for row in rows if column in row and str(row[column]) not in ("-", "nan")]
    return sum(values) if values else float("nan")


def _core_zones(device_log):
    """Per run host ID, each RISC's kernel zone in ns: mean and max over cores, and the core count."""
    with open(device_log) as log:
        header = log.readline()
    # A simulator reports no clock; its zones are then in cycles.
    mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0]) or 1000.0
    log = pd.read_csv(device_log, skiprows=1, skipinitialspace=True)
    log.columns = [column.strip() for column in log.columns]
    log = log[log["zone name"].isin(["BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"])]
    start = log[log["type"] == "ZONE_START"].groupby(["run host ID", "core_x", "core_y", "RISC processor type"])
    end = log[log["type"] == "ZONE_END"].groupby(["run host ID", "core_x", "core_y", "RISC processor type"])
    zone = (end["time[cycles since reset]"].max() - start["time[cycles since reset]"].min()) * 1000.0 / mhz
    zone = zone.rename("ns").reset_index()
    out = {}
    for (run, risc), group in zone.groupby(["run host ID", "RISC processor type"]):
        out.setdefault(int(run), {})[risc.replace("_", "")] = (group["ns"].mean(), group["ns"].max(), len(group))
    return out


def _breakdown(regions, device_log, board):
    """Log, per op, memory and shape, each path's median per-RISC durations."""
    try:
        zones = _core_zones(device_log) if device_log else {}
    except Exception as error:  # the breakdown is diagnostic; a parse failure must not hide the timing
        logger.warning(f"BREAKDOWN no per-core zones from {device_log}: {error}")
        zones = {}

    def median(values):
        values = sorted(v for v in values if v == v)
        return values[len(values) // 2] if values else float("nan")

    for key in _calls(regions):
        name, path, memory, shape = key.split("-")
        calls = [regions.get(f"{key}-{r}", []) for r in range(REPEATS)]
        if not all(calls):
            logger.info(f"BREAKDOWN {name} {path} {board} {memory} {shape} not profiled")
            continue
        first = median(_column(c, "DEVICE KERNEL DURATION [ns]") for c in calls)
        cells = [f"kernel={first:.0f}"]
        for label, column in (
            ("core_avg", "DEVICE KERNEL DURATION PER CORE AVG [ns]"),
            ("core_max", "DEVICE KERNEL DURATION PER CORE MAX [ns]"),
            ("start_skew", "DEVICE KERNEL FIRST TO LAST START [ns]"),
        ):
            cells.append(f"{label}={median(_column(c, column) for c in calls):.0f}")
        for risc in RISCS:
            span = median(_column(c, f"DEVICE {risc} KERNEL DURATION [ns]") for c in calls)
            cells.append(f"{risc}={span:.0f}")
        per_core = []
        for risc in RISCS:
            means, maxes, cores = [], [], 0
            for c in calls:
                runs = [zones.get(int(row["GLOBAL CALL COUNT"]), {}).get(risc) for row in c]
                if runs and all(runs):
                    means.append(sum(r[0] for r in runs))
                    maxes.append(sum(r[1] for r in runs))
                    cores = runs[0][2]
            if means:
                per_core.append(f"{risc}={median(means):.0f}/{median(maxes):.0f}")
        if per_core:
            cells.append(f"per-core mean/max over {cores} cores: " + " ".join(per_core))
        logger.info(f"BREAKDOWN {name} {path} {board} {memory} {shape} ns " + " ".join(cells))


def _cb_waits(regions, board):
    def median(values):
        values = sorted(v for v in values if v == v)
        return values[len(values) // 2] if values else float("nan")

    for key in _calls(regions):
        name, path, memory, shape = key.split("-")
        calls = [regions.get(f"{key}-{r}", []) for r in range(REPEATS)]
        if not all(calls):
            continue
        cells = [
            f"{label}={median(_column(c, column) for c in calls):.0f}"
            for label, column in (
                ("kernel", "DEVICE KERNEL DURATION [ns]"),
                ("unpack_wait_front_sum", "DEVICE COMPUTE CB WAIT FRONT [ns]"),
                ("pack_reserve_back_sum", "DEVICE COMPUTE CB RESERVE BACK [ns]"),
            )
        ]
        logger.info(f"CBWAIT {name} {path} {board} {memory} {shape} ns " + " ".join(cells))


@pytest.mark.models_device_performance_bare_metal
def test_pack_relu_device_perf():
    from tracy.common import clear_profiler_runtime_artifacts
    from tracy.process_model_log import get_latest_ops_log_filename, get_profiler_folder, run_device_profiler

    board = _board()
    subdir = "pack_relu_device_perf"
    clear_profiler_runtime_artifacts()
    run_device_profiler(f"pytest {PATH}::test_pack_relu_profiled_calls", subdir, device_analysis_types=ANALYSES)
    timed = [name for name, (_, _, boards) in CASES.items() if board in boards]
    for name in CASES:
        if name not in timed:
            logger.info(f"DEVICE_PERF {name} {board} keeps TT-NN's path; the reference equals it")
    regions = _regions(get_latest_ops_log_filename(subdir))
    logs = glob.glob(os.path.join(str(get_profiler_folder(subdir)), "**", "profile_log_device.csv"), recursive=True)
    _breakdown(regions, logs[0] if logs else None, board)
    # A second run counts, per call, the cycles the unpacker waits for input tiles and the packer for
    # output space, summed over cores: the time compute spends starved by data movement.
    run_device_profiler(
        f"pytest {PATH}::test_pack_relu_profiled_calls",
        f"{subdir}_cb",
        device_analysis_types=[
            "device_kernel_duration",
            "device_compute_cb_wait_front",
            "device_compute_cb_reserve_back",
        ],
        sum_profiling=True,
    )
    _cb_waits(_regions(get_latest_ops_log_filename(f"{subdir}_cb")), board)
    if not timed:
        return
    slow = []
    for name in timed:
        for height, width in SHAPES:
            per_call = {}
            for path in ("old", "fused"):
                measured = [regions[f"{name}-{path}-dram-{height}x{width}-{repeat}"] for repeat in range(REPEATS)]
                assert all(measured), f"a {path} call of {name} launched no program"
                ns = [_column(rows, "DEVICE KERNEL DURATION [ns]") for rows in measured]
                per_call[path] = (sorted(ns)[REPEATS // 2], len(measured[0]))
            (old, old_programs), (fused, fused_programs) = per_call["old"], per_call["fused"]
            logger.info(
                f"DEVICE_PERF {name} {board} {height}x{width} old={old:.0f}ns ({old_programs} programs) "
                f"fused={fused:.0f}ns ({fused_programs} programs) speedup={old / fused:.2f}x"
            )
            if fused > old * (1 - MARGIN):
                slow.append(f"{name} {height}x{width}: fused {fused:.0f} ns, old {old:.0f} ns")
    assert not slow, f"the generated program is not {MARGIN:.0%} faster than the path it replaces: {slow}"
