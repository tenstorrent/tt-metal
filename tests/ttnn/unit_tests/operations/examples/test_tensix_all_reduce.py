# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling for rectangular Tensix all-reduce."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import datetime
import socket
import statistics
import subprocess
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.examples.tensix_all_reduce import (
    ALL_ROW_VECTOR_VARIANTS,
    ROW_VECTOR_VARIANTS,
    VARIANTS,
    all_reduce,
    build_group_layout,
    build_member_layout,
    create_row_vector_memory_config,
    create_sharded_memory_config,
    row_sum_all_reduce,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
_DEVICE_CSV = Path(os.environ.get("TT_METAL_HOME", ".")) / "generated/profiler/.logs/profile_log_device.csv"


def _chip_freq_mhz():
    """AICLK the profiler stamped on its device log (header line), or None if there is no log yet."""
    try:
        with open(_DEVICE_CSV) as f:
            header = f.readline()
    except OSError:
        return None
    for part in header.split(","):
        if "CHIP_FREQ" in part:
            return float(part.split(":")[1])
    return None


def _selected_variants():
    selected = tuple(part for part in os.environ.get("AR_VARIANTS", ",".join(VARIANTS)).split(",") if part)
    unknown = set(selected) - set(VARIANTS)
    if unknown:
        raise ValueError(f"unknown AR_VARIANTS: {sorted(unknown)}")
    return selected


def _custom_case():
    raw = os.environ.get("AR_GROUP_SHAPE")
    if raw is None:
        return None
    rows, cols = (int(part) for part in raw.split(","))
    return ("custom", (rows, cols), int(os.environ.get("AR_NUM_GROUPS", "1")))


def _default_perf_cases(device):
    grid = device.compute_with_storage_grid_size()
    half_x = max(2, grid.x // 2)
    half_y = max(2, grid.y // 2)
    candidates = [
        ("whole_rows", (1, grid.x), grid.y),
        ("whole_columns", (grid.y, 1), grid.x),
        ("half_rows", (1, half_x), (grid.x // half_x) * grid.y),
        ("two_rows", (min(2, grid.y), grid.x), max(1, grid.y // min(2, grid.y))),
        ("two_columns", (grid.y, min(2, grid.x)), max(1, grid.x // min(2, grid.x))),
        ("quad", (half_y, half_x), (grid.x // half_x) * (grid.y // half_y)),
    ]
    unique = []
    seen = set()
    for case in candidates:
        _, shape, groups = case
        key = (shape, groups)
        if shape[0] * shape[1] >= 2 and groups >= 1 and key not in seen:
            unique.append(case)
            seen.add(key)
    return unique


def _cases(device):
    custom = _custom_case()
    return [custom] if custom is not None else _default_perf_cases(device)


def _make_input(device, group_shape, num_groups, num_tiles):
    layout = build_group_layout(device, group_shape, num_groups)
    shape = (layout.num_cores * ttnn.TILE_SIZE, num_tiles * ttnn.TILE_SIZE)
    torch_input = torch.empty(shape, dtype=torch.float32)
    shard_index = {core: index for index, core in enumerate(layout.active_cores)}

    element_pattern = (torch.arange(num_tiles * ttnn.TILE_SIZE, dtype=torch.float32) % 17).reshape(1, -1) / 256.0
    for group in layout.groups:
        for member, core in enumerate(group.cores):
            index = shard_index[core]
            value = (group.index + 1) * 0.5 + (member + 1) / 64.0
            torch_input[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] = value + element_pattern

    quantized = torch_input.to(torch.bfloat16).to(torch.float32)
    expected = torch.empty_like(quantized)
    for group in layout.groups:
        indices = [shard_index[core] for core in group.cores]
        reduced = sum(quantized[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] for index in indices)
        for index in indices:
            expected[index * ttnn.TILE_SIZE : (index + 1) * ttnn.TILE_SIZE] = reduced

    memory_config = create_sharded_memory_config(device, group_shape, num_groups, num_tiles)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=memory_config,
    )
    return tt_input, expected


def _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters=1):
    output = all_reduce(
        tt_input,
        variant=variant,
        group_shape=group_shape,
        num_groups=num_groups,
        num_tiles=num_tiles,
        kernel_iters=kernel_iters,
    )
    actual = ttnn.to_torch(output).to(torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.125)
    return output


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total = 0.0
    found = False
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure_matrix(device, runners, trials, kernel_iters):
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)

    samples = {variant: [] for variant in runners}
    for trial in range(trials + 1):
        for variant, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"device profiler produced no duration for {variant}"
            if trial > 0:
                samples[variant].append(duration / kernel_iters)
    return samples


def _format_report(results, *, arch, box, num_tiles, trials, kernel_iters):
    lines = [
        "# Tensix rectangular all-reduce report",
        "",
        f"box={box}  arch={arch}  N={trials} (median)  kernel-iters={kernel_iters}  num-tiles={num_tiles}",
        "",
        "| Placement | Group | Groups | Cores | Method | Median ns/all-reduce | Std / median | vs best |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for case_name, group_shape, num_groups, case_results in results:
        # Ratio against the FASTEST variant in this case, never a fixed one. Anchoring on a
        # known-slow method makes every other row look good and hides the real gap between the
        # best algorithm and the second best.
        best_median = min(statistics.median(v) for v in case_results.values())
        # Best row first, so the table reads as a ranking.
        for variant, values in sorted(case_results.items(), key=lambda kv: statistics.median(kv[1])):
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            noise = std / median * 100.0 if median else float("nan")
            noise_text = f"{noise:.1f}%" + (" (noisy)" if noise >= 5.0 else "")
            ratio = best_median / median
            rows, cols = group_shape
            lines.append(
                f"| {case_name} | {rows}x{cols} | {num_groups} | {rows * cols * num_groups} | "
                f"{variant} | {median:.1f} | {noise_text} | {ratio:.2f}x |"
            )
    return "\n".join(lines) + "\n"


def test_tensix_all_reduce_correctness(device):
    num_tiles = int(os.environ.get("AR_NUM_TILES", "6"))
    kernel_iters = int(os.environ.get("AR_KERNEL_ITERS", "1"))
    for case_name, group_shape, num_groups in _cases(device):
        tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
        for variant in _selected_variants():
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
            logger.info(
                f"PASS case={case_name} group={group_shape[0]}x{group_shape[1]} groups={num_groups} "
                f"tiles={num_tiles} variant={variant}"
            )


def test_tensix_all_reduce_device_perf(device):
    """Correctness-gate every variant, then measure kernel duration; perf never fails directionally."""
    num_tiles = int(os.environ.get("AR_NUM_TILES", "6"))
    kernel_iters = int(os.environ.get("AR_KERNEL_ITERS", "1"))
    trials = int(os.environ.get("AR_TRIALS", "5"))
    variants = _selected_variants()
    results = []

    for case_name, group_shape, num_groups in _cases(device):
        tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
        for variant in variants:
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)

        runners = {
            variant: (
                lambda variant=variant: all_reduce(
                    tt_input,
                    variant=variant,
                    group_shape=group_shape,
                    num_groups=num_groups,
                    num_tiles=num_tiles,
                    kernel_iters=kernel_iters,
                )
            )
            for variant in variants
        }
        samples = _measure_matrix(device, runners, trials, kernel_iters)
        results.append((case_name, group_shape, num_groups, samples))

    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    report = _format_report(
        results,
        arch=arch,
        box=socket.gethostname(),
        num_tiles=num_tiles,
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    report_path = os.environ.get("AR_REPORT")
    if report_path:
        Path(report_path).write_text(report)


# A regime: (id, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters).
# `group_shape` may be "whole_row" (1 x grid width), resolved on the device; a row that needs a longer
# line than the grid has is skipped, never bent to fit.
# Thresholds: helps -> T_help = max(1.05, 1 + 0.5 * (measured - 1)); does_not_help -> <= 1.0 for a measured
# loss, <= 1.05 for a measured null. Push variants run at kernel-iters 1 only: push has no consumer
# back-pressure, so in-kernel repeats let contributors run ahead of the workers.
_ROOT, _PULL, _PUSH = "reduce_root_mcast", "reduce_scatter_mcast", "reduce_scatter_push"
_REGIMES = [
    # Root reduction vs reduce-scatter. Thresholds of the first three hold on WH B0 n150 and BH p150b.
    ("helps_1x8_x8groups_6tiles", (1, 8), 8, 6, "helps", 1.21, _ROOT, _PULL, 10),  # WH 1.68x, BH 1.43x
    (
        "does_not_help_1x4_x16groups_6tiles",
        (1, 4),
        16,
        6,
        "does_not_help",
        1.0,
        _ROOT,
        _PULL,
        10,
    ),  # WH 0.78-0.85x, BH 0.55x
    ("does_not_help_1x8_x8groups_1tile", (1, 8), 8, 1, "does_not_help", 1.0, _ROOT, _PULL, 10),  # WH 0.81x, BH 0.75x
    # Large payloads (BH p150b only; WH B0 not measured).
    ("helps_4x4_x1group_32tiles", (4, 4), 1, 32, "helps", 2.5, _ROOT, _PULL, 10),  # BH 4.04x
    ("helps_1x8_x1group_32tiles_push", (1, 8), 1, 32, "helps", 1.15, _ROOT, _PUSH, 1),  # BH 1.30x
    (
        "does_not_help_wholerow_x1group_16tiles_pull",
        "whole_row",
        1,
        16,
        "does_not_help",
        1.0,
        _ROOT,
        _PULL,
        10,
    ),  # BH 1x13 0.82x
    ("helps_wholerow_x1group_16tiles_push", "whole_row", 1, 16, "helps", 1.11, _ROOT, _PUSH, 1),  # BH 1x13 1.22x
]
# Barrier before signal (baseline) vs flush before signal (lever). BH p150b only; WH B0 not measured.
_FLUSH_REGIMES = [
    ("helps_push_1x8_x1group_1tile", (1, 8), 1, 1, "helps", 1.09, _PUSH, _PUSH + "_flush", 1),  # BH 1.19x
    ("helps_push_1x8_x1group_6tiles", (1, 8), 1, 6, "helps", 1.05, _PUSH, _PUSH + "_flush", 1),  # BH 1.09x
    ("helps_pull_1x8_x1group_1tile", (1, 8), 1, 1, "helps", 1.05, _PULL, _PULL + "_flush", 1),  # BH 1.07x
    (
        "does_not_help_push_1x8_x1group_32tiles",
        (1, 8),
        1,
        32,
        "does_not_help",
        1.05,
        _PUSH,
        _PUSH + "_flush",
        1,
    ),  # 1.02x
    (
        "does_not_help_pull_1x8_x1group_32tiles",
        (1, 8),
        1,
        32,
        "does_not_help",
        1.05,
        _PULL,
        _PULL + "_flush",
        1,
    ),  # 1.00x
]


def _assert_regime(device, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters):
    """Interleaved A/B of baseline vs lever, median of 11 retained trials, direction asserted."""
    if group_shape == "whole_row":
        width = device.compute_with_storage_grid_size().x
        if width < 10:
            pytest.skip(f"needs a line of >= 10 Tensix cores; this grid row has {width}")
        group_shape = (1, width)
    trials = 11
    tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
    for variant in (baseline, lever):
        _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
    runners = {
        variant: lambda variant=variant: all_reduce(
            tt_input,
            variant=variant,
            group_shape=group_shape,
            num_groups=num_groups,
            num_tiles=num_tiles,
            kernel_iters=kernel_iters,
        )
        for variant in (baseline, lever)
    }
    samples = _measure_matrix(device, runners, trials, kernel_iters)
    base_ns = statistics.median(samples[baseline])
    lever_ns = statistics.median(samples[lever])
    speedup = base_ns / lever_ns
    logger.info(
        f"REGIME {kind} group={group_shape[0]}x{group_shape[1]} groups={num_groups} tiles/core={num_tiles} "
        f"kernel-iters={kernel_iters} "
        f"{baseline}={base_ns:.1f}ns [{min(samples[baseline]):.1f}-{max(samples[baseline]):.1f}] "
        f"{lever}={lever_ns:.1f}ns [{min(samples[lever]):.1f}-{max(samples[lever]):.1f}] "
        f"speedup={speedup:.3f}x trials={trials} arch={device.arch()} chip_freq={_chip_freq_mhz()}MHz"
    )
    if kind == "helps":
        assert speedup >= threshold, f"expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"expected <= {threshold}x, got {speedup:.3f}x"


_REGIME_ARGS = "group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters"


@pytest.mark.parametrize(_REGIME_ARGS, [r[1:] for r in _REGIMES], ids=[r[0] for r in _REGIMES])
def test_tensix_all_reduce_regimes(
    device, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters
):
    """Root reduction (baseline) vs reduce-scatter (lever)."""
    _assert_regime(device, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters)


@pytest.mark.parametrize(_REGIME_ARGS, [r[1:] for r in _FLUSH_REGIMES], ids=[r[0] for r in _FLUSH_REGIMES])
def test_tensix_all_reduce_flush_regimes(
    device, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters
):
    """Barrier before the publishing semaphore increment (baseline) vs flush before it (lever)."""
    _assert_regime(device, group_shape, num_groups, num_tiles, kind, threshold, baseline, lever, kernel_iters)


# ---------------------------------------------------------------------------------------------
# Sweeps behind the README's tables. Skipped unless AR_SWEEPS=1 (they compile ~100 programs).
# Both are interleaved A/B per cell (so ratios survive a clock change mid-session), median of
# AR_SWEEP_TRIALS retained trials, and log one table; AR_SWEEP_REPORT=<path> also writes it.
# ---------------------------------------------------------------------------------------------

_sweeps = pytest.mark.skipif(os.environ.get("AR_SWEEPS") != "1", reason="set AR_SWEEPS=1 to run the sweeps")


def _full_grid_groups(device, group_shape):
    grid = device.compute_with_storage_grid_size()
    return (grid.x // group_shape[1]) * (grid.y // group_shape[0])


def _ab_cell(device, variants, group_shape, num_groups, num_tiles, kernel_iters, trials):
    """Correctness-gate each variant, then measure them interleaved. Returns {variant: [ns, ...]} for the
    variants that fit in L1 and {variant: reason} for the ones that do not."""
    tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
    runnable, skipped = [], {}
    for variant in variants:
        try:
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
            runnable.append(variant)
        except RuntimeError as error:  # CB allocation clash: does not fit in L1 at this size
            if "clash" not in str(error) and "L1" not in str(error):
                raise
            skipped[variant] = "L1"
    runners = {
        variant: lambda variant=variant: all_reduce(
            tt_input,
            variant=variant,
            group_shape=group_shape,
            num_groups=num_groups,
            num_tiles=num_tiles,
            kernel_iters=kernel_iters,
        )
        for variant in runnable
    }
    samples = _measure_matrix(device, runners, trials, kernel_iters) if runners else {}
    return samples, skipped


def _sweep(device, cells, pairs, *, title):
    """cells: [(group_shape, num_groups, num_tiles, kernel_iters)]; pairs: [(baseline, lever)]."""
    trials = int(os.environ.get("AR_SWEEP_TRIALS", "5"))
    variants = tuple(dict.fromkeys(v for pair in pairs for v in pair))
    lines = [
        f"### {title}",
        "",
        "STAMP",
        "",
        "| Group | Members | Groups | Tiles/core | Kernel-iters | Baseline | Baseline ns [min-max] | Lever | "
        "Lever ns [min-max] | Speedup |",
        "|---|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for group_shape, num_groups, num_tiles, kernel_iters in cells:
        samples, skipped = _ab_cell(device, variants, group_shape, num_groups, num_tiles, kernel_iters, trials)

        def cell(variant):
            if variant in skipped:
                return f"does not fit ({skipped[variant]})", None
            values = samples[variant]
            median = statistics.median(values)
            return f"{median:.1f} [{min(values):.1f}-{max(values):.1f}]", median

        rows, cols = group_shape
        for baseline, lever in pairs:
            base_text, base_ns = cell(baseline)
            lever_text, lever_ns = cell(lever)
            speedup = f"{base_ns / lever_ns:.3f}x" if base_ns and lever_ns else "-"
            line = (
                f"| {rows}x{cols} | {rows * cols} | {num_groups} | {num_tiles} | {kernel_iters} | {baseline} | "
                f"{base_text} | {lever} | {lever_text} | {speedup} |"
            )
            lines.append(line)
            logger.info(line)
    # Read after the measurements: the profiler (re)creates its device log when the session's first
    # results are dumped, so the header carries this session's AICLK only from then on.
    lines[2] = (
        f"box={socket.gethostname()}  arch={device.arch()}  N={trials} (median, interleaved)  "
        f"CHIP_FREQ={_chip_freq_mhz()} MHz (profiler log header)"
    )
    report = "\n".join(lines) + "\n"
    logger.info("\n" + report)
    path = os.environ.get("AR_SWEEP_REPORT")
    if path:
        with open(path, "a") as f:
            f.write(report + "\n")


def _flush_cells(device):
    grid = device.compute_with_storage_grid_size()
    shapes = [(1, 4), (1, 8), (1, grid.x), (grid.y, 1), (2, grid.x)]
    cells = []
    for shape in shapes:
        for num_groups in (1, _full_grid_groups(device, shape)):
            for num_tiles in (1, 6, 32):
                cells.append((shape, num_groups, num_tiles, 1))
    return list(dict.fromkeys(cells))


@_sweeps
def test_tensix_all_reduce_flush_sweep(device):
    """Barrier before signal (baseline) vs flush before signal (lever), pull and push twins, kernel-iters 1."""
    pairs = [
        ("reduce_scatter_mcast", "reduce_scatter_mcast_flush"),
        ("reduce_scatter_push", "reduce_scatter_push_flush"),
    ]
    _sweep(device, _flush_cells(device), pairs, title="Signal after a flush instead of a barrier")


def _boundary_cells(device):
    grid = device.compute_with_storage_grid_size()
    shapes = [(1, 8), (1, grid.x), (grid.y, 1), (2, 8), (4, 4), (2, grid.x), (grid.y // 2, grid.x // 2)]
    tiles = [int(t) for t in os.environ.get("AR_BOUNDARY_TILES", "8,16,24,32,48,64,128,256").split(",")]
    kernel_iters = int(os.environ.get("AR_BOUNDARY_KERNEL_ITERS", "10"))
    cells = []
    groupings = os.environ.get("AR_BOUNDARY_GROUPINGS", "isolated,full_grid").split(",")
    for shape in shapes:
        group_counts = [1] if "isolated" in groupings else []
        group_counts += [_full_grid_groups(device, shape)] if "full_grid" in groupings else []
        for num_groups in group_counts:
            for num_tiles in tiles:
                cells.append((shape, num_groups, num_tiles, kernel_iters))
    return list(dict.fromkeys(cells))


@_sweeps
def test_tensix_all_reduce_boundary_sweep(device):
    """Root reduction (baseline) vs reduce-scatter (lever) out to the L1 ceiling of each."""
    levers = os.environ.get("AR_BOUNDARY_LEVERS", "reduce_scatter_mcast").split(",")
    pairs = [("reduce_root_mcast", lever) for lever in levers]
    _sweep(device, _boundary_cells(device), pairs, title="Root reduction vs reduce-scatter, large payloads")


# =============================================================================================
# Per-row-vector payload (`row_sum_all_reduce`): bytes per member on the wire, both directions,
# and the return path. A row is WIDTH_SHARDED over K members, b tile-rows each; every member ends
# with the whole-row sums as column 0 of a float32 tile per tile-row.
# =============================================================================================

_RV_BASELINE = "tile_partial_unicast"


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _ints(name, default):
    return [int(v) for v in os.environ.get(name, default).split(",") if v]


def _rv_selected_variants():
    selected = tuple(v for v in os.environ.get("AR_RV_VARIANTS", ",".join(ROW_VECTOR_VARIANTS)).split(",") if v)
    unknown = set(selected) - set(ALL_ROW_VECTOR_VARIANTS)
    if unknown:
        raise ValueError(f"unknown AR_RV_VARIANTS: {sorted(unknown)}")
    return selected


def _rv_make_input(device, num_members, tile_rows, width_tiles):
    """0/1 input (about 1 in 8 ones): every row sum is a small integer, exact in float32 and through
    the tf32 source registers, so the check is exact and a missing or misplaced partial (a change of
    at least 1 in some row sum) fails it at any K."""
    shape = (tile_rows * ttnn.TILE_SIZE, num_members * width_tiles * ttnn.TILE_SIZE)
    gen = torch.Generator().manual_seed(1234 + 7 * num_members + 3 * tile_rows + width_tiles)
    x = (torch.randint(0, 8, shape, generator=gen) == 0).to(torch.float32)
    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_row_vector_memory_config(device, num_members, tile_rows, width_tiles),
    )
    return tt_input, x.sum(dim=1)


def _rv_run(tt_input, variant, num_members, tile_rows, width_tiles, kernel_iters=1, output_tensor=None):
    return row_sum_all_reduce(
        tt_input,
        variant=variant,
        num_members=num_members,
        tile_rows=tile_rows,
        width_tiles=width_tiles,
        kernel_iters=kernel_iters,
        output_tensor=output_tensor,
    )


def _rv_run_checked(tt_input, row_sums, variant, num_members, tile_rows, width_tiles, kernel_iters=1):
    # Output pre-filled with a sentinel, so a variant that fails to deliver a result cannot pass on
    # the previous variant's output left at the same L1 address.
    device = tt_input.device()
    sentinel = ttnn.from_torch(
        torch.full((tile_rows * ttnn.TILE_SIZE, num_members * ttnn.TILE_SIZE), -1.0),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_row_vector_memory_config(device, num_members, tile_rows, 1),
    )
    out = ttnn.to_torch(_rv_run(tt_input, variant, num_members, tile_rows, width_tiles, kernel_iters, sentinel))
    column0 = out.to(torch.float32)[:, 0 :: ttnn.TILE_SIZE]  # (b x 32, K): column 0 of every member's tile
    expected = row_sums.reshape(-1, 1).expand(-1, num_members)
    torch.testing.assert_close(column0, expected, rtol=0.0, atol=0.0, msg=lambda m: f"{variant}: {m}")


def _rv_cell(device, variants, num_members, tile_rows, width_tiles, trials, kernel_iters):
    tt_input, row_sums = _rv_make_input(device, num_members, tile_rows, width_tiles)
    for variant in variants:
        _rv_run_checked(tt_input, row_sums, variant, num_members, tile_rows, width_tiles, kernel_iters)
    runners = {
        v: (lambda v=v: _rv_run(tt_input, v, num_members, tile_rows, width_tiles, kernel_iters)) for v in variants
    }
    return _measure_matrix(device, runners, trials, kernel_iters)


def _rv_correctness_cases(device):
    grid = device.compute_with_storage_grid_size()
    capacity = grid.x * grid.y
    # (K, b, w, kernel-iters): odd K, K not a multiple of 32, more than one gather tile, a member set
    # that is not a rectangle (multicast reaches non-members), several in-kernel iterations.
    cases = [(2, 1, 1, 1), (2, 2, 1, 3), (5, 4, 2, 1), (11, 4, 1, 2), (12, 2, 1, 3), (16, 4, 2, 1), (40, 3, 1, 2)]
    cases += [(64, 4, 2, 1), (min(128, capacity), 4, 1, 2), (capacity - 1, 2, 1, 1)]
    return [c for c in dict.fromkeys(cases) if 2 <= c[0] <= capacity]


def test_tensix_all_reduce_row_vector_correctness(device):
    """Every row-vector variant, exact vs torch, column 0 of every member's output tile."""
    variants = os.environ.get("AR_RV_CORRECTNESS_VARIANTS")
    variants = tuple(variants.split(",")) if variants else ALL_ROW_VECTOR_VARIANTS
    for num_members, tile_rows, width_tiles, kernel_iters in _rv_correctness_cases(device):
        tt_input, row_sums = _rv_make_input(device, num_members, tile_rows, width_tiles)
        placement = build_member_layout(device, num_members).placement
        for variant in variants:
            _rv_run_checked(tt_input, row_sums, variant, num_members, tile_rows, width_tiles, kernel_iters)
        logger.info(
            f"PASS K={num_members} placement={placement} b={tile_rows} w={width_tiles} "
            f"kernel-iters={kernel_iters} variants={len(variants)}"
        )


def _rv_stamp(device, trials, kernel_iters):
    return (
        f"box={socket.gethostname()}  arch={device.arch()}  clock={_chip_freq_mhz()} MHz (profiler log header)  "
        f"date={datetime.date.today().isoformat()}  sha={_git_sha()}  N={trials} (median, interleaved)  "
        f"kernel-iters={kernel_iters} ({'per-launch latency' if kernel_iters == 1 else 'steady state'})"
    )


# (title, baseline, lever): each lever with the other concept held fixed.
_RV_PAIRS = [
    ("tile -> faces, unicast", "tile_partial_unicast", "faces_partial_unicast"),
    ("tile -> faces, mcast", "tile_partial_mcast", "faces_partial_mcast"),
    ("faces -> row, unicast", "faces_partial_unicast", "row_partial_unicast"),
    ("faces -> row, mcast", "faces_partial_mcast", "row_partial_mcast"),
    ("unicast -> mcast, tile", "tile_partial_unicast", "tile_partial_mcast"),
    ("unicast -> mcast, faces", "faces_partial_unicast", "faces_partial_mcast"),
    ("unicast -> mcast, row", "row_partial_unicast", "row_partial_mcast"),
    ("faces return, unicast", "faces_partial_unicast", "faces_partial_faces_return_unicast"),
    ("faces return, mcast", "faces_partial_mcast", "faces_partial_faces_return_mcast"),
    ("both: tile+unicast -> row+mcast", "tile_partial_unicast", "row_partial_mcast"),
]


def _rv_format(device, results, trials, kernel_iters):
    lines = [
        "### Every variant",
        "",
        "| K (members) | Placement | b (tile-rows / member) | w (tiles / member / tile-row) | Method | ns | vs "
        f"{_RV_BASELINE} |",
        "|---:|---|---:|---:|---|---:|---:|",
    ]
    for num_members, tile_rows, width_tiles, samples in results:
        placement = build_member_layout(device, num_members).placement
        base = statistics.median(samples[_RV_BASELINE]) if _RV_BASELINE in samples else None
        for variant, values in samples.items():
            median = statistics.median(values)
            ratio = f"{base / median:.2f}x" if base else "-"
            lines.append(
                f"| {num_members} | {placement} | {tile_rows} | {width_tiles} | {variant} | {median:.0f} | {ratio} |"
            )
    medians = [(k, b, w, {v: statistics.median(s) for v, s in samples.items()}) for k, b, w, samples in results]
    present = [p for p in _RV_PAIRS if all(p[1] in m and p[2] in m for *_, m in medians)]
    if present:
        lines += [
            "",
            "### Lever speedups (baseline ns / lever ns; < 1.00x = the lever is slower)",
            "",
            "| K | b | w | " + " | ".join(f"{name}" for name, _, _ in present) + " | fastest |",
            "|---:|---:|---:|" + "---:|" * len(present) + "---|",
        ]
        for k, b, w, m in medians:
            cells = " | ".join(f"{m[a] / m[l]:.2f}x" for _, a, l in present)
            fastest = min(m, key=m.get)
            lines.append(f"| {k} | {b} | {w} | {cells} | `{fastest}` {m[fastest]:.0f} ns |")
    return "\n".join([_rv_stamp(device, trials, kernel_iters), ""] + lines) + "\n"


_rv_sweep = pytest.mark.skipif(
    os.environ.get("AR_RV_SWEEP") != "1" and os.environ.get("AR_SWEEPS") != "1",
    reason="set AR_RV_SWEEP=1 (or AR_SWEEPS=1) to run the row-vector sweep",
)


@_rv_sweep
def test_tensix_all_reduce_row_vector_sweep(device):
    """Correctness-gate every variant per cell, then measure them interleaved. Perf never fails the test."""
    grid = device.compute_with_storage_grid_size()
    members = [k for k in _ints("AR_RV_MEMBERS", "2,4,8,12,16,32,64,128") if 2 <= k <= grid.x * grid.y]
    tile_rows_list = _ints("AR_RV_TILE_ROWS", "1,4")
    width_list = _ints("AR_RV_WIDTH_TILES", "1")
    trials = int(os.environ.get("AR_RV_TRIALS", "5"))
    kernel_iters = int(os.environ.get("AR_KERNEL_ITERS", "1"))
    variants = _rv_selected_variants()

    results = []
    for width_tiles in width_list:
        for tile_rows in tile_rows_list:
            for num_members in members:
                if tile_rows > num_members:
                    continue  # one owner per tile-row
                samples = _rv_cell(device, variants, num_members, tile_rows, width_tiles, trials, kernel_iters)
                results.append((num_members, tile_rows, width_tiles, samples))
                logger.info(
                    f"K={num_members} b={tile_rows} w={width_tiles} "
                    + "  ".join(f"{v}={statistics.median(s):.0f}ns" for v, s in samples.items())
                )
    report = _rv_format(device, results, trials, kernel_iters)
    logger.info("\n" + report)
    path = os.environ.get("AR_RV_REPORT")
    if path:
        with open(path, "a") as f:
            f.write(report + "\n")


# Regimes: the catalog's per-row-vector boundary claims, asserted. Interleaved A/B, median of 11.
# helps -> speedup >= max(1.05, 1 + half the measured gain); does_not_help -> speedup <= 1.05.
# (id, K, b, w, kind, threshold, baseline, lever)
_TU, _FU, _RU = "tile_partial_unicast", "faces_partial_unicast", "row_partial_unicast"
_TM, _FM, _RM = "tile_partial_mcast", "faces_partial_mcast", "row_partial_mcast"
_FRU, _FRM = "faces_partial_faces_return_unicast", "faces_partial_faces_return_mcast"
_RV_REGIMES = [
    # Measured BH p150b, 13x10 grid, 1350 MHz, w = 1, kernel-iters 1, N = 5 sweep (report.md, 2026-10-31).
    # Partial size on the wire: whole tile (4096 B) -> faces 0/2 (2048 B) -> one row (128 B).
    ("faces_over_tile_helps_K128_b4_mcast", 128, 4, 1, "helps", 1.21, _TM, _FM),  # 13 261 -> 9 355 ns, 1.42x
    ("faces_over_tile_helps_K8_b4_unicast", 8, 4, 1, "helps", 1.08, _TU, _FU),  # 4 483 -> 3 859 ns, 1.16x
    ("faces_over_tile_does_not_help_K2_b1_unicast", 2, 1, 1, "does_not_help", 1.05, _TU, _FU),  # 1.02x
    ("row_over_faces_helps_K64_b1_mcast", 64, 1, 1, "helps", 1.22, _FM, _RM),  # 5 156 -> 3 570 ns, 1.44x
    ("row_over_faces_helps_K12_b4_mcast", 12, 4, 1, "helps", 1.07, _FM, _RM),  # 3 919 -> 3 428 ns, 1.14x
    ("row_over_faces_does_not_help_K4_b1_unicast", 4, 1, 1, "does_not_help", 1.05, _FU, _RU),  # 0.89x
    ("row_over_faces_does_not_help_K8_b1_mcast", 8, 1, 1, "does_not_help", 1.05, _FM, _RM),  # 0.94x
    ("row_over_faces_does_not_help_K12_b1_mcast", 12, 1, 1, "does_not_help", 1.05, _FM, _RM),  # 0.99x
    # Return path: per-owner unicast to all K members -> one root multicast.
    ("mcast_helps_K64_b1_tile", 64, 1, 1, "helps", 1.31, _TU, _TM),  # 10 948 -> 6 736 ns, 1.63x
    ("mcast_helps_K16_b4_faces", 16, 4, 1, "helps", 1.12, _FU, _FM),  # 5 276 -> 4 225 ns, 1.25x
    ("mcast_does_not_help_K4_b1_tile", 4, 1, 1, "does_not_help", 1.05, _TU, _TM),  # 0.93x
    ("mcast_does_not_help_K4_b4_faces", 4, 4, 1, "does_not_help", 1.05, _FU, _FM),  # 0.92x
    # Return payload: the whole column-0 tile -> only its faces 0/2.
    ("faces_return_helps_K16_b4_unicast", 16, 4, 1, "helps", 1.07, _FU, _FRU),  # 5 276 -> 4 595 ns, 1.15x
    ("faces_return_does_not_help_K16_b4_mcast", 16, 4, 1, "does_not_help", 1.05, _FM, _FRM),  # 0.67x
    ("faces_return_does_not_help_K64_b1_unicast", 64, 1, 1, "does_not_help", 1.05, _FU, _FRU),  # 0.95x
    # Both levers.
    ("both_help_K128_b4", 128, 4, 1, "helps", 2.85, _TU, _RM),  # 25 546 -> 5 424 ns, 4.71x
    ("both_do_not_help_K4_b1", 4, 1, 1, "does_not_help", 1.05, _TU, _RM),  # 0.86x
]


@pytest.mark.parametrize(
    "num_members, tile_rows, width_tiles, kind, threshold, baseline, lever",
    [r[1:] for r in _RV_REGIMES],
    ids=[r[0] for r in _RV_REGIMES],
)
def test_tensix_all_reduce_row_vector_regimes(
    device, num_members, tile_rows, width_tiles, kind, threshold, baseline, lever
):
    """Per-row-vector payload: partial size on the wire, return payload and return path."""
    grid = device.compute_with_storage_grid_size()
    if num_members > grid.x * grid.y:
        pytest.skip(f"needs {num_members} Tensix cores; this grid has {grid.x * grid.y}")
    trials = 11
    samples = _rv_cell(device, (baseline, lever), num_members, tile_rows, width_tiles, trials, 1)
    base_ns = statistics.median(samples[baseline])
    lever_ns = statistics.median(samples[lever])
    speedup = base_ns / lever_ns
    logger.info(
        f"REGIME {kind} K={num_members} b={tile_rows} w={width_tiles} {baseline}={base_ns:.0f}ns "
        f"{lever}={lever_ns:.0f}ns speedup={speedup:.3f}x trials={trials} arch={device.arch()} "
        f"clock={_chip_freq_mhz()}MHz"
    )
    if kind == "helps":
        assert speedup >= threshold, f"expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"expected <= {threshold}x, got {speedup:.3f}x"
