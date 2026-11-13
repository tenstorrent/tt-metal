# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling for rectangular Tensix all-reduce."""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import socket
import statistics
from pathlib import Path

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    check_regime,
    chip_freq_mhz,
    grid_xy,
    interleaved_ns,
    require_grid,
    sweep_gate,
)
from ttnn.operations.examples.tensix_all_reduce import (
    VARIANTS,
    all_reduce,
    build_group_layout,
    create_sharded_memory_config,
)


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


def _runners(tt_input, variants, group_shape, num_groups, num_tiles, kernel_iters):
    """{variant: zero-arg launch} for `perf_test_utils.interleaved_ns`."""
    return {
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

        runners = _runners(tt_input, variants, group_shape, num_groups, num_tiles, kernel_iters)
        samples = interleaved_ns(device, runners, trials=trials, per_launch_divisor=kernel_iters)
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


def _require_groups(device, group_shape, num_groups, what):
    """Skip unless `build_group_layout` can place `num_groups` groups of rows x cols Tensix cores. It
    packs them row-major: grid.x // cols groups side by side in each band of `rows` grid rows, then the
    next band, so the case needs `cols` columns and `rows` x (number of bands) rows."""
    rows, cols = group_shape
    per_band = max(1, grid_xy(device)[0] // cols)
    require_grid(device, cols, rows * -(-num_groups // per_band), what=what)


# A regime: (id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters).
# `group_shape` may be "whole_row" (1 x grid width), resolved on the device: it needs a line of at
# least 10 Tensix cores and is skipped on a narrower grid. Every fixed group shape x group count is
# skipped when the grid cannot place it (`_require_groups`; (1, 8) x 8 and (1, 4) x 16 need 8x8 on
# an 8-column grid), never bent to fit.
# `measured` is the speedup (baseline ns / lever ns) measured per architecture: "wh" = Wormhole B0
# n150, "bh" = Blackhole p150b, a (low, high) pair for a range over runs. `check_regime` asserts it
# against a bound derived from the number measured on the running architecture
# (`perf_test_utils.regime_bound`), only logs on an architecture with no measured number, and
# re-measures once (pooling both passes) before failing.
# Push variants run at kernel-iters 1 only: push has no consumer back-pressure, so in-kernel repeats
# let contributors run ahead of the workers.
_ROOT, _PULL, _PUSH = "reduce_root_mcast", "reduce_scatter_mcast", "reduce_scatter_push"
_REGIMES = [
    # Root reduction vs reduce-scatter, measured on WH B0 n150 and BH p150b.
    ("helps_1x8_x8groups_6tiles", (1, 8), 8, 6, "helps", {"wh": 1.68, "bh": 1.43}, _ROOT, _PULL, 10),
    (
        "does_not_help_1x4_x16groups_6tiles",
        (1, 4),
        16,
        6,
        "does_not_help",
        {"wh": (0.78, 0.85), "bh": 0.55},
        _ROOT,
        _PULL,
        10,
    ),
    (
        "does_not_help_1x8_x8groups_1tile",
        (1, 8),
        8,
        1,
        "does_not_help",
        {"wh": 0.81, "bh": 0.75},
        _ROOT,
        _PULL,
        10,
    ),
    # Large payloads (BH p150b only; WH B0 not measured).
    ("helps_4x4_x1group_32tiles", (4, 4), 1, 32, "helps", {"bh": 4.04}, _ROOT, _PULL, 10),
    ("helps_1x8_x1group_32tiles_push", (1, 8), 1, 32, "helps", {"bh": 1.30}, _ROOT, _PUSH, 1),
    (
        "does_not_help_wholerow_x1group_16tiles_pull",
        "whole_row",
        1,
        16,
        "does_not_help",
        {"bh": 0.82},  # 1x13
        _ROOT,
        _PULL,
        10,
    ),
    ("helps_wholerow_x1group_16tiles_push", "whole_row", 1, 16, "helps", {"bh": 1.22}, _ROOT, _PUSH, 1),  # 1x13
]
# Barrier before signal (baseline) vs flush before signal (lever). BH p150b only; WH B0 not measured.
_FLUSH_REGIMES = [
    ("helps_push_1x8_x1group_1tile", (1, 8), 1, 1, "helps", {"bh": 1.19}, _PUSH, _PUSH + "_flush", 1),
    ("helps_push_1x8_x1group_6tiles", (1, 8), 1, 6, "helps", {"bh": 1.09}, _PUSH, _PUSH + "_flush", 1),
    ("helps_pull_1x8_x1group_1tile", (1, 8), 1, 1, "helps", {"bh": 1.07}, _PULL, _PULL + "_flush", 1),
    (
        "does_not_help_push_1x8_x1group_32tiles",
        (1, 8),
        1,
        32,
        "does_not_help",
        {"bh": 1.02},
        _PUSH,
        _PUSH + "_flush",
        1,
    ),
    (
        "does_not_help_pull_1x8_x1group_32tiles",
        (1, 8),
        1,
        32,
        "does_not_help",
        {"bh": 1.00},
        _PULL,
        _PULL + "_flush",
        1,
    ),
]
_REGIME_TRIALS = 11


def _check_regime(device, regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters):
    """Correctness-gate baseline and lever, then interleaved A/B through `perf_test_utils.check_regime`."""
    if group_shape == "whole_row":
        require_grid(device, 10, 1, what=f"{regime_id} (a line of >= 10 Tensix cores)")
        group_shape = (1, grid_xy(device)[0])
    else:
        _require_groups(device, group_shape, num_groups, what=regime_id)
    tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
    for variant in (baseline, lever):
        _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
    runners = _runners(tt_input, (baseline, lever), group_shape, num_groups, num_tiles, kernel_iters)

    def measure():
        samples = interleaved_ns(device, runners, trials=_REGIME_TRIALS, per_launch_divisor=kernel_iters)
        return samples[baseline], samples[lever]

    rows, cols = group_shape
    check_regime(
        device,
        regime_id,
        kind,
        measured,
        measure,
        detail=(
            f"{baseline} -> {lever}, group {rows}x{cols} x {num_groups} = {rows * cols * num_groups} Tensix cores, "
            f"{num_tiles} tiles per Tensix core, kernel-iters {kernel_iters}, ns per all-reduce"
        ),
    )


_REGIME_ARGS = "regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters"


@pytest.mark.parametrize(_REGIME_ARGS, _REGIMES, ids=[r[0] for r in _REGIMES])
def test_tensix_all_reduce_regimes(
    device, regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters
):
    """Root reduction (baseline) vs reduce-scatter (lever)."""
    _check_regime(device, regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters)


@pytest.mark.parametrize(_REGIME_ARGS, _FLUSH_REGIMES, ids=[r[0] for r in _FLUSH_REGIMES])
def test_tensix_all_reduce_flush_regimes(
    device, regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters
):
    """Barrier before the publishing semaphore increment (baseline) vs flush before it (lever)."""
    _check_regime(device, regime_id, group_shape, num_groups, num_tiles, kind, measured, baseline, lever, kernel_iters)


# ---------------------------------------------------------------------------------------------
# Measurement sweeps behind the README's tables. Opt-in (`perf_test_utils.sweep_gate`): skipped
# unless AR_SWEEPS=1 or EXAMPLES_SWEEPS=1 (they compile ~100 programs). Both are interleaved A/B
# per cell (so ratios survive a clock change mid-session), median of AR_SWEEP_TRIALS trials, and
# log one table; AR_SWEEP_REPORT=<path> also appends it to that file.
# ---------------------------------------------------------------------------------------------

# The two TT_THROW messages of the program's circular-buffer validation (tt_metal/impl/program/program.cpp)
# that mean "this variant's statically allocated circular buffers do not fit in L1 at this size". Only
# these drop a variant from a sweep table; any other error re-raises, even one whose text mentions L1.
_CB_DOES_NOT_FIT = (
    ("Statically allocated circular buffers on core range", "beyond max L1 size"),
    ("Statically allocated circular buffers in program", "clash with L1 buffers"),
)


def _cb_does_not_fit(error):
    text = str(error)
    return any(all(part in text for part in message) for message in _CB_DOES_NOT_FIT)


def _full_grid_groups(device, group_shape):
    grid = device.compute_with_storage_grid_size()
    return (grid.x // group_shape[1]) * (grid.y // group_shape[0])


def _fits_grid(device, group_shape):
    grid_x, grid_y = grid_xy(device)
    return group_shape[0] <= grid_y and group_shape[1] <= grid_x and group_shape[0] * group_shape[1] >= 2


def _ab_cell(device, variants, group_shape, num_groups, num_tiles, kernel_iters, trials):
    """Correctness-gate each variant, then measure them interleaved. Returns {variant: [ns, ...]} for the
    variants whose circular buffers fit in L1 and {variant: reason} for the ones that do not."""
    tt_input, expected = _make_input(device, group_shape, num_groups, num_tiles)
    runnable, skipped = [], {}
    for variant in variants:
        try:
            _run_checked(tt_input, expected, variant, group_shape, num_groups, num_tiles, kernel_iters)
            runnable.append(variant)
        except RuntimeError as error:
            if not _cb_does_not_fit(error):
                raise
            skipped[variant] = "L1"
    runners = _runners(tt_input, runnable, group_shape, num_groups, num_tiles, kernel_iters)
    samples = interleaved_ns(device, runners, trials=trials, per_launch_divisor=kernel_iters) if runners else {}
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
        "| Group | Members | Groups | Tiles per Tensix core | Kernel-iters | Baseline | Baseline ns [min-max] | "
        "Lever | Lever ns [min-max] | Speedup |",
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
        f"CHIP_FREQ={chip_freq_mhz()} MHz (profiler log header)"
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
    for shape in (s for s in shapes if _fits_grid(device, s)):
        for num_groups in (1, _full_grid_groups(device, shape)):
            for num_tiles in (1, 6, 32):
                cells.append((shape, num_groups, num_tiles, 1))
    return list(dict.fromkeys(cells))


@sweep_gate("AR_SWEEPS")
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
    for shape in (s for s in shapes if _fits_grid(device, s)):
        group_counts = [1] if "isolated" in groupings else []
        group_counts += [_full_grid_groups(device, shape)] if "full_grid" in groupings else []
        for num_groups in group_counts:
            for num_tiles in tiles:
                cells.append((shape, num_groups, num_tiles, kernel_iters))
    return list(dict.fromkeys(cells))


@sweep_gate("AR_SWEEPS")
def test_tensix_all_reduce_boundary_sweep(device):
    """Root reduction (baseline) vs reduce-scatter (lever) out to the L1 ceiling of each."""
    levers = os.environ.get("AR_BOUNDARY_LEVERS", "reduce_scatter_mcast").split(",")
    pairs = [("reduce_root_mcast", lever) for lever in levers]
    _sweep(device, _boundary_cells(device), pairs, title="Root reduction vs reduce-scatter, large payloads")
