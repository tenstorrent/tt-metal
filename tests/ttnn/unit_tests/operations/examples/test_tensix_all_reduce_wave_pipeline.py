# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and in-process device profiling: one-shot vs wave-pipelined row all-reduce."""

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

from ttnn.operations.examples.tensix_all_reduce_wave_pipeline import (
    create_sharded_memory_config,
    row_all_reduce,
    wave_plan,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
TILE = 32

# (case, rows, width_tiles): tile-rows per Tensix core and tiles per tile-row per Tensix core.
_DEFAULT_CASES = [
    ("tall", 32, 4),  # 128 tiles per Tensix core
    ("wide", 8, 16),  # 128 tiles per Tensix core, few rows
    ("short_narrow", 8, 1),  # 8 tiles per Tensix core
]
_DEFAULT_WAVES = (1, 2, 4, 8)


def _int(name, default):
    return int(os.environ.get(name, default))


def _group(device):
    grid = device.compute_with_storage_grid_size()
    return _int("WP_GROUP_SIZE", min(8, grid.x)), _int("WP_NUM_GROUPS", min(8, grid.y))


def _cases():
    if "WP_ROWS" in os.environ:
        return [("custom", _int("WP_ROWS", 32), _int("WP_WIDTH_TILES", 4))]
    return _DEFAULT_CASES


def _wave_counts():
    raw = os.environ.get("WP_WAVES")
    return tuple(int(w) for w in raw.split(",")) if raw else _DEFAULT_WAVES


def _methods(rows, group_size):
    """(label, variant, waves) for every wave count this geometry splits evenly; one_shot first."""
    methods = [("one_shot", "one_shot", 1)]
    for waves in _wave_counts():
        if waves == 1:
            continue
        try:
            wave_plan(rows, group_size, waves)
        except ValueError:
            continue
        methods.append((f"waves={waves}", "waves", waves))
    return methods


def _make_input(device, group_size, num_groups, rows, width_tiles, seed=7):
    # Values in {-1, 0, 1} / 8: every partial, total and output is k / 8 with |k| <= width_tiles *
    # group_size + 1, exact in bf16 (8 significant bits) up to 255, so the check below can be exact.
    torch.manual_seed(seed)
    cores = group_size * num_groups
    x = torch.randint(-1, 2, (cores * rows * TILE, width_tiles * TILE)).to(torch.float32) / 8
    # [group, member, row, 32, w, 32]; shard index = group * group_size + member.
    tiles = x.reshape(num_groups, group_size, rows, TILE, width_tiles, TILE)
    total = tiles.sum(dim=4).sum(dim=1)  # [g, r, 32, 32]
    expected = (tiles + total[:, None, :, :, None, :]).reshape(x.shape)
    tt_input = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(device, group_size, num_groups, rows, width_tiles),
    )
    return tt_input, expected


def _run(tt_input, variant, waves, group_size, num_groups, rows, width_tiles, kernel_iters=1):
    return row_all_reduce(
        tt_input,
        variant=variant,
        group_size=group_size,
        num_groups=num_groups,
        rows=rows,
        width_tiles=width_tiles,
        waves=waves,
        kernel_iters=kernel_iters,
    )


def _check_all(tt_input, expected, methods, group_size, num_groups, rows, width_tiles, kernel_iters=1):
    """Every method matches torch, and every method matches one_shot bit for bit (same arithmetic order)."""
    reference = None
    for label, variant, waves in methods:
        out = ttnn.to_torch(_run(tt_input, variant, waves, group_size, num_groups, rows, width_tiles, kernel_iters))
        out = out.to(torch.float32)
        if width_tiles * group_size + 1 <= 255:
            torch.testing.assert_close(out, expected, rtol=0, atol=0, msg=lambda m: f"{label}: {m}")
        else:  # sums past 255 / 8 round in bf16; allow a couple of bf16 steps of the total
            torch.testing.assert_close(out, expected, rtol=0.02, atol=1.0, msg=lambda m: f"{label}: {m}")
        if reference is None:
            reference = out
        else:
            assert torch.equal(out, reference), f"{label} differs from one_shot"


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total, found = 0.0, False
    for programs in per_chip.values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _measure(device, runners, trials, kernel_iters):
    """{label: [ns per all-reduce, ...]}; the first launch of each runner (JIT) is discarded."""
    for run in runners.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = {label: [] for label in runners}
    for _ in range(trials):
        for label, run in runners.items():
            run()
            duration = _read_kernel_ns(device)
            assert duration is not None, f"device profiler produced no duration for {label}"
            samples[label].append(duration / kernel_iters)
    return samples


def _arch_label(device):
    arch = str(device.arch()).rsplit(".", 1)[-1]
    return {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(arch, arch)


def _format_report(results, *, box, arch, group_size, num_groups, trials, kernel_iters):
    cores = group_size * num_groups
    lines = [
        "# Row all-reduce: one-shot vs wave-pipelined exchange",
        "",
        f"box={box}  arch={arch}  N={trials} (median)  kernel-iters={kernel_iters}  "
        f"group={group_size} Tensix cores along a grid row  groups={num_groups}  cores={cores}",
        "",
        "| Case | Tile-rows x tiles per Tensix core | Method | Tiles per wave | ns per all-reduce | vs one_shot |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for case, rows, width_tiles, case_results in results:
        base = statistics.median(case_results["one_shot"][1])
        for label, (waves, values) in case_results.items():
            median = statistics.median(values)
            lines.append(
                f"| {case} | {rows} x {width_tiles} | {label} | {rows // waves * width_tiles} | "
                f"{median:.0f} | {base / median:.2f}x |"
            )
    return "\n".join(lines) + "\n"


def test_tensix_all_reduce_wave_pipeline_correctness(device):
    group_size, num_groups = _group(device)
    for case, rows, width_tiles in _cases():
        methods = _methods(rows, group_size)
        tt_input, expected = _make_input(device, group_size, num_groups, rows, width_tiles)
        _check_all(tt_input, expected, methods, group_size, num_groups, rows, width_tiles)
        # kernel_iters > 1 re-uses every CB and semaphore across iterations inside one launch.
        _check_all(tt_input, expected, methods, group_size, num_groups, rows, width_tiles, kernel_iters=3)
        logger.info(f"PASS {case} rows={rows} width_tiles={width_tiles} methods={[m[0] for m in methods]}")


def test_tensix_all_reduce_wave_pipeline_device_perf(device):
    """Correctness-gate every method, then measure kernel duration; perf never fails a test."""
    group_size, num_groups = _group(device)
    trials = _int("WP_TRIALS", 1)
    kernel_iters = _int("WP_KERNEL_ITERS", 1)
    results = []
    for case, rows, width_tiles in _cases():
        methods = _methods(rows, group_size)
        tt_input, expected = _make_input(device, group_size, num_groups, rows, width_tiles)
        _check_all(tt_input, expected, methods, group_size, num_groups, rows, width_tiles, kernel_iters)
        runners = {
            label: (
                lambda variant=variant, waves=waves: _run(
                    tt_input, variant, waves, group_size, num_groups, rows, width_tiles, kernel_iters
                )
            )
            for label, variant, waves in methods
        }
        samples = _measure(device, runners, trials, kernel_iters)
        results.append((case, rows, width_tiles, {label: (waves, samples[label]) for label, _, waves in methods}))

    report = _format_report(
        results,
        box=socket.gethostname(),
        arch=_arch_label(device),
        group_size=group_size,
        num_groups=num_groups,
        trials=trials,
        kernel_iters=kernel_iters,
    )
    logger.info("\n" + report)
    if path := os.environ.get("WP_REPORT"):
        Path(path).write_text(report)


# (id, rows, width_tiles, waves, kind, threshold): baseline one_shot vs the lever at `waves`.
# helps -> speedup >= max(1.05, 1 + half the measured gain); does_not_help -> speedup <= 1.05.
_REGIMES = [
    ("helps_tall_32x4_4waves", 32, 4, 4, "helps", 1.17),  # BH p150b 1.34x
    ("does_not_help_short_8x1_4waves", 8, 1, 4, "does_not_help", 1.05),  # BH p150b 0.92x (2 tiles per wave)
]


@pytest.mark.parametrize(
    "rows, width_tiles, waves, kind, threshold",
    [r[1:] for r in _REGIMES],
    ids=[r[0] for r in _REGIMES],
)
def test_tensix_all_reduce_wave_pipeline_regimes(device, rows, width_tiles, waves, kind, threshold):
    """Interleaved A/B of one_shot vs `waves` waves; median of 5 trials."""
    group_size, num_groups = _group(device)
    methods = [("one_shot", "one_shot", 1), (f"waves={waves}", "waves", waves)]
    tt_input, expected = _make_input(device, group_size, num_groups, rows, width_tiles)
    _check_all(tt_input, expected, methods, group_size, num_groups, rows, width_tiles)
    runners = {
        label: (lambda variant=variant, w=waves_: _run(tt_input, variant, w, group_size, num_groups, rows, width_tiles))
        for label, variant, waves_ in methods
    }
    samples = _measure(device, runners, 5, 1)
    base = statistics.median(samples["one_shot"])
    lever = statistics.median(samples[f"waves={waves}"])
    speedup = base / lever
    logger.info(
        f"REGIME {kind} rows={rows} width_tiles={width_tiles} waves={waves} one_shot={base:.0f}ns "
        f"waves={lever:.0f}ns speedup={speedup:.3f}x arch={_arch_label(device)}"
    )
    if kind == "helps":
        assert speedup >= threshold, f"expected >= {threshold}x, got {speedup:.3f}x"
    else:
        assert speedup <= threshold, f"expected <= {threshold}x, got {speedup:.3f}x"
