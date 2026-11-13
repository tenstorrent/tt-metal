# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-row-vector all-reduce (`row_sum_all_reduce`): correctness, regimes and the measurement sweep.

Bytes per member on the wire, both directions, and the return path. A row is WIDTH_SHARDED over K
members (K Tensix cores), b tile-rows each; every member ends with the whole-row sums as column 0 of a
float32 tile per tile-row.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import datetime
import socket
import statistics
import subprocess

import pytest
import torch
import ttnn
from loguru import logger

from tests.ttnn.unit_tests.operations.examples.perf_test_utils import (
    check_regime,
    chip_freq_mhz,
    grid_cores,
    interleaved_ns,
    require_cores,
    sweep_gate,
)
from ttnn.operations.examples.tensix_all_reduce_row_vector import (
    ALL_ROW_VECTOR_VARIANTS,
    ROW_VECTOR_VARIANTS,
    build_member_layout,
    create_row_vector_memory_config,
    row_sum_all_reduce,
)

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


def _rv_prepare(device, variants, num_members, tile_rows, width_tiles, kernel_iters):
    """Correctness-gate every variant on one input, then return {variant: zero-arg launch}."""
    tt_input, row_sums = _rv_make_input(device, num_members, tile_rows, width_tiles)
    for variant in variants:
        _rv_run_checked(tt_input, row_sums, variant, num_members, tile_rows, width_tiles, kernel_iters)
    return {v: (lambda v=v: _rv_run(tt_input, v, num_members, tile_rows, width_tiles, kernel_iters)) for v in variants}


def _rv_cell(device, variants, num_members, tile_rows, width_tiles, trials, kernel_iters):
    runners = _rv_prepare(device, variants, num_members, tile_rows, width_tiles, kernel_iters)
    return interleaved_ns(device, runners, trials=trials, per_launch_divisor=kernel_iters)


def _rv_correctness_cases(device):
    capacity = grid_cores(device)
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
        f"box={socket.gethostname()}  arch={device.arch()}  clock={chip_freq_mhz()} MHz (profiler log header)  "
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


@sweep_gate("AR_RV_SWEEP", "AR_SWEEPS")
def test_tensix_all_reduce_row_vector_sweep(device):
    """Correctness-gate every variant per cell, then measure them interleaved. Perf never fails the test."""
    capacity = grid_cores(device)
    members = [k for k in _ints("AR_RV_MEMBERS", "2,4,8,12,16,32,64,128") if 2 <= k <= capacity]
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


# Regimes: the per-row-vector boundary claims, asserted. Interleaved A/B, median of 11 trials per
# variant, kernel-iters 1. `measured` is the speedup (baseline ns / lever ns) per architecture ("bh" =
# Blackhole p150b; none was measured on Wormhole B0, so these only log there). `check_regime` asserts it
# against a bound derived from the number measured on the running architecture
# (`perf_test_utils.regime_bound`) and re-measures once (pooling both passes) before failing. A regime
# needs K Tensix cores and is skipped on a smaller grid.
# Measured BH p150b, 13x10 grid, 1350 MHz, w = 1, kernel-iters 1, N = 5 sweep (report.md, 2026-10-31).
# Partial size on the wire: whole tile (4096 B) -> faces 0/2 (2048 B) -> one row (128 B).
# (id, K, b, w, kind, measured, baseline, lever)
_TU, _FU, _RU = "tile_partial_unicast", "faces_partial_unicast", "row_partial_unicast"
_TM, _FM, _RM = "tile_partial_mcast", "faces_partial_mcast", "row_partial_mcast"
_FRU, _FRM = "faces_partial_faces_return_unicast", "faces_partial_faces_return_mcast"
_RV_REGIMES = [
    ("faces_over_tile_helps_K128_b4_mcast", 128, 4, 1, "helps", {"bh": 1.42}, _TM, _FM),  # 13 261 -> 9 355 ns
    ("faces_over_tile_helps_K8_b4_unicast", 8, 4, 1, "helps", {"bh": 1.16}, _TU, _FU),  # 4 483 -> 3 859 ns
    ("faces_over_tile_does_not_help_K2_b1_unicast", 2, 1, 1, "does_not_help", {"bh": 1.02}, _TU, _FU),
    ("row_over_faces_helps_K64_b1_mcast", 64, 1, 1, "helps", {"bh": 1.44}, _FM, _RM),  # 5 156 -> 3 570 ns
    ("row_over_faces_helps_K12_b4_mcast", 12, 4, 1, "helps", {"bh": 1.14}, _FM, _RM),  # 3 919 -> 3 428 ns
    ("row_over_faces_does_not_help_K4_b1_unicast", 4, 1, 1, "does_not_help", {"bh": 0.89}, _FU, _RU),
    ("row_over_faces_does_not_help_K8_b1_mcast", 8, 1, 1, "does_not_help", {"bh": 0.94}, _FM, _RM),
    ("row_over_faces_does_not_help_K12_b1_mcast", 12, 1, 1, "does_not_help", {"bh": 0.99}, _FM, _RM),
    # Return path: per-owner unicast to all K members -> one root multicast.
    ("mcast_helps_K64_b1_tile", 64, 1, 1, "helps", {"bh": 1.63}, _TU, _TM),  # 10 948 -> 6 736 ns
    ("mcast_helps_K16_b4_faces", 16, 4, 1, "helps", {"bh": 1.25}, _FU, _FM),  # 5 276 -> 4 225 ns
    ("mcast_does_not_help_K4_b1_tile", 4, 1, 1, "does_not_help", {"bh": 0.93}, _TU, _TM),
    ("mcast_does_not_help_K4_b4_faces", 4, 4, 1, "does_not_help", {"bh": 0.92}, _FU, _FM),
    # Return payload: the whole column-0 tile -> only its faces 0/2.
    ("faces_return_helps_K16_b4_unicast", 16, 4, 1, "helps", {"bh": 1.15}, _FU, _FRU),  # 5 276 -> 4 595 ns
    ("faces_return_does_not_help_K16_b4_mcast", 16, 4, 1, "does_not_help", {"bh": 0.67}, _FM, _FRM),
    ("faces_return_does_not_help_K64_b1_unicast", 64, 1, 1, "does_not_help", {"bh": 0.95}, _FU, _FRU),
    # Both levers.
    ("both_help_K128_b4", 128, 4, 1, "helps", {"bh": 4.71}, _TU, _RM),  # 25 546 -> 5 424 ns
    ("both_do_not_help_K4_b1", 4, 1, 1, "does_not_help", {"bh": 0.86}, _TU, _RM),
]
_RV_REGIME_TRIALS = 11


@pytest.mark.parametrize(
    "regime_id, num_members, tile_rows, width_tiles, kind, measured, baseline, lever",
    _RV_REGIMES,
    ids=[r[0] for r in _RV_REGIMES],
)
def test_tensix_all_reduce_row_vector_regimes(
    device, regime_id, num_members, tile_rows, width_tiles, kind, measured, baseline, lever
):
    """Per-row-vector payload: partial size on the wire, return payload and return path."""
    require_cores(device, num_members, what=f"{regime_id} (K = {num_members} members)")
    runners = _rv_prepare(device, (baseline, lever), num_members, tile_rows, width_tiles, 1)

    def measure():
        samples = interleaved_ns(device, runners, trials=_RV_REGIME_TRIALS)
        return samples[baseline], samples[lever]

    placement = build_member_layout(device, num_members).placement
    check_regime(
        device,
        regime_id,
        kind,
        measured,
        measure,
        detail=(
            f"{baseline} -> {lever}, K={num_members} Tensix cores ({placement}), b={tile_rows} tile-rows, "
            f"w={width_tiles} tiles, kernel-iters 1, ns per launch"
        ),
    )
