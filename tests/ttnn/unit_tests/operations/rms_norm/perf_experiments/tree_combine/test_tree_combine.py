# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Correctness gate + device measurement for the rms_norm combine bake-off.

Correctness is the ONLY pass/fail; every duration is recorded, never asserted.

    scripts/run_safe_pytest.sh --run-all \
      tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/tree_combine/test_tree_combine.py

`TC_CASES=flagged|fanin|sweep|extra|crossover|line|all`, or a comma-separated list of
case names, selects which geometries run (default `all`).

MEASURED — Blackhole p150b @1350 MHz, median of 3 fresh dispatches per variant
(spread < 3% everywhere), bf16 activations / float32 stat tiles / HiFi2 /
fp32_dest_acc_en=False on BOTH variants.  `s*B` is the flat root's gathered TILE
count, which is what the win tracks:

    geometry            s   B  s*B    K   flat_root   tree    ratio
    8x4  g1 (W=5120)   32   1   32    8      4467     3473    1.286   <- flagged cell
    8x4  g1             32   1   32    4      4493     3533    1.272
    8x4  g1             32   1   32   16      4484     3847    1.166
    8x4  g1             32   1   32    2      4498     3956    1.137
    7x4  g1 (W=7168)   28   1   28    7      4195     3461    1.212   <- flagged cell
    8x8  g1             64   1   64    8      6736     3822    1.762
    6x3  g1 (W=2304)   18   1   18    6      3404     3056    1.114
    4x4  g1             16   1   16    4      3233     3093    1.045
    2x8  g1             16   1   16    4      3267     3088    1.058
    2x8  g1             16   1   16    8      3233     3123    1.035
    2x8  g1             16   1   16    2      3253     3250    1.001  (K artifact, not a floor)
    8x4  g1             32   8  256    8     27045    17593    1.537
    4x4  g1             16   8  128    4     18864    15404    1.225
    8x1  g8 (W=1024)     8   8   64    4     14867    13222    1.124   <- prefill focus cell
    8x1  g1              8   8   64    4     14842    13114    1.132
    8x1  g8              8   8   64    2     14873    14374    1.035
    8x1  g1              8   4   32    4      7883     7250    1.087
    8x1  g1              8   2   16    4      4385     4118    1.065
    8x1  g1              8   1    8    4      2648     2858    0.927  <- REGRESSION
    8x1  g1              8   1    8    2      2679     3057    0.876  <- REGRESSION
    1x8  g1              8   1    8    4      2507     2616    0.958  <- REGRESSION
    8x1  g8              8   1    8    4      2743     2899    0.946  <- REGRESSION

Reading: the tree wins wherever the flat root gathers >= ~16 stat tiles and loses
by 4-12% at 8 tiles (the extra round trip is not repaid).  The best level-1 fan-in
is the divisor of `s` nearest sqrt(s) — for the flagged rects that IS `rect_w`
(the grid-row split), and the curve is U-shaped in K (K=2 and K=16 both lose ~10
points to K=8 at s=32), i.e. the cost tracks max(K, L): it is the FAN-IN that is
being cut, not the root's math.
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import statistics

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.rms_norm.perf_experiments.tree_combine.tree_combine import (
    VARIANTS,
    Geometry,
    build_layout,
    combine,
    create_stat_memory_config,
    reference_rms_recip,
)

_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"
TILE = 32
EPS = 1e-6
TRIALS = int(os.environ.get("TC_TRIALS", "3"))

# --------------------------------------------------------------------------
# Geometries.  `flagged` are the op's two perf-flagged 2-D sharded cells and the
# tournament's focus prefill cell, reproduced exactly (s, B, S, W all match).
# --------------------------------------------------------------------------
CASE_GROUPS = {
    "flagged": [
        # (1,1,32,5120) WIDTH [32,160] on (8,4): one row-group, s=32, B=1, S=5, W=5120
        ("w5120_8x4", Geometry(rect_w=8, rect_h=4, num_groups=1, block_rows=1, fanin=8, hidden_tiles=5)),
        # (1,1,32,7168) WIDTH [32,256] on (7,4): one row-group, s=28, B=1, S=8, W=7168
        ("w7168_7x4", Geometry(rect_w=7, rect_h=4, num_groups=1, block_rows=1, fanin=7, hidden_tiles=8)),
        # (1,1,8192,1024) BLOCK [1024,128] on (8,8): 8 row-groups of a 1-D line of 8,
        # B=8, S=4, W=1024.  A 2-level tree on a LINE (K=4 -> L=2).
        ("w1024_8x1_g8", Geometry(rect_w=8, rect_h=1, num_groups=8, block_rows=8, fanin=4, hidden_tiles=4)),
    ],
    "fanin": [
        ("w5120_8x4_K4", Geometry(rect_w=8, rect_h=4, num_groups=1, block_rows=1, fanin=4, hidden_tiles=5)),
        ("w5120_8x4_K16", Geometry(rect_w=8, rect_h=4, num_groups=1, block_rows=1, fanin=16, hidden_tiles=5)),
        ("w5120_8x4_K2", Geometry(rect_w=8, rect_h=4, num_groups=1, block_rows=1, fanin=2, hidden_tiles=5)),
    ],
    "sweep": [
        ("rect_4x4", Geometry(rect_w=4, rect_h=4, num_groups=1, block_rows=1, fanin=4, hidden_tiles=5)),
        ("rect_8x8", Geometry(rect_w=8, rect_h=8, num_groups=1, block_rows=1, fanin=8, hidden_tiles=5)),
        ("rect_2x8", Geometry(rect_w=2, rect_h=8, num_groups=1, block_rows=1, fanin=2, hidden_tiles=5)),
        ("rect_4x4_B8", Geometry(rect_w=4, rect_h=4, num_groups=1, block_rows=8, fanin=4, hidden_tiles=5)),
        ("rect_8x4_B8", Geometry(rect_w=8, rect_h=4, num_groups=1, block_rows=8, fanin=8, hidden_tiles=5)),
    ],
    # Is a flat/regressing cell a property of the GEOMETRY or of the K I picked for
    # it?  Same geometries, better-balanced level-1 fan-in.
    "extra": [
        ("rect_2x8_K8", Geometry(rect_w=2, rect_h=8, num_groups=1, block_rows=1, fanin=8, hidden_tiles=5)),
        ("rect_2x8_K4", Geometry(rect_w=2, rect_h=8, num_groups=1, block_rows=1, fanin=4, hidden_tiles=5)),
        ("line_8x1_K2", Geometry(rect_w=8, rect_h=1, num_groups=1, block_rows=1, fanin=2, hidden_tiles=4)),
        ("w1024_8x1_g8_K2", Geometry(rect_w=8, rect_h=1, num_groups=8, block_rows=8, fanin=2, hidden_tiles=4)),
        ("w1024_8x1_g8_K8x1_K4_b1", Geometry(rect_w=8, rect_h=1, num_groups=8, block_rows=1, fanin=4, hidden_tiles=4)),
    ],
    # Where is the crossover?  s*B is the flat root's gathered TILE count.
    "crossover": [
        ("rect_6x3", Geometry(rect_w=6, rect_h=3, num_groups=1, block_rows=1, fanin=6, hidden_tiles=4)),
        ("line_8x1_B2", Geometry(rect_w=8, rect_h=1, num_groups=1, block_rows=2, fanin=4, hidden_tiles=4)),
        ("line_8x1_B4", Geometry(rect_w=8, rect_h=1, num_groups=1, block_rows=4, fanin=4, hidden_tiles=4)),
    ],
    "line": [
        ("line_8x1", Geometry(rect_w=8, rect_h=1, num_groups=1, block_rows=1, fanin=4, hidden_tiles=4)),
        ("line_1x8", Geometry(rect_w=1, rect_h=8, num_groups=1, block_rows=1, fanin=4, hidden_tiles=4)),
        ("line_8x1_B8", Geometry(rect_w=8, rect_h=1, num_groups=1, block_rows=8, fanin=4, hidden_tiles=4)),
    ],
}


def _selected_cases():
    key = os.environ.get("TC_CASES", "all")
    if key == "all":
        cases = []
        for group in ("flagged", "fanin", "sweep", "line"):
            cases.extend(CASE_GROUPS[group])
        return cases
    if key in CASE_GROUPS:
        return CASE_GROUPS[key]
    wanted = set(key.split(","))
    picked = [case for group in CASE_GROUPS.values() for case in group if case[0] in wanted]
    if not picked:
        raise ValueError(f"TC_CASES must be a group {sorted(CASE_GROUPS)}, 'all', or case names; got {key!r}")
    return picked


def _make_stats(device, geometry: Geometry):
    layout = build_layout(device, geometry)
    ncores = len(layout.active_cores)
    B = geometry.block_rows
    torch.manual_seed(1234)
    # Stat-tile magnitudes as the op produces them: entry (i, j) of a partial tile is
    # Sum over the core's S hidden tiles of x^2, so mean ~ S.  Total per row is then
    # ~32*s*S == W, i.e. mean(x^2) ~ 1 and 1/rms ~ 1 — the op's real operating point.
    stats = torch.rand((ncores * TILE, B * TILE), dtype=torch.float32) * (2.0 * geometry.hidden_tiles)
    # Give the 32 rows of every tile a 100x spread of scales, so the 32 combined
    # results span a decade.  Without it every row-sum is ~W and 1/rms ~ 1 for all
    # 32 rows: PCC against a near-constant reference is dominated by rounding and
    # reads as ~0.95 even when the max relative error is 0.7% (measured).
    row_scale = torch.logspace(-1.0, 1.0, TILE, dtype=torch.float32).repeat(ncores)
    stats *= row_scale.unsqueeze(1)
    expected = reference_rms_recip(stats, layout, epsilon=EPS)
    tt_stats = ttnn.from_torch(
        stats,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_stat_memory_config(device, geometry),
    )
    return layout, tt_stats, expected


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def _check(actual_tensor, expected, geometry: Geometry):
    """Column 0 of every output tile carries the REDUCE_ROW result."""
    actual = ttnn.to_torch(actual_tensor).to(torch.float32)
    got = torch.stack([actual[:, r * TILE] for r in range(geometry.block_rows)], dim=1)
    rel = ((got - expected).abs() / expected.abs().clamp_min(1e-12)).max().item()
    return got, _pcc(got, expected), rel


def _read_kernel_ns(device):
    """(summed ns, program count) since the previous read — one dispatch per read."""
    ttnn.synchronize_device(device)
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data() or {}
    total = 0.0
    count = 0
    for programs in per_chip.values():
        for program in programs:
            analyses = getattr(program, "program_analyses_results", None) or {}
            entry = analyses.get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                count += 1
    return (total, count)


def test_tree_combine(device):
    rows = []
    for name, geometry in _selected_cases():
        try:
            layout, tt_stats, expected = _make_stats(device, geometry)
        except ValueError as exc:  # geometry does not fit this grid
            logger.warning(f"SKIP {name}: {exc}")
            continue

        samples = {}
        quality = {}
        for variant in VARIANTS:
            out = combine(tt_stats, variant=variant, geometry=geometry, epsilon=EPS)
            _got, pcc, rel = _check(out, expected, geometry)
            quality[variant] = (pcc, rel)
            logger.info(f"{name} [{geometry.label}] {variant}: pcc={pcc:.6f} max_rel_err={rel:.3e}")
            # Both gates: PCC catches a structurally wrong combine, max relative
            # error catches a precision loss PCC would hide (and vice versa).
            assert pcc > 0.999, f"{name}/{variant}: combine is wrong (pcc {pcc})"
            assert rel < 0.03, f"{name}/{variant}: combine is imprecise (max rel err {rel})"

            _read_kernel_ns(device)  # drain the correctness dispatch
            durations = []
            for _ in range(TRIALS):
                combine(tt_stats, variant=variant, geometry=geometry, epsilon=EPS)
                ns, programs = _read_kernel_ns(device)
                assert programs == 1, f"{name}/{variant}: profiler returned {programs} programs, want 1"
                durations.append(ns)
            samples[variant] = durations

        base = statistics.median(samples["flat_root"])
        cand = statistics.median(samples["tree"])
        rows.append((name, geometry, base, cand, samples, quality))
        logger.info(
            f"RESULT {name} [{geometry.label}] flat_root={base:.0f} ns "
            f"tree={cand:.0f} ns  ratio={base / cand:.3f}x "
            f"(flat {samples['flat_root']}, tree {samples['tree']})"
        )

    logger.info("=" * 100)
    logger.info(
        f"{'case':<20} {'geometry':<34} {'flat_root ns':>13} {'tree ns':>10} {'speedup':>8} {'pcc flat/tree':>22}"
    )
    for name, geometry, base, cand, samples, quality in rows:
        logger.info(
            f"{name:<20} {geometry.label:<34} {base:>13.0f} {cand:>10.0f} {base / cand:>7.3f}x "
            f"{quality['flat_root'][0]:>10.6f}/{quality['tree'][0]:.6f}"
        )
    logger.info("=" * 100)
