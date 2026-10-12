# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `eltwise_l1_vs_dest_accumulate` compute example.

A running L1 accumulator is built by folding B input tiles onto it, ranked by accumulator L1
traffic: `rmw` round-trips `acc` through unpack+add+pack every tile; `pack_l1_acc` lets the PACK
engine fold two-tile DEST sums onto `acc` in place (never unpacks it); `dest_acc` keeps the running
sum in DEST and packs once at the end. Same fp32 sum, single core, sharded L1. See
ttnn/ttnn/operations/examples/eltwise_l1_vs_dest_accumulate/README.md.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_eltwise_l1_vs_dest_accumulate.py::test_eltwise_l1_vs_dest_accumulate_device_perf
"""

import os

os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from ttnn.operations.examples.eltwise_l1_vs_dest_accumulate import (
    eltwise_l1_vs_dest_accumulate,
    create_sharded_memory_config,
    VARIANTS,
)
from ttnn.operations.examples.eltwise_l1_vs_dest_accumulate.eltwise_l1_vs_dest_accumulate import (
    create_program_descriptor,
)

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


TILE = 32

# Single L1 accumulator tile (the helper's L1Accumulation lifecycle folds the stream onto one tile).
TILES_PER_BLOCK = 1
NUM_BLOCKS = int(os.environ.get("ELDA_B", "64"))
KERNEL_ITERS = int(os.environ.get("ELDA_ITERS", "100"))
N_WARMUP = 3
N_PROFILE_ITERS = int(os.environ.get("ELDA_TRIALS", "10"))
_INNER = 5
VARIANT_LIST = tuple(os.environ.get("ELDA_VARIANTS", ",".join(VARIANTS)).split(","))
REPORT_PATH = os.environ.get(
    "ELDA_REPORT",
    str(Path(__file__).resolve().parents[5] / "ttnn/ttnn/operations/examples/eltwise_l1_vs_dest_accumulate/report.md"),
)
PCC = 0.9999
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _make_input(device, b=NUM_BLOCKS, t=TILES_PER_BLOCK):
    torch.manual_seed(0)
    x = torch.randn(b * TILE, t * TILE, dtype=torch.float32) * 0.1
    tt_x = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=create_sharded_memory_config(b, t)
    )
    return tt_x, x


def _expected(x, b=NUM_BLOCKS, t=TILES_PER_BLOCK):
    return x.view(b, TILE, t * TILE).sum(0)


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure_ns(device, run_fn):
    """Median ns/op over rounds (each round = mean of _INNER launches), plus std%."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)
    samples = []
    for _ in range(N_PROFILE_ITERS):
        for _ in range(_INNER):
            run_fn()
        total = _read_kernel_ns(device)
        if total is None:
            return None, None
        samples.append(total / _INNER)
    med = statistics.median(samples)
    return med, (statistics.pstdev(samples) / med * 100.0 if med else float("nan"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_eltwise_l1_vs_dest_accumulate_correctness(device, variant):
    """Both variants compute the identical, correct fp32 running sum."""
    tt_x, x = _make_input(device)
    expected = _expected(x)
    out = ttnn.to_torch(eltwise_l1_vs_dest_accumulate(tt_x, variant=variant, num_blocks=NUM_BLOCKS, kernel_iters=1)).to(
        torch.float32
    )
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, PCC)


def test_eltwise_l1_vs_dest_accumulate_device_perf(device):
    """Device kernel duration: rmw vs pack_l1_acc (perf is evidence, not a pass/fail)."""
    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    box = socket.gethostname()

    tt_x, _ = _make_input(device)
    tt_acc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILES_PER_BLOCK * TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(1, TILES_PER_BLOCK),
    )

    ns = {}
    for variant in VARIANT_LIST:
        desc = create_program_descriptor(
            tt_x,
            tt_acc,
            variant=variant,
            num_blocks=NUM_BLOCKS,
            kernel_iters=KERNEL_ITERS,
        )
        run_fn = lambda x=tt_x, a=tt_acc, d=desc: ttnn.generic_op([x, a], d)
        med, std = _measure_ns(device, run_fn)
        assert med is not None, f"profiler produced no data for {variant} (profiler-enabled build?)"
        ns[variant] = (med, std)

    base = ns.get("rmw", next(iter(ns.values())))[0]
    steps = NUM_BLOCKS * KERNEL_ITERS
    lines = [
        "",
        "=== eltwise_l1_vs_dest_accumulate device perf (single-core sharded-L1 running sum) — accumulate mechanism ===",
        f"    box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"blocks={NUM_BLOCKS}  tiles/block={TILES_PER_BLOCK}  iters={KERNEL_ITERS}  rounds={N_PROFILE_ITERS}x{_INNER}",
        f"    {'method':<14}  {'ns/op':>12}  {'±%':>5}  {'vs rmw':>8}",
    ]
    for variant in VARIANT_LIST:
        m, s = ns[variant]
        tag = "  (base)" if variant == "rmw" else f"  {base/m:5.2f}x"
        lines.append(f"    {variant:<14}  {m:>12.1f}  {s:>5.1f}{tag}")
    logger.info("\n".join(lines))

    md = [
        "# eltwise_l1_vs_dest_accumulate — device report",
        "",
        f"- box: `{box}`",
        f"- arch: {arch}",
        f"- shape: {NUM_BLOCKS} blocks x {TILES_PER_BLOCK} tiles (fp32); kernel_iters={KERNEL_ITERS}; rounds={N_PROFILE_ITERS}x{_INNER}",
        f"- accumulation steps per launch: {steps}",
        "- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)",
        "- takeaway: the win tracks how much L1 traffic the accumulator pays. rmw round-trips acc through",
        "  unpack+add+pack every tile; pack_l1_acc only packs acc (packer folds DEST onto it, never",
        "  unpacks) once per pair; dest_acc keeps the running sum in DEST and touches L1 once, at the end.",
        "",
        "| method | ns/op | ±% | vs rmw |",
        "|--------|-------|----|--------|",
    ]
    for variant in VARIANT_LIST:
        m, s = ns[variant]
        md.append(f"| {variant} | {m:.1f} | {s:.1f} | {base/m:.2f}x |")
    try:
        Path(REPORT_PATH).write_text("\n".join(md) + "\n")
        logger.info(f"[eltwise_l1_vs_dest_accumulate] wrote {REPORT_PATH}")
    except OSError as e:
        logger.warning(f"[eltwise_l1_vs_dest_accumulate] could not write report: {e}")


# =============================================================================
# Regimes (Wormhole B0 n150): interleaved A,B,A,B... medians of >= 10 trials per variant
# =============================================================================
_REGIME_TRIALS = 11
_REGIME_ITERS = 100


def _regime_ns(device, tt_x, tt_acc, base_variant, lever_variant, num_blocks):
    """Interleaved per-evaluation device-kernel ns for two variants -> (base_samples, lever_samples)."""
    runs = {}
    for name, variant in (("base", base_variant), ("lever", lever_variant)):
        desc = create_program_descriptor(
            tt_x, tt_acc, variant=variant, num_blocks=num_blocks, kernel_iters=_REGIME_ITERS
        )
        runs[name] = lambda d=desc: ttnn.generic_op([tt_x, tt_acc], d)
    for run in runs.values():
        run()
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # discard warmup window
    samples = {"base": [], "lever": []}
    for trial in range(_REGIME_TRIALS + 1):
        for name, run in runs.items():
            run()
            ns = _read_kernel_ns(device)
            assert ns is not None, f"no profiler data for {name}"
            if trial:  # discard first timed pass
                samples[name].append(ns / _REGIME_ITERS)
    return samples["base"], samples["lever"]


# (steps B, baseline, lever, verdict, measured speedup)
#   helps:          B=64 single-tile steps, rmw -> pack_l1_acc (rmw unpacks+packs acc every tile; the packer folds
#                   a pair per step instead). Measured ~4.1x on Wormhole B0 (5.09x was Blackhole).
#   does_not_help:  B=2 (one pair): pack_l1_acc -> dest_acc. Keeping the sum in DEST instead of folding in L1 saves
#                   nothing when there is a single pair to fold (~1.00x); dest_acc only pays as B grows (~1.3x at B=64).
_REGIMES = [
    pytest.param(64, "rmw", "pack_l1_acc", "helps", 4.10, id="helps_B64_rmw_to_pack_l1_acc"),
    pytest.param(2, "pack_l1_acc", "dest_acc", "does_not_help", 1.00, id="flat_B2_pack_l1_acc_to_dest_acc"),
]


@pytest.mark.parametrize("num_blocks,base_variant,lever_variant,verdict,measured", _REGIMES)
def test_eltwise_l1_vs_dest_accumulate_regimes(device, num_blocks, base_variant, lever_variant, verdict, measured):
    # The regime thresholds below were calibrated on Wormhole B0 (n150). The boundary between
    # "helps" and "does not help" moves with the arch (grid size, DRAM banks, NoC, clock), so on
    # another arch re-measure the regimes and add its own thresholds before enabling this test.
    if not ttnn.device.is_wormhole_b0(device):
        pytest.skip("regime thresholds are calibrated on Wormhole B0 only; re-measure on this arch first")
    tt_x, x = _make_input(device, b=num_blocks)
    expected = _expected(x, b=num_blocks)
    tt_acc = ttnn.allocate_tensor_on_device(
        ttnn.Shape([TILE, TILES_PER_BLOCK * TILE]),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(1, TILES_PER_BLOCK),
    )
    for variant in (base_variant, lever_variant):  # correctness gate
        out = ttnn.to_torch(
            eltwise_l1_vs_dest_accumulate(tt_x, variant=variant, num_blocks=num_blocks, kernel_iters=1)
        ).to(torch.float32)
        assert_with_pcc(expected, out, PCC)

    base, lever = _regime_ns(device, tt_x, tt_acc, base_variant, lever_variant, num_blocks)
    b, l = statistics.median(base), statistics.median(lever)
    speedup = b / l
    arch = {"WORMHOLE_B0": "WH_B0", "BLACKHOLE": "BH"}.get(str(device.arch()).rsplit(".", 1)[-1], str(device.arch()))
    logger.info(
        f"[{arch}] {verdict} B={num_blocks} steps (1 tile each, fp32): {base_variant} {b:.0f} ns "
        f"[{min(base):.0f}..{max(base):.0f}] -> {lever_variant} {l:.0f} ns [{min(lever):.0f}..{max(lever):.0f}] "
        f"= {speedup:.2f}x (trials={len(base)}, ns per accumulation pass)"
    )
    if verdict == "helps":
        assert speedup >= max(1.05, 1 + 0.5 * (measured - 1)), f"expected a clear win, got {speedup:.2f}x"
    else:
        assert speedup <= 1.05, f"expected no meaningful gain (<= 1.05x), got {speedup:.2f}x"
