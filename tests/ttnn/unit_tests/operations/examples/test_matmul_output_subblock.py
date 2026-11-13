# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the `matmul_output_subblock` compute example.

A single-core, sharded-L1 matmul (via the matmul_block helper) isolates the output-subblock SHAPE
and the SRC-register argument reuse it buys: a wide subblock (1x8) reuses one A-tile across 8 B-tiles;
a tall one (8x1) reuses one B-tile across 8 A-rows; 1x1 reuses nothing. Same correct C. See
ttnn/ttnn/operations/examples/matmul_output_subblock/README.md.

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_correctness
    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/examples/test_matmul_output_subblock.py::test_matmul_output_subblock_device_perf

The device-perf test writes its markdown report only when MOS_REPORT names a path.
"""

import os

# Enable the on-device profiler IN-PROCESS (all three, before the device opens). Module-scoped
# via setdefault so it neither perturbs other examples nor overrides an outer tracy run.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import socket
import statistics
from pathlib import Path

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.examples.perf_test_utils import check_regime, interleaved_ns, read_kernel_ns
from ttnn.operations.examples.matmul_output_subblock import (
    matmul_output_subblock,
    create_sharded_memory_config,
    VARIANTS,
    SUBBLOCK,
)
from ttnn.operations.examples.matmul_output_subblock.matmul_output_subblock import create_program_descriptor

from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = 32

# Kt=1 (a single K-tile) isolates the output-block argument reuse from K accumulation, and frees the
# full 8-tile DEST for wide/tall subblocks. Mt/Nt sized for meaningful single-core time.
MT = int(os.environ.get("MOS_MT", "16"))  # output tile rows
NT = int(os.environ.get("MOS_NT", "16"))  # output tile cols
KT = int(os.environ.get("MOS_KT", "1"))  # contraction tiles (single K-block)
KERNEL_ITERS = int(os.environ.get("MOS_ITERS", "100"))  # in-kernel repeat (steady-state)
N_WARMUP = 3
N_PROFILE_ITERS = int(os.environ.get("MOS_TRIALS", "10"))
_INNER = 5
VARIANT_LIST = tuple(os.environ.get("MOS_VARIANTS", ",".join(VARIANTS)).split(","))
REPORT_PATH = os.environ.get("MOS_REPORT")  # a report file is written only when the caller names a path
PCC = 0.99


def _make_inputs(device):
    torch.manual_seed(0)
    a = torch.randn(MT * TILE, KT * TILE, dtype=torch.float32) * 0.1
    b = torch.randn(KT * TILE, NT * TILE, dtype=torch.float32) * 0.1
    tt_a = ttnn.from_torch(
        a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(MT, KT),
    )
    tt_b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(KT, NT),
    )
    return tt_a, tt_b, a, b


def _measure_ns(device, run_fn):
    """Median ns/op over N_PROFILE_ITERS rounds (each round = mean of _INNER launches), plus std%."""
    for _ in range(N_WARMUP):
        run_fn()
    ttnn.synchronize_device(device)
    read_kernel_ns(device)  # flush warmup window
    samples = []
    for _ in range(N_PROFILE_ITERS):
        for _ in range(_INNER):
            run_fn()
        total = read_kernel_ns(device)
        if total is None:
            return None, None
        samples.append(total / _INNER)
    med = statistics.median(samples)
    return med, (statistics.pstdev(samples) / med * 100.0 if med else float("nan"))


@pytest.mark.parametrize("variant", VARIANTS)
def test_matmul_output_subblock_correctness(device, variant):
    """Every subblock shape computes the identical, correct C = A @ B."""
    tt_a, tt_b, a, b = _make_inputs(device)
    expected = a @ b
    out = ttnn.to_torch(matmul_output_subblock(tt_a, tt_b, variant=variant, kernel_iters=1)).to(torch.float32)
    assert list(out.shape) == list(expected.shape), f"{out.shape} != {expected.shape}"
    assert_with_pcc(expected, out, PCC)


def test_matmul_output_subblock_device_perf(device):
    """Device kernel duration across output-subblock shapes (perf is evidence, not a pass/fail).

    Wide (1x8, 2x4) reuse an A-tile across B-tiles; tall (8x1, 4x2) reuse a B-tile across A-rows;
    1x1 reuses nothing. Bigger subblock -> fewer SRC operand loads (+ fewer per-subblock cycles).
    """
    arch = os.environ.get("ARCH_NAME", str(device.arch()))
    box = socket.gethostname()

    tt_a, tt_b, _, _ = _make_inputs(device)
    tt_c = ttnn.allocate_tensor_on_device(
        ttnn.Shape([MT * TILE, NT * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(MT, NT),
    )

    ns = {}  # variant -> (median_ns, std_pct)
    for variant in VARIANT_LIST:
        desc = create_program_descriptor(tt_a, tt_b, tt_c, variant=variant, kernel_iters=KERNEL_ITERS)
        run_fn = lambda a=tt_a, b=tt_b, c=tt_c, d=desc: ttnn.generic_op([a, b, c], d)
        med, std = _measure_ns(device, run_fn)
        assert med is not None, f"profiler produced no data for {variant} (profiler-enabled build?)"
        ns[variant] = (med, std)

    base = ns.get("sb_1x1", next(iter(ns.values())))[0]
    lines = [
        "",
        "=== matmul_output_subblock device perf (single-core sharded-L1 matmul) — subblock shape / SRC reuse ===",
        f"    box={box}  arch={arch}  cores=1  placement=single-core sharded-L1  "
        f"M={MT}t N={NT}t K={KT}t  iters={KERNEL_ITERS}  rounds={N_PROFILE_ITERS}x{_INNER}",
        f"    {MT*NT} output tiles; wide reuses A across B, tall reuses B across A; DEST=8 tiles",
        f"    {'variant':<10}  {'sb(hxw)':>8}  {'reuse':>6}  {'ns/op':>12}  {'±%':>5}  {'vs 1x1':>8}",
    ]
    for variant in VARIANT_LIST:
        sb_h, sb_w = SUBBLOCK[variant]
        reuse = "-" if (sb_h == 1 and sb_w == 1) else ("B" if sb_h > sb_w else "A")
        m, s = ns[variant]
        tag = "  (base)" if variant == "sb_1x1" else f"  {base/m:5.2f}x"
        lines.append(f"    {variant:<10}  {f'{sb_h}x{sb_w}':>8}  {reuse:>6}  {m:>12.1f}  {s:>5.1f}{tag}")
    logger.info("\n".join(lines))

    md = [
        "# matmul_output_subblock — device report",
        "",
        f"- box: `{box}`",
        f"- arch: {arch}",
        f"- shape: M={MT} N={NT} K={KT} tiles (single K-block); kernel_iters={KERNEL_ITERS}; rounds={N_PROFILE_ITERS}x{_INNER}",
        f"- output tiles: {MT*NT}; DEST capacity = {8} tiles (fp16, no fp32 accumulator)",
        "- metric: DEVICE KERNEL DURATION [ns], median over rounds (±% = pstdev/median)",
        "- takeaway: the matmul_block SRC-register operand reuse is the lever. A wide subblock (sb_w)",
        "  loads one A-tile and reuses it across sb_w B-tiles; a tall subblock (sb_h) reuses one B-tile",
        "  across sb_h A-rows; 1x1 reloads both operands per output tile. Bigger subblock -> fewer SRC",
        "  loads (+ fewer per-subblock cycles) -> faster, up to the DEST budget (8 tiles).",
        "",
        "| variant | sb (h×w) | reuses | ns/op | ±% | vs 1x1 |",
        "|---------|----------|--------|-------|----|--------|",
    ]
    for variant in VARIANT_LIST:
        sb_h, sb_w = SUBBLOCK[variant]
        reuse = "none" if (sb_h == 1 and sb_w == 1) else ("B (across A)" if sb_h > sb_w else "A (across B)")
        m, s = ns[variant]
        md.append(f"| {variant} | {sb_h}×{sb_w} | {reuse} | {m:.1f} | {s:.1f} | {base/m:.2f}x |")
    if REPORT_PATH:
        try:
            Path(REPORT_PATH).write_text("\n".join(md) + "\n")
            logger.info(f"[matmul_output_subblock] wrote {REPORT_PATH}")
        except OSError as e:
            logger.warning(f"[matmul_output_subblock] could not write report: {e}")


# =============================================================================
# Regimes: interleaved A,B,A,B... trials per variant, verdict from `perf_test_utils.check_regime`
# =============================================================================
_REGIME_TRIALS = 11
_REGIME_ITERS = 20


def _regime_inputs(device, mt, nt, kt):
    torch.manual_seed(0)
    a = torch.randn(mt * TILE, kt * TILE, dtype=torch.float32) * 0.1
    b = torch.randn(kt * TILE, nt * TILE, dtype=torch.float32) * 0.1
    mk = lambda t, h, w: ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(h, w),
    )
    tt_c = ttnn.allocate_tensor_on_device(
        ttnn.Shape([mt * TILE, nt * TILE]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        create_sharded_memory_config(mt, nt),
    )
    return mk(a, mt, kt), mk(b, kt, nt), tt_c, a @ b


# (Mt, Nt, Kt tiles, baseline, lever, kind, measured speedup per architecture)
#   helps:          16x16 output tiles, Kt=1: sb_1x1 -> sb_2x4 (8-tile subblock, A reused across 4 B-tiles).
#   measure:        tall sb_8x1 -> wide sb_1x8 (same problem), and Kt=4 sb_1x1 -> sb_2x4: the verdict flips between boards.
#   does_not_help:  2x2 output tiles, Kt=1: sb_1x1 -> sb_2x2. Only 4 output tiles per pass, so there is no
#                   operand reload left to amortize.
# The bound each regime is asserted against is derived from the speedup measured on the architecture it runs on
# (`perf_test_utils.regime_bound`): "helps" asserts half the measured gain but never closer than 10% to the
# measurement; "does_not_help" asserts <= max(1.0, 1.10 x the highest measured value) (so the Wormhole 1.02x
# bound is 1.122x). "measure" regimes only log, as does an architecture with no number below; a missed bound is
# re-measured once and re-checked on the pooled samples. Numbers per board in README.md. Single Tensix core, so no
# grid guard is needed.
_REGIMES = [
    pytest.param(16, 16, 1, "sb_1x1", "sb_2x4", "helps", {"wh": 1.75, "bh": 1.46}, id="helps_16x16t_Kt1_1x1_to_2x4"),
    pytest.param(
        16,
        16,
        1,
        "sb_8x1",
        "sb_1x8",
        "measure",
        {"wh": 1.40, "bh": 1.00},
        id="measure_16x16t_Kt1_tall8x1_to_wide1x8",
    ),
    pytest.param(
        16, 16, 4, "sb_1x1", "sb_2x4", "measure", {"wh": 1.69, "bh": 1.10}, id="measure_16x16t_Kt4_1x1_to_2x4"
    ),
    pytest.param(2, 2, 1, "sb_1x1", "sb_2x2", "does_not_help", {"wh": 1.02, "bh": 0.97}, id="flat_2x2t_Kt1_1x1_to_2x2"),
]


@pytest.mark.parametrize("mt,nt,kt,base_variant,lever_variant,kind,measured", _REGIMES)
def test_matmul_output_subblock_regimes(device, mt, nt, kt, base_variant, lever_variant, kind, measured):
    tt_a, tt_b, tt_c, expected = _regime_inputs(device, mt, nt, kt)
    for variant in (base_variant, lever_variant):  # correctness gate
        out = ttnn.to_torch(matmul_output_subblock(tt_a, tt_b, variant=variant, kernel_iters=1)).to(torch.float32)
        assert_with_pcc(expected, out, PCC)

    runners = {}
    for name, variant in (("base", base_variant), ("lever", lever_variant)):
        desc = create_program_descriptor(tt_a, tt_b, tt_c, variant=variant, kernel_iters=_REGIME_ITERS)
        runners[name] = lambda d=desc: ttnn.generic_op([tt_a, tt_b, tt_c], d)

    def measure():
        s = interleaved_ns(device, runners, trials=_REGIME_TRIALS, per_launch_divisor=_REGIME_ITERS)
        return s["base"], s["lever"]

    check_regime(
        device,
        f"matmul_output_subblock M={mt} N={nt} K={kt} {base_variant}->{lever_variant}",
        kind,
        measured,
        measure,
        detail=f"M={mt} N={nt} K={kt} tiles, 1 Tensix core, ns per matmul",
    )
