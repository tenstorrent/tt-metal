# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device-profiler perf tests for eltwise_chain, measured as A/B ratios AGAINST HAND-WRITTEN RAW LLK
(board/clock-independent, unlike an absolute-ns baseline). Each test isolates ONE chain feature:

  1. HOISTING — helper's hoisted init (uniform chain) must be FASTER than raw that re-inits per tile
     (raw_exp_unhoisted.cpp) and ~EQUAL to raw that also hoists (raw_exp_hoisted.cpp) [no tax].

  2. BLOCKING — what block_size buys when the init is an EXPENSIVE, non-hoistable SFPU LUT. Chain
     gelu(A)*B (mirrors moe_compute): {Gelu, MulBinary} is non-uniform, so gelu's LUT init is emitted
     per block-iteration; block_size=N loads it once per N tiles. helper blk=1 vs blk=CMP_BLK => GAIN;
     helper blk=1 vs raw_gelu_noblock => no tax. All Bulk, so each CB holds all n (n stays L1-sized).

Functional configs run the op ITERS times + a correctness sanity; perf tests re-run them under the
device profiler (run_device_perf_detailed, DEVICE KERNEL ns for GenericOpDeviceOperation). Sweep N so
the trend is robust to per-launch jitter (profiler STD ~0.02%).
"""

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

OP = "GenericOpDeviceOperation"
ITERS = 20
PERF = "tests/ttnn/unit_tests/kernel_lib/test_chain_perf.py"

AXES = "ttnn/cpp/ttnn/kernel_lib/tests/axes"
# Helper chains
HOIST_SINGLE = f"{AXES}/hoist_single_call.cpp"  # helper, streaming, init hoisted once
HOIST_PERTILE = f"{AXES}/hoist_per_tile.cpp"  # helper, streaming, init per tile
BLOCK_CHUNKED = f"{AXES}/block_exp_chunked.cpp"  # helper, Chunked block-capable (functional only)
BLOCK_GELU = f"{AXES}/block_gelu.cpp"  # helper gelu(A)*B — expensive non-hoistable LUT init, block arg
FUSED = f"{AXES}/fused_chain.cpp"  # FPU add + Exp + DestReuse mul (functional only)
# Hand-written raw-LLK baselines (same compute, differ only in the one axis under test)
RAW_HOIST_HOISTED = f"{AXES}/raw_exp_hoisted.cpp"  # raw, streaming, init hoisted once
RAW_HOIST_UNHOISTED = f"{AXES}/raw_exp_unhoisted.cpp"  # raw, streaming, init per tile
RAW_GELU_NOBLOCK = f"{AXES}/raw_gelu_noblock.cpp"  # raw gelu(A)*B, Bulk, LUT init per tile, NO blocking

# block_size = 4: gelu(A)*B uses 2 DEST slots/tile (D0=gelu(A), D1=B), so lane_width=2 and
# block_size*2 must fit the 8 half-sync DEST slots — 4 is the max without full-sync.
CMP_BLK = 4

# Init-hoisting A/B: helper vs raw, hoisted vs unhoisted. All four compute the identical exp(x).
HOIST_VARIANTS = {
    "helper_hoisted": HOIST_SINGLE,
    "helper_unhoisted": HOIST_PERTILE,
    "raw_hoisted": RAW_HOIST_HOISTED,
    "raw_unhoisted": RAW_HOIST_UNHOISTED,
}
# Block A/B on the EXPENSIVE-init chain gelu(A)*B — three variants, all Bulk lifecycle:
#   helper_block   : helper with block_size=CMP_BLK  (gelu LUT init once per block-iteration)
#   helper_noblock : helper with block_size=1        (gelu LUT init per tile) -> isolates block_size
#   raw_noblock    : hand-written, block_size=1, LUT init per tile -> the no-abstraction-tax reference
# helper_noblock vs helper_block => what block_size buys; helper_noblock vs raw_noblock => no tax.
BLOCK_VARIANTS = {
    "helper_block": (BLOCK_GELU, lambda n: [n, CMP_BLK]),
    "helper_noblock": (BLOCK_GELU, lambda n: [n, 1]),
    "raw_noblock": (RAW_GELU_NOBLOCK, lambda n: [n]),
}

# Tile-count sweeps: small (overhead-dominated) -> large (work-dominated).
HOIST_N = [64, 512, 4096]
BLOCK_N = [64, 512, 2048]  # Chunked functional sweep (CB stays small)
BLOCK_SIZES = [1, 8]  # helper-only functional block_size sweep
# The block PERF comparison is Bulk (waited upfront); gelu(A)*B has 2 input CBs + 1 output, each
# holding all n tiles -> keep n L1-sized (192 tiles bf16 * 3 CBs ~ 1.18 MB; 256 overflows the 1.5 MB L1).
# All values divisible by CMP_BLK so the block helper needs no remainder handling.
BLOCK_CMP_N = [64, 128, 192]

# Lifecycle functional configs (fused exp(A+B)*C) — correctness coverage only; NO perf test.
LIFECYCLE_NS = [64, 128, 1024]
MAX_CHUNK = 8
BULK_BATCH = 64  # Bulk window per chain call (bounded; CB = 2*BULK_BATCH pages, independent of N)

# A/B invariants (ratios, board-independent). Directional, not absolute-ns baselines.
GAIN_MIN = 0.015  # hoisting must be >= 1.5% faster than unhoisted raw (observed ~2.5%)
TAX_TOL = 0.04  # helper within 4% of hand-written raw -> "near-zero abstraction tax"
BLOCK_GAIN_MIN = 0.01  # block_size amortizing the gelu LUT init measures ~1.5-1.8% (blk4); guard >= 1%


# =============================================================================
# Functional configs (profiled below). Each runs the op ITERS times + a correctness sanity.
# =============================================================================
def _run_hoist(device, n, kernel):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=2002)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel, [n], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op([tt_in, tt_out], program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    assert torch.allclose(torch.exp(torch_in.to(torch.float32)), res, atol=0.1, rtol=0.1)


def _run_block(device, n, block_size, kernel):
    """Chunked block path — CB holds only ~block_size tiles (functional block_size sweep)."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=2001)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    pages = max(4, 2 * block_size)
    cbs = [lib.cb_descriptor(0, dt, pages, cg), lib.cb_descriptor(16, dt, pages, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel, [n, block_size], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op([tt_in, tt_out], program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    assert torch.allclose(torch.exp(torch_in.to(torch.float32)), res, atol=0.1, rtol=0.1)


def _run_gelu_bulk(device, n, kernel, cta):
    """BULK gelu(A)*B — input+output waited upfront, so each of the 2 input CBs + the output CB holds
    all n tiles. `cta` is the compute kernel's compile-time args (helper: [n, blk]; raw: [n]). Used for
    every block variant so the ONLY difference measured is block_size (gelu's LUT init per block vs per
    tile). gelu is the WH tanh-ish approximation, so the golden uses approximate='tanh' + loose tol."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    ta, tt_a = lib.make_input(shape, dt, device, seed=2007)
    tb, tt_b = lib.make_input(shape, dt, device, seed=2008)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, n, cg), lib.cb_descriptor(1, dt, n, cg), lib.cb_descriptor(16, dt, n, cg)]
    reader = lib.build_reader_kernel([tt_a, tt_b], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel, cta, cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    golden = torch.nn.functional.gelu(ta.to(torch.float32), approximate="tanh") * tb.to(torch.float32)
    assert torch.allclose(golden, res, atol=0.2, rtol=0.2), "gelu(A)*B mismatch"


def _run_lifecycle(device, mode, n):
    """Fused chain out = exp(A+B)*C over n tiles; `mode` selects the lifecycle + block_size + CB sizing.
    Functional correctness only (no perf test): exercises a realistic fused FPU+SFPU+DestReuse chain."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    ta, tt_a = lib.make_input(shape, dt, device, seed=2003)
    tb, tt_b = lib.make_input(shape, dt, device, seed=2004)
    tc, tt_c = lib.make_input(shape, dt, device, seed=2005)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    if mode == "bulk1":  # Bulk, batched window, block_size 1 (no blocking)
        cta, pages = [n, 1, 0, BULK_BATCH], 2 * BULK_BATCH
    elif mode == "bulk8":  # Bulk, batched window, block_size max (blocking within Bulk)
        cta, pages = [n, MAX_CHUNK, 0, BULK_BATCH], 2 * BULK_BATCH
    else:  # chunk8: Chunked, single call, block_size max
        cta, pages = [n, MAX_CHUNK, 1, 0], 2 * MAX_CHUNK
    cbs = [lib.cb_descriptor(i, dt, pages, cg) for i in (0, 1, 2)] + [lib.cb_descriptor(16, dt, pages, cg)]
    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(FUSED, cta, cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op([tt_a, tt_b, tt_c, tt_out], program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    golden = torch.exp((ta + tb).to(torch.float32)) * tc.to(torch.float32)
    assert torch.allclose(golden, res, atol=0.15, rtol=0.15), "fused exp(A+B)*C mismatch"


@pytest.mark.parametrize("n", HOIST_N)
@pytest.mark.parametrize("variant", list(HOIST_VARIANTS))
def test_func_hoist(device, variant, n):
    _run_hoist(device, n, HOIST_VARIANTS[variant])


@pytest.mark.parametrize("n", BLOCK_N)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_func_block(device, n, block_size):
    _run_block(device, n, block_size, BLOCK_CHUNKED)


@pytest.mark.parametrize("n", BLOCK_CMP_N)
@pytest.mark.parametrize("variant", list(BLOCK_VARIANTS))
def test_func_block_cmp(device, variant, n):
    kernel, cta_fn = BLOCK_VARIANTS[variant]
    _run_gelu_bulk(device, n, kernel, cta_fn(n))


@pytest.mark.parametrize("n", LIFECYCLE_NS)
@pytest.mark.parametrize("mode", ["bulk1", "bulk8", "chunk8"])
def test_func_lifecycle(device, mode, n):
    _run_lifecycle(device, mode, n)


# =============================================================================
# Perf tests — profile each config across the n sweep and compare real DEVICE KERNEL ns AGAINST
# hand-written raw LLK. Assertions are A/B ratios (design invariants), not absolute-ns baselines.
# =============================================================================
def _device_kernel_ns(node, subdir):
    from models.perf.device_perf_utils import run_device_perf_detailed

    command = f'pytest "{PERF}::{node}" -v'
    results = run_device_perf_detailed(command=command, subdir=subdir, cols=["DEVICE KERNEL"], op_name=OP)
    return results["DEVICE KERNEL"]["AVG"]


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("n", HOIST_N)
def test_perf_hoisting_device(n):
    """Helper's hoisted init vs hand-written raw LLK. Two invariants at tile count n:
    (1) GAIN   — helper_hoisted is faster than raw_unhoisted (init per tile) by > GAIN_MIN;
    (2) NO TAX — helper_hoisted is within ±TAX_TOL of raw_hoisted (init hoisted by hand)."""

    def ns(variant):
        return _device_kernel_ns(f"test_func_hoist[variant={variant}-n={n}]", f"eltwise_hoist_{variant}_n{n}")

    helper_h = ns("helper_hoisted")
    raw_h = ns("raw_hoisted")
    raw_u = ns("raw_unhoisted")
    logger.info(
        f"[hoist n={n}] helper_hoisted {helper_h:.0f} | raw_hoisted {raw_h:.0f} | raw_unhoisted {raw_u:.0f} | "
        f"GAIN(raw_unhoisted/helper) x{raw_u / helper_h:.3f} | TAX(helper/raw_hoisted) x{helper_h / raw_h:.3f}"
    )
    assert raw_u / helper_h > 1 + GAIN_MIN, (
        f"hoisting gain too small at n={n}: helper_hoisted {helper_h:.0f} vs raw_unhoisted {raw_u:.0f} "
        f"(x{raw_u / helper_h:.3f}, need > {1 + GAIN_MIN:.3f}). Hoisting stopped paying off."
    )
    assert abs(helper_h / raw_h - 1) < TAX_TOL, (
        f"abstraction tax at n={n}: helper_hoisted {helper_h:.0f} vs raw_hoisted {raw_h:.0f} "
        f"(x{helper_h / raw_h:.3f}, need within ±{TAX_TOL * 100:.0f}%). Helper diverged from hand-written."
    )


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("n", BLOCK_CMP_N)
def test_perf_blocking_device(n):
    """What block_size buys when the init is an expensive, non-hoistable SFPU LUT (chain gelu(A)*B).
    (1) GAIN — helper blk=1 (LUT per tile) vs helper blk=CMP_BLK (LUT per block) > BLOCK_GAIN_MIN faster;
    (2) NO TAX — helper blk=1 vs hand-written raw_noblock (both LUT per tile) within TAX_TOL. All Bulk."""

    def ns(variant):
        return _device_kernel_ns(f"test_func_block_cmp[variant={variant}-n={n}]", f"eltwise_gelu_{variant}_n{n}")

    helper_b = ns("helper_block")  # block_size = CMP_BLK
    helper_nb = ns("helper_noblock")  # block_size = 1
    raw_nb = ns("raw_noblock")  # hand-written, block_size = 1
    logger.info(
        f"[gelu-block n={n}] helper_blk{CMP_BLK} {helper_b:.0f} | helper_blk1 {helper_nb:.0f} | raw_blk1 {raw_nb:.0f} | "
        f"GAIN(blk1/blk{CMP_BLK}) x{helper_nb / helper_b:.3f} | TAX(helper_blk1/raw) x{helper_nb / raw_nb:.3f}"
    )
    assert helper_nb / helper_b > 1 + BLOCK_GAIN_MIN, (
        f"block_size gain too small at n={n}: blk1 {helper_nb:.0f} vs blk{CMP_BLK} {helper_b:.0f} "
        f"(x{helper_nb / helper_b:.3f}, need > {1 + BLOCK_GAIN_MIN:.3f}). block_size stopped amortizing the LUT init."
    )
    assert abs(helper_nb / raw_nb - 1) < TAX_TOL, (
        f"abstraction tax at n={n}: helper blk1 {helper_nb:.0f} vs raw blk1 {raw_nb:.0f} "
        f"(x{helper_nb / raw_nb:.3f}, need within ±{TAX_TOL * 100:.0f}%)."
    )
