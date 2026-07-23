# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Device-profiler performance tests for the eltwise_chain helper, swept across tile count.

Follows the SDXL op-unit perf pattern (models/demos/stable_diffusion_xl_base/tests/
test_sdxl_op_unit_test_perf.py): a functional test runs the config; a perf test marked
@models_device_performance_bare_metal re-runs it under the device profiler via
run_device_perf_detailed and reads the real DEVICE KERNEL duration (ns) for "GenericOpDeviceOperation".

WHY SWEEP N: a single small tile count is dominated by fixed per-launch overhead and is not
representative — it can't show how a knob (blocking, init hoisting) scales. We measure SMALL,
MEDIUM and LARGE n so the trend is visible and any conclusion is robust to per-launch jitter
(the profiler value is already very stable: STD ~0.02% at n=512). Blocking uses a CHUNKED
block-capable chain so the CB stays small and n can scale to thousands of tiles (the Bulk+Block
variant is L1-bound to a few hundred). Hoisting is streaming, so n scales freely.

Comparisons are A/B at each n (no hard-coded baseline). The op runs ITERS times per profile.
"""

import pytest
import torch
import ttnn
from loguru import logger

import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

OP = "GenericOpDeviceOperation"
ITERS = 20
PERF = "tests/ttnn/unit_tests/kernel_lib/test_chain_perf.py"

BLOCK_CHUNKED = "ttnn/cpp/ttnn/kernel_lib/tests/axes/block_exp_chunked.cpp"
BLOCK_BULK = "ttnn/cpp/ttnn/kernel_lib/tests/axes/block_exp.cpp"  # Bulk + Block, block_size param
HOIST_SINGLE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_single_call.cpp"  # streaming per-tile
HOIST_PERTILE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/hoist_per_tile.cpp"
FUSED = "ttnn/cpp/ttnn/kernel_lib/tests/axes/fused_chain.cpp"  # FPU add + Exp + DestReuse mul

# Chunked-vs-Bulk comparison on a REALISTIC fused chain (out = exp(A+B)*C: FPU add + Exp + DestReuse
# mul). Neither lifecycle stages the whole window — both keep a BOUNDED CB and process N over many
# iterations, so N scales to thousands (1024 tiles). Three configs isolate the two effects:
#   bulk1  -> Bulk, batched (window=BULK_BATCH), block_size=1   : no blocking, batched-upfront baseline
#   bulk8  -> Bulk, batched (window=BULK_BATCH), block_size=max : blocking WITHIN Bulk (bulk1 -> bulk8)
#   chunk8 -> Chunked, single call, block_size=max             : lifecycle gain (bulk8 -> chunk8)
LIFECYCLE_NS = [64, 128, 1024]
MAX_CHUNK = 8
BULK_BATCH = 64  # Bulk window per chain call (bounded; CB = 2*BULK_BATCH pages, independent of N)

# Tile-count sweep: small (overhead-dominated) -> large (work-dominated).
HOIST_N = [64, 512, 4096]
BLOCK_N = [64, 512, 2048]
BLOCK_SIZES = [1, 8]


# =============================================================================
# Functional configs (profiled below). Each runs the op ITERS times + a correctness sanity.
# =============================================================================
def _run_block_chunked(device, n, block_size):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=2001)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    pages = max(4, 2 * block_size)  # Chunked: CB holds ~block_size tiles, double-buffered
    cbs = [lib.cb_descriptor(0, dt, pages, cg), lib.cb_descriptor(16, dt, pages, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(BLOCK_CHUNKED, [n, block_size], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op([tt_in, tt_out], program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    assert torch.allclose(torch.exp(torch_in.to(torch.float32)), res, atol=0.1, rtol=0.1)


def _run_hoist(device, n, kernel):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    _, tt_in = lib.make_input(shape, dt, device, seed=2002)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel, [n], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    for _ in range(ITERS):
        ttnn.generic_op([tt_in, tt_out], program)
    ttnn.synchronize_device(device)


@pytest.mark.parametrize("n", BLOCK_N)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_func_block(device, n, block_size):
    _run_block_chunked(device, n, block_size)


@pytest.mark.parametrize("n", HOIST_N)
@pytest.mark.parametrize("mode", ["single", "pertile"])
def test_func_hoist(device, n, mode):
    _run_hoist(device, n, HOIST_SINGLE if mode == "single" else HOIST_PERTILE)


def _run_lifecycle(device, mode, n):
    """Fused chain out = exp(A+B)*C over n tiles; `mode` selects the lifecycle + block_size + CB sizing.
    life 0=Bulk (whole window resident, pages=n), 1=Chunked (pages ~2*block_size)."""
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


@pytest.mark.parametrize("n", LIFECYCLE_NS)
@pytest.mark.parametrize("mode", ["bulk1", "bulk8", "chunk8"])
def test_func_lifecycle(device, mode, n):
    _run_lifecycle(device, mode, n)


# =============================================================================
# Perf tests — profile each config across the n sweep and compare real DEVICE KERNEL ns.
# =============================================================================
def _device_kernel_ns(node, subdir):
    from models.perf.device_perf_utils import run_device_perf_detailed

    command = f'pytest "{PERF}::{node}" -v'
    results = run_device_perf_detailed(command=command, subdir=subdir, cols=["DEVICE KERNEL"], op_name=OP)
    return results["DEVICE KERNEL"]["AVG"]


# Recorded device-kernel baselines (ns), measured 2026-06-09 on a single Wormhole core, build_Release
# with device profiler. Assert each measurement within MARGIN of its baseline (SDXL-style regression
# guard) — NOT a directional "blocking helps" claim (it doesn't for this workload; see the table the
# test logs). Profiler is stable to ~0.02%; MARGIN is generous for build/condition drift.
MARGIN = 0.08
HOIST_BASELINE_NS = {
    (64, "single"): 47449,
    (64, "pertile"): 49030,
    (512, "single"): 368649,
    (512, "pertile"): 381460,
    (4096, "single"): 2938401,
    (4096, "pertile"): 3040781,
}
BLOCK_BASELINE_NS = {
    (64, 1): 47458,
    (64, 8): 52653,
    (512, 1): 368676,
    (512, 8): 375529,
    (2048, 1): 1469962,
    (2048, 8): 1482795,
}


def _check_baseline(measured, baseline, label):
    lo, hi = baseline * (1 - MARGIN), baseline * (1 + MARGIN)
    assert lo <= measured <= hi, (
        f"{label}: {measured:.0f} ns outside {baseline} ± {MARGIN*100:.0f}% ({lo:.0f}-{hi:.0f}). "
        f"A real device-perf change — update the baseline if intentional."
    )


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("n", HOIST_N)
def test_perf_hoisting_device(n):
    """Init-once vs init-per-tile device-kernel ns at tile count n. Logs the ratio (the helper-design
    signal: hoisting is ~3.5% faster across the whole n range) and regression-guards each config."""
    ns_single = _device_kernel_ns(f"test_func_hoist[mode=single-n={n}]", f"eltwise_hoist_single_n{n}")
    ns_pertile = _device_kernel_ns(f"test_func_hoist[mode=pertile-n={n}]", f"eltwise_hoist_pertile_n{n}")
    logger.info(
        f"[n={n}] DEVICE KERNEL ns | hoist-single {ns_single:.0f} | per-tile {ns_pertile:.0f} | "
        f"hoist x{ns_pertile/ns_single:.3f}"
    )
    _check_baseline(ns_single, HOIST_BASELINE_NS[(n, "single")], f"hoist-single n={n}")
    _check_baseline(ns_pertile, HOIST_BASELINE_NS[(n, "pertile")], f"hoist-pertile n={n}")


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("n", BLOCK_N)
def test_perf_blocking_device(n):
    """block_size 1 vs 8 device-kernel ns at tile count n (Chunked block-capable chain). Logs the
    ratio — blocking is neutral-to-slightly-negative for this Exp chain and the penalty amortizes as n
    grows (x0.90 @ n=64 -> x0.99 @ n=2048), which is why the n sweep matters. Regression-guards each config."""
    ns_b1 = _device_kernel_ns(f"test_func_block[block_size=1-n={n}]", f"eltwise_block1_n{n}")
    ns_b8 = _device_kernel_ns(f"test_func_block[block_size=8-n={n}]", f"eltwise_block8_n{n}")
    logger.info(
        f"[n={n}] DEVICE KERNEL ns | block=1 {ns_b1:.0f} | block=8 {ns_b8:.0f} | block8/block1 x{ns_b8/ns_b1:.3f}"
    )
    _check_baseline(ns_b1, BLOCK_BASELINE_NS[(n, 1)], f"block=1 n={n}")
    _check_baseline(ns_b8, BLOCK_BASELINE_NS[(n, 8)], f"block=8 n={n}")


# Chunked-vs-Bulk baselines (ns), measured 2026-06-09. Findings:
#   - blocking within Bulk does NOTHING: bulk1 ≈ bulk8 (x0.99-1.00) — upfront serialization is the
#     bottleneck, not loop overhead.
#   - the REAL gain is Chunked vs Bulk at the same block size: ~1.7x (n=64) -> ~1.9x (n=256) faster,
#     because Chunked overlaps per-chunk read with compute; Bulk waits the whole window first.
# Bounded-CB fused chain out=exp(A+B)*C, batched Bulk (window=64) vs single-call Chunked, 2026-06-09.
# KEY FINDING: when Bulk is used REALISTICALLY (batched with a bounded window, not whole-window
# staging), the Chunked advantage shrinks sharply with N: x1.77 @ n=64 (1 batch = whole window) ->
# x1.41 @ n=128 -> only x1.05 @ n=1024 (16 batches). Double-buffered batches recover cross-batch
# pipelining, so at scale batched-Bulk ≈ Chunked (~5% behind). Blocking within Bulk stays ~1-2%.
LIFECYCLE_BASELINE_NS = {
    (64, "bulk1"): 115359,
    (64, "bulk8"): 114173,
    (64, "chunk8"): 64773,
    (128, "bulk1"): 173789,
    (128, "bulk8"): 170937,
    (128, "chunk8"): 121282,
    (1024, "bulk1"): 986744,
    (1024, "bulk8"): 962285,
    (1024, "chunk8"): 912314,
}


@pytest.mark.models_device_performance_bare_metal
@pytest.mark.parametrize("n", LIFECYCLE_NS)
def test_perf_lifecycle_compare(n):
    """Same exp(x) over n tiles: Bulk(blk=1) vs Bulk(blk=max) vs Chunked(blk=max), real device-kernel ns.
    bulk1->bulk8 isolates the blocking gain within the Bulk lifecycle; bulk8->chunk8 isolates the
    Chunked-vs-Bulk lifecycle gain at the same block size. Compute is identical across all three."""
    modes = ("bulk1", "bulk8", "chunk8")
    ns = {}
    for mode in modes:
        ns[mode] = _device_kernel_ns(f"test_func_lifecycle[mode={mode}-n={n}]", f"eltwise_life_{mode}_n{n}")
    fastest = min(ns.values())
    for mode in modes:
        logger.info(f"[chunk-vs-bulk n={n}] {mode:7s} {ns[mode]:.0f} ns | x{ns[mode]/fastest:.3f} vs fastest")
    logger.info(
        f"[chunk-vs-bulk n={n}] blocking gain (bulk1/bulk8) x{ns['bulk1']/ns['bulk8']:.3f} | "
        f"Chunked-vs-Bulk gain (bulk8/chunk8) x{ns['bulk8']/ns['chunk8']:.3f}"
    )
    for mode in modes:
        if (n, mode) in LIFECYCLE_BASELINE_NS:
            _check_baseline(ns[mode], LIFECYCLE_BASELINE_NS[(n, mode)], f"{mode} n={n}")
