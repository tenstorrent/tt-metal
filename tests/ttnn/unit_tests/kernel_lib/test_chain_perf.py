# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Real-time-profiler performance tests for the eltwise_chain helper, swept across tile count.

Functional tests run the configs; the marked perf tests use the in-process real-time profiler to
collect GenericOp device-program durations in nanoseconds.

WHY SWEEP N: a single small tile count is dominated by fixed per-launch overhead and is not
representative. Hoisting retains the small/large endpoints. Lifecycle retains the small and large
endpoints to verify that PerBlockSize's advantage survives as the bounded Bulk workload scales.

Each measurement is checked against a recorded Wormhole RT baseline; the op runs ITERS times per
profile and uses the median duration.
"""

import statistics

import pytest
import torch
import ttnn
from loguru import logger

from models.common.utility_functions import is_wormhole_b0
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib
from tests.ttnn.profiling.realtime_profiler_utils import collect_op_durations_merged, require_realtime_profiler

ITERS = 20

BLOCK_CHUNKED = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/block_exp_chunked.cpp"
HOIST = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/hoist.cpp"
FUSED = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/fused_chain.cpp"  # FPU add + Exp + DestReuse mul
DEEP_FUSED = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/fused_chain_deep.cpp"

# PerBlockSize-vs-Bulk comparison on a REALISTIC fused chain (out = exp(A+B)*C: FPU add + Exp + DestReuse
# mul). Both retain a bounded CB and process N over many iterations, so N scales to thousands. The
# lifecycle test holds block_size=8 constant and isolates only when the Bulk chain waits for a batch
# versus when PerBlockSize overlaps work. A separate upfront-Bulk test varies block_size fairly.
LIFECYCLE_FUNCTIONAL_NS = [64, 128, 1024]
LIFECYCLE_PERF_NS = [64, 1024]
MAX_CHUNK = 8
BULK_BATCH = 64  # Bulk window per chain call (bounded; CB = 2*BULK_BATCH pages, independent of N)
UPFRONT_BULK_N = 64  # Full-window CB fits in single-core WH L1: 4 CBs * 2 * 64 BF16 tiles = 1 MiB.
UPFRONT_BULK_BLOCK_SIZES = [1, 2, 4, 8]

# Tile-count sweep: small (overhead-dominated) -> large (work-dominated).
HOIST_N = [64, 4096]
BLOCK_N = [64, 2048]
BLOCK_SIZES = [1, 8]


# =============================================================================
# Functional configs (profiled below). Each runs the op ITERS times + a correctness sanity.
# =============================================================================
def _make_block_chunked(device, n, block_size):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=2001)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    pages = max(4, 2 * block_size)  # PerBlockSize: CB holds ~block_size tiles, double-buffered
    cbs = [lib.cb_descriptor(0, dt, pages, cg), lib.cb_descriptor(16, dt, pages, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(BLOCK_CHUNKED, [n, block_size], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return torch_in, [tt_in, tt_out], program


def _run_block_chunked(device, n, block_size):
    torch_in, tensors, program = _make_block_chunked(device, n, block_size)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    assert torch.allclose(torch.exp(torch_in.to(torch.float32)), res, atol=0.1, rtol=0.1)


def _make_hoist(device, n, mode):
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=2002)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    mode_id = {"single": 0, "pertile": 1, "caller": 2}[mode]
    compute = lib.build_compute_kernel(HOIST, [n, mode_id], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return torch_in, [tt_in, tt_out], program


def _run_hoist(device, n, mode, iters=ITERS):
    torch_in, tensors, program = _make_hoist(device, n, mode)
    out = None
    for _ in range(iters):
        out = ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out).to(torch.float32)
    assert torch.allclose(torch.exp(torch_in.to(torch.float32)), result, atol=0.1, rtol=0.1)
    return result


@pytest.mark.parametrize("n", BLOCK_N)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_func_block(device, n, block_size):
    _run_block_chunked(device, n, block_size)


@pytest.mark.parametrize("n", HOIST_N)
@pytest.mark.parametrize("mode", ["single", "pertile"])
def test_func_hoist(device, n, mode):
    _run_hoist(device, n, mode)


def test_setup_placements_are_bit_identical(device):
    """Hoisted, per-tile, and caller-owned setup must produce the same bits."""
    single = _run_hoist(device, 16, "single", iters=1)
    per_tile = _run_hoist(device, 16, "pertile", iters=1)
    caller = _run_hoist(device, 16, "caller", iters=1)
    assert torch.equal(single, per_tile), "hoisted and per-tile initialization produced different results"
    assert torch.equal(single, caller), "hoisted and caller-owned initialization produced different results"


def _make_fused_chain(device, n, block_size, per_block_size, bulk_batch, kernel_path=FUSED):
    """Fused chain out = exp(A+B)*C with explicit lifecycle and CB-window controls."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()
    ta, tt_a = lib.make_input(shape, dt, device, seed=2003)
    tb, tt_b = lib.make_input(shape, dt, device, seed=2004)
    tc, tt_c = lib.make_input(shape, dt, device, seed=2005)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cta = [n, block_size, int(per_block_size), bulk_batch]
    pages = 2 * (block_size if per_block_size else bulk_batch)
    cbs = [lib.cb_descriptor(i, dt, pages, cg) for i in (0, 1, 2)] + [lib.cb_descriptor(16, dt, pages, cg)]
    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, n, cg)
    compute = lib.build_compute_kernel(kernel_path, cta, cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return (ta, tb, tc), [tt_a, tt_b, tt_c, tt_out], program


def _make_lifecycle(device, mode, n):
    if mode == "bulk1":
        return _make_fused_chain(device, n, 1, False, BULK_BATCH)
    if mode == "bulk8":
        return _make_fused_chain(device, n, MAX_CHUNK, False, BULK_BATCH)
    return _make_fused_chain(device, n, MAX_CHUNK, True, 0)


def _make_upfront_bulk(device, block_size):
    """Deep, upfront-waited Bulk chain; only its calculation block size varies."""
    return _make_fused_chain(device, UPFRONT_BULK_N, block_size, False, UPFRONT_BULK_N, kernel_path=DEEP_FUSED)


def _run_lifecycle(device, mode, n):
    (ta, tb, tc), tensors, program = _make_lifecycle(device, mode, n)
    out = None
    for _ in range(ITERS):
        out = ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    res = ttnn.to_torch(out).to(torch.float32)
    golden = torch.exp((ta + tb).to(torch.float32)) * tc.to(torch.float32)
    assert torch.allclose(golden, res, atol=0.15, rtol=0.15), "fused exp(A+B)*C mismatch"


def _run_upfront_bulk(device, block_size):
    (ta, tb, tc), tensors, program = _make_upfront_bulk(device, block_size)
    out = ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    result = ttnn.to_torch(out).to(torch.float32)
    golden = torch.relu(torch.sigmoid(torch.tanh(torch.exp((ta + tb).to(torch.float32)) * tc.to(torch.float32))))
    assert torch.allclose(golden, result, atol=0.15, rtol=0.15), "deep upfront-Bulk chain mismatch"


@pytest.mark.parametrize("n", LIFECYCLE_FUNCTIONAL_NS)
@pytest.mark.parametrize("mode", ["bulk1", "bulk8", "chunk8"])
def test_func_lifecycle(device, mode, n):
    _run_lifecycle(device, mode, n)


@pytest.mark.parametrize("block_size", UPFRONT_BULK_BLOCK_SIZES)
def test_func_upfront_bulk(device, block_size):
    _run_upfront_bulk(device, block_size)


# =============================================================================
# Perf tests — profile each config in-process and compare real-time device-program ns.
# =============================================================================
def _realtime_program_ns(device, make_program, kernel_path):
    """Median RT duration of freshly dispatched GenericOp programs."""
    require_realtime_profiler("eltwise-chain performance coverage")
    _, tensors, program = make_program()

    # RT callback subscriptions can replay older records, so the helper keeps the newest ITERS
    # matching dispatches. Do not warm up: the RT record already excludes host-side compilation.
    durations = collect_op_durations_merged(
        device,
        lambda: ttnn.generic_op(tensors, program),
        kernel_path,
        iters=ITERS,
        allow_stale_prefix=True,
    )
    return statistics.median(durations)


RT_BASELINE_MARGIN = 0.02
wormhole_rt_baseline = pytest.mark.skipif(not is_wormhole_b0(), reason="RT baselines are recorded on Wormhole B0")
HOIST_RT_BASELINE_NS = {
    (64, "single"): 47788,
    (64, "pertile"): 49686,
    (4096, "single"): 2938758,
    (4096, "pertile"): 3061605,
}
LIFECYCLE_RT_BASELINE_NS = {
    (64, "bulk8"): 119078,
    (64, "chunk8"): 66058,
    (1024, "bulk8"): 968640,
    (1024, "chunk8"): 916342,
}
UPFRONT_BULK_RT_BASELINE_NS = {
    1: 272510,
    2: 268703,
    4: 266763,
    8: 265009,
}


def _assert_rt_baseline(measured_ns, baseline_ns, label):
    lower = baseline_ns * (1 - RT_BASELINE_MARGIN)
    upper = baseline_ns * (1 + RT_BASELINE_MARGIN)
    assert lower <= measured_ns <= upper, (
        f"{label}: {measured_ns:.0f} ns outside the RT baseline {baseline_ns} ± "
        f"{RT_BASELINE_MARGIN * 100:.0f}% ({lower:.0f}-{upper:.0f} ns)"
    )


@pytest.mark.models_device_performance_bare_metal
@wormhole_rt_baseline
@pytest.mark.parametrize("n", HOIST_N)
def test_perf_hoisting_device(device, n):
    """RT program timing for init-once vs init-per-tile at tile count n."""
    ns_single = _realtime_program_ns(device, lambda: _make_hoist(device, n, "single"), HOIST)
    ns_pertile = _realtime_program_ns(device, lambda: _make_hoist(device, n, "pertile"), HOIST)
    logger.info(
        f"[n={n}] RT program ns | hoist-single {ns_single:.0f} | per-tile {ns_pertile:.0f} | "
        f"hoist x{ns_pertile/ns_single:.3f}"
    )
    _assert_rt_baseline(ns_single, HOIST_RT_BASELINE_NS[(n, "single")], f"hoist-single n={n}")
    _assert_rt_baseline(ns_pertile, HOIST_RT_BASELINE_NS[(n, "pertile")], f"hoist-pertile n={n}")
    assert (
        ns_single < ns_pertile * 0.99
    ), f"hoisting lost its expected device-program benefit: {ns_single:.0f} ns vs {ns_pertile:.0f} ns"


@pytest.mark.models_device_performance_bare_metal
@wormhole_rt_baseline
@pytest.mark.parametrize("n", LIFECYCLE_PERF_NS)
def test_perf_lifecycle_compare(device, n):
    """Bulk versus PerBlockSize at the same block size (8)."""
    modes = ("bulk8", "chunk8")
    ns = {
        mode: _realtime_program_ns(device, lambda mode=mode: _make_lifecycle(device, mode, n), FUSED) for mode in modes
    }
    fastest = min(ns.values())
    for mode in modes:
        logger.info(f"[chunk-vs-bulk n={n}] {mode:7s} {ns[mode]:.0f} RT ns | x{ns[mode]/fastest:.3f} vs fastest")
    logger.info(f"[chunk-vs-bulk n={n}] PerBlockSize-vs-Bulk gain (bulk8/chunk8) x{ns['bulk8']/ns['chunk8']:.3f}")
    for mode in modes:
        _assert_rt_baseline(ns[mode], LIFECYCLE_RT_BASELINE_NS[(n, mode)], f"{mode} n={n}")
    assert ns["chunk8"] < ns["bulk8"], f"PerBlockSize lost its lifecycle benefit at n={n}"


@pytest.mark.models_device_performance_bare_metal
@wormhole_rt_baseline
def test_perf_upfront_bulk_block_size(device):
    """Upfront-waited Bulk(block=1) versus Bulk(block=2/4/8) on a deep, one-DEST chain."""
    ns = {
        block_size: _realtime_program_ns(
            device, lambda block_size=block_size: _make_upfront_bulk(device, block_size), DEEP_FUSED
        )
        for block_size in UPFRONT_BULK_BLOCK_SIZES
    }
    for block_size, duration_ns in ns.items():
        logger.info(f"[upfront-bulk n={UPFRONT_BULK_N}] block={block_size} RT program ns | {duration_ns:.0f}")
    for block_size, duration_ns in ns.items():
        _assert_rt_baseline(
            duration_ns,
            UPFRONT_BULK_RT_BASELINE_NS[block_size],
            f"upfront-bulk n={UPFRONT_BULK_N} block={block_size}",
        )
    for block_size in UPFRONT_BULK_BLOCK_SIZES[1:]:
        assert ns[block_size] < ns[1] * 0.99, (
            f"upfront Bulk(block={block_size}) lost its expected block-size benefit: "
            f"{ns[block_size]:.0f} ns vs block=1 {ns[1]:.0f} ns"
        )
