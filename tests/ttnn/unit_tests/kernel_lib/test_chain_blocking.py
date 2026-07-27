# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Blocking — correctness + throughput for eltwise_chain (block_size axis).

block_size processes multiple tiles per inner iter across DEST lanes. It is a loop-structure
optimization: it must NOT change the per-tile result, only reduce loop/DEST-sync overhead.
  - test_blocking_correctness : block_size {1,2,4,8} all produce bit-identical exp(x).
  - test_blocking_throughput  : logs tiles/sec + speedup and guards against gross regression
                                (block=8 not dramatically slower than block=1). Wall-clock smoke
                                signal only — real perf gating belongs in a device-profiler job.
"""

import time
import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/block_exp.cpp"
FIXED_BLOCK_TAIL_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/block_exp_chunked_fixed_tail.cpp"


def _build(device, n, block_size):
    """Returns (program, [tensors], torch_in) for a Bulk+Block exp chain at the given block_size."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()
    torch_in, tt_in = lib.make_input(shape, dt, device, seed=601)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    # Bulk stages the full window upfront -> size both CBs for all n tiles.
    cbs = [lib.cb_descriptor(0, dt, n, core_grid), lib.cb_descriptor(16, dt, n, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n, block_size], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    return program, [tt_in, tt_out], torch_in


def _run_once(device, n, block_size):
    program, tensors, torch_in = _build(device, n, block_size)
    output = ttnn.generic_op(tensors, program)
    return torch_in.to(torch.float32), ttnn.to_torch(output).to(torch.float32)


# =============================================================================
# Correctness — blocking must not change the per-tile result.
# =============================================================================
@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_blocking_correctness(device, block_size):
    n = 8
    torch_in, out = _run_once(device, n, block_size)
    golden = torch.exp(torch_in)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"blocking correctness block_size={block_size} | {msg}")
    assert pcc_ok, msg


def test_blocking_identical_across_sizes(device):
    """Every block size must yield a BIT-IDENTICAL result to block_size=1 (loop structure only)."""
    n = 8
    _, base = _run_once(device, n, 1)
    for bs in (2, 4, 8):
        _, out = _run_once(device, n, bs)
        max_diff = (out - base).abs().max().item()
        logger.info(f"blocking identical: block_size={bs} vs 1 -> max abs diff {max_diff}")
        assert torch.equal(out, base), f"block_size={bs} diverged from block_size=1 (max diff {max_diff})"


@pytest.mark.parametrize("n", [1, 3, 7, 8, 9, 15])
def test_fixed_block_tail_executes_only_valid_tiles(device, n):
    """A fixed-size physical tail synchronizes a full chunk while math covers only valid tiles."""
    block_size = 8
    physical_n = ((n + block_size - 1) // block_size) * block_size
    dt = ttnn.bfloat16
    physical_shape = [1, 1, 32, 32 * physical_n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(physical_shape, dt, device, seed=602 + n)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(physical_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    # Both dataflow sides exchange the complete physical chunk. Only the first n output tiles
    # contain defined results; the tail pages exist to exercise the full-block CB contract.
    pages = 2 * block_size
    cbs = [lib.cb_descriptor(0, dt, pages, core_grid), lib.cb_descriptor(16, dt, pages, core_grid)]
    reader = lib.build_reader_kernel([tt_in], physical_n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, physical_n, core_grid)
    compute = lib.build_compute_kernel(FIXED_BLOCK_TAIL_KERNEL, [1, n, block_size, 1], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in, tt_out], program)
    valid_columns = 32 * n
    actual = ttnn.to_torch(output).to(torch.float32)[..., :valid_columns]
    golden = torch.exp(torch_in.to(torch.float32)[..., :valid_columns])
    pcc_ok, msg = comp_pcc(golden, actual, lib.pcc_threshold([dt]))
    logger.info(f"padded Chunked tail n={n}, physical_n={physical_n} | {msg}")
    assert pcc_ok, msg


@pytest.mark.parametrize("Ht,Wt", [(2, 3), (2, 7), (2, 9), (3, 5)])
def test_fixed_block_tail_is_synchronized_per_row(device, Ht, Wt):
    """Every logical row gets its own full physical tail chunk."""
    block_size = 8
    physical_Wt = ((Wt + block_size - 1) // block_size) * block_size
    physical_n = Ht * physical_Wt
    dt = ttnn.bfloat16
    physical_shape = [1, 1, 32 * Ht, 32 * physical_Wt]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(physical_shape, dt, device, seed=702 + Ht * 10 + Wt)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(physical_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    pages = 2 * block_size
    cbs = [lib.cb_descriptor(0, dt, pages, core_grid), lib.cb_descriptor(16, dt, pages, core_grid)]
    reader = lib.build_reader_kernel([tt_in], physical_n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, physical_n, core_grid)
    compute = lib.build_compute_kernel(FIXED_BLOCK_TAIL_KERNEL, [Ht, Wt, block_size, 1], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in, tt_out], program)
    valid_columns = 32 * Wt
    actual = ttnn.to_torch(output).to(torch.float32)[..., :valid_columns]
    golden = torch.exp(torch_in.to(torch.float32)[..., :valid_columns])
    pcc_ok, msg = comp_pcc(golden, actual, lib.pcc_threshold([dt]))
    logger.info(f"row-blocked tail Ht={Ht}, Wt={Wt}, physical_Wt={physical_Wt} | {msg}")
    assert pcc_ok, msg


@pytest.mark.parametrize("Ht,Wt", [(1, 3), (1, 9), (2, 3), (2, 9)])
def test_clamped_block_tail_synchronizes_only_valid_tiles(device, Ht, Wt):
    """ValidTiles mode clamps both Chunked synchronization and math to each logical row tail."""
    block_size = 8
    n = Ht * Wt
    dt = ttnn.bfloat16
    shape = [1, 1, 32 * Ht, 32 * Wt]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=802 + Ht * 10 + Wt)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)

    # A row-clamped grid repeats after Wt logical pages, so keep the CB ring row-aligned.
    # A block-sized ring would let the partial first-row tail misalign a later full chunk
    # across the physical ring boundary.
    pages = 2 * Wt
    cbs = [lib.cb_descriptor(0, dt, pages, core_grid), lib.cb_descriptor(16, dt, pages, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(FIXED_BLOCK_TAIL_KERNEL, [Ht, Wt, block_size, 0], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    output = ttnn.generic_op([tt_in, tt_out], program)
    actual = ttnn.to_torch(output).to(torch.float32)
    golden = torch.exp(torch_in.to(torch.float32))
    pcc_ok, msg = comp_pcc(golden, actual, lib.pcc_threshold([dt]))
    logger.info(f"clamped Chunked tail Ht={Ht}, Wt={Wt} | {msg}")
    assert pcc_ok, msg


# =============================================================================
# Throughput — wall-clock smoke signal across block sizes (informational + gross-regression guard).
# =============================================================================
def _median_time(device, n, block_size, iters=15, warmup=3):
    program, tensors, _ = _build(device, n, block_size)
    for _ in range(warmup):
        ttnn.generic_op(tensors, program)
    ttnn.synchronize_device(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ttnn.generic_op(tensors, program)
        ttnn.synchronize_device(device)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def test_blocking_throughput(device):
    n = 64
    t1 = _median_time(device, n, 1)
    t8 = _median_time(device, n, 8)
    tps1 = n / t1
    tps8 = n / t8
    speedup = t1 / t8
    logger.info(
        f"blocking throughput n={n}: block=1 {t1*1e3:.3f}ms ({tps1:,.0f} tiles/s) | "
        f"block=8 {t8*1e3:.3f}ms ({tps8:,.0f} tiles/s) | speedup x{speedup:.2f}"
    )
    # Correctness still holds at the large size.
    _, out = _run_once(device, n, 8)
    golden = torch.exp(_run_once(device, n, 1)[0])
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    assert pcc_ok, msg
    # Gross-regression guard only (wall-clock is host-dominated and noisy; do NOT assert a tight
    # speedup). block=8 must not be dramatically slower than block=1.
    assert t8 < t1 * 1.5, f"block=8 ({t8*1e3:.3f}ms) is far slower than block=1 ({t1*1e3:.3f}ms) — regression?"
