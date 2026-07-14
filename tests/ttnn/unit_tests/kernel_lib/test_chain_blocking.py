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
