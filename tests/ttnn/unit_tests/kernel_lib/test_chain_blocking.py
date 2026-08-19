# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Blocking correctness for eltwise_chain.

block_size processes multiple tiles per inner iter across DEST lanes. It is a loop-structure
optimization: every supported size must match exp(x) and remain bit-identical to block_size=1.
Tail tests separately cover the two physical synchronization contracts.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc, comp_ulp
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/block_exp.cpp"
FIXED_BLOCK_TAIL_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/block_exp_chunked_fixed_tail.cpp"

# Accuracy bound for the Exp<> (Approx::Exact) chains below, in bf16 ULPs of the float32
# golden. PCC is scale-invariant, so it cannot catch a systematic accuracy loss on its own.
EXP_ULP_THRESHOLD = 2


def _assert_matches_golden(golden, actual, label):
    """Correlation (PCC) plus accuracy (ULP) against the float32 exp golden.

    The device result is bf16, so the ULP error is measured in bf16 ULPs: comp_ulp keeps the
    float32 golden as the higher-precision reference and sizes one ULP from its bf16 cast.
    """
    pcc_ok, msg = comp_pcc(golden, actual, lib.pcc_threshold([ttnn.bfloat16]))
    logger.debug(f"{label} | {msg}")
    assert pcc_ok, msg
    ulp_ok, ulp_msg = comp_ulp(golden, actual.to(torch.bfloat16), EXP_ULP_THRESHOLD)
    logger.debug(f"{label} | {ulp_msg}")
    assert ulp_ok, ulp_msg


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
def test_blocking_correctness_and_equivalence(device):
    """One execution matrix checks both the math golden and the stronger cross-size invariant."""
    n = 8
    results = {}
    for block_size in (1, 2, 4, 8):
        torch_in, out = _run_once(device, n, block_size)
        golden = torch.exp(torch_in)
        _assert_matches_golden(golden, out, f"blocking correctness block_size={block_size}")
        results[block_size] = out

    base = results[1]
    for bs, out in results.items():
        max_diff = (out - base).abs().max().item()
        logger.debug(f"blocking identical: block_size={bs} vs 1 -> max abs diff {max_diff}")
        assert torch.equal(out, base), f"block_size={bs} diverged from block_size=1 (max diff {max_diff})"


@pytest.mark.parametrize("n", [1, 7, 8, 9, 15])
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
    _assert_matches_golden(golden, actual, f"padded PerBlockSize tail n={n}, physical_n={physical_n}")


@pytest.mark.parametrize("Ht,Wt", [(2, 3), (2, 9)])
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
    _assert_matches_golden(golden, actual, f"row-blocked tail Ht={Ht}, Wt={Wt}, physical_Wt={physical_Wt}")


@pytest.mark.parametrize("Ht,Wt", [(2, 3), (2, 9)])
def test_clamped_block_tail_synchronizes_only_valid_tiles(device, Ht, Wt):
    """ValidTiles mode clamps both PerBlockSize synchronization and math to each logical row tail."""
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
    _assert_matches_golden(golden, actual, f"clamped PerBlockSize tail Ht={Ht}, Wt={Wt}")
