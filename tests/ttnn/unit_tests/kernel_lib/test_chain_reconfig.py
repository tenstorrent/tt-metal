# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Validation for the eltwise_chain data-format reconfig emission.

The format-transition kernels exercise 4-arg _with_dt, 2-arg combined, mixed-prev, single-side,
and pack-side behavior. Per-CB dtypes are chosen so should_reconfigure_cbs sees mismatched formats
and the reprogram actually fires. A separate repeated-same-CB case validates streaming accounting
without claiming to observe compile-time elision.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/reconfig"
KERNEL = f"{KERNEL_DIR}/scenarios.cpp"


def _assert_each_tile(golden, actual, threshold, label):
    """A multi-tile run covers first-use and steady state without allowing one bad tile to hide in PCC."""
    assert golden.shape == actual.shape
    num_tiles = golden.shape[-1] // 32
    for tile in range(num_tiles):
        tile_slice = slice(32 * tile, 32 * (tile + 1))
        ok, message = comp_pcc(golden[..., tile_slice], actual[..., tile_slice], threshold)
        assert ok, f"{label}, tile={tile}: {message}"
    logger.debug(f"{label} | all {num_tiles} tiles passed independently")


# =============================================================================
# 4-arg reconfig_data_format(prev_a, curr_a, prev_b, curr_b) (_with_dt)
# =============================================================================
# Chain: BinaryFpu(CbA,CbB) -> BinaryFpu(CbC,CbD) -> PackTile(CbOut).
# At element 1: srca rotates CbA->CbC with prev set AND srcb rotates CbB->CbD with prev set.
# CbA=bfp8, CbB=bf16, CbC=bf16, CbD=fp32 produces dual format delta on both sides simultaneously.
# Net semantic = CbC + CbD (first add overwritten in D0).
def test_4arg_with_dt(device):
    num_tiles = 8
    fp32_dest_acc_en = False
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_c, dt_d, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, ttnn.bfloat16

    _, tt_a = lib.make_input(shape, dt_a, device, seed=11)
    _, tt_b = lib.make_input(shape, dt_b, device, seed=22)
    torch_c, tt_c = lib.make_input(shape, dt_c, device, seed=33)
    torch_d, tt_d = lib.make_input(shape, dt_d, device, seed=44)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(1, dt_b, 2, core_grid),
        lib.cb_descriptor(2, dt_c, 2, core_grid),
        lib.cb_descriptor(3, dt_d, 2, core_grid),
        lib.cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c, tt_d], num_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_tiles, 0], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_d, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_c.to(torch.float32) + torch_d.to(torch.float32)
    _assert_each_tile(
        golden,
        torch_out,
        lib.pcc_threshold([dt_a, dt_b, dt_c, dt_d, dt_out]),
        f"4arg with-dt fp32_dest={fp32_dest_acc_en}",
    )


# =============================================================================
# 2-arg combined reconfig_data_format(curr_a, curr_b) (no _with_dt)
# =============================================================================
# Chain: BinaryFpu(CbA,CbB) -> PackTile(CbOut), first chain element.
# Both srca and srcb are first-emit on the BinaryFpu, neither has prev. 2-arg combined fires.
# CbA=bfp8, CbB=fp32 maxes format delta to catch argument-routing regressions.
def test_2arg_combined(device):
    num_tiles = 8
    fp32_dest_acc_en = True
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_out = ttnn.bfloat8_b, ttnn.float32, ttnn.bfloat16

    torch_a, tt_a = lib.make_input(shape, dt_a, device, seed=51)
    torch_b, tt_b = lib.make_input(shape, dt_b, device, seed=52)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(1, dt_b, 2, core_grid),
        lib.cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = lib.build_reader_kernel([tt_a, tt_b], num_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_tiles, 1], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32)
    _assert_each_tile(
        golden,
        torch_out,
        lib.pcc_threshold([dt_a, dt_b, dt_out]),
        f"2arg combined fp32_dest={fp32_dest_acc_en}",
    )


# =============================================================================
# Mixed prev (srca has prev, srcb first-emit)
# =============================================================================
# Chain: CopyTile(CbA->D0) -> BinaryFpu(CbB,CbC->D1) -> AddBinary(D0+D1->D0) -> PackTile(CbOut).
# At BinaryFpu: prev_a=CbA (from CopyTile), curr_a=CbB → srca _with_dt; prev_b=NO_PREV_CB, curr_b=CbC
# → srcb single-arg first-emit. Every result feeds the output — net = CbA + (CbB + CbC) — so a
# botched srca reconfig (CbA->CbB) drops PCC (CbA is load-bearing, not discarded).
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_mixed_prev(device, fp32_dest_acc_en):
    num_tiles = 8
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_c, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32, ttnn.bfloat16

    torch_a, tt_a = lib.make_input(shape, dt_a, device, seed=61)
    torch_b, tt_b = lib.make_input(shape, dt_b, device, seed=62)
    torch_c, tt_c = lib.make_input(shape, dt_c, device, seed=63)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(1, dt_b, 2, core_grid),
        lib.cb_descriptor(2, dt_c, 2, core_grid),
        lib.cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c], num_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_tiles, 2], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32) + torch_c.to(torch.float32)
    _assert_each_tile(
        golden,
        torch_out,
        lib.pcc_threshold([dt_a, dt_b, dt_c, dt_out]),
        f"mixed-prev fp32_dest={fp32_dest_acc_en}",
    )


# =============================================================================
# Single-side _with_dt on srca
# =============================================================================
# Chain: CopyTile(CbA, D0) -> CopyTile(CbB, D0) -> PackTile(CbOut).
# At element 1: prev_a=CbA, curr_a=CbB → srca per-side _with_dt fires. srcb untouched throughout.
# CbA=bfp8, CbB=bf16 spans block-float -> IEEE on srca. Net semantic = CbB.
def test_singleside(device):
    num_tiles = 8
    fp32_dest_acc_en = False
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16

    _, tt_a = lib.make_input(shape, dt_a, device, seed=71)
    torch_b, tt_b = lib.make_input(shape, dt_b, device, seed=72)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(1, dt_b, 2, core_grid),
        lib.cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = lib.build_reader_kernel([tt_a, tt_b], num_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_tiles, 3], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_b.to(torch.float32)
    _assert_each_tile(
        golden,
        torch_out,
        lib.pcc_threshold([dt_a, dt_b, dt_out]),
        f"single-side fp32_dest={fp32_dest_acc_en}",
    )


# =============================================================================
# Pack-side _with_dt: multi-pack heterogeneous output chain
# =============================================================================
# Chain: CopyTile(CbA, D0) -> PackTile(CbOut1, D0) -> PackTile(CbOut2, D0).
# Both PackTiles read D0 (the CopyTile result = CbA) and pack to their respective output CBs with
# different dtypes (CbOut1=bf16, CbOut2=bfp8). Heterogeneous pack CBs trigger the per-stage emission
# path: boot programs only the first opt-in pack
# site; subsequent sites emit the 2-arg `pack_reconfig_data_format(prev_p, curr_p)` form before
# their per-iter pack work, with wraparound for site 0 to handle iter-to-iter cycling.
def test_pack_to_bfp8(device):
    num_tiles = 8
    fp32_dest_acc_en = True
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_out1, dt_out2 = ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b

    torch_a, tt_a = lib.make_input(shape, dt_a, device, seed=81)

    tt_out1 = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out1, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out2 = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out2, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(16, dt_out1, 2, core_grid),
        lib.cb_descriptor(17, dt_out2, 2, core_grid),
    ]

    reader = lib.build_reader_kernel([tt_a], num_tiles, core_grid)
    writer = lib.build_writer_2out_kernel([tt_out1, tt_out2], num_tiles, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_tiles, 4], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_a, tt_out1, tt_out2], program)
    out1 = ttnn.to_torch(tt_out1).to(torch.float32)
    out2 = ttnn.to_torch(tt_out2).to(torch.float32)

    golden = torch_a.to(torch.float32)
    _assert_each_tile(
        golden,
        out1,
        lib.pcc_threshold([dt_a, dt_out1]),
        f"heterogeneous pack bf16 fp32_dest={fp32_dest_acc_en}",
    )
    _assert_each_tile(
        golden,
        out2,
        lib.pcc_threshold([dt_a, dt_out2]),
        f"heterogeneous pack bfp8 fp32_dest={fp32_dest_acc_en}",
    )


# =============================================================================
# Repeated streaming reads from the same CB
# =============================================================================
# Chain: CopyTile(CbA, D0) x3 -> PackTile(CbOut). Each CopyTile must independently consume
# one tile while the final copy overwrites D0. This validates repeated same-CB reader accounting;
# whether redundant reconfiguration was compile-time-elided requires a separate codegen oracle.
#
# Note on tile accounting: each of the 3 CopyTiles is on Streaming lifecycle and consumes one
# CbA tile per outer iter. Total CbA tiles consumed = 3 * num_iters. Output is 1 tile per outer
# iter (PackTile). Each output tile = the 3rd CopyTile's D0 value = the 3rd tile of the triplet
# (since each CopyTile overwrites D0). The input tensor is sized to hold the full 3*num_iters
# tiles, and the golden picks every 3rd input tile.
def test_repeated_same_cb_reads(device):
    num_iters = 8
    fp32_dest_acc_en = False
    tiles_consumed_per_iter = 3
    total_input_tiles = tiles_consumed_per_iter * num_iters
    input_shape = [1, 1, 32, 32 * total_input_tiles]
    output_shape = [1, 1, 32, 32 * num_iters]
    dt_a, dt_out = ttnn.bfloat16, ttnn.bfloat16

    torch_a, tt_a = lib.make_input(input_shape, dt_a, device, seed=91)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = lib.single_core_grid()
    cbs = [
        lib.cb_descriptor(0, dt_a, 2, core_grid),
        lib.cb_descriptor(16, dt_out, 2, core_grid),
    ]

    # Reader pushes total_input_tiles tiles to CbA; chain consumes 3 per outer iter.
    reader = lib.build_reader_kernel([tt_a], total_input_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_iters, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [num_iters, 5], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    # Per-tile golden: output[i] = input[3*i + 2] (the third CopyTile in iter i overwrites D0 last).
    torch_a_f32 = torch_a.to(torch.float32)
    # Reshape input into (..., 32, num_iters, 3, 32), keep the last of the 3 along the triplet axis.
    a_view = torch_a_f32.view(1, 1, 32, num_iters, 3, 32)
    golden = a_view[..., 2, :].contiguous().view(1, 1, 32, 32 * num_iters)

    _assert_each_tile(
        golden,
        torch_out,
        lib.pcc_threshold([dt_a, dt_out]),
        f"repeated same-CB reads fp32_dest={fp32_dest_acc_en}",
    )
