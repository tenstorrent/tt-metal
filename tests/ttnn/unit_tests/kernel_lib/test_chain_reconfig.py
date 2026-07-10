# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Validation for the eltwise_chain data-format reconfig emission.

Each kernel under ttnn/cpp/ttnn/kernel_lib/tests/chain_reconfig/ exercises one emission case (4-arg
_with_dt, 2-arg combined, mixed-prev, single-side, pack-side, compile-time elision). Per-CB dtypes
are chosen so the LLK's should_reconfigure_cbs sees mismatched formats and the reprogram actually
fires (not the fast-path skip).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc

# Tile size in bytes per dtype (from tt_metal/api/tt-metalium/tt_backend_api_types.hpp).
DTYPE_TILE_BYTES = {
    ttnn.bfloat16: 2048,
    ttnn.float32: 4096,
    ttnn.bfloat8_b: 1088,
}

KERNEL_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/chain_reconfig"
WRITER_1OUT = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"


def _torch_dtype_for(ttnn_dtype):
    if ttnn_dtype == ttnn.float32:
        return torch.float32
    if ttnn_dtype == ttnn.bfloat16:
        return torch.bfloat16
    # bfp8_b and bfp4_b host-side: kept as float32 so the precision loss happens during ttnn quantization.
    return torch.float32


def _make_input(shape, ttnn_dtype, device, seed):
    torch.manual_seed(seed)
    torch_t = torch.randn(shape, dtype=torch.float32) * 0.5 + 0.25
    torch_t = torch_t.to(_torch_dtype_for(ttnn_dtype))
    tt_t = ttnn.from_torch(
        torch_t,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return torch_t, tt_t


def _pcc_threshold(dtypes):
    if any(d == ttnn.bfloat8_b for d in dtypes):
        return 0.99
    if any(d == ttnn.float32 for d in dtypes):
        return 0.999
    return 0.9999


def _cb_descriptor(cb_id, dtype, num_pages, core_grid):
    page_size = DTYPE_TILE_BYTES[dtype]
    fmt = ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=page_size)
    return ttnn.CBDescriptor(
        total_size=page_size * num_pages,
        core_ranges=core_grid,
        format_descriptors=[fmt],
    )


def _single_core_grid():
    core = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])


def _build_reader_kernel(reader_path, input_tensors, num_tiles, core_grid):
    cta = []
    for t in input_tensors:
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in input_tensors] + [num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=reader_path,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


def _build_writer_1out_kernel(output_tensor, num_tiles, core_grid):
    cta = [16] + ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [output_tensor.buffer_address(), num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=WRITER_1OUT,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def _build_writer_2out_kernel(output_tensors, num_tiles, core_grid):
    cta = []
    for t in output_tensors:
        cta.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [t.buffer_address() for t in output_tensors] + [num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/writer_2_outputs.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def _build_compute_kernel(kernel_name, num_tiles, fp32_dest_acc_en, core_grid):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/{kernel_name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[num_tiles],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en),
    )


# =============================================================================
# 4-arg reconfig_data_format(prev_a, curr_a, prev_b, curr_b) (_with_dt)
# =============================================================================
# Chain: BinaryFpu(CbA,CbB) -> BinaryFpu(CbC,CbD) -> PackTile(CbOut).
# At element 1: srca rotates CbA->CbC with prev set AND srcb rotates CbB->CbD with prev set.
# CbA=bfp8, CbB=bf16, CbC=bf16, CbD=fp32 produces dual format delta on both sides simultaneously.
# Net semantic = CbC + CbD (first add overwritten in D0).
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_4arg_with_dt(device, num_tiles, fp32_dest_acc_en):
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_c, dt_d, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16, ttnn.float32, ttnn.bfloat16

    _, tt_a = _make_input(shape, dt_a, device, seed=11)
    _, tt_b = _make_input(shape, dt_b, device, seed=22)
    torch_c, tt_c = _make_input(shape, dt_c, device, seed=33)
    torch_d, tt_d = _make_input(shape, dt_d, device, seed=44)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(1, dt_b, 2, core_grid),
        _cb_descriptor(2, dt_c, 2, core_grid),
        _cb_descriptor(3, dt_d, 2, core_grid),
        _cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_4_inputs.cpp", [tt_a, tt_b, tt_c, tt_d], num_tiles, core_grid)
    writer = _build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = _build_compute_kernel("chain_4arg_with_dt.cpp", num_tiles, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_d, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_c.to(torch.float32) + torch_d.to(torch.float32)
    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, _pcc_threshold([dt_a, dt_b, dt_c, dt_d, dt_out]))
    logger.info(f"case A | num_tiles={num_tiles} | fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg


# =============================================================================
# 2-arg combined reconfig_data_format(curr_a, curr_b) (no _with_dt)
# =============================================================================
# Chain: BinaryFpu(CbA,CbB) -> PackTile(CbOut), first chain element.
# Both srca and srcb are first-emit on the BinaryFpu, neither has prev. 2-arg combined fires.
# CbA=bfp8, CbB=fp32 maxes format delta to catch argument-routing regressions.
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_2arg_combined(device, num_tiles, fp32_dest_acc_en):
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_out = ttnn.bfloat8_b, ttnn.float32, ttnn.bfloat16

    torch_a, tt_a = _make_input(shape, dt_a, device, seed=51)
    torch_b, tt_b = _make_input(shape, dt_b, device, seed=52)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(1, dt_b, 2, core_grid),
        _cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_2_inputs.cpp", [tt_a, tt_b], num_tiles, core_grid)
    writer = _build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = _build_compute_kernel("chain_2arg_combined.cpp", num_tiles, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32)
    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, _pcc_threshold([dt_a, dt_b, dt_out]))
    logger.info(f"case B | num_tiles={num_tiles} | fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg


# =============================================================================
# Mixed prev (srca has prev, srcb first-emit)
# =============================================================================
# Chain: CopyTile(CbA->D0) -> BinaryFpu(CbB,CbC->D1) -> AddBinary(D0+D1->D0) -> PackTile(CbOut).
# At BinaryFpu: prev_a=CbA (from CopyTile), curr_a=CbB → srca _with_dt; prev_b=NO_PREV_CB, curr_b=CbC
# → srcb single-arg first-emit. Every result feeds the output — net = CbA + (CbB + CbC) — so a
# botched srca reconfig (CbA->CbB) drops PCC (CbA is load-bearing, not discarded).
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_mixed_prev(device, num_tiles, fp32_dest_acc_en):
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_c, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.float32, ttnn.bfloat16

    torch_a, tt_a = _make_input(shape, dt_a, device, seed=61)
    torch_b, tt_b = _make_input(shape, dt_b, device, seed=62)
    torch_c, tt_c = _make_input(shape, dt_c, device, seed=63)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(1, dt_b, 2, core_grid),
        _cb_descriptor(2, dt_c, 2, core_grid),
        _cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_3_inputs.cpp", [tt_a, tt_b, tt_c], num_tiles, core_grid)
    writer = _build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = _build_compute_kernel("chain_mixed_prev.cpp", num_tiles, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_a.to(torch.float32) + torch_b.to(torch.float32) + torch_c.to(torch.float32)
    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, _pcc_threshold([dt_a, dt_b, dt_c, dt_out]))
    logger.info(f"mixed-prev | num_tiles={num_tiles} | fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg


# =============================================================================
# Single-side _with_dt on srca
# =============================================================================
# Chain: CopyTile(CbA, D0) -> CopyTile(CbB, D0) -> PackTile(CbOut).
# At element 1: prev_a=CbA, curr_a=CbB → srca per-side _with_dt fires. srcb untouched throughout.
# CbA=bfp8, CbB=bf16 spans block-float -> IEEE on srca. Net semantic = CbB.
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_singleside(device, num_tiles, fp32_dest_acc_en):
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_b, dt_out = ttnn.bfloat8_b, ttnn.bfloat16, ttnn.bfloat16

    _, tt_a = _make_input(shape, dt_a, device, seed=71)
    torch_b, tt_b = _make_input(shape, dt_b, device, seed=72)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(1, dt_b, 2, core_grid),
        _cb_descriptor(16, dt_out, 2, core_grid),
    ]

    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_2_inputs.cpp", [tt_a, tt_b], num_tiles, core_grid)
    writer = _build_writer_1out_kernel(tt_out, num_tiles, core_grid)
    compute = _build_compute_kernel("chain_singleside.cpp", num_tiles, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    golden = torch_b.to(torch.float32)
    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, _pcc_threshold([dt_a, dt_b, dt_out]))
    logger.info(f"case D | num_tiles={num_tiles} | fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg


# =============================================================================
# Pack-side _with_dt: multi-pack heterogeneous output chain
# =============================================================================
# Chain: CopyTile(CbA, D0) -> PackTile(CbOut1, D0) -> PackTile(CbOut2, D0).
# Both PackTiles read D0 (the CopyTile result = CbA) and pack to their respective output CBs with
# different dtypes (CbOut1=bf16, CbOut2=bfp8). Heterogeneous pack CBs trigger the per-stage emission
# path: boot programs only the first opt-in pack
# site; subsequent sites emit the 2-arg `pack_reconfig_data_format(prev_p, curr_p)` form before
# their per-iter pack work, with wraparound for site 0 to handle iter-to-iter cycling.
@pytest.mark.parametrize("num_tiles", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_pack_to_bfp8(device, num_tiles, fp32_dest_acc_en):
    shape = [1, 1, 32, 32 * num_tiles]
    dt_a, dt_out1, dt_out2 = ttnn.bfloat16, ttnn.bfloat16, ttnn.bfloat8_b

    torch_a, tt_a = _make_input(shape, dt_a, device, seed=81)

    tt_out1 = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out1, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    tt_out2 = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape), dt_out2, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(16, dt_out1, 2, core_grid),
        _cb_descriptor(17, dt_out2, 2, core_grid),
    ]

    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_1_input.cpp", [tt_a], num_tiles, core_grid)
    writer = _build_writer_2out_kernel([tt_out1, tt_out2], num_tiles, core_grid)
    compute = _build_compute_kernel("chain_pack_to_bfp8.cpp", num_tiles, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    ttnn.generic_op([tt_a, tt_out1, tt_out2], program)
    out1 = ttnn.to_torch(tt_out1).to(torch.float32)
    out2 = ttnn.to_torch(tt_out2).to(torch.float32)

    golden = torch_a.to(torch.float32)
    pcc_ok1, pcc_msg1 = comp_pcc(golden, out1, _pcc_threshold([dt_a, dt_out1]))
    pcc_ok2, pcc_msg2 = comp_pcc(golden, out2, _pcc_threshold([dt_a, dt_out2]))
    logger.info(
        f"case E | num_tiles={num_tiles} | fp32_dest_acc_en={fp32_dest_acc_en} | out1={pcc_msg1} | out2={pcc_msg2}"
    )
    assert pcc_ok1, f"out1 (bf16): {pcc_msg1}"
    assert pcc_ok2, f"out2 (bfp8): {pcc_msg2}"


# =============================================================================
# Compile-time elision (same CB on srca across consecutive elements)
# =============================================================================
# Chain: CopyTile(CbA, D0) x3 -> PackTile(CbOut). curr_a == prev_a == CbA at elements 1, 2 →
# reconf_a evaluates to false at compile time, no LLK emission past element 0.
# Verifies the refactored fold preserves the `if constexpr` elision path.
#
# Note on tile accounting: each of the 3 CopyTiles is on Streaming lifecycle and consumes one
# CbA tile per outer iter. Total CbA tiles consumed = 3 * num_iters. Output is 1 tile per outer
# iter (PackTile). Each output tile = the 3rd CopyTile's D0 value = the 3rd tile of the triplet
# (since each CopyTile overwrites D0). The input tensor is sized to hold the full 3*num_iters
# tiles, and the golden picks every 3rd input tile.
@pytest.mark.parametrize("num_iters", [1, 8, 64])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_elide(device, num_iters, fp32_dest_acc_en):
    tiles_consumed_per_iter = 3
    total_input_tiles = tiles_consumed_per_iter * num_iters
    input_shape = [1, 1, 32, 32 * total_input_tiles]
    output_shape = [1, 1, 32, 32 * num_iters]
    dt_a, dt_out = ttnn.bfloat16, ttnn.bfloat16

    torch_a, tt_a = _make_input(input_shape, dt_a, device, seed=91)

    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dt_out, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    core_grid = _single_core_grid()
    cbs = [
        _cb_descriptor(0, dt_a, 2, core_grid),
        _cb_descriptor(16, dt_out, 2, core_grid),
    ]

    # Reader pushes total_input_tiles tiles to CbA; chain consumes 3 per outer iter.
    reader = _build_reader_kernel(f"{KERNEL_DIR}/reader_1_input.cpp", [tt_a], total_input_tiles, core_grid)
    writer = _build_writer_1out_kernel(tt_out, num_iters, core_grid)
    compute = _build_compute_kernel("chain_elide.cpp", num_iters, fp32_dest_acc_en, core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    # Per-tile golden: output[i] = input[3*i + 2] (the third CopyTile in iter i overwrites D0 last).
    torch_a_f32 = torch_a.to(torch.float32)
    # Reshape input into (..., 32, num_iters, 3, 32), keep the last of the 3 along the triplet axis.
    a_view = torch_a_f32.view(1, 1, 32, num_iters, 3, 32)
    golden = a_view[..., 2, :].contiguous().view(1, 1, 32, 32 * num_iters)

    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, _pcc_threshold([dt_a, dt_out]))
    logger.info(f"case F | num_iters={num_iters} | fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg
