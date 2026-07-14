# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Functional validation for InputLifecycle::OuterStream — the streamed outer-axis broadcast.

A BinaryFpu chain over grid(Ht, Wt):
  cb_a: Streaming   + Scalar  -> one tile per (ht, wt), front-read, popped per tile
  cb_b: OuterStream + Scalar  -> ONE tile per row, front-read across the row's Wt cols,
                                 waited at row entry / popped at row exit (shallow 2-deep CB)
Net: out[ht*Wt + wt] = a[ht*Wt + wt] + b[ht]. This is the only path exercising the per-row wait/pop
hooks + the O(1)-L1 streamed broadcast (a resident Bulk+Col operand would need an Ht-deep CB).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc

KERNEL_DIR = "ttnn/cpp/ttnn/kernel_lib/tests/outer_stream"
WRITER_1OUT = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp"

DTYPE_TILE_BYTES = {ttnn.bfloat16: 2048, ttnn.float32: 4096}


def _make_input(shape, ttnn_dtype, device, seed):
    torch.manual_seed(seed)
    torch_t = torch.randn(shape, dtype=torch.float32) * 0.5 + 0.25
    host = torch_t.to(torch.bfloat16) if ttnn_dtype == ttnn.bfloat16 else torch_t
    tt_t = ttnn.from_torch(
        host, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    return torch_t, tt_t


def _cb(cb_id, dtype, num_pages, grid):
    ps = DTYPE_TILE_BYTES[dtype]
    return ttnn.CBDescriptor(
        total_size=ps * num_pages,
        core_ranges=grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=cb_id, data_format=dtype, page_size=ps)],
    )


def _single_core_grid():
    c = ttnn.CoreCoord(0, 0)
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c)])


def _build_reader(tt_a, tt_b, Ht, Wt, grid):
    cta = list(ttnn.TensorAccessorArgs(tt_a).get_compile_time_args())
    cta += list(ttnn.TensorAccessorArgs(tt_b).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [tt_a.buffer_address(), tt_b.buffer_address(), Ht, Wt]
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/reader_a_full_b_per_row.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor(),
    )


def _build_writer(tt_out, num_tiles, grid):
    cta = [16] + list(ttnn.TensorAccessorArgs(tt_out).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    rt[0][0] = [tt_out.buffer_address(), num_tiles, 0]
    return ttnn.KernelDescriptor(
        kernel_source=WRITER_1OUT,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=cta,
        runtime_args=rt,
        config=ttnn.WriterConfigDescriptor(),
    )


def _build_compute(Ht, Wt, fp32_dest_acc_en, grid):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KERNEL_DIR}/chain_outer_stream.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=grid,
        compile_time_args=[Ht, Wt],
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=fp32_dest_acc_en),
    )


# (Ht, Wt): Ht>1 exercises the per-row advance; Wt>1 exercises the broadcast across cols.
@pytest.mark.parametrize("Ht,Wt", [(2, 1), (2, 3), (4, 8), (8, 5), (3, 16)])
@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_outer_stream_broadcast(device, Ht, Wt, fp32_dest_acc_en):
    dt = ttnn.bfloat16
    a_shape = [1, 1, 32, 32 * Ht * Wt]
    b_shape = [1, 1, 32, 32 * Ht]

    torch_a, tt_a = _make_input(a_shape, dt, device, seed=101)
    torch_b, tt_b = _make_input(b_shape, dt, device, seed=202)

    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(a_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    grid = _single_core_grid()
    cbs = [_cb(0, dt, 2, grid), _cb(1, dt, 2, grid), _cb(16, dt, 2, grid)]

    program = ttnn.ProgramDescriptor(
        kernels=[
            _build_reader(tt_a, tt_b, Ht, Wt, grid),
            _build_writer(tt_out, Ht * Wt, grid),
            _build_compute(Ht, Wt, fp32_dest_acc_en, grid),
        ],
        semaphores=[],
        cbs=cbs,
    )
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    torch_out = ttnn.to_torch(output).to(torch.float32)

    # Tiles lie along W: tile i = columns [i*32, (i+1)*32). out tile (ht, wt) = a tile (ht, wt) + b tile (ht).
    a_v = torch_a.to(torch.float32).view(1, 1, 32, Ht, Wt, 32)
    b_v = torch_b.to(torch.float32).view(1, 1, 32, Ht, 1, 32)  # broadcast over Wt
    golden = (a_v + b_v).reshape(1, 1, 32, 32 * Ht * Wt)

    pcc_ok, pcc_msg = comp_pcc(golden, torch_out, 0.999)
    logger.info(f"OuterStream | Ht={Ht} Wt={Wt} fp32_dest_acc_en={fp32_dest_acc_en} | {pcc_msg}")
    assert pcc_ok, pcc_msg
