# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
import ttnn
import numpy as np
from loguru import logger

from pykernel.ast import *
from pykernel.types import *


# / ------------------------------------------- /
# / ----------------- PYKERNELS --------------- /
# / ------------------------------------------- /
@ttkernel_tensix_compile(verbose=True)
def eltwise_sfpu(cb_in: CircularBuffer, cb_out: CircularBuffer, ct_args=[]):
    per_core_block_cnt = ct_args[0]
    per_core_block_dim = ct_args[1]

    unary_op_init_common(cb_in, cb_out)
    for i in range(0, per_core_block_cnt, 1):
        cb_reserve_back(cb_out, per_core_block_dim)
        for j in range(0, per_core_block_dim, 1):
            tile_regs_acquire()
            cb_wait_front(cb_in, 1)

            copy_tile(cb_in, 0, 0)

            exp_tile_init()
            exp_tile(0)

            tile_regs_commit()
            tile_regs_wait()
            pack_tile(0, cb_out, 0)

            cb_pop_front(cb_in, 1)
            tile_regs_release()

        cb_push_back(cb_out, per_core_block_dim)
    return


@ttkernel_noc_compile(verbose=True)
def writer_unary_interleaved(cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]):
    dst_addr: int = rt_args[0]
    num_tiles = rt_args[1]
    start_id = rt_args[2]

    dst_is_dram = ct_args[1]
    onetile = 1
    tile_bytes = get_tile_size(cb_out)
    dataformat = get_dataformat(cb_out)

    s0 = get_interleaved_addr_gen_fast(dst_is_dram, dst_addr, tile_bytes, dataformat)

    end_id = start_id + num_tiles
    ii: int = start_id
    for i in range(start_id, end_id, onetile):
        cb_wait_front(cb_out, onetile)
        l1_read_addr = get_read_ptr(cb_out)
        noc_async_write_tile(ii, s0, l1_read_addr)
        noc_async_write_barrier()
        cb_pop_front(cb_out, onetile)
        ii += onetile
    return


@ttkernel_noc_compile(verbose=True)
def reader_unary_interleaved(cb_in: CircularBuffer, cb_out: CircularBuffer, rt_args, ct_args=[]):
    src_addr: int = rt_args[0]
    num_tiles = rt_args[1]
    start_id = rt_args[2]

    src_is_dram = ct_args[0]  # True
    onetile = 1
    tile_bytes = get_tile_size(cb_in)
    dataformat = get_dataformat(cb_in)

    s0 = get_interleaved_addr_gen_fast(src_is_dram, src_addr, tile_bytes, dataformat)

    end_id = start_id + num_tiles
    ii: int = start_id
    for i in range(start_id, end_id, onetile):
        cb_reserve_back(cb_in, onetile)
        l1_write_addr = get_write_ptr(cb_in)
        noc_async_read_tile(ii, s0, l1_write_addr)
        noc_async_read_barrier()
        cb_push_back(cb_in, onetile)
        ii += onetile
    return


def run_pykernel_demo(device):
    # / ------------------------------------------- /
    # / ----------------- HOST CODE --------------- /
    # / ------------------------------------------- /
    num_tiles = 4
    src_bank_id = 0
    dst_bank_id = 0

    shape = [1, num_tiles, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )
    io_tensors = [input_tensor, output_tensor]

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    input_cb_data_format = ttnn.bfloat16
    cb_total_size = 2 * 2 * 1024
    cb_page_size = 2 * 1024

    input_cb_attributes = ttnn.CircularBufferAttributes(
        core_spec=core_grid,
        total_size=cb_total_size,
        page_size=cb_page_size,
        dtype=input_cb_data_format,
    )
    output_cb_attributes = ttnn.CircularBufferAttributes(
        core_spec=core_grid,
        total_size=cb_total_size,
        page_size=cb_page_size,
        dtype=input_cb_data_format,
    )

    cb_in_id = 0  # ttnn.CBIndex.c_0
    cb_out_id = 16  # ttnn.CBIndex.c_16
    cb_attributes = {
        cb_in_id: input_cb_attributes,
        cb_out_id: output_cb_attributes,
    }

    is_dram_input = True
    reader_compile_time_args = [is_dram_input]
    writer_compile_time_args = [cb_out_id, is_dram_input]
    compute_compile_time_args = [num_tiles, 1]
    reader_rt_args = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = [output_tensor.buffer_address(), num_tiles, 0]

    # Create and dump pykernels to files
    cb_in = CircularBuffer(cb_in_id)
    cb_out = CircularBuffer(cb_out_id)
    reader_string = reader_unary_interleaved(cb_in, cb_out, reader_rt_args, ct_args=reader_compile_time_args)
    writer_string = writer_unary_interleaved(cb_in, cb_out, writer_rt_args, ct_args=writer_compile_time_args)
    compute_string = eltwise_sfpu(cb_in, cb_out, ct_args=compute_compile_time_args)

    reader_kernel = Kernel("reader_unary", reader_string)
    writer_kernel = Kernel("writer_unary", writer_string)
    compute_kernel = Kernel("eltwise_sfpu", compute_string)

    reader_pykernel_path = reader_kernel.dump_to_file()
    writer_pykernel_path = writer_kernel.dump_to_file()
    compute_pykernel_path = compute_kernel.dump_to_file()

    # Create program attributes
    reader_attributes = ttnn.DataMovementAttributes(
        core_spec=core_grid,
        kernel_path=reader_pykernel_path,  # "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        config=reader_compile_time_args,
        runtime_args_per_core={core: reader_rt_args},
        is_reader=True,
    )
    writer_attributes = ttnn.DataMovementAttributes(
        core_spec=core_grid,
        kernel_path=writer_pykernel_path,  # "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        config=writer_compile_time_args,
        runtime_args_per_core={core: writer_rt_args},
        is_reader=False,
    )

    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        compile_args=compute_compile_time_args,
    )
    compute_attributes = ttnn.ComputeAttributes(
        core_spec=core_grid,
        kernel_path=compute_pykernel_path,  # "tt_metal/kernels/compute/eltwise_sfpu.cpp",
        config=compute_config,
    )

    program_attributes = ttnn.ProgramAttributes(
        cb_attributes,
        [reader_attributes, writer_attributes],
        [compute_attributes],
    )

    output = ttnn.generic_op(io_tensors, program_attributes)

    golden = ttnn.exp(input_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)

    logger.info("Input Tensor: {}", input_tensor)
    logger.info("Golden Tensor: {}", torch_golden)
    logger.info("Output Tensor: {}", torch_output)

    matching = torch.allclose(torch_golden, torch_output)
    logger.info("Tensors are matching: {}", matching)
    assert matching


device = ttnn.open_device(device_id=0)

run_pykernel_demo(device)

ttnn.close_device(device)
