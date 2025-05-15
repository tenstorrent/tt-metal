# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from loguru import logger

from models.utility_functions import skip_for_blackhole


def _test_eltwise_exp(device):
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

    input_cb_data_format = ttnn.bfloat16  # this will be mapped tt::DataFormat::Float16_b
    cb_total_size = 2 * 2 * 1024  # tt::DataFormat::Float16_b hard coded to have size 2 * 1024
    cb_page_size = 2 * 1024

    in_cb = 0
    out_cb = 16
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb,
        data_format=input_cb_data_format,
        page_size=cb_page_size,
    )
    in_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[in_cb_format],
    )
    out_cb_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=core_grid,
        format_descriptors=[out_cb_format],
    )

    is_dram_input = 1
    reader_compile_time_args = [is_dram_input]
    writer_compile_time_args = [out_cb, is_dram_input]
    compute_compile_time_args = [num_tiles, 1]
    reader_rt_args = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = [output_tensor.buffer_address(), num_tiles, 0]

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=[[reader_rt_args]],
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=[[writer_rt_args]],
        config=ttnn.WriterConfigDescriptor(),
    )

    sfpu_defines = [("SFPU_OP_EXP_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")]
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/eltwise_sfpu.cpp",
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[[[]]],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
        semaphores=[],
        cbs=[in_cb_descriptor, out_cb_descriptor],
    )

    output = ttnn.generic_op(io_tensors, program_descriptor)
    golden = ttnn.exp(input_tensor)

    torch_golden = ttnn.to_torch(golden)
    torch_output = ttnn.to_torch(output)
    logger.info(f"input_tensor: {input_tensor}")
    logger.info(f"torch_golden: {torch_golden}")
    logger.info(f"torch_output: {torch_output}")

    matching = torch.allclose(torch_golden, torch_output)
    logger.info(f"Tensors are matching: {matching}")
    assert matching


@skip_for_blackhole("Not tested / built for Blackhole")
def test_generic_op():
    # Choose not to parametrize the input tensors
    # this was chosen to highlight the operation of the Generic Op instead of testing Eltwise Op's func
    device = ttnn.open_device(device_id=0)

    _test_eltwise_exp(device)

    ttnn.close_device(device)
