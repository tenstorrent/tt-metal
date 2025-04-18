# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
    get_lib_dtype,
)

from loguru import logger


def run_generic_op():
    pass


def test_eltwise_exp(device):
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

    in_cb = ttnn.CBIndex.c_0  # these can also just be integers
    out_cb = ttnn.CBIndex.c_16
    cb_attributes = {
        in_cb: input_cb_attributes,
        out_cb: output_cb_attributes,
    }

    # should we expose a get_buffer_type function?
    is_dram_input = 1
    reader_compile_time_args = [is_dram_input]
    writer_compile_time_args = [out_cb, is_dram_input]
    reader_rt_args = [input_tensor.buffer_address(), num_tiles, 0]
    writer_rt_args = [output_tensor.buffer_address(), num_tiles, 0]

    reader_attributes = ttnn.DataMovementAttributes(
        core_spec=core_grid,
        kernel_path="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        config=reader_compile_time_args,
        runtime_args_per_core={core: reader_rt_args},
        is_reader=True,
    )
    writer_attributes = ttnn.DataMovementAttributes(
        core_spec=core_grid,
        kernel_path="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        config=writer_compile_time_args,
        runtime_args_per_core={core: writer_rt_args},
        is_reader=False,
    )

    sfpu_defines = {"SFPU_OP_EXP_INCLUDE": "1", "SFPU_OP_CHAIN_0": "exp_tile_init(); exp_tile(0);"}
    compute_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        compile_args=[num_tiles, 1],
        defines=sfpu_defines,
    )
    compute_attributes = ttnn.ComputeAttributes(
        core_spec=core_grid,
        kernel_path="tt_metal/kernels/compute/eltwise_sfpu.cpp",
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
    assert torch.allclose(torch_golden, torch_output)


def run():
    device = ttnn.open_device(device_id=0)

    test_eltwise_exp(device)

    ttnn.close_device(device)


run()
