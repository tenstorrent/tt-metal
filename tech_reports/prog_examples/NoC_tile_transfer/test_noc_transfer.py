# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from loguru import logger


@pytest.mark.parametrize("num_tiles", [1])
def test_noc_tile_transfer(device, num_tiles):
    """
    Test NoC L1-to-L1 data transfer between two cores using ttnn.generic_op.
    This example demonstrates pure data movement without compute operations.

    Data flow:
    1. Core 0 reads data from DRAM to its local CB0
    2. Core 0 sends data over NoC to Core 1's CB1
    3. Core 1 receives data in CB1
    4. Core 1 writes data to DRAM
    """
    shape = [1, num_tiles, 32, 32]
    data = torch.full(shape, 14.0).to(torch.bfloat16)  # Using 14 as in the C++ example

    dram_memory_config = ttnn.DRAM_MEMORY_CONFIG

    # Create input tensor on device
    input_tensor = ttnn.from_torch(
        data,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=dram_memory_config,
    )

    # Allocate output tensor on device
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(shape),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        device,
        dram_memory_config,
    )

    io_tensors = [input_tensor, output_tensor]

    # Core configuration - using 2 specific cores
    core0 = ttnn.CoreCoord(0, 0)
    core1 = ttnn.CoreCoord(0, 1)

    core0_range = ttnn.CoreRangeSet([ttnn.CoreRange(core0, core0)])
    core1_range = ttnn.CoreRangeSet([ttnn.CoreRange(core1, core1)])
    both_cores_range = ttnn.CoreRangeSet([ttnn.CoreRange(core0, core1)])

    # Get physical coordinates for NoC addressing
    core0_physical = device.worker_core_from_logical_core(core0)
    core1_physical = device.worker_core_from_logical_core(core1)

    # Circular buffer configuration
    cb_data_format = ttnn.bfloat16
    cb_page_size = 2 * 1024  # tile size for bfloat16
    cb_total_size = 2 * cb_page_size  # double buffering

    # CB0 - used by Core 0
    cb0_index = 0
    cb0_format = ttnn.CBFormatDescriptor(
        buffer_index=cb0_index,
        data_format=cb_data_format,
        page_size=cb_page_size,
    )
    cb0_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=both_cores_range,  # Both cores need access for NoC transfer
        format_descriptors=[cb0_format],
    )

    # CB1 - used by Core 1
    cb1_index = 1
    cb1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb1_index,
        data_format=cb_data_format,
        page_size=cb_page_size,
    )
    cb1_descriptor = ttnn.CBDescriptor(
        total_size=cb_total_size,
        core_ranges=both_cores_range,  # Both cores need access for NoC transfer
        format_descriptors=[cb1_format],
    )

    # Semaphore for inter-core synchronization
    semaphore_id = 0
    semaphore_descriptor = ttnn.SemaphoreDescriptor(
        id=semaphore_id,
        core_type=ttnn.CoreType.WORKER,
        core_ranges=both_cores_range,
        initial_value=0,
    )

    # Core 0 - Reader kernel (reads from DRAM to CB0)
    reader0_compile_time_args = [cb0_index]
    reader0_compile_time_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader0_rt_args = ttnn.RuntimeArgs()
    reader0_rt_args[core0.x][core0.y] = [input_tensor.buffer_address()]

    core0_reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/programming_examples/NoC_tile_transfer/kernels/dataflow/reader0.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core0_range,
        compile_time_args=reader0_compile_time_args,
        runtime_args=reader0_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Core 0 - Writer kernel (sends data to Core 1 via NoC)
    writer0_compile_time_args = [cb0_index, cb1_index]

    writer0_rt_args = ttnn.RuntimeArgs()
    writer0_rt_args[core0.x][core0.y] = [core1_physical.x, core1_physical.y, semaphore_id]

    core0_writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/programming_examples/NoC_tile_transfer/kernels/dataflow/writer0.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core0_range,
        compile_time_args=writer0_compile_time_args,
        runtime_args=writer0_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Core 1 - Reader kernel (receives data from Core 0 via NoC)
    reader1_compile_time_args = [cb0_index, cb1_index]

    reader1_rt_args = ttnn.RuntimeArgs()
    reader1_rt_args[core1.x][core1.y] = [core0_physical.x, core0_physical.y, semaphore_id]

    core1_reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/programming_examples/NoC_tile_transfer/kernels/dataflow/reader1.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core1_range,
        compile_time_args=reader1_compile_time_args,
        runtime_args=reader1_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # Core 1 - Writer kernel (writes to DRAM from CB1)
    writer1_compile_time_args = [cb1_index]
    writer1_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer1_rt_args = ttnn.RuntimeArgs()
    writer1_rt_args[core1.x][core1.y] = [output_tensor.buffer_address()]

    core1_writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/programming_examples/NoC_tile_transfer/kernels/dataflow/writer1.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core1_range,
        compile_time_args=writer1_compile_time_args,
        runtime_args=writer1_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Create program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[core0_reader_kernel, core0_writer_kernel, core1_reader_kernel, core1_writer_kernel],
        semaphores=[semaphore_descriptor],
        cbs=[cb0_descriptor, cb1_descriptor],
    )

    # Execute the operation
    output = ttnn.generic_op(io_tensors, program_descriptor)

    # Verify results
    torch_input = ttnn.to_torch(input_tensor)
    torch_output = ttnn.to_torch(output)

    logger.info(f"input_tensor: {input_tensor}")
    logger.info(f"torch_input: {torch_input}")
    logger.info(f"torch_output: {torch_output}")

    matching = torch.allclose(torch_input, torch_output)
    logger.info(f"Tensors are matching: {matching}")
    assert matching, "Output should match input (passthrough via NoC transfer)"
