# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""generic_op compiled ahead of time (ttnn.experimental.compile_generic_op) and run with
ttnn.experimental.run_compiled_program, optionally from a kernel config buffer.

The kernel-config-buffer test needs slow dispatch and the HYBRID allocator, both set before pytest
starts: TT_METAL_SLOW_DISPATCH_MODE=1 TT_METAL_ALLOCATOR_MODE_HYBRID=1.
"""

import os

import pytest
import torch
import ttnn

NUM_TILES = 8
TILE_BYTES = 2 * 1024  # one bfloat16 tile
CORES = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))])
SHAPE = [1, NUM_TILES, 32, 32]

needs_kernel_config_buffer = pytest.mark.skipif(
    os.environ.get("TT_METAL_SLOW_DISPATCH_MODE") != "1" or os.environ.get("TT_METAL_ALLOCATOR_MODE_HYBRID") != "1",
    reason="a kernel config buffer needs TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_ALLOCATOR_MODE_HYBRID=1",
)


def make_io(device):
    """Random input data, its DRAM tensor, and an empty DRAM output tensor."""
    data = torch.rand(SHAPE).to(torch.bfloat16)
    input_tensor = ttnn.from_torch(
        data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(SHAPE), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    return data, input_tensor, output_tensor


def plain_cb():
    return ttnn.CBDescriptor(
        total_size=2 * TILE_BYTES,
        core_ranges=CORES,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=TILE_BYTES)],
    )


def copy_descriptor(input_tensor, output_tensor, cb, compile_time_arg=0):
    """A DRAM-to-DRAM copy of NUM_TILES tiles over CORES: a reader and a writer sharing CB 0 (`cb`).

    `compile_time_arg` is the writer's CB index argument; changing it changes the program cache key."""
    tiles_per_core = NUM_TILES // CORES.num_cores()
    reader_args = ttnn.RuntimeArgs()
    writer_args = ttnn.RuntimeArgs()
    start = 0
    for core_range in CORES.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_args[x][y] = [input_tensor.buffer_address(), tiles_per_core, start]
                writer_args[x][y] = [output_tensor.buffer_address(), tiles_per_core, start]
                start += tiles_per_core

    kernels = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow"
    reader = ttnn.KernelDescriptor(
        kernel_source=f"{kernels}/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=CORES,
        compile_time_args=ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args(),
        runtime_args=reader_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=f"{kernels}/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=CORES,
        compile_time_args=[compile_time_arg] + ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args(),
        runtime_args=writer_args,
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer], semaphores=[], cbs=[cb])


def output_of(device, output_tensor):
    ttnn.synchronize_device(device)
    return ttnn.to_torch(output_tensor)


def test_run_compiled_program(device):
    data, input_tensor, output_tensor = make_io(device)
    compiled = ttnn.experimental.compile_generic_op(
        [input_tensor, output_tensor], copy_descriptor(input_tensor, output_tensor, plain_cb())
    )
    assert "GenericOpDeviceOperation" in compiled.operation_name

    # Compiling enqueues nothing.
    ttnn.synchronize_device(device)
    assert not torch.equal(ttnn.to_torch(output_tensor), data)

    ttnn.experimental.run_compiled_program(
        compiled, [input_tensor, output_tensor], copy_descriptor(input_tensor, output_tensor, plain_cb())
    )
    assert torch.equal(output_of(device, output_tensor), data)

    # Run again with other tensors: their addresses are runtime args, so the program is the same.
    other_data, other_input, other_output = make_io(device)
    ttnn.experimental.run_compiled_program(
        compiled, [other_input, other_output], copy_descriptor(other_input, other_output, plain_cb())
    )
    assert torch.equal(output_of(device, other_output), other_data)


def test_run_rejects_a_different_program(device, expect_error):
    _, input_tensor, output_tensor = make_io(device)
    io_tensors = [input_tensor, output_tensor]
    compiled = ttnn.experimental.compile_generic_op(
        io_tensors, copy_descriptor(input_tensor, output_tensor, plain_cb())
    )

    # A different compile-time arg needs a different program.
    different = copy_descriptor(input_tensor, output_tensor, plain_cb(), compile_time_arg=1)
    with expect_error(RuntimeError, "different program cache key"):
        ttnn.experimental.run_compiled_program(compiled, io_tensors, different)


@needs_kernel_config_buffer
def test_run_from_kernel_config_buffer(device, expect_error):
    data, input_tensor, output_tensor = make_io(device)
    io_tensors = [input_tensor, output_tensor]
    descriptor = copy_descriptor(input_tensor, output_tensor, plain_cb())
    compiled = ttnn.experimental.compile_generic_op(io_tensors, descriptor)
    buffer = ttnn.experimental.create_kernel_config_buffer(compiled)
    assert buffer.max_size > 0

    ttnn.experimental.run_compiled_program(compiled, io_tensors, descriptor, kernel_config_buffer=buffer)
    assert torch.equal(output_of(device, output_tensor), data)

    buffer.deallocate()
    with expect_error(RuntimeError, "deallocated"):
        ttnn.experimental.run_compiled_program(compiled, io_tensors, descriptor, kernel_config_buffer=buffer)
