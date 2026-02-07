import torch
import pytest
import ttnn
from loguru import logger


@pytest.mark.timeout(30)  # Fail if test hangs for more than 30 seconds
def test_deadlock_repro(device):
    """
    Reproduces the deadlock issue from tt-mlir Archive-6848/deadlock.mlir

    This test creates a minimal program with:
    - 3 L1 buffers at specific addresses (matching MLIR)
    - A compute kernel that does tilize_init, tilize_block, tilize_uninit
    - Two dataflow kernels (reader and writer)
    - Single core execution on (0,0)

    The MLIR shows:
    - Buffer 0 @ 102208: tile<32x32, f32> output
    - Buffer 1 @ 106304: 5408x32 si32 input
    - Buffer 2 @ 798528: tile<32x32, si32> intermediate
    """

    # Create input data - 32x32 = 1024 elements in row-major format
    input_shape = [1, 1, 32, 32]
    input_data = torch.arange(1024, dtype=torch.int32).reshape(input_shape)

    # Create input tensor - si32 (signed int32) matching MLIR Buffer 1
    input_tensor = ttnn.from_torch(
        input_data,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Create output tensor for tilized result - f32 matching MLIR Buffer 0
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_shape),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    # Single core at (0,0) matching MLIR
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB indices matching MLIR: cb_ports = [0, 1, 2]
    # CB 0: output (f32 tiles)
    # CB 1: intermediate (si32 tiles)
    # CB 2: input (row-major si32)
    in_cb_id = 2
    intermediate_cb_id = 1
    out_cb_id = 0

    # CB configurations
    int32_tile_size_bytes = 4 * 1024  # int32: 4 bytes * 32 * 32
    float32_tile_size_bytes = 4 * 1024  # float32: 4 bytes * 32 * 32
    cb_num_tiles = 2

    # Input CB (row-major si32 data)
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb_id,
        data_format=ttnn.int32,
        page_size=int32_tile_size_bytes,
    )
    in_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * int32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[in_cb_format],
    )

    # Intermediate CB (si32 tiles)
    intermediate_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=intermediate_cb_id,
        data_format=ttnn.int32,
        page_size=int32_tile_size_bytes,
    )
    intermediate_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * int32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[intermediate_cb_format],
    )

    # Output CB (f32 tiles)
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb_id,
        data_format=ttnn.float32,
        page_size=float32_tile_size_bytes,
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * float32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[out_cb_format],
    )

    # DataFlow0 kernel (reader) - reads from input buffer to CB 2
    reader_ct_args = [in_cb_id]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    num_tiles = 1
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_tiles,
        0,  # start tile
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/deadlock_dataflow_0_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # DataFlow1 kernel (writer) - writes from CB 0 to output buffer
    writer_ct_args = [out_cb_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_tiles,
        0,  # start tile
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/deadlock_dataflow_1_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel - does tilize operations
    # Matching MLIR: tilize_init, tilize_block, tilize_uninit
    # ct_args in MLIR: [cb_port[0], cb_port[1], cb_port[2]]
    # The kernel expects CB indices as compile time args
    compute_ct_args = [
        out_cb_id,  # cb_port[0] = 0
        in_cb_id,  # cb_port[1] = 2
        intermediate_cb_id,  # cb_port[2] = 1
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/deadlock_compute_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # Create program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, intermediate_cb_desc, out_cb_desc],
    )

    logger.info("Running generic_op with tilize program...")

    # Execute the program
    # If this completes without hanging/timeout, there's no deadlock
    ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    logger.info("Program completed - no deadlock detected")
    # Note: Output values are garbage because dataflow kernels are empty stubs
    # This test only verifies deadlock behavior, not functional correctness

    # Ensure device is fully synchronized before cleanup
    ttnn.synchronize_device(device)


@pytest.mark.timeout(30)  # Fail if test hangs for more than 30 seconds
def test_no_deadlock_repro(device):
    """
    Reproduces the no_deadlock case from tt-mlir no_deadlock-opt.mlir

    This test is similar to test_deadlock_repro but with a key difference:
    - Buffer 0 (output) is si32 instead of f32

    The MLIR shows:
    - Buffer 0 @ 103712: tile<32x32, si32> output (si32 not f32!)
    - Buffer 1 @ 106304: 5408x32 si32 input
    - Buffer 2 @ 798528: tile<32x32, si32> intermediate
    """

    # Create input data - 32x32 = 1024 elements in row-major format
    input_shape = [1, 1, 32, 32]
    input_data = torch.arange(1024, dtype=torch.int32).reshape(input_shape)

    # Create input tensor - si32 (signed int32) matching MLIR Buffer 1
    input_tensor = ttnn.from_torch(
        input_data,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Create output tensor - si32 matching MLIR Buffer 0 (KEY DIFFERENCE!)
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_shape),
        ttnn.int32,  # si32 instead of f32
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    # Single core at (0,0) matching MLIR
    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB indices matching MLIR: cb_ports = [0, 1, 2]
    # CB 0: output (si32 tiles)
    # CB 1: intermediate (si32 tiles)
    # CB 2: input (row-major si32)
    in_cb_id = 2
    intermediate_cb_id = 1
    out_cb_id = 0

    # CB configurations - all si32 now
    int32_tile_size_bytes = 4 * 1024  # int32: 4 bytes * 32 * 32
    cb_num_tiles = 2

    # Input CB (row-major si32 data)
    in_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=in_cb_id,
        data_format=ttnn.int32,
        page_size=int32_tile_size_bytes,
    )
    in_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * int32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[in_cb_format],
    )

    # Intermediate CB (si32 tiles)
    intermediate_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=intermediate_cb_id,
        data_format=ttnn.int32,
        page_size=int32_tile_size_bytes,
    )
    intermediate_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * int32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[intermediate_cb_format],
    )

    # Output CB (si32 tiles - not f32!)
    out_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=out_cb_id,
        data_format=ttnn.int32,
        page_size=int32_tile_size_bytes,
    )
    out_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_tiles * int32_tile_size_bytes,
        core_ranges=core_range,
        format_descriptors=[out_cb_format],
    )

    # DataFlow0 kernel (reader) - reads from input buffer to CB 2
    reader_ct_args = [in_cb_id]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    num_tiles = 1
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_tiles,
        0,  # start tile
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/no_deadlock_dataflow_0_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # DataFlow1 kernel (writer) - writes from CB 0 to output buffer
    writer_ct_args = [out_cb_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_tiles,
        0,  # start tile
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/no_deadlock_dataflow_1_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel - does tilize operations
    # Matching MLIR: tilize_init, tilize_block, tilize_uninit
    # ct_args in MLIR: [cb_port[0], cb_port[1], cb_port[2]]
    # The kernel expects CB indices as compile time args
    compute_ct_args = [
        out_cb_id,  # cb_port[0] = 0
        in_cb_id,  # cb_port[1] = 2
        intermediate_cb_id,  # cb_port[2] = 1
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/no_deadlock_compute_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    # Create program descriptor
    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, intermediate_cb_desc, out_cb_desc],
    )

    logger.info("Running generic_op with no_deadlock tilize program...")

    # Execute the program
    # If this completes without hanging/timeout, there's no deadlock
    ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    logger.info("Program completed - no deadlock detected")
    # Note: Output values are garbage because dataflow kernels are empty stubs
    # This test only verifies deadlock behavior, not functional correctness

    # Ensure device is fully synchronized before cleanup
    ttnn.synchronize_device(device)
