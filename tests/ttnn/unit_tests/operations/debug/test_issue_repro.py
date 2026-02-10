import torch
import pytest
import ttnn
from loguru import logger


def test_deadlock_repro(device):
    """
    Reproduces deadlock from deadlock.mlir.
    Issue: https://github.com/tenstorrent/tt-mlir/issues/6848
    f32 output buffer with si32 input causes type mismatch deadlock.
    """

    logger.info("Starting test_deadlock_repro")

    input_shape = [1, 1, 32, 32]
    input_data = torch.arange(1024, dtype=torch.int32).reshape(input_shape)

    input_tensor = ttnn.from_torch(
        input_data,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # f32 output
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_shape),
        ttnn.float32,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # CB 0: f32 output, CB 1: si32 intermediate, CB 2: si32 input
    in_cb_id = 2
    intermediate_cb_id = 1
    out_cb_id = 0

    int32_tile_size_bytes = 4 * 1024
    float32_tile_size_bytes = 4 * 1024
    cb_num_tiles = 2

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

    # DataFlow0 kernel
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

    # DataFlow1 kernel
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

    compute_ct_args = [
        out_cb_id,
        in_cb_id,
        intermediate_cb_id,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/deadlock_compute_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, intermediate_cb_desc, out_cb_desc],
    )

    ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    ttnn.synchronize_device(device)
    logger.info("Completed test_deadlock_repro")


def test_no_deadlock_repro(device):
    """
    Reproduces no_deadlock case from no_deadlock.mlir.
    Issue: https://github.com/tenstorrent/tt-mlir/issues/6848
    si32 output buffer matches si32 input
    """

    logger.info("Starting test_no_deadlock_repro")
    input_shape = [1, 1, 32, 32]
    input_data = torch.arange(1024, dtype=torch.int32).reshape(input_shape)

    input_tensor = ttnn.from_torch(
        input_data,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # si32 output
    output_tensor = ttnn.allocate_tensor_on_device(
        ttnn.Shape(input_shape),
        ttnn.int32,
        ttnn.TILE_LAYOUT,
        device,
        ttnn.L1_MEMORY_CONFIG,
    )

    core = ttnn.CoreCoord(0, 0)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])

    # All CBs use si32
    in_cb_id = 2
    intermediate_cb_id = 1
    out_cb_id = 0

    int32_tile_size_bytes = 4 * 1024
    cb_num_tiles = 2

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

    reader_ct_args = [in_cb_id]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    num_tiles = 1
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        num_tiles,
        0,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/no_deadlock_dataflow_0_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_ct_args = [out_cb_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        num_tiles,
        0,
    ]

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/no_deadlock_dataflow_1_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    compute_ct_args = [
        out_cb_id,
        in_cb_id,
        intermediate_cb_id,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/no_deadlock_compute_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[in_cb_desc, intermediate_cb_desc, out_cb_desc],
    )

    ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
    ttnn.synchronize_device(device)
    logger.info("Completed test_no_deadlock_repro")


def test_issue_6849_repro(device):
    """
    Attempt to reproduce issue from issue_6849.mlir. Not seeing the incorrect values reported in the issue.
    Issue: https//github.com/tenstorrent/tt-mlir/issues/6849
    Multi-core (4x2) program with multiple CB ports and buffer configurations.
    To see output, run export TT_METAL_DPRINT_CORES=0,0
    DPRINTs are located in /tt-metal/tt_metal/kernels/compute/issue_6849_compute_kernel.cpp
    """

    logger.info("Starting test_issue_6849_repro")

    # 4x2 grid = 8 cores
    num_cores_y = 4
    num_cores_x = 2

    # MLIR shows buffers are shaped 4x2x...x... (grid_height x grid_width x ...)
    # This represents sharded data across the 4x2 grid
    # For single core data: 1 tile = 32x32, so:
    # - 4x2x1x1 tiles = 4 rows * 2 cols * 1 * 1 = 8 tiles total
    # - 4x2x3x1 tiles = 4 rows * 2 cols * 3 * 1 = 24 tiles total

    # Per MLIR operands: we need 5 tensors matching (%0, %0, %1, %3, %3)
    # %0 = 4x2x1x1 f32 tiles (address 103712)
    # %1 = 4x2x3x1 si32 tiles (address 106304)
    # %3 = 4x2x3x1 f32 tiles (address 118592)

    # Create shapes for the entire sharded buffer

    # 4x2x1x1 -> [128, 64] size tensor
    tensor_shape_4x2x1x1 = [128, 64]
    # 4x2x3x1 -> [384, 64] size tensor
    tensor_shape_4x2x3x1 = [384, 64]

    # Input data - %0: f32 buffer (used twice as operands 0 and 1)
    input_f32_data = torch.randn(tensor_shape_4x2x1x1, dtype=torch.float32)

    # Input data - %1: si32 buffer (operand 2)
    input_si32_data = torch.randint(-100, 100, tensor_shape_4x2x3x1, dtype=torch.int32)

    # Output data - %3: f32 buffer (used twice as operands 3 and 4)
    output_f32_data = torch.zeros(tensor_shape_4x2x3x1, dtype=torch.float32)

    # Create sharded memory configs
    tensor_4x2x1x1_mem_cfg = ttnn.create_sharded_memory_config(
        tensor_shape_4x2x1x1,
        ttnn.CoreGrid(y=4, x=2),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )

    tensor_4x2x3x1_mem_cfg = ttnn.create_sharded_memory_config(
        tensor_shape_4x2x3x1,
        ttnn.CoreGrid(y=4, x=2),
        strategy=ttnn.ShardStrategy.BLOCK,
        use_height_and_width_as_shard_shape=False,
    )

    # Create tensors on device
    tensor_f32 = ttnn.from_torch(
        input_f32_data,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_4x2x1x1_mem_cfg,
    )

    tensor_si32 = ttnn.from_torch(
        input_si32_data,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_4x2x3x1_mem_cfg,
    )

    tensor_f32_output = ttnn.from_torch(
        output_f32_data,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=tensor_4x2x3x1_mem_cfg,
    )

    # Core grid 4x2
    start_core = ttnn.CoreCoord(0, 0)
    end_core = ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1)
    core_range = ttnn.CoreRangeSet([ttnn.CoreRange(start_core, end_core)])

    # CB configuration, cb_ports = [31, 0, 1, 2, 3]
    cb_31 = 31  # f32
    cb_0 = 0  # f32
    cb_1 = 1  # si32
    cb_2 = 2  # f32
    cb_3 = 3  # f32

    float32_tile_size = 4 * 1024
    int32_tile_size = 4 * 1024
    cb_31_num_tiles = 1
    cb_0_num_tiles = 1
    cb_1_num_tiles = 3
    cb_2_num_tiles = 3
    cb_3_num_tiles = 3

    # CB 31: f32
    cb_31_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_31,
        data_format=ttnn.float32,
        page_size=float32_tile_size,
    )
    cb_31_desc = ttnn.CBDescriptor(
        total_size=cb_31_num_tiles * float32_tile_size,
        core_ranges=core_range,
        format_descriptors=[cb_31_format],
    )

    # CB 0: f32
    cb_0_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_0,
        data_format=ttnn.float32,
        page_size=float32_tile_size,
    )
    cb_0_desc = ttnn.CBDescriptor(
        total_size=cb_0_num_tiles * float32_tile_size,
        core_ranges=core_range,
        format_descriptors=[cb_0_format],
    )

    # CB 1: si32
    cb_1_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_1,
        data_format=ttnn.int32,
        page_size=int32_tile_size,
    )
    cb_1_desc = ttnn.CBDescriptor(
        total_size=cb_1_num_tiles * int32_tile_size,
        core_ranges=core_range,
        format_descriptors=[cb_1_format],
    )

    # CB 2: f32
    cb_2_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_2,
        data_format=ttnn.float32,
        page_size=float32_tile_size,
    )
    cb_2_desc = ttnn.CBDescriptor(
        total_size=cb_2_num_tiles * float32_tile_size,
        core_ranges=core_range,
        format_descriptors=[cb_2_format],
    )

    # CB 3: f32
    cb_3_format = ttnn.CBFormatDescriptor(
        buffer_index=cb_3,
        data_format=ttnn.float32,
        page_size=float32_tile_size,
    )
    cb_3_desc = ttnn.CBDescriptor(
        total_size=cb_3_num_tiles * float32_tile_size,
        core_ranges=core_range,
        format_descriptors=[cb_3_format],
    )

    # DataFlow0 kernel: ct_args = [cb_port[1], buffer_address[operand_2]]
    reader_ct_args = [cb_0]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(tensor_si32).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    for y in range(num_cores_y):
        for x in range(num_cores_x):
            # Kernel is empty but we still need to provide proper args structure
            reader_rt_args[x][y] = []

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/issue_6849_dataflow_0_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # DataFlow1 kernel: ct_args = [cb_port[3]] = [cb_2]
    writer_ct_args = [cb_2]

    writer_rt_args = ttnn.RuntimeArgs()
    for y in range(num_cores_y):
        for x in range(num_cores_x):
            writer_rt_args[x][y] = []

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/dataflow/issue_6849_dataflow_1_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # Compute kernel: ct_args = [cb_port[0], cb_port[1], cb_port[2], cb_port[3], cb_port[4]]
    # = [cb_31, cb_0, cb_1, cb_2, cb_3]
    compute_ct_args = [
        cb_31,
        cb_0,
        cb_1,
        cb_2,
        cb_3,
    ]

    compute_kernel = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/issue_6849_compute_kernel.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_range,
        compile_time_args=compute_ct_args,
        runtime_args=[],
        config=ttnn.ComputeConfigDescriptor(),
    )

    program_descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=[cb_31_desc, cb_0_desc, cb_1_desc, cb_2_desc, cb_3_desc],
    )

    ttnn.generic_op([tensor_f32, tensor_f32, tensor_si32, tensor_f32_output, tensor_f32_output], program_descriptor)

    ttnn.synchronize_device(device)
    logger.info("Completed test_issue_6849_repro")
