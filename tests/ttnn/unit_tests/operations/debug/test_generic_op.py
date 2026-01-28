# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
import numpy as np
from math import prod
from loguru import logger

from models.common.utility_functions import skip_for_blackhole


@skip_for_blackhole("Not tested / built for Blackhole")
@pytest.mark.parametrize("num_tiles", [1, 12, 64, 128])
def test_eltwise_exp(device, num_tiles):
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

    max_core = ttnn.CoreCoord(7, 7)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
    (_, core_grid, core_group_1, core_group_2, work_per_core1, _) = ttnn.split_work_to_cores(all_cores, num_tiles)
    assert (
        len(core_group_2.ranges()) == 0
    ), "tt_metal/kernels/compute/eltwise_sfpu.cpp kernel has number of tiles to compile as compile time arg, does not support 2 core groups"

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

    reader_compile_time_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    writer_compile_time_args = [out_cb]
    writer_compile_time_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    compute_compile_time_args = [work_per_core1, 1]

    reader_rt_args = ttnn.RuntimeArgs()
    writer_rt_args = ttnn.RuntimeArgs()
    current_tile = 0
    for core_range in core_group_1.ranges():
        for x in range(core_range.start.x, core_range.end.x + 1):
            for y in range(core_range.start.y, core_range.end.y + 1):
                reader_rt_args[x][y] = [input_tensor.buffer_address(), work_per_core1, current_tile]
                writer_rt_args[x][y] = [output_tensor.buffer_address(), work_per_core1, current_tile]
                current_tile += work_per_core1

    reader_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=reader_compile_time_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=writer_compile_time_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    sfpu_defines = [("SFPU_OP_EXP_INCLUDE", "1"), ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")]
    compute_kernel_descriptor = ttnn.KernelDescriptor(
        kernel_source="tt_metal/kernels/compute/eltwise_sfpu.cpp",
        # source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH, expecting this to be the default value
        core_ranges=core_grid,
        compile_time_args=compute_compile_time_args,
        defines=sfpu_defines,
        runtime_args=[],
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


@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
def test_point_to_point(mesh_device):
    logger.info("Running test_point_to_point: sending data from device (0,0) to device (0,1)")
    mesh_shape = mesh_device.shape
    num_devices = prod(mesh_shape)

    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)

    full_shape = (1, num_devices, 32, 64)
    dtype = ttnn.bfloat16

    input_torch = torch.rand(full_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        input_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=1),
    )
    ttnn.synchronize_device(mesh_device)

    # Get sender's data for verification
    input_shards = ttnn.get_device_tensors(input_tensor)
    sender_data = ttnn.to_torch(input_shards[0])
    original_receiver_data = ttnn.to_torch(input_shards[1])
    logger.info(f"Original shard data at receiver: {original_receiver_data}")

    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    input_num_pages = (prod(input_tensor.padded_shape) + 1024 - 1) // 1024  # = 2
    input_page_size_bytes = input_num_pages * 1024
    l1_alignment = 16
    aligned_input_page_size_bytes = ((input_page_size_bytes + l1_alignment - 1) // l1_alignment) * l1_alignment

    fabric_max_packet_size_bytes = 4416

    max_packet_size_bytes = 1 << (fabric_max_packet_size_bytes.bit_length() - 1)  # bit_floor equivalent.. = 4096
    num_pages_per_packet = min(max_packet_size_bytes // aligned_input_page_size_bytes, input_num_pages)  # = 2
    packet_size_bytes = aligned_input_page_size_bytes * num_pages_per_packet  # = 4096
    num_page_segments = 1

    intermediate_spec = ttnn._ttnn.operations.point_to_point.p2p_compute_intermediate_tensor_spec(
        input_tensor, receiver_coord, sender_coord, ttnn.Topology.Linear
    )
    intermediate_tensor = ttnn.allocate_tensor_on_device(intermediate_spec, mesh_device)

    # Use single core for simplicity
    sender_core = ttnn.CoreCoord(0, 0)
    receiver_core = ttnn.CoreCoord(0, 0)
    sender_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
    receiver_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(receiver_core, receiver_core)])

    input_dataformat = dtype
    packet_header_size_bytes = 64

    grid = mesh_device.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    global_semaphore = ttnn.create_global_semaphore(mesh_device, available_cores, 0)
    global_semaphore_address = ttnn.get_global_semaphore_address(global_semaphore)
    ttnn.synchronize_device(mesh_device)

    # Determine routing direction and next fabric node ID
    # hardcoded for point_to_point::detail::fabric_1d_routing FABRIC_1D + Linear
    sender_fabric_id = mesh_device.get_fabric_node_id(sender_coord)
    receiver_fabric_id = mesh_device.get_fabric_node_id(receiver_coord)
    num_hops_sender = 1
    dst_is_forward = False
    next_fabric_id_sender = mesh_device.get_fabric_node_id(receiver_coord)
    num_hops_receiver = 1
    sender_is_forward = True
    next_fabric_id_receiver = mesh_device.get_fabric_node_id(sender_coord)
    link_idx = 0

    mesh_program_descriptor = ttnn.MeshProgramDescriptor()
    # ----- SENDER PROGRAM -----
    sender_cb_id = 0
    packet_header_cb_id = 1
    packet_cb_id = 2
    cb_num_pages = 2

    sender_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=sender_cb_id,
        data_format=input_dataformat,
        page_size=aligned_input_page_size_bytes,
    )
    sender_cb_desc = ttnn.CBDescriptor(
        total_size=cb_num_pages * aligned_input_page_size_bytes,
        core_ranges=sender_core_set,
        format_descriptors=[sender_cb_format],
    )

    packet_header_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=packet_header_cb_id,
        data_format=ttnn.uint32,
        page_size=packet_header_size_bytes,
    )
    packet_header_cb_desc = ttnn.CBDescriptor(
        total_size=2 * 2 * packet_header_size_bytes,  # 2 headers * 2 buffering
        core_ranges=sender_core_set,
        format_descriptors=[packet_header_cb_format],
    )

    packet_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=packet_cb_id,
        data_format=input_dataformat,
        page_size=packet_size_bytes,
    )
    packet_cb_desc = ttnn.CBDescriptor(
        total_size=packet_size_bytes,
        core_ranges=sender_core_set,
        format_descriptors=[packet_cb_format],
    )

    reader_ct_args = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[sender_core.x][sender_core.y] = [
        input_tensor.buffer_address(),
        input_num_pages,
        0,  # page_idx_start
        input_page_size_bytes,
    ]

    writer_ct_args = [
        sender_cb_id,
        packet_header_cb_id,
        packet_cb_id,
        l1_alignment,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[sender_core.x][sender_core.y] = [
        intermediate_tensor.buffer_address(),
        0,  # page_idx_start
        input_num_pages,  # page_idx_end
        num_hops_sender,
        input_page_size_bytes,
        packet_size_bytes,
        num_pages_per_packet,
        num_page_segments,
        global_semaphore_address,
        int(dst_is_forward),
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp",
        core_ranges=sender_core_set,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_send.cpp",
        core_ranges=sender_core_set,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    sender_program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel],
        semaphores=[],
        cbs=[sender_cb_desc, packet_header_cb_desc, packet_cb_desc],
    )

    # Append fabric connection args to writer kernel
    writer_rt_args_ref = sender_program.kernels[1].runtime_args[sender_core.x][sender_core.y]

    # dst_is_forward = False
    writer_rt_args_ref.append(int(not dst_is_forward))
    fabric_args = ttnn.setup_fabric_connection(
        sender_fabric_id, next_fabric_id_sender, link_idx, sender_program, sender_core
    )
    writer_rt_args_ref.extend(fabric_args)

    mesh_program_descriptor[ttnn.MeshCoordinateRange(sender_coord, sender_coord)] = sender_program

    # ----- RECEIVER PROGRAM -----
    packet_header_cb_id = 0
    packet_cb_id = 1
    receiver_cb_id = 2

    packet_header_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=packet_header_cb_id,
        data_format=ttnn.uint32,
        page_size=packet_header_size_bytes,
    )
    packet_header_cb_desc = ttnn.CBDescriptor(
        total_size=2 * 2 * packet_header_size_bytes,
        core_ranges=receiver_core_set,
        format_descriptors=[packet_header_cb_format],
    )

    packet_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=packet_cb_id,
        data_format=input_dataformat,
        page_size=packet_size_bytes,
    )
    packet_cb_desc = ttnn.CBDescriptor(
        total_size=packet_size_bytes,
        core_ranges=receiver_core_set,
        format_descriptors=[packet_cb_format],
    )

    receiver_cb_num_pages = 3 * num_pages_per_packet
    receiver_cb_format = ttnn.CBFormatDescriptor(
        buffer_index=receiver_cb_id,
        data_format=input_dataformat,
        page_size=input_page_size_bytes,
    )
    receiver_cb_desc = ttnn.CBDescriptor(
        total_size=receiver_cb_num_pages * input_page_size_bytes,
        core_ranges=receiver_core_set,
        format_descriptors=[receiver_cb_format],
    )

    reader_ct_args = [
        packet_header_cb_id,
        packet_cb_id,
        receiver_cb_id,
        l1_alignment,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(intermediate_tensor).get_compile_time_args())

    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[receiver_core.x][receiver_core.y] = [
        0,  # page_idx_start
        input_num_pages,  # page_idx_end
        num_pages_per_packet,
        intermediate_tensor.buffer_address(),
        packet_size_bytes,
        input_page_size_bytes,
        num_page_segments,
        global_semaphore_address,
        num_hops_receiver,
        int(sender_is_forward),
    ]

    writer_ct_args = [receiver_cb_id]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[receiver_core.x][receiver_core.y] = [
        output_tensor.buffer_address(),
        input_num_pages,
        0,  # page_idx_start
        input_page_size_bytes,
    ]

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_receive.cpp",
        core_ranges=receiver_core_set,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    writer_kernel = ttnn.KernelDescriptor(
        kernel_source="ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/writer_unary_interleaved_start_id_gen.cpp",
        core_ranges=receiver_core_set,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    receiver_program = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel],
        semaphores=[],
        cbs=[packet_header_cb_desc, packet_cb_desc, receiver_cb_desc],
    )

    # Append fabric connection args to reader kernel
    reader_rt_args_ref = receiver_program.kernels[0].runtime_args[receiver_core.x][receiver_core.y]

    # sender_is_forward = True
    fabric_args = ttnn.setup_fabric_connection(
        receiver_fabric_id, next_fabric_id_receiver, link_idx, receiver_program, receiver_core
    )
    reader_rt_args_ref.extend(fabric_args)
    reader_rt_args_ref.append(int(not sender_is_forward))

    mesh_program_descriptor[ttnn.MeshCoordinateRange(receiver_coord, receiver_coord)] = receiver_program

    # Execute generic_op
    ttnn.generic_op([input_tensor, intermediate_tensor, output_tensor], mesh_program_descriptor)
    ttnn.synchronize_device(mesh_device)

    # Verify output
    output_data = ttnn.get_device_tensors(output_tensor)
    output_receiver_idx = 1
    output_tensor_cpu = ttnn.to_torch(output_data[output_receiver_idx])

    logger.info(f"Sender data: {sender_data}")
    logger.info(f"Output at receiver: {output_tensor_cpu}")

    assert torch.allclose(output_tensor_cpu, sender_data), "Receiver should have sender's data after transfer"


def create_fabric_router_config(max_payload_size):
    """Helper to create FabricRouterConfig with custom max payload size."""
    config = ttnn._ttnn.fabric.FabricRouterConfig()
    config.max_packet_payload_size_bytes = max_payload_size
    return config


@pytest.mark.parametrize("mesh_device", [(4, 2)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(15232),
        }
    ],
    indirect=True,
)
def test_deepseek_minimal_broadcast_dual_axis(mesh_device):
    """
    Test dual-axis broadcast on a 4x2 mesh using the generic op infrastructure.

    Sender at (1,0) broadcasts:
    1. First across secondary axis (axis 1) to the device at (1,1) - the secondary sender
    2. Then both sender (1,0) and secondary sender (1,1) broadcast along primary axis (axis 0)
       to all devices in their respective columns

    This test implements the broadcast_tile_writer_batch1.cpp and broadcast_tile_reader_batch1.cpp
    kernels entirely through Python using the generic op API.
    """
    logger.info("Running test_deepseek_minimal_broadcast_dual_axis")

    mesh_rows = 4
    mesh_cols = 2
    num_devices = mesh_rows * mesh_cols
    sender_row = 1
    sender_col = 0

    # Tensor configuration
    output_shape = [1, 7168]
    input_shard_shape = (1, 7168)
    input_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    tensor_mem_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    layout = ttnn.TILE_LAYOUT
    input_dtype = ttnn.bfloat16
    num_links = 1
    cluster_axis = 0
    secondary_cluster_axis = 1
    topology = ttnn.Topology.Linear

    # Create submesh
    mesh_shape = mesh_device.shape
    if mesh_shape[0] * mesh_shape[1] < num_devices:
        pytest.skip("Test requires more devices than are available on this platform")

    submesh = mesh_device.create_submesh(ttnn.MeshShape((mesh_rows, mesh_cols)))

    # Set up sub-device
    compute_grid_size = submesh.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device = ttnn.SubDevice([ccl_sub_device_crs])
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]
    sub_device_manager = submesh.create_sub_device_manager([worker_sub_device], 0)
    submesh.load_sub_device_manager(sub_device_manager)
    submesh.set_sub_device_stall_group(sub_device_stall_group)

    # Create global semaphores
    grid = submesh.compute_with_storage_grid_size()
    num_cores = grid.x * grid.y
    available_cores = ttnn.num_cores_to_corerangeset(num_cores, grid, row_wise=True)
    out_ready_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    barrier_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    secondary_sync_semaphore = ttnn.create_global_semaphore(submesh, available_cores, 0)
    ttnn.synchronize_device(submesh)

    out_ready_sem_addr = ttnn.get_global_semaphore_address(out_ready_semaphore)
    barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_semaphore)
    secondary_sync_sem_addr = ttnn.get_global_semaphore_address(secondary_sync_semaphore)

    # Set up sharded memory config
    input_shard_spec = ttnn.ShardSpec(
        input_shard_grid,
        input_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_mem_config = ttnn.MemoryConfig(tensor_mem_layout, buffer_type=ttnn.BufferType.L1, shard_spec=input_shard_spec)
    output_mem_config = input_mem_config

    # Create sender tensor
    sender_tensor = torch.rand(output_shape, dtype=torch.bfloat16)

    # Create mesh tensor with sender's tensor at sender_coord, zeros elsewhere
    device_tensors = []
    for row in range(mesh_rows):
        if row == sender_row:
            device_tensors.append(sender_tensor)
        else:
            device_tensors.append(torch.zeros_like(sender_tensor))

    mesh_tensor_torch = torch.cat(device_tensors, dim=0)
    mesh_mapper_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementReplicate()], submesh.shape)
    input_tensor_mesh = ttnn.from_torch(
        mesh_tensor_torch,
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=input_mem_config,
        mesh_mapper=ttnn.create_mesh_mapper(submesh, mesh_mapper_config),
    )

    # Create output tensor
    output_tensor = ttnn.from_torch(
        torch.zeros(output_shape, dtype=torch.bfloat16),
        device=submesh,
        layout=layout,
        tile=ttnn.Tile((1, 32)),
        dtype=input_dtype,
        memory_config=output_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
    )

    # Get tensor specs
    input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
    output_tensors_per_device = ttnn.get_device_tensors(output_tensor)

    # Calculate packet size and page info
    packet_size_bytes = 14336  # tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes()
    # For width-sharded tiny tiles (1, 32) with bfloat16:
    # page_size = tile_height * tile_width * element_size = 1 * 32 * 2 = 64 bytes
    tile_height = 1
    tile_width = 32
    element_size = 2  # bfloat16
    page_size_bytes = tile_height * tile_width * element_size  # 64 bytes per tile
    # Number of tiles = shard_width / tile_width = 7168 / 32 = 224 tiles
    input_num_pages = output_shape[1] // tile_width  # 224 pages/tiles
    num_pages_per_packet = packet_size_bytes // page_size_bytes  # 14336 / 64 = 224

    # CB indices
    src0_cb_index = 0

    # Create mesh program descriptor
    mesh_program_descriptor = ttnn.MeshProgramDescriptor()

    # For each device in the mesh, create appropriate program
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            coord = ttnn.MeshCoordinate(row, col)
            is_sender = (row == sender_row) and (col == sender_col)
            is_secondary_sender = (row == sender_row) and (col != sender_col)
            is_receiver = not is_sender and not is_secondary_sender

            # Get the device's input and output tensors
            device_idx = row * mesh_cols + col
            input_tensor_device = input_tensors_per_device[device_idx]
            output_tensor_device = output_tensors_per_device[device_idx]

            # Worker core is the data core
            worker_core = ttnn.CoreCoord(0, 0)
            worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])

            # Get physical core for NOC addressing
            device = input_tensor_device.device()
            data_core_physical = device.worker_core_from_logical_core(worker_core)
            core_noc_x = data_core_physical.x
            core_noc_y = data_core_physical.y

            # Calculate ring index and targets for primary axis (column)
            ring_size = mesh_rows
            ring_index = row
            # For Linear topology, calculate forward and backward targets
            num_targets_forward = ring_size - ring_index - 1
            num_targets_backward = ring_index

            # Determine if this device has secondary axis connections
            has_secondary_target = is_sender and (mesh_cols > 1)
            has_reverse_secondary_connection = is_secondary_sender

            # Calculate mcast distances
            start_distance_forward = 1 if num_targets_forward > 0 else 0
            range_hops_forward = num_targets_forward
            start_distance_backward = 1 if num_targets_backward > 0 else 0
            range_hops_backward = num_targets_backward

            # Reader compile-time args
            reader_compile_args = [
                src0_cb_index,  # cb0_id
                num_pages_per_packet,  # packet_size_in_pages
                page_size_bytes,  # tensor0_page_size
                int(is_sender),  # is_sender
                core_noc_x,
                core_noc_y,
                int(is_secondary_sender),  # is_secondary_sender
                int(is_sender or is_secondary_sender),  # is_active_broadcaster
            ]

            # Writer compile-time args
            writer_compile_args = [
                src0_cb_index,  # cb0_id
                num_pages_per_packet,  # packet_size_in_pages
                page_size_bytes,  # tensor0_page_size
                num_targets_forward,  # num_targets_forward_direction
                num_targets_backward,  # num_targets_backward_direction
                int(is_sender),  # is_sender
                core_noc_x,
                core_noc_y,
                int(is_secondary_sender),  # is_secondary_sender
                int(has_secondary_target),  # has_secondary_target
                int(has_reverse_secondary_connection),  # has_reverse_secondary_connection
                start_distance_forward,  # start_distance_in_hops_forward
                range_hops_forward,  # range_hops_forward
                start_distance_backward,  # start_distance_in_hops_backward
                range_hops_backward,  # range_hops_backward
                0,  # using_persistent_buffers = False
            ]

            # Reader runtime args
            reader_rt_args = ttnn.RuntimeArgs()
            reader_rt_args[worker_core.x][worker_core.y] = [
                input_tensor_device.buffer_address(),  # tensor_address0
                0,  # tile_id_start
                input_num_pages,  # tile_id_end
            ]

            # Writer runtime args
            wait_output_semaphore = is_secondary_sender or is_receiver
            reset_global_semaphore = is_secondary_sender or is_receiver
            out_ready_sem_wait_value = 1 * num_links

            writer_rt_args = ttnn.RuntimeArgs()
            writer_rt_args[worker_core.x][worker_core.y] = [
                output_tensor_device.buffer_address(),  # tensor_address0
                out_ready_sem_addr,  # out_ready_sem_bank_addr
                0,  # tile_id_start
                input_num_pages,  # tile_id_end
                int(wait_output_semaphore),  # wait_output_semaphore
                int(reset_global_semaphore),  # reset_global_semaphore
                core_noc_x,  # out_ready_sem_noc0_x (drain_sync_core)
                core_noc_y,  # out_ready_sem_noc0_y
                out_ready_sem_wait_value,  # out_ready_sem_wait_value
                barrier_sem_addr,  # barrier_sem
                core_noc_x,  # barrier_sem_noc0_x
                core_noc_y,  # barrier_sem_noc0_y
                ring_index,
                secondary_sync_sem_addr,  # secondary_sync_sem
            ]

            # Determine fabric connections
            fabric_node_id = submesh.get_fabric_node_id(coord)
            dst_nodes = []

            # Primary axis connections (forward and backward in column)
            if num_targets_forward > 0:
                forward_coord = ttnn.MeshCoordinate(row + 1, col)
                dst_nodes.append(submesh.get_fabric_node_id(forward_coord))

            if num_targets_backward > 0:
                backward_coord = ttnn.MeshCoordinate(row - 1, col)
                dst_nodes.append(submesh.get_fabric_node_id(backward_coord))

            # Secondary axis connection (for sender to secondary sender)
            if has_secondary_target:
                secondary_coord = ttnn.MeshCoordinate(row, 1)  # Other column
                dst_nodes.append(submesh.get_fabric_node_id(secondary_coord))

            # Reverse secondary connection (for secondary sender back to sender)
            if has_reverse_secondary_connection:
                sender_coord_back = ttnn.MeshCoordinate(sender_row, sender_col)
                dst_nodes.append(submesh.get_fabric_node_id(sender_coord_back))

            num_connections = len(dst_nodes)
            writer_rt_args[worker_core.x][worker_core.y].append(num_connections)

            # Create CB config
            df = input_dtype
            cb_config = ttnn.CBFormatDescriptor(
                buffer_index=src0_cb_index,
                data_format=df,
                page_size=page_size_bytes,
            )
            cb_desc = ttnn.CBDescriptor(
                total_size=num_pages_per_packet * page_size_bytes,
                core_ranges=worker_core_set,
                format_descriptors=[cb_config],
            )

            # Create reader kernel
            reader_kernel = ttnn.KernelDescriptor(
                kernel_source="ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/kernels/broadcast_tile_reader_batch1.cpp",
                core_ranges=worker_core_set,
                compile_time_args=reader_compile_args,
                runtime_args=reader_rt_args,
                config=ttnn.ReaderConfigDescriptor(),
            )

            # Create writer kernel
            writer_kernel = ttnn.KernelDescriptor(
                kernel_source="ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_minimal_broadcast/device/kernels/broadcast_tile_writer_batch1.cpp",
                core_ranges=worker_core_set,
                compile_time_args=writer_compile_args,
                runtime_args=writer_rt_args,
                config=ttnn.WriterConfigDescriptor(),
            )

            # Create program descriptor
            program = ttnn.ProgramDescriptor(
                kernels=[reader_kernel, writer_kernel],
                semaphores=[],
                cbs=[cb_desc],
            )

            # Append fabric connection args to writer kernel if there are connections
            if num_connections > 0:
                writer_rt_args_ref = program.kernels[1].runtime_args[worker_core.x][worker_core.y]
                fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    dst_nodes,
                    [0],  # link_idx for all connections
                    program,
                    1,  # kernel_idx (writer kernel)
                    worker_core,
                    ttnn.FabricApiType.Linear,
                )
                writer_rt_args_ref.extend(fabric_args)

            mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

    # Execute generic_op
    logger.info("Executing dual-axis broadcast via generic_op...")
    ttnn.generic_op([input_tensor_mesh, output_tensor], mesh_program_descriptor)
    ttnn.synchronize_device(submesh)

    # Verify output - all devices should have the sender's data
    logger.info("Verifying output...")
    output_tensor_torch = ttnn.to_torch(
        output_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(submesh, dim=0),
    )

    slice_size = output_shape[0]
    all_passed = True
    for device_idx in range(num_devices):
        start = device_idx * slice_size
        end = start + slice_size
        received = output_tensor_torch[start:end, :]
        assert received.shape == sender_tensor.shape, f"Shape mismatch at device {device_idx}"

        if not torch.allclose(received, sender_tensor, rtol=1e-3, atol=1e-3):
            logger.error(f"Output mismatch for device {device_idx}")
            all_passed = False
        else:
            logger.info(f"Device {device_idx}: PASSED")

    # Cleanup
    submesh.reset_sub_device_stall_group()
    submesh.clear_loaded_sub_device_manager()

    assert all_passed, "Not all devices received the correct broadcast data"
    logger.info("test_deepseek_minimal_broadcast_dual_axis PASSED")
