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
