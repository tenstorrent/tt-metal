# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
CCL All-Reduce Operation using ttnn.generic_op

This module implements a multi-device all-reduce operation where each device
sends its data to its neighbor and receives data from its neighbor, then
performs a local reduction (sum).

For 2-device all-reduce with Linear topology:
- Neighbour exchange of data between devices 0 and 1
- Each device sums its local data with the received data
- Residual tensor is added to the sum if we are fusing with the next residual add block
"""

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class DeepseekMinimalAllReduce:
    """
    Multi-device all-reduce implementation using ttnn.generic_op.

    This class implements all-reduce across devices in a mesh using the fabric
    infrastructure. Each device exchanges data with its neighbor and performs
    a local sum reduction.
    """

    @staticmethod
    def golden(input_tensors, residual_tensor=None):
        """
        PyTorch reference implementation of all-reduce for validation.

        Args:
            input_tensors: List of input tensors (torch.Tensor), one per device
            residual_tensor: Optional residual tensor to add after reduction

        Returns:
            Output tensor that would be on each device after all-reduce (sum of all inputs)
        """
        result = torch.sum(torch.stack(input_tensors), dim=0)
        if residual_tensor is not None:
            result += residual_tensor
        return result

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor,
        semaphores,
        cluster_axis=0,
        num_links=2,
        persistent_output_tensor=None,
        residual_tensor_mesh=None,
    ):
        """
        Execute all-reduce operation using generic_op.

        Args:
            input_tensor_mesh: Input tensor mesh (each device has its own data)
            intermediate_tensor: Pre-allocated intermediate tensor mesh for receiving data
                                 Must have standard tiles (32x32)
            cluster_axis: Axis for all-reduce (default 0)
            num_links: Number of links to use (default 2 for all-reduce)
            persistent_output_tensor: Optional pre-allocated output tensor mesh.
            residual_tensor_mesh: Optional tensor mesh for residuals.
            semaphores: List of two global semaphores for synchronization.

        Returns:
            Output tensor with all-reduced data on all devices
        """

        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor)
        if residual_tensor_mesh is not None:
            residual_tensors_per_device = ttnn.get_device_tensors(residual_tensor_mesh)

        semaphore1 = semaphores[0]
        semaphore2 = semaphores[1]

        semaphore1_addr = ttnn.get_global_semaphore_address(semaphore1)
        semaphore2_addr = ttnn.get_global_semaphore_address(semaphore2)

        # Get tile and page info from input tensor (tiny tiles 1x32)
        input_tensor_sample = input_tensors_per_device[0]
        tile = input_tensor_sample.tile
        tile_height, tile_width = tile.tile_shape

        if persistent_output_tensor is not None:
            output_tensor = persistent_output_tensor
        else:
            output_tensor = ttnn.allocate_tensor_on_device(input_tensor_sample.spec, mesh_device)

        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        # Get tile info from intermediate tensor (standard tiles 32x32)
        intermediate_tensor_sample = intermediate_tensors_per_device[0]
        intermediate_tile = intermediate_tensor_sample.tile
        intermediate_tile_height, intermediate_tile_width = intermediate_tile.tile_shape

        # Element size based on dtype
        dtype = input_tensor_sample.dtype
        element_size = 2

        page_size_bytes = tile_height * tile_width * element_size
        standard_tile_size_bytes = intermediate_tile_height * intermediate_tile_width * element_size

        # Get shard shape to calculate number of pages
        shard_spec = input_tensor_sample.memory_config().shard_spec
        shard_width = shard_spec.shape[1]
        input_num_pages = shard_width // tile_width

        # Standard tile calculation
        tiny_tiles_per_standard_tile = 32
        num_standard_tiles = (input_num_pages + tiny_tiles_per_standard_tile - 1) // tiny_tiles_per_standard_tile

        # Packet and alignment info
        l1_alignment = 16  # L1 alignment
        packet_size_bytes = input_num_pages * page_size_bytes

        # CB indices
        src0_cb_index = 0  # For sender data (tiny tiles)
        compute_cb_in1 = 1  # For remote data (standard tiles, intermediate tensor)
        compute_cb_in2 = 2  # For local data (standard tiles, re-interpreted input)
        compute_cb_out = 3  # For output (standard tiles, re-interpreted output)
        packet_header_cb_id = 4  # For fabric packet headers
        packet_cb_id = 5  # For fabric packets
        compute_cb_residual = 6  # For fused residual add
        compute_cb_temp = 7

        # Packet header size
        packet_header_size_bytes = 32  # Size of one packet header

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Unified kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_all_reduce/kernels/all_reduce_kernel.cpp"

        # For each device in the mesh, create appropriate program
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                # Ring index along the cluster axis
                ring_index = row if cluster_axis == 0 else col
                is_first_chip = ring_index == 0

                # Get the device's tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]

                device = input_tensor_device.device()

                # Worker cores: data core (receiver) and second core (sender)
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                data_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
                if data_core.x > 0:
                    second_core = ttnn.CoreCoord(data_core.x - 1, data_core.y)
                else:
                    second_core = ttnn.CoreCoord(data_core.x + 1, data_core.y)

                sender_core = second_core
                receiver_core = data_core

                worker_core_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(sender_core, sender_core),
                        ttnn.CoreRange(receiver_core, receiver_core),
                    ]
                )
                sender_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(sender_core, sender_core)])
                receiver_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(receiver_core, receiver_core)])

                # Get physical cores for NOC addressing
                data_core_physical = device.worker_core_from_logical_core(data_core)
                core_noc_x = data_core_physical.x
                core_noc_y = data_core_physical.y

                sender_physical = device.worker_core_from_logical_core(sender_core)
                receiver_physical = device.worker_core_from_logical_core(receiver_core)

                remote_sender_noc_x = sender_physical.x
                remote_sender_noc_y = sender_physical.y
                remote_receiver_noc_x = receiver_physical.x
                remote_receiver_noc_y = receiver_physical.y

                # Determine neighbor and semaphores based on position
                sender_link = 0 if is_first_chip else 1
                receiver_link = 1 if is_first_chip else 0
                sender_semaphore_addr = semaphore1_addr if is_first_chip else semaphore2_addr
                receiver_semaphore_addr = semaphore2_addr if is_first_chip else semaphore1_addr

                # Calculate neighbor coordinate
                if is_first_chip:
                    neighbor_row = row + 1 if cluster_axis == 0 else row
                    neighbor_col = col if cluster_axis == 0 else col + 1
                else:
                    neighbor_row = row - 1 if cluster_axis == 0 else row
                    neighbor_col = col if cluster_axis == 0 else col - 1

                dst_num_hops = 1
                num_connections = 1
                using_persistent_buffers = 1 if persistent_output_tensor is not None else 0
                has_residual = 1 if residual_tensor_mesh is not None else 0

                # === NAMED COMPILE TIME ARGS ===
                # Sender kernel: NCRISC (reader) + BRISC (writer)
                sender_ncrisc_ct_args = [
                    ("cb0_id", src0_cb_index),
                    ("num_tiles", input_num_pages),
                    ("tensor_page_size", page_size_bytes),
                    ("core_noc_x", core_noc_x),
                    ("core_noc_y", core_noc_y),
                ]

                sender_brisc_ct_args = [
                    ("packet_header_cb_id", packet_header_cb_id),
                    ("packet_cb_id", src0_cb_index),
                    ("l1_alignment", l1_alignment),
                    ("input_num_tiles", input_num_pages),
                    ("page_size_bytes", page_size_bytes),
                    ("payload_size_bytes", packet_size_bytes),
                    ("data_noc_x", core_noc_x),
                    ("data_noc_y", core_noc_y),
                    ("remote_receiver_noc_x", remote_receiver_noc_x),
                    ("remote_receiver_noc_y", remote_receiver_noc_y),
                    ("dst_num_hops", dst_num_hops),
                    ("num_connections", num_connections),
                    ("using_persistent_buffer", using_persistent_buffers),
                ]

                # Receiver kernel: NCRISC (reader) + TRISC (compute)
                receiver_ncrisc_ct_args = [
                    ("packet_header_cb_id", packet_header_cb_id),
                    ("cb_in1", compute_cb_in1),
                    ("l1_alignment", l1_alignment),
                    ("cb_in2", compute_cb_in2),
                    ("remote_sender_noc_x", remote_sender_noc_x),
                    ("remote_sender_noc_y", remote_sender_noc_y),
                    ("num_standard_tiles", num_standard_tiles),
                    ("cb_residual", compute_cb_residual),
                    ("has_residual", has_residual),
                    ("using_persistent_buffer", using_persistent_buffers),
                ]

                receiver_trisc_ct_args = [
                    ("cb_in0", compute_cb_in1),
                    ("cb_in1", compute_cb_in2),
                    ("cb_out0", compute_cb_out),
                    ("cb_residual", compute_cb_residual),
                    ("cb_temp", compute_cb_temp),
                    ("has_residual", has_residual),
                    ("num_tiles", num_standard_tiles),
                ]

                # === COMMON RUNTIME ARGS ===
                # Sender NCRISC common runtime args (arg 0)
                sender_ncrisc_common_rt_args = [
                    input_tensor_device.buffer_address(),  # tensor_address
                ]

                # Sender BRISC common runtime args (args 0-1)
                sender_brisc_common_rt_args = [
                    intermediate_tensor_device.buffer_address(),  # receiver_base_address
                    sender_semaphore_addr,  # receive_semaphore_addr
                ]

                # Receiver NCRISC common runtime args (arg 0)
                receiver_ncrisc_common_rt_args = [
                    receiver_semaphore_addr,  # sender_semaphore_addr
                ]

                # === CB DESCRIPTORS ===
                cb0_format = ttnn.CBFormatDescriptor(
                    buffer_index=src0_cb_index,
                    data_format=dtype,
                    page_size=page_size_bytes,
                    tile=ttnn.TileDescriptor(tile_height, tile_width),
                )
                cb0_desc = ttnn.CBDescriptor(
                    total_size=input_num_pages * page_size_bytes,
                    core_ranges=worker_core_set,
                    format_descriptors=[cb0_format],
                )

                # CB1: Remote data for compute (standard tiles, intermediate tensor)
                cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(compute_cb_in1, intermediate_tensor_device)
                cb1_desc.core_ranges = receiver_core_set

                # CB2: Local data for compute - backed by input tensor but with standard tile format
                cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(compute_cb_in2, input_tensor_device)
                cb2_desc.core_ranges = receiver_core_set
                cb2_desc.total_size = num_standard_tiles * standard_tile_size_bytes
                cb2_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=compute_cb_in2,
                        data_format=dtype,
                        page_size=standard_tile_size_bytes,
                        tile=ttnn.TileDescriptor(intermediate_tile_height, intermediate_tile_width),
                    )
                ]

                # CB3: Output for compute - backed by output tensor with standard tile format
                cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(compute_cb_out, output_tensor_device)
                cb3_desc.core_ranges = receiver_core_set
                cb3_desc.total_size = num_standard_tiles * standard_tile_size_bytes
                cb3_desc.format_descriptors = [
                    ttnn.CBFormatDescriptor(
                        buffer_index=compute_cb_out,
                        data_format=dtype,
                        page_size=standard_tile_size_bytes,
                        tile=ttnn.TileDescriptor(intermediate_tile_height, intermediate_tile_width),
                    )
                ]

                # CB4: Packet headers
                cb4_format = ttnn.CBFormatDescriptor(
                    buffer_index=packet_header_cb_id,
                    data_format=ttnn.uint32,
                    page_size=packet_header_size_bytes,
                )
                cb4_desc = ttnn.CBDescriptor(
                    total_size=2 * packet_header_size_bytes,
                    core_ranges=worker_core_set,
                    format_descriptors=[cb4_format],
                )

                # CB5: Packet data for sender
                cb5_format = ttnn.CBFormatDescriptor(
                    buffer_index=packet_cb_id,
                    data_format=dtype,
                    page_size=page_size_bytes,
                )
                cb5_desc = ttnn.CBDescriptor(
                    total_size=packet_size_bytes,
                    core_ranges=worker_core_set,
                    format_descriptors=[cb5_format],
                )
                cb_list = [cb0_desc, cb1_desc, cb2_desc, cb3_desc, cb4_desc, cb5_desc]
                if residual_tensor_mesh is not None:
                    cb6_desc = ttnn.cb_descriptor_from_sharded_tensor(
                        compute_cb_residual, residual_tensors_per_device[device_idx]
                    )
                    cb6_desc.core_ranges = receiver_core_set
                    cb6_desc.total_size = num_standard_tiles * standard_tile_size_bytes
                    cb6_desc.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=compute_cb_residual,
                            data_format=dtype,
                            page_size=standard_tile_size_bytes,
                            tile=ttnn.TileDescriptor(intermediate_tile_height, intermediate_tile_width),
                        )
                    ]
                    cb_list.append(cb6_desc)
                    # CB7: Temp scratch buffer (not backed by tensor)
                    cb7_desc = ttnn.CBDescriptor(
                        total_size=num_standard_tiles * standard_tile_size_bytes,
                        core_ranges=receiver_core_set,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=compute_cb_temp,
                                data_format=dtype,
                                page_size=standard_tile_size_bytes,
                                tile=ttnn.TileDescriptor(intermediate_tile_height, intermediate_tile_width),
                            )
                        ],
                    )
                    cb_list.append(cb7_desc)

                # === KERNEL DESCRIPTORS using UnifiedKernelDescriptor ===
                # Single unified kernel with is_sender compile-time arg to differentiate roles
                # Sender core: NCRISC reads tensor, BRISC writes via fabric, TRISC no-op
                # Receiver core: NCRISC waits for data, BRISC no-op, TRISC computes reduction

                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=kernel_path,
                    core_ranges=worker_core_set,
                    ncrisc_named_compile_time_args=sender_ncrisc_ct_args + receiver_ncrisc_ct_args,
                    brisc_named_compile_time_args=sender_brisc_ct_args,
                    trisc_named_compile_time_args=receiver_trisc_ct_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        fp32_dest_acc_en=True,
                        math_approx_mode=False,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_sender",
                            core_range=sender_core_set,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        # Sender core: NCRISC and BRISC need per-core runtime args for fabric
                        # Receiver core: NCRISC needs per-core runtime args for fabric
                        ncrisc_args=[(sender_core, []), (receiver_core, [])],
                        brisc_args=[(sender_core, [])],
                    ),
                )

                # Get kernel descriptors - generates separate kernels for sender and receiver groups
                kernel_result = unified_kernel.get_kernel_descriptors()

                # Get kernel indices by role using compile-time arg values
                sender_group = kernel_result.get_group_by_arg("is_sender", 1)
                receiver_group = kernel_result.get_group_by_arg("is_sender", 0)

                # Set common runtime args for each kernel group
                kernel_result.kernels[
                    sender_group.ncrisc_kernel_index
                ].common_runtime_args = sender_ncrisc_common_rt_args
                kernel_result.kernels[sender_group.brisc_kernel_index].common_runtime_args = sender_brisc_common_rt_args
                kernel_result.kernels[
                    receiver_group.ncrisc_kernel_index
                ].common_runtime_args = receiver_ncrisc_common_rt_args

                # === PROGRAM DESCRIPTOR ===
                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    semaphores=[],
                    cbs=cb_list,
                )

                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                neighbor_coord = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
                neighbor_fabric_node_id = mesh_device.get_fabric_node_id(neighbor_coord)

                # Sender NCRISC fabric connection (reader needs fabric for signaling)
                sender_reader_kernel_idx = sender_group.ncrisc_kernel_index
                sender_reader_rt_args_ref = program.kernels[sender_reader_kernel_idx].runtime_args[sender_core.x][
                    sender_core.y
                ]
                sender_reader_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [sender_link],
                    program,
                    sender_reader_kernel_idx,
                    sender_core,
                )
                sender_reader_rt_args_ref.extend(sender_reader_fabric_args)

                # Sender BRISC fabric connection (writer sends data via fabric)
                sender_writer_kernel_idx = sender_group.brisc_kernel_index
                sender_writer_rt_args_ref = program.kernels[sender_writer_kernel_idx].runtime_args[sender_core.x][
                    sender_core.y
                ]
                sender_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [sender_link],
                    program,
                    sender_writer_kernel_idx,
                    sender_core,
                )
                sender_writer_rt_args_ref.extend(sender_fabric_args)

                receiver_reader_kernel_idx = receiver_group.ncrisc_kernel_index
                receiver_reader_rt_args_ref = program.kernels[receiver_reader_kernel_idx].runtime_args[receiver_core.x][
                    receiver_core.y
                ]
                receiver_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [receiver_link],
                    program,
                    receiver_reader_kernel_idx,
                    receiver_core,
                )
                receiver_reader_rt_args_ref.extend(receiver_fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        input_list = [input_tensor_mesh, output_tensor, intermediate_tensor]
        if residual_tensor_mesh is not None:
            input_list.append(residual_tensor_mesh)
        logger.debug("Executing CCL all-reduce via generic_op...")
        ttnn.generic_op(input_list, mesh_program_descriptor)

        # generic_op modifies tensors in place; return the output tensor
        return output_tensor
