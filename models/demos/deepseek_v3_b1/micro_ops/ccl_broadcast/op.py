# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
CCL Broadcast Operation using ttnn.generic_op
This module implements a multi-device broadcast operation where a sender device
broadcasts data to all other devices in the mesh. Supports both single-axis
and dual-axis broadcast configurations.
For dual-axis broadcast on a 2D mesh:
1. Primary sender broadcasts across secondary axis to create a secondary sender
2. Both sender and secondary sender broadcast along the primary axis to their columns
"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor


class DeepseekMinimalBroadcast:
    """
    Multi-device broadcast implementation using ttnn.generic_op.
    This class implements broadcast from a sender device to all other devices
    in a mesh using the fabric infrastructure.
    """

    @staticmethod
    def golden(input_tensor):
        """
        PyTorch reference implementation of broadcast for validation.
        Args:
            input_tensor: Input tensor (torch.Tensor) - the data at sender
        Returns:
            Output tensor that would be on each device after broadcast
        """
        # All devices should have the sender's data
        return input_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        output_tensor,
        sender_coord,
        semaphores,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
    ):
        """
        Execute broadcast operation using generic_op.
        Args:
            input_tensor_mesh: Input tensor mesh (sender has data, others have zeros)
            output_tensor: Pre-allocated output tensor mesh
            sender_coord: ttnn.MeshCoordinate of the sender device
            semaphores: List of pre-created semaphores
            cluster_axis: Primary axis for broadcast (default 0)
            secondary_cluster_axis: Secondary axis for dual-axis broadcast (optional)
            num_links: Number of links to use (default 1)
        Returns:
            Output tensor with broadcast data on all devices
        """

        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        sender_row = sender_coord[0]
        sender_col = sender_coord[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)

        out_ready_semaphore = semaphores[0]
        barrier_semaphore = semaphores[1]
        secondary_sync_semaphore = semaphores[2]

        out_ready_sem_addr = ttnn.get_global_semaphore_address(out_ready_semaphore)
        barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_semaphore)
        secondary_sync_sem_addr = ttnn.get_global_semaphore_address(secondary_sync_semaphore)

        # Calculate packet size and page info
        packet_size_bytes = 14336  # 14 KB packets for (1, 7168) input

        # Get tile info from input tensor
        input_tensor_sample = input_tensors_per_device[0]
        tile = input_tensor_sample.tile
        tile_height, tile_width = tile.tile_shape

        dtype = input_tensor_sample.dtype
        element_size = 2
        page_size_bytes = tile_height * tile_width * element_size

        # Get shard shape to calculate number of pages
        shard_spec = input_tensor_sample.memory_config().shard_spec
        shard_width = shard_spec.shape[1]
        input_num_pages = shard_width // tile_width
        num_pages_per_packet = packet_size_bytes // page_size_bytes

        # CB index
        src0_cb_index = 0

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # Kernel paths
        # Use unified kernel for both reader and writer roles
        ccl_kernel_path = "models/demos/deepseek_v3_b1/micro_ops/ccl_broadcast/kernels/ccl_broadcast_kernel.cpp"

        # For each device in the mesh, create appropriate program
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                is_sender = (row == sender_row) and (col == sender_col)
                is_secondary_sender = secondary_cluster_axis is not None and (row == sender_row) and (col != sender_col)
                is_receiver = not is_sender and not is_secondary_sender

                # Get the device's input and output tensors
                device_idx = row * mesh_cols + col
                input_tensor_device = input_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]

                # Worker core is the data core (from shard grid)
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)
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
                has_secondary_target = is_sender and (mesh_cols > 1) and (secondary_cluster_axis is not None)

                # Calculate mcast distances
                start_distance_forward = 1 if num_targets_forward > 0 else 0
                range_hops_forward = num_targets_forward
                start_distance_backward = 1 if num_targets_backward > 0 else 0
                range_hops_backward = num_targets_backward
                num_pages_to_read = input_num_pages

                # Reader named compile-time args
                reader_named_compile_time_args = [
                    ("cb0_id", src0_cb_index),
                    ("num_pages_to_read", num_pages_to_read),
                    ("is_sender", 1 if is_sender else 0),
                ]

                # Writer named compile-time args
                writer_named_compile_time_args = [
                    ("cb0_id", src0_cb_index),
                    ("num_pages_to_read", num_pages_to_read),
                    ("tensor0_page_size", page_size_bytes),
                    ("num_targets_forward_direction", num_targets_forward),
                    ("num_targets_backward_direction", num_targets_backward),
                    ("is_sender", 1 if is_sender else 0),
                    ("core_noc_x", core_noc_x),
                    ("core_noc_y", core_noc_y),
                    ("is_secondary_sender", 1 if is_secondary_sender else 0),
                    ("has_secondary_target", 1 if has_secondary_target else 0),
                    ("start_distance_in_hops_forward", start_distance_forward),
                    ("range_hops_forward", range_hops_forward),
                    ("start_distance_in_hops_backward", start_distance_backward),
                    ("range_hops_backward", range_hops_backward),
                ]

                union_named_compile_time_args = []
                _seen_ct = set()
                for name, val in reader_named_compile_time_args + writer_named_compile_time_args:
                    if name not in _seen_ct:
                        _seen_ct.add(name)
                        union_named_compile_time_args.append((name, val))

                # Determine fabric connections
                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                dst_nodes = []

                # Primary axis connections (forward and backward in column)
                if num_targets_forward > 0:
                    forward_coord = ttnn.MeshCoordinate(row + 1, col)
                    dst_nodes.append(mesh_device.get_fabric_node_id(forward_coord))

                if num_targets_backward > 0:
                    backward_coord = ttnn.MeshCoordinate(row - 1, col)
                    dst_nodes.append(mesh_device.get_fabric_node_id(backward_coord))

                # Secondary axis connection (for sender to secondary sender)
                if has_secondary_target:
                    secondary_coord = ttnn.MeshCoordinate(row, 1)  # Other column
                    dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))

                num_connections = len(dst_nodes)

                # Writer runtime args - moved to common args since CCL only uses one core
                wait_output_semaphore = is_secondary_sender or is_receiver
                reset_global_semaphore = is_secondary_sender or is_receiver
                out_ready_sem_wait_value = 1 * num_links

                writer_common_rt_args = [
                    int(output_tensor_device.buffer_address()),  # tensor_address0
                    int(out_ready_sem_addr),  # out_ready_sem_bank_addr
                    int(wait_output_semaphore),  # wait_output_semaphore
                    int(reset_global_semaphore),  # reset_global_semaphore
                    core_noc_x,  # out_ready_sem_noc0_x (drain_sync_core)
                    core_noc_y,  # out_ready_sem_noc0_y
                    out_ready_sem_wait_value,  # out_ready_sem_wait_value
                    int(barrier_sem_addr),  # barrier_sem
                    core_noc_x,  # barrier_sem_noc0_x
                    core_noc_y,  # barrier_sem_noc0_y
                    ring_index,
                    int(secondary_sync_sem_addr),  # secondary_sync_sem
                    num_connections,  # num_connections (computed from len(dst_nodes))
                ]

                # Create CB config
                cb_desc = ttnn.cb_descriptor_from_sharded_tensor(src0_cb_index, input_tensor_device)
                # Create unified kernel descriptor for CCL broadcast
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source=ccl_kernel_path,
                    core_ranges=worker_core_set,
                    ncrisc_named_compile_time_args=union_named_compile_time_args,
                    brisc_named_compile_time_args=union_named_compile_time_args,
                    ncrisc_common_runtime_args=writer_common_rt_args,
                    # Per-core runtime args: empty for NCRISC (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],  # Fabric args appended after program creation
                    ),
                )

                # Create program descriptor (only reader and writer, no compute)
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels[:2],
                    semaphores=[],
                    cbs=[cb_desc],
                )

                # Append fabric connection args to NCRISC kernel if needed
                # Runtime args are already initialized by UnifiedKernelDescriptor via per_core_runtime_args_descriptors
                if num_connections > 0:
                    writer_rt_args_ref = program.kernels[0].runtime_args[worker_core.x][worker_core.y]
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id,
                        dst_nodes,
                        [0],
                        program,
                        0,  # kernel_idx (writer kernel)
                        worker_core,
                    )
                    writer_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op([input_tensor_mesh, output_tensor], mesh_program_descriptor)

        return result
