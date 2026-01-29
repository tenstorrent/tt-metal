# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_bfloat16_packed, float_to_uint32


class BroadcastRMSNorm:
    """
    Fused Broadcast + RMSNorm operation using ttnn.generic_op.
    NCRISC: ccl_broadcast reader kernel
    BRISC: ccl_broadcast writer + RMSNorm reader
    TRISC: RMSNorm compute
    """

    @staticmethod
    def golden(input_tensor, gamma_tensor, epsilon=1e-6):
        """
        PyTorch reference implementation of RMS norm for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            gamma_tensor: Gamma/weight tensor (torch.Tensor)
            epsilon: Small value to avoid division by zero

        Returns:
            Output tensor with RMS norm applied
        """
        variance = input_tensor.pow(2).mean(-1, keepdim=True)
        normalized = input_tensor * torch.rsqrt(variance + epsilon)
        return normalized * gamma_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores,
        cluster_axis=0,
        secondary_cluster_axis=None,
        topology=None,
        num_links=1,
        using_persistent_buffers=True,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        rsqrt_fast_approx=False,
    ):
        if topology is None:
            topology = ttnn.Topology.Linear

        sender_row = sender_coord[0]
        sender_col = sender_coord[1]

        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)

        # Create global semaphores
        grid = mesh_device.compute_with_storage_grid_size()

        out_ready_semaphore = semaphores[0]
        barrier_semaphore = semaphores[1]
        secondary_sync_semaphore = semaphores[2]

        out_ready_sem_addr = ttnn.get_global_semaphore_address(out_ready_semaphore)
        barrier_sem_addr = ttnn.get_global_semaphore_address(barrier_semaphore)
        secondary_sync_sem_addr = ttnn.get_global_semaphore_address(secondary_sync_semaphore)

        # Calculate packet size and page info
        packet_size_bytes = 14336  # 14 KB packets for (1, 7168) input

        # Get tile info from input tensor (use a sample device tensor)
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

        # CB indices for rms norm
        input_cb = 0
        scalars_cb = 1
        interm_cb = 2
        gamma_cb = 3
        output_cb = 4
        # CB index for broadcast
        src0_cb_index = 5

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # for rms norm: interpret tile sizes
        input_shape = input_tensor_sample.shape
        data_format = dtype
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        num_faces = interpreted_tile.num_faces
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Number of elements
        numel = input_tensor_mesh.logical_volume()

        # Worker core for rms is same as broadcast per-shard mapping (we will compute per-device below)

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_bfloat16_packed(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

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
                has_reverse_secondary_connection = is_secondary_sender

                # Calculate mcast distances
                start_distance_forward = 1 if num_targets_forward > 0 else 0
                range_hops_forward = num_targets_forward
                start_distance_backward = 1 if num_targets_backward > 0 else 0
                range_hops_backward = num_targets_backward

                # Named compile-time args for NCRISC (reader/broadcast)
                ncrisc_named_compile_time_args = [
                    ("cb0_id", src0_cb_index),
                    ("packet_size_in_pages", num_pages_per_packet),
                    ("tensor0_page_size", page_size_bytes),
                    ("is_sender", int(is_sender)),
                    ("core_noc_x", core_noc_x),
                    ("core_noc_y", core_noc_y),
                    ("is_secondary_sender", int(is_secondary_sender)),
                    ("is_active_broadcaster", int(is_sender or is_secondary_sender)),
                ]

                # Named compile-time args for BRISC (writer + rms reader)
                brisc_named_compile_time_args = [
                    ("cb0_id", src0_cb_index),
                    ("packet_size_in_pages", num_pages_per_packet),
                    ("tensor0_page_size", page_size_bytes),
                    ("num_targets_forward_direction", num_targets_forward),
                    ("num_targets_backward_direction", num_targets_backward),
                    ("is_sender", int(is_sender)),
                    ("core_noc_x", core_noc_x),
                    ("core_noc_y", core_noc_y),
                    ("is_secondary_sender", int(is_secondary_sender)),
                    ("has_secondary_target", int(has_secondary_target)),
                    ("has_reverse_secondary_connection", int(has_reverse_secondary_connection)),
                    ("start_distance_in_hops_forward", start_distance_forward),
                    ("range_hops_forward", range_hops_forward),
                    ("start_distance_in_hops_backward", start_distance_backward),
                    ("range_hops_backward", range_hops_backward),
                    ("using_persistent_buffers", 1 if using_persistent_buffers else 0),
                    # RMSNorm reader specific CTArgs
                    ("rmsnorm_num_faces", num_faces),
                    ("rmsnorm_scalars_cb", scalars_cb),
                ]

                # Ensure BRISC named compile-time args include rmsnorm keys required by kernel
                required_brisc_ctags = {"rmsnorm_num_faces", "rmsnorm_scalars_cb"}
                provided_brisc_ctags = {name for name, _ in brisc_named_compile_time_args}
                missing = required_brisc_ctags - provided_brisc_ctags
                if missing:
                    raise RuntimeError(f"Missing BRISC named compile-time args required by kernel: {missing}")

                # Named compile-time args for TRISC (compute)
                trisc_named_compile_time_args = [
                    ("rmsnorm_input_cb", input_cb),
                    ("rmsnorm_scalars_cb", scalars_cb),
                    ("rmsnorm_interm_cb", interm_cb),
                    ("rmsnorm_gamma_cb", gamma_cb),
                    ("rmsnorm_output_cb", output_cb),
                    ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
                    ("rmsnorm_num_tiles", num_tiles),
                    ("rmsnorm_rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
                ]

                # Reader runtime args
                reader_rt_args = ttnn.RuntimeArgs()
                # Insert placeholder at index 0 so kernel get_arg_val(1..3) map correctly
                reader_rt_args[worker_core.x][worker_core.y] = [
                    0,  # placeholder (kernel reads tensor at index 1)
                    int(input_tensor_device.buffer_address()),  # tensor_address0 (32-bit)
                    0,  # tile_id_start
                    input_num_pages,  # tile_id_end
                ]

                # Writer runtime args
                wait_output_semaphore = is_secondary_sender or is_receiver
                reset_global_semaphore = is_secondary_sender or is_receiver
                out_ready_sem_wait_value = 1 * num_links

                writer_rt_args = ttnn.RuntimeArgs()
                writer_rt_args[worker_core.x][worker_core.y] = [
                    int(output_tensor_device.buffer_address()),  # tensor_address0 (32-bit)
                    int(out_ready_sem_addr),  # out_ready_sem_bank_addr (32-bit)
                    0,  # tile_id_start
                    input_num_pages,  # tile_id_end
                    int(wait_output_semaphore),  # wait_output_semaphore
                    int(reset_global_semaphore),  # reset_global_semaphore
                    core_noc_x,  # out_ready_sem_noc0_x (drain_sync_core)
                    core_noc_y,  # out_ready_sem_noc0_y
                    out_ready_sem_wait_value,  # out_ready_sem_wait_value
                    int(barrier_sem_addr),  # barrier_sem (32-bit)
                    core_noc_x,  # barrier_sem_noc0_x
                    core_noc_y,  # barrier_sem_noc0_y
                    ring_index,
                    int(secondary_sync_sem_addr),  # secondary_sync_sem (32-bit)
                ]

                # Append writer-side fields that the BRISC kernel expects at indices 14 and 15:
                # index 14: num_connections
                # index 15: rmsnorm scalar (packed bfloat16)
                # Make sure these are present before we assign runtime_args to program.kernels
                # (Note: num_connections is computed after dst_nodes is populated below)

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

                # Reverse secondary connection (for secondary sender back to sender)
                if has_reverse_secondary_connection:
                    sender_coord_back = ttnn.MeshCoordinate(sender_row, sender_col)
                    dst_nodes.append(mesh_device.get_fabric_node_id(sender_coord_back))

                num_connections = len(dst_nodes)

                # Append writer-side fields that the BRISC kernel expects at indices 14 and 15:
                # index 14: num_connections
                # index 15: rmsnorm scalar (packed bfloat16)
                # Make sure these are present before we assign runtime_args to program.kernels
                writer_rt_args[worker_core.x][worker_core.y].append(int(num_connections))
                writer_rt_args[worker_core.x][worker_core.y].append(int(scalar_packed))

                # Validate that BRISC kernel will receive the expected RTArgs length (0..15)
                per_core_writer_list = writer_rt_args[worker_core.x][worker_core.y]
                if len(per_core_writer_list) < 16:
                    raise RuntimeError(
                        f"BRISC runtime_args too short for core {worker_core}: expected >=16 entries, got {len(per_core_writer_list)}"
                    )

                # Create tile descriptor for proper tile dimensions
                tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

                # Create circular buffer descriptors
                # CB 0: Input (created from sharded tensor mesh)
                in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor_mesh)
                in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 1: Scalars (epsilon and reduction scalar)
                scalars_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=scalars_cb,
                    data_format=data_format,
                    page_size=cb_page_size,
                    tile=tile_descriptor,
                )
                scalars_cb_descriptor = ttnn.CBDescriptor(
                    total_size=cb_page_size,
                    core_ranges=worker_core_set,
                    format_descriptors=[scalars_cb_format],
                )

                # CB 2: Intermediate buffer
                interm_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=interm_cb,
                    data_format=data_format,
                    page_size=cb_page_size,
                    tile=tile_descriptor,
                )
                interm_cb_descriptor = ttnn.CBDescriptor(
                    total_size=num_tiles * cb_page_size,
                    core_ranges=worker_core_set,
                    format_descriptors=[interm_cb_format],
                )

                # CB 3: Gamma (created from sharded gamma tensor)
                gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor)
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 4: Output (created from sharded tensor)
                out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
                out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                out_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # Create CB config for broadcast
                cb_config = ttnn.CBFormatDescriptor(
                    buffer_index=src0_cb_index,
                    data_format=dtype,
                    page_size=page_size_bytes,
                )
                cb_desc = ttnn.CBDescriptor(
                    total_size=num_pages_per_packet * page_size_bytes,
                    core_ranges=worker_core_set,
                    format_descriptors=[cb_config],
                )

                # Unified kernel descriptor for fused op
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/broadcast_rms/kernels/broadcast_rms_kernel.cpp",
                    core_ranges=worker_core_set,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    # Provide a placeholder scalar slot at index 0 for NCRISC so
                    # the kernel's get_arg_val(1..3) tensor indices align with
                    # the host's reader_rt_args (tensor_address at index 1).
                    ncrisc_common_runtime_args=[0],
                    trisc_common_runtime_args=[epsilon_packed],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_active_core",
                            core_range=worker_core_set,
                            value=1,
                            other_value=0,
                        ),
                    ],
                )

                # Program descriptor
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors(),
                    cbs=[
                        in_cb_descriptor,
                        scalars_cb_descriptor,
                        interm_cb_descriptor,
                        gamma_cb_descriptor,
                        out_cb_descriptor,
                        cb_desc,
                    ],
                    semaphores=[],
                )

                # Ensure per-kernel runtime args are set for reader/writer kernels before any append operations
                # kernels ordering: [ncrisc_reader, brisc_writer, trisc_compute]
                if len(program.kernels) >= 1:
                    program.kernels[0].runtime_args = reader_rt_args
                if len(program.kernels) >= 2:
                    program.kernels[1].runtime_args = writer_rt_args

                # Explicitly set TRISC runtime args (epsilon at index 0) to match kernel get_arg_val(0)
                if len(program.kernels) >= 3:
                    compute_rt_args = ttnn.RuntimeArgs()
                    compute_rt_args[worker_core.x][worker_core.y] = [int(epsilon_packed)]
                    program.kernels[2].runtime_args = compute_rt_args

                # Append fabric connection args to writer kernel if there are connections
                if num_connections > 0:
                    # writer kernel is index 1 in the unified kernel descriptor list
                    writer_rt_args_ref = program.kernels[1].runtime_args[worker_core.x][worker_core.y]
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id, dst_nodes, [0], program, 1, worker_core  # kernel_idx (writer kernel)
                    )
                    writer_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op([input_tensor_mesh, gamma_tensor, output_tensor], mesh_program_descriptor)

        return result
