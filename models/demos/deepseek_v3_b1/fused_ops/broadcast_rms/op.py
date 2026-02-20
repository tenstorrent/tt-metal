# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreRuntimeArgsDescriptor, UnifiedKernelDescriptor
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class BroadcastRMSNorm:
    """
    Fused Broadcast + RMSNorm operation using ttnn.generic_op.
    NCRISC: ccl_broadcast reader kernel + RMSNorm reader (or just RMSNorm if skip_ccl)
    BRISC: ccl_broadcast writer (or no-op if skip_ccl)
    TRISC: RMSNorm compute

    When skip_ccl=True, the operation runs on a single device without CCL broadcast,
    effectively performing only the RMSNorm computation.
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
        intermediate_tensor_mesh,
        gamma_tensor,
        sender_coord,
        output_tensor,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        rsqrt_fast_approx=False,
        skip_ccl=False,
    ):
        """
        Execute fused Broadcast+RMSNorm operation.

        Args:
            skip_ccl: If True, skip CCL broadcast and run RMSNorm only (single-device mode).
                      In this mode, semaphores and sender_coord are ignored.
        """

        sender_row = sender_coord[0]
        sender_col = sender_coord[1]

        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor_mesh)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)

        # Semaphore addresses (only needed for CCL mode)
        out_ready_sem_addr = 0
        barrier_sem_addr = 0
        secondary_sync_sem_addr = 0
        if not skip_ccl and semaphores is not None:
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
        tile_id_start = 0

        # bcast cb info
        input_shape = input_tensor_sample.shape
        page_size_bytes = 32 * 32 * element_size  # interpret it as 32x32 tile to use the same cb as rmsnorm
        input_num_pages = input_shape[0] * input_shape[1] * element_size // page_size_bytes
        num_pages_per_packet = packet_size_bytes // page_size_bytes

        # CB indices for rms norm
        input_cb = 0
        pkt_cb = 1
        gamma_cb = 2
        output_cb = 3

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # for rms norm: interpret tile sizes
        data_format = dtype
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Number of elements
        numel = input_tensor_mesh.logical_volume()

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_uint32(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

        # For each device in the mesh, create appropriate program
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)

                # CCL role calculation (only matters if not skipping CCL)
                if skip_ccl:
                    is_sender = False
                    is_secondary_sender = False
                    is_receiver = False
                else:
                    is_sender = (row == sender_row) and (col == sender_col)
                    is_secondary_sender = (
                        secondary_cluster_axis is not None and (row == sender_row) and (col != sender_col)
                    )
                    is_receiver = not is_sender and not is_secondary_sender

                # Get the device's input and output tensors
                device_idx = row * mesh_cols + col
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]

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

                rmsnorm_input_source_cb = input_cb
                num_pages_to_read = input_num_pages

                ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    # CCL broadcast writer args (dummy values when skip_ccl)
                    ("cb0_id", pkt_cb if not skip_ccl else 0),
                    ("num_pages_to_read", num_pages_to_read if not skip_ccl else 0),
                    ("tensor0_page_size", page_size_bytes if not skip_ccl else 0),
                    ("num_targets_forward_direction", num_targets_forward if not skip_ccl else 0),
                    ("num_targets_backward_direction", num_targets_backward if not skip_ccl else 0),
                    ("is_sender", int(is_sender) if not skip_ccl else 0),
                    ("core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("has_secondary_target", int(has_secondary_target) if not skip_ccl else 0),
                    ("start_distance_in_hops_forward", start_distance_forward if not skip_ccl else 0),
                    ("range_hops_forward", range_hops_forward if not skip_ccl else 0),
                    ("start_distance_in_hops_backward", start_distance_backward if not skip_ccl else 0),
                    ("range_hops_backward", range_hops_backward if not skip_ccl else 0),
                    ("rmsnorm_input_cb", rmsnorm_input_source_cb),
                    ("rmsnorm_num_tiles", num_tiles),
                    ("intermediate_cb", input_cb if not skip_ccl else pkt_cb),
                    ("gamma_cb", gamma_cb),
                ]

                brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("cb0_id", pkt_cb if not skip_ccl else 0),
                    ("num_pages_to_read", num_pages_to_read if not skip_ccl else 0),
                    ("is_sender", int(is_sender) if not skip_ccl else 0),
                ]

                # Named compile-time args for TRISC (rmsnorm compute)
                trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("rmsnorm_input_cb", rmsnorm_input_source_cb),
                    ("rmsnorm_gamma_cb", gamma_cb),
                    ("rmsnorm_output_cb", output_cb),
                    ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
                    ("rmsnorm_num_tiles", num_tiles),
                    ("rmsnorm_rsqrt_fast_approx", 1 if rsqrt_fast_approx else 0),
                ]

                fabric_node_id = None
                dst_nodes = []
                num_connections = 0
                if not skip_ccl:
                    fabric_node_id = mesh_device.get_fabric_node_id(coord)

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

                # Common runtime args for writer (broadcast args shared across cores)
                writer_common_rt_args = []
                if not skip_ccl:
                    # Multi-device mode: CCL writer args
                    wait_output_semaphore = is_secondary_sender or is_receiver
                    reset_global_semaphore = is_secondary_sender or is_receiver
                    out_ready_sem_wait_value = 1 * num_links

                    writer_common_rt_args = [
                        int(intermediate_tensor_device.buffer_address()),  # tensor_address0
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
                        int(num_connections),  # num_connections
                    ]

                # Create tile descriptor for proper tile dimensions
                tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

                # Create circular buffer descriptors
                # CB 0: In multi-device mode, backed by intermediate_tensor_mesh (broadcast destination)
                #       In single-device mode, backed by input_tensor_mesh (direct input)
                cb0_backing_tensor = input_tensor_mesh if skip_ccl else intermediate_tensor_mesh
                in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, cb0_backing_tensor)
                in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 2: ccl buffer
                pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(pkt_cb, input_tensor_mesh)

                # CB 3: Gamma (created from sharded gamma tensor)
                gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor)
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB 4: Output (created from sharded tensor)
                out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
                out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                out_cb_descriptor.format_descriptors[0].page_size = cb_page_size
                kernel_defines = [("SKIP_CCL", "1")] if skip_ccl else []

                # Unified kernel descriptor for fused op
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/broadcast_rms/kernels/broadcast_rms_kernel.cpp",
                    core_ranges=worker_core_set,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    ncrisc_common_runtime_args=writer_common_rt_args,
                    trisc_common_runtime_args=[epsilon_packed, scalar_packed],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    defines=kernel_defines,
                    # Per-core runtime args: empty for BRISC (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],  # Fabric args appended after program creation
                    ),
                )

                # Program descriptor
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=[
                        in_cb_descriptor,
                        pkt_cb_descriptor,
                        gamma_cb_descriptor,
                        out_cb_descriptor,
                    ],
                    semaphores=[],
                )

                # Append fabric connection args to NCRISC kernel if needed (CCL mode only)
                # Runtime args are already initialized by UnifiedKernelDescriptor via per_core_runtime_args_descriptors
                if not skip_ccl and num_connections > 0:
                    # NCRISC writer kernel is index 0 in the unified kernel descriptor list
                    writer_rt_args_ref = program.kernels[0].runtime_args[worker_core.x][worker_core.y]
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id, dst_nodes, [0], program, 0, worker_core
                    )
                    writer_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        result = ttnn.generic_op(
            [input_tensor_mesh, intermediate_tensor_mesh, gamma_tensor, output_tensor], mesh_program_descriptor
        )

        return result
