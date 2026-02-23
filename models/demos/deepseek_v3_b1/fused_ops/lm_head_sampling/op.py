# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LM Head Sampling: CCL Broadcast + Mcast + Matmul fused operation for LM head vocab projection.

In multi-device mode, CCL broadcasts input_tensor from the sender device to all devices,
then on each device multicasts from the sender core to all cores in the device grid,
then each matmul core computes a local matmul with its vocab weight shard.

In single-device mode (skip_ccl=True), the CCL broadcast is skipped and the
input_tensor is used directly.

- input_tensor (in0): [1, K] on sender core
- vocab_tensor (in1): [K, N_total] width-sharded across matmul cores as [K, N_per_core]
- output: [1, N_total] width-sharded across matmul cores as [1, N_per_core]

CB Layout:
- CB 0:  mcast_src (input_tensor on sender core, tensor-backed)
- CB 1:  mcast_dst / matmul_in0 (all device grid cores, intermediate)
- CB 2:  matmul_in1 (vocab weights on matmul cores, tensor-backed)
- CB 16: matmul_out (output on matmul cores, tensor-backed)
- CB 30: bcast_pkt (CCL broadcast packet buffer, only in multi-device mode)
"""


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class LMHeadSampling:
    """
    LM head sampling vocab projection: CCL broadcast + mcast + matmul via ttnn.generic_op.

    In multi-device mode, CCL broadcasts input_tensor from the sender device to all
    devices, then on each device the sender core multicasts to all device cores, and
    each matmul core computes [1, K] x [K, N_per_core] with its width-sharded vocab
    weight shard.
    """

    @staticmethod
    def golden(input_tensor, vocab_tensor):
        """
        PyTorch reference implementation of matmul for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [M, K]
            vocab_tensor: Vocab tensor (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N]
        """
        return input_tensor @ vocab_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        vocab_tensor,
        output_tensor,
        sender_coord,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
        fp32_dest_acc_en=False,
        skip_ccl=False,
    ):
        """
        Execute LM head sampling CCL broadcast + mcast + matmul operation using generic_op.

        In multi-device mode, CCL broadcasts input_tensor from the sender device to all
        devices via the fabric, then on each device the sender core multicasts to all
        device cores, and each matmul core computes a local matmul with its vocab weight
        shard.

        Args:
            input_tensor_mesh: Input tensor mesh [1, K] height-sharded in L1 on a single sender core
            intermediate_tensor_mesh: Intermediate mesh tensor for CCL broadcast destination
            vocab_tensor: Vocab weights [K, N_total] width-sharded across matmul cores as [K, N_per_core]
            output_tensor: Pre-allocated output [1, N_total] width-sharded across matmul cores
            sender_coord: Tuple (row, col) of sender device in mesh
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            num_links: Number of fabric links for CCL
            fp32_dest_acc_en: Whether to enable FP32 accumulation
            skip_ccl: Whether to skip CCL broadcast (single-device mode)
        Returns:
            Output tensor with matmul result
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
        vocab_tensors_per_device = ttnn.get_device_tensors(vocab_tensor)
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

        # Calculate packet size and page info for CCL broadcast
        packet_size_bytes = 14336  # 14 KB packets for (1, 7168) input

        # Get tile info from input tensor (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        in0_tile = input_tensor_sample.get_tile()

        input_shape = input_tensor_sample.shape
        data_format = input_tensor_sample.dtype
        element_size = 2
        tile_id_start = 0

        # CCL broadcast page info
        bcast_page_size_bytes = 32 * 32 * element_size  # interpret as 32x32 tile
        bcast_num_pages = input_shape[0] * input_shape[1] * element_size // bcast_page_size_bytes
        num_pages_per_packet = packet_size_bytes // bcast_page_size_bytes

        # Matmul shape info from input and vocab tensors
        num_tiles_k = input_shape[1] // in0_tile.tile_shape[1]

        # Get output tile info
        output_tensor_sample = output_tensors_per_device[0]
        out_tile = output_tensor_sample.get_tile()

        # Get vocab weights info (per-core output width)
        vocab_tensor_sample = vocab_tensors_per_device[0]
        weights_shard_spec = vocab_tensor_sample.memory_config().shard_spec
        n_per_core = weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // out_tile.tile_shape[1]

        # Input tile size for mcast data
        input_tile_size = in0_tile.get_tile_size(data_format)
        mcast_data_size_bytes = num_tiles_k * input_tile_size

        # ====================================================================
        # CB indices
        # ====================================================================
        mcast_src_cb = 0  # input_tensor on sender core (tensor-backed)
        mcast_dst_cb = 1  # Mcast destination = matmul in0 (all mcast grid cores, intermediate)
        matmul_in1_cb = 2  # vocab_tensor weights on matmul cores (tensor-backed)
        matmul_out_cb = 16  # Matmul output on matmul cores (tensor-backed)

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 30  # Packet buffer for CCL broadcast

        # ====================================================================
        # Semaphore IDs (for intra-device mcast)
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

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

                # Get per-device tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                vocab_tensor_device = vocab_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]

                # Get device handle
                device = input_tensor_device.device()

                # ================================================================
                # CCL broadcast: physical core and routing info
                # ================================================================
                # Worker core from input tensor shard grid (single core)
                input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                shard_grid_start = input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(shard_grid_start.x, shard_grid_start.y)

                # Get physical core for NOC addressing
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

                # ================================================================
                # Core grid configuration (per-device)
                # ================================================================
                # Sender core: from input_tensor (must be single core)
                mcast_sender_core_grid = input_tensor_device.memory_config().shard_spec.grid
                assert mcast_sender_core_grid.num_cores() == 1, "input_tensor must be sharded on a single sender core"
                mcast_sender_core = list(mcast_sender_core_grid.ranges())[0].start

                # Matmul cores: from vocab_tensor (multiple cores with weight shards)
                matmul_core_grid = vocab_tensor_device.memory_config().shard_spec.grid

                # Mcast grid = full device compute grid (includes sender + all matmul cores)
                device_grid_size = device.compute_with_storage_grid_size()
                mcast_grid = ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1)
                )

                mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
                num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y

                # Build mcast receiver grid = mcast grid minus sender core
                mcast_receiver_ranges = []
                for r in range(mcast_grid.start.y, mcast_grid.end.y + 1):
                    for c in range(mcast_grid.start.x, mcast_grid.end.x + 1):
                        if c == mcast_sender_core.x and r == mcast_sender_core.y:
                            continue
                        mcast_receiver_ranges.append(ttnn.CoreRange(ttnn.CoreCoord(c, r), ttnn.CoreCoord(c, r)))
                mcast_receiver_grid = ttnn.CoreRangeSet(mcast_receiver_ranges)

                # All cores = mcast grid (sender is already included)
                all_cores = mcast_grid_set

                # Determine if sender is part of the mcast rectangle
                is_part_of_receiver_grid = mcast_grid.contains(mcast_sender_core)

                # Get NOC coordinates for mcast destination
                mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
                mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)
                bcast_num_pages_to_read = bcast_num_pages

                # ================================================================
                # NCRISC compile-time args
                # ================================================================
                ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_num_pages_to_read", bcast_num_pages_to_read if not skip_ccl else 0),
                    ("bcast_tensor0_page_size", bcast_page_size_bytes if not skip_ccl else 0),
                    ("bcast_num_targets_forward_direction", num_targets_forward if not skip_ccl else 0),
                    ("bcast_num_targets_backward_direction", num_targets_backward if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    ("bcast_core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("bcast_core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("bcast_is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("bcast_has_secondary_target", int(has_secondary_target) if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_forward", start_distance_forward if not skip_ccl else 0),
                    ("bcast_range_hops_forward", range_hops_forward if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_backward", start_distance_backward if not skip_ccl else 0),
                    ("bcast_range_hops_backward", range_hops_backward if not skip_ccl else 0),
                    # Mcast source (for setup_sharded_buffer on sender core)
                    ("mcast_src_cb", mcast_src_cb),
                    ("mcast_src_num_pages", num_tiles_k),
                    # Mcast receiver
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_dst_cb", mcast_dst_cb),
                    ("mcast_dst_num_pages", num_tiles_k),
                    # Matmul
                    ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
                    ("matmul_in1", matmul_in1_cb),
                    ("matmul_k_num_tiles", num_tiles_k),
                    ("matmul_out_w", out_w_per_core),
                ]

                # ================================================================
                # BRISC compile-time args
                # ================================================================
                brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_num_pages_to_read", bcast_num_pages_to_read if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    # Mcast sender
                    ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
                    ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
                    ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
                    ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
                    ("mcast_num_cores", num_mcast_cores),
                    ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_data_size_bytes", mcast_data_size_bytes),
                    ("mcast_src_cb", mcast_src_cb),
                    ("mcast_src_num_pages", num_tiles_k),
                    ("mcast_dst_cb", mcast_dst_cb),
                    ("mcast_is_part_of_receiver_grid", is_part_of_receiver_grid),
                ]

                # ================================================================
                # TRISC compile-time args
                # ================================================================
                trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("matmul_in0", mcast_dst_cb),  # Matmul reads from mcast destination
                    ("matmul_in1", matmul_in1_cb),
                    ("matmul_out", matmul_out_cb),
                    ("matmul_k_num_tiles", num_tiles_k),
                    ("matmul_out_w", out_w_per_core),
                ]

                # ================================================================
                # CCL Broadcast common runtime args
                # ================================================================
                if skip_ccl:
                    ncrisc_bcast_common_args = []
                    dst_nodes = []
                    fabric_node_id = None
                else:
                    wait_output_semaphore = is_secondary_sender or is_receiver
                    reset_global_semaphore = is_secondary_sender or is_receiver
                    out_ready_sem_wait_value = 1 * num_links

                    # Build dst_nodes to compute num_connections = len(dst_nodes)
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
                        secondary_coord = ttnn.MeshCoordinate(row, 1)
                        dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))

                    num_connections = len(dst_nodes)

                    ncrisc_bcast_common_args = [
                        int(intermediate_tensor_device.buffer_address()),  # tensor_address0
                        int(out_ready_sem_addr),  # out_ready_sem_bank_addr
                        int(wait_output_semaphore),  # wait_output_semaphore
                        int(reset_global_semaphore),  # reset_global_semaphore
                        core_noc_x,  # out_ready_sem_noc0_x
                        core_noc_y,  # out_ready_sem_noc0_y
                        out_ready_sem_wait_value,  # out_ready_sem_wait_value
                        int(barrier_sem_addr),  # barrier_sem
                        core_noc_x,  # barrier_sem_noc0_x
                        core_noc_y,  # barrier_sem_noc0_y
                        ring_index,  # ring_index
                        int(secondary_sync_sem_addr),  # secondary_sync_sem
                        num_connections,  # num_connections
                    ]

                # ================================================================
                # Circular buffer descriptors
                # ================================================================
                # CB 0: Mcast source — In multi-device mode, backed by intermediate_tensor
                #       (where CCL broadcast placed the data). In single-device mode,
                #       backed by input_tensor directly.
                mcast_src_backing_tensor = input_tensor_device if skip_ccl else intermediate_tensor_device
                mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(mcast_src_cb, mcast_src_backing_tensor)

                # CB 1: Mcast destination — intermediate buffer on all device grid cores
                mcast_dst_tile_descriptor = ttnn.TileDescriptor(in0_tile)
                mcast_dst_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=mcast_dst_cb,
                    data_format=data_format,
                    page_size=input_tile_size,
                    tile=mcast_dst_tile_descriptor,
                )
                mcast_dst_cb_descriptor = ttnn.CBDescriptor(
                    total_size=num_tiles_k * input_tile_size,
                    core_ranges=all_cores,
                    format_descriptors=[mcast_dst_cb_format],
                )

                # CB 2: Matmul weights — vocab_tensor, tensor-backed on matmul cores
                matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, vocab_tensor_device)

                # CB 16: Matmul output — tensor-backed on matmul cores
                matmul_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_out_cb, output_tensor_device)

                # CB list
                cbs_list = [
                    mcast_src_cb_descriptor,
                    mcast_dst_cb_descriptor,
                    matmul_in1_cb_descriptor,
                    matmul_out_cb_descriptor,
                ]

                # CB 30: CCL broadcast packet buffer (only in multi-device mode)
                if not skip_ccl:
                    bcast_pkt_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bcast_pkt_cb, input_tensor_device)
                    cbs_list.append(bcast_pkt_cb_descriptor)

                # ================================================================
                # Semaphore descriptors (for intra-device mcast)
                # ================================================================
                semaphore_descriptors = [
                    ttnn.SemaphoreDescriptor(
                        id=mcast_data_sender_semaphore_id,
                        core_ranges=all_cores,
                        initial_value=0,
                    ),
                    ttnn.SemaphoreDescriptor(
                        id=mcast_data_receiver_semaphore_id,
                        core_ranges=all_cores,
                        initial_value=0,
                    ),
                ]

                # ================================================================
                # Unified kernel descriptor
                # ================================================================
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/lm_head_sampling/kernels/lm_head_sampling_kernel.cpp",
                    core_ranges=all_cores,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    ncrisc_common_runtime_args=ncrisc_bcast_common_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_input_core",
                            core_range=mcast_sender_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mcast_receiver_core",
                            core_range=mcast_receiver_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul_core",
                            core_range=matmul_core_grid,
                            value=1,
                            other_value=0,
                        ),
                    ],
                    # Per-core runtime args: empty for BRISC on worker_core (fabric args appended later)
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],
                    ),
                )

                # ================================================================
                # Program descriptor
                # ================================================================
                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors().kernels,
                    cbs=cbs_list,
                    semaphores=semaphore_descriptors,
                )

                # Append fabric connection args to BRISC kernel if needed (CCL mode only)
                if not skip_ccl and num_connections > 0:
                    for idx, kernel in enumerate(program.kernels):
                        if kernel.core_ranges.contains(worker_core) and (
                            isinstance(kernel.config, ttnn.ReaderConfigDescriptor)
                            or (
                                isinstance(kernel.config, ttnn.DataMovementConfigDescriptor)
                                and kernel.config.processor == ttnn.DataMovementProcessor.RISCV_1
                            )
                        ):
                            writer_rt_args_ref = kernel.runtime_args[worker_core.x][worker_core.y]
                            fabric_args = ttnn.setup_routing_plane_connection(
                                fabric_node_id, dst_nodes, [0], program, idx, worker_core
                            )
                            writer_rt_args_ref.extend(fabric_args)
                            break

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic op
        result = ttnn.generic_op(
            [input_tensor_mesh, intermediate_tensor_mesh, vocab_tensor, output_tensor],
            mesh_program_descriptor,
        )

        return result
