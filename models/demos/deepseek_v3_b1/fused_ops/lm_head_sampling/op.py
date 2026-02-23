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


import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    PerCoreRuntimeArgsDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _is_singleton_prefix_shape(shape, expected_last_dim: int) -> bool:
    dims = tuple(int(d) for d in shape)
    if not dims or dims[-1] != expected_last_dim:
        return False
    return all(d == 1 for d in dims[:-1])


class LMHeadSampling:
    """
    LM head sampling vocab projection: CCL broadcast + mcast + matmul via ttnn.generic_op.

    In multi-device mode, CCL broadcasts input_tensor from the sender device to all
    devices, then on each device the sender core multicasts to all device cores, and
    each matmul core computes [1, K] x [K, N_per_core] with its width-sharded vocab
    weight shard.
    """

    @staticmethod
    def golden(input_tensor, vocab_tensor, indices: torch.Tensor | None = None, k: int = 1, p: float = 1.0):
        """
        PyTorch reference implementation for fused LM-head + sampling golden.

        Args:
            input_tensor: Input tensor (torch.Tensor) [M, K]
            vocab_tensor: Vocab tensor (torch.Tensor) [K, N]
            indices: Optional indices tensor used by fused sampling. If provided,
                golden returns sampled index tensor [1, 1]. If omitted, returns scores.
            k: Sampling k; currently only k=1 supported when indices is provided.
            p: Top-p threshold (unused for k=1 path).

        Returns:
            - If indices is None: output scores tensor [M, N]
            - If indices is provided: sampled index tensor [1, 1] (uint32)
        """
        scores = input_tensor @ vocab_tensor

        if k != 1:
            raise NotImplementedError("LMHeadSampling fused golden currently supports only k=1")

        # p is intentionally unused in k=1 path; keep for API compatibility.
        _ = p

        scores_f32 = scores.float().reshape(-1)
        indices_i64 = indices.to(torch.int64).reshape(-1)
        if scores_f32.numel() != indices_i64.numel():
            raise ValueError(
                f"scores and indices must have the same number of elements, got {scores_f32.numel()} and {indices_i64.numel()}"
            )

        max_score = torch.max(scores_f32)
        tied_mask = scores_f32 == max_score
        selected_index = torch.min(indices_i64[tied_mask]).to(torch.uint32)
        return selected_index.reshape(1, 1)

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        vocab_tensor,
        output_tensor,
        sender_coord,
        indices_tensor=None,
        output_index_tensor=None,
        argmax_final_core_coord=None,
        argmax_final_mesh_coord=None,
        global_semaphore=None,
        global_stage2_semaphore=None,
        fabric_scratch_tensor=None,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=None,
        num_links=1,
        fp32_dest_acc_en=False,
        skip_ccl=None,
        socket_output=None,
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
            indices_tensor: Optional pre-cached global indices tensor, width-sharded like output scores
            output_index_tensor: Optional pre-allocated [1, 1] uint32 tensor for fused argmax output
            argmax_final_core_coord: Optional final core for fused argmax reduction (defaults to first matmul core)
            sender_coord: Tuple (row, col) of sender device in mesh
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            num_links: Number of fabric links for CCL
            fp32_dest_acc_en: Whether to enable FP32 accumulation
            skip_ccl: Whether to skip CCL broadcast. If None, defaults to True for single-device meshes.
            socket_output: Optional socket output endpoint. Supports ttnn.D2HSocket (host output) and
                ttnn.MeshSocket sender endpoint (D2D output).
        Returns:
            Output tensor with matmul result. If fused argmax is enabled, output_index_tensor is written in-place.
        """
        # LMHeadSampling is always fused with k=1 sampling (argmax fast path).
        enable_argmax = True
        socket_mode_none = 0
        socket_mode_d2h = 1
        socket_mode_d2d = 2
        socket_page_size_bytes = 64
        if socket_output is None:
            socket_mode_selected = socket_mode_none
        elif isinstance(socket_output, ttnn.D2HSocket):
            socket_mode_selected = socket_mode_d2h
        elif isinstance(socket_output, ttnn.MeshSocket):
            socket_mode_selected = socket_mode_d2d
        else:
            raise TypeError(
                f"Unsupported socket_output type for lm_head_sampling: {type(socket_output)}. "
                "Expected ttnn.D2HSocket or ttnn.MeshSocket."
            )
        enable_socket_output = socket_output is not None
        if indices_tensor is None or output_index_tensor is None:
            raise ValueError("indices_tensor and output_index_tensor are required for fused LM-head + sampling")

        # Get mesh/device info
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]
        if skip_ccl is None:
            skip_ccl = mesh_rows * mesh_cols == 1
        if enable_socket_output:
            # only for d2h sockets
            if isinstance(socket_output, ttnn.D2HSocket):
                socket_output.set_page_size(socket_page_size_bytes)
            active_socket_cores = socket_output.get_active_cores()
            if len(active_socket_cores) != 1:
                raise ValueError("socket output for lm_head_sampling must have exactly one active core")
            socket_core = active_socket_cores[0]
        # In multi-column meshes, enable secondary-axis relay by default so broadcast reaches
        # all clusters. Callers can still override explicitly if needed.
        if not skip_ccl and mesh_cols > 1 and secondary_cluster_axis is None:
            secondary_cluster_axis = 1

        sender_row = sender_coord[0]
        sender_col = sender_coord[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor_mesh)
        vocab_tensors_per_device = ttnn.get_device_tensors(vocab_tensor)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        indices_tensors_per_device = ttnn.get_device_tensors(indices_tensor) if enable_argmax else None
        output_index_tensors_per_device = ttnn.get_device_tensors(output_index_tensor) if enable_argmax else None
        scratch_tensors_per_device = (
            ttnn.get_device_tensors(fabric_scratch_tensor) if (enable_argmax and not skip_ccl) else None
        )

        if enable_argmax and not skip_ccl:
            if global_semaphore is None or global_stage2_semaphore is None or fabric_scratch_tensor is None:
                raise ValueError(
                    "global_semaphore, global_stage2_semaphore, and fabric_scratch_tensor are required for mesh argmax"
                )
            if mesh_rows < 2 or mesh_cols != 2:
                raise NotImplementedError(
                    f"Fused LM-head mesh argmax currently supports only (R,2) with R>=2, got {mesh_shape}"
                )
            if argmax_final_mesh_coord is None:
                raise ValueError("argmax_final_mesh_coord is required for mesh argmax")

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

        global_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )
        global_stage2_sem_addr = (
            int(ttnn.get_global_semaphore_address(global_stage2_semaphore)) if (enable_argmax and not skip_ccl) else 0
        )

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
        argmax_winner_cb = 3
        argmax_gather_cb = 4
        argmax_indices_cb = 5
        argmax_socket_cb = 6
        matmul_out_cb = 16  # Matmul output on matmul cores (tensor-backed)

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 30  # Packet buffer for CCL broadcast

        # ====================================================================
        # Semaphore IDs (for intra-device mcast)
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        argmax_receiver_semaphore_id = 2
        argmax_local_ready_semaphore_id = 3

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
                indices_tensor_device = indices_tensors_per_device[device_idx] if enable_argmax else None
                output_index_tensor_device = output_index_tensors_per_device[device_idx] if enable_argmax else None

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
                argmax_core_grid = matmul_core_grid
                argmax_cores_row_wise = ttnn.corerange_to_cores(argmax_core_grid, row_wise=True)

                # Mcast grid = bounding box of (matmul participants U sender core).
                # This avoids reserving the full device grid while still ensuring
                # is_input_core is inside unified kernel core_ranges.
                matmul_bbox = matmul_core_grid.bounding_box()
                mcast_grid = ttnn.CoreRange(
                    ttnn.CoreCoord(
                        min(matmul_bbox.start.x, mcast_sender_core.x),
                        min(matmul_bbox.start.y, mcast_sender_core.y),
                    ),
                    ttnn.CoreCoord(
                        max(matmul_bbox.end.x, mcast_sender_core.x),
                        max(matmul_bbox.end.y, mcast_sender_core.y),
                    ),
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

                if enable_argmax:
                    if indices_tensor_device.memory_config().shard_spec.grid != argmax_core_grid:
                        raise ValueError("indices_tensor must be width-sharded on the same core grid as LM-head scores")
                    if (
                        indices_tensor_device.memory_config().shard_spec.shape
                        != output_tensor_device.memory_config().shard_spec.shape
                    ):
                        raise ValueError("indices_tensor shard shape must match output_tensor shard shape")
                    if indices_tensor_device.dtype != ttnn.uint32:
                        raise ValueError("indices_tensor must be uint32")
                    if output_index_tensor_device.memory_config().shard_spec.grid.num_cores() != 1:
                        raise ValueError("output_index_tensor must be sharded on a single final core")
                    if not _is_singleton_prefix_shape(output_index_tensor_device.shape, 1):
                        raise ValueError(
                            "output_index_tensor must be singleton-prefix with last dim 1 (per-device logical shape (1,1))"
                        )

                    output_index_core = output_index_tensor_device.memory_config().shard_spec.grid.ranges()[0].start
                    argmax_final_core = (
                        output_index_core if argmax_final_core_coord is None else argmax_final_core_coord
                    )
                    if not any(
                        c.x == argmax_final_core.x and c.y == argmax_final_core.y for c in argmax_cores_row_wise
                    ):
                        raise ValueError("argmax_final_core_coord must be within the matmul core grid")
                    if output_index_core.x != argmax_final_core.x or output_index_core.y != argmax_final_core.y:
                        raise ValueError("output_index_tensor shard core must match argmax_final_core_coord")
                    emit_socket_on_this_device = bool(enable_socket_output)
                    argmax_num_values = n_per_core
                    argmax_num_senders = len(argmax_cores_row_wise)
                    argmax_expected_remote_incs = argmax_num_senders - 1
                    argmax_sender_idx_lookup = {(c.x, c.y): idx for idx, c in enumerate(argmax_cores_row_wise)}
                    argmax_winner_page_bytes = 16
                    argmax_mesh_mode = 0
                    argmax_stage1_sender = 0
                    argmax_stage1_receiver = 0
                    argmax_stage2_sender = 0
                    argmax_stage2_receiver = 0
                    argmax_stage1_slot_base_offset = 0
                    argmax_stage1_num_slots = 0
                    argmax_stage1_expected_remote_incs = 0
                    argmax_stage1_local_slot_offset = 0
                    argmax_stage2_slot_base_offset = 0
                    argmax_stage2_num_slots = 0
                    argmax_stage2_expected_remote_incs = 0
                    argmax_stage2_local_slot_offset = 0
                    argmax_mesh_local_send_slot_offset = 0
                    is_argmax_mesh_sender_core = False
                    sender_link_idx = 0
                    dest_coord = ttnn.MeshCoordinate(row, col)
                    per_core_brisc_runtime_args = []

                    if not skip_ccl:
                        target_row = int(argmax_final_mesh_coord[0])
                        target_col = int(argmax_final_mesh_coord[1])
                        if not (0 <= target_row < mesh_rows and 0 <= target_col < mesh_cols):
                            raise ValueError(
                                f"argmax_final_mesh_coord {argmax_final_mesh_coord} out of bounds for mesh shape {mesh_shape}"
                            )
                        emit_socket_on_this_device = bool(
                            enable_socket_output and row == target_row and col == target_col
                        )

                        def _x_axis_link_idx_for_stage1_sender(sender_row_local: int) -> int:
                            linear_distance = abs(int(sender_row_local) - target_row)
                            ring_distance = min(linear_distance, mesh_rows - linear_distance)
                            max_ring_distance = mesh_rows // 2
                            first_half_threshold = (max_ring_distance + 1) // 2
                            return 0 if ring_distance <= first_half_threshold else 1

                        argmax_mesh_mode = 1
                        argmax_stage1_slot_base_offset = 0
                        argmax_stage1_num_slots = mesh_rows
                        argmax_stage2_slot_base_offset = (
                            argmax_stage1_slot_base_offset + argmax_stage1_num_slots * argmax_winner_page_bytes
                        )
                        argmax_stage2_num_slots = mesh_cols
                        argmax_stage1_expected_remote_incs = mesh_rows - 1
                        argmax_stage2_expected_remote_incs = mesh_cols - 1
                        argmax_stage1_sender = 1 if row != target_row else 0
                        argmax_stage1_receiver = 1 if row == target_row else 0
                        argmax_stage2_sender = 1 if (row == target_row and col != target_col) else 0
                        argmax_stage2_receiver = 1 if (row == target_row and col == target_col) else 0
                        argmax_stage1_local_slot_offset = (
                            argmax_stage1_slot_base_offset + row * argmax_winner_page_bytes
                        )
                        argmax_stage2_local_slot_offset = (
                            argmax_stage2_slot_base_offset + col * argmax_winner_page_bytes
                        )
                        is_argmax_mesh_sender_core = bool(argmax_stage1_sender or argmax_stage2_sender)
                        argmax_mesh_local_send_slot_offset = (
                            argmax_stage1_local_slot_offset if argmax_stage1_sender else argmax_stage2_local_slot_offset
                        )

                        if is_argmax_mesh_sender_core:
                            if argmax_stage1_sender:
                                dest_coord = ttnn.MeshCoordinate(target_row, col)
                                send_slot_offset = argmax_stage1_slot_base_offset + row * argmax_winner_page_bytes
                                sender_dst_sem_addr = global_sem_addr
                                sender_link_idx = _x_axis_link_idx_for_stage1_sender(row)
                            else:
                                dest_coord = ttnn.MeshCoordinate(target_row, target_col)
                                send_slot_offset = argmax_stage2_slot_base_offset + col * argmax_winner_page_bytes
                                sender_dst_sem_addr = global_stage2_sem_addr
                                sender_link_idx = 0

                            dest_idx = int(dest_coord[0]) * mesh_cols + int(dest_coord[1])
                            per_core_brisc_runtime_args.append(
                                (
                                    argmax_final_core,
                                    [
                                        int(argmax_mesh_local_send_slot_offset),
                                        int(mesh_device.get_fabric_node_id(dest_coord).mesh_id),
                                        int(mesh_device.get_fabric_node_id(dest_coord).chip_id),
                                        int(scratch_tensors_per_device[dest_idx].buffer_address())
                                        + int(send_slot_offset),
                                        int(sender_dst_sem_addr),
                                    ],
                                )
                            )
                    if emit_socket_on_this_device:
                        if (
                            socket_core.device_coord != ttnn.MeshCoordinate(row, col)
                            or socket_core.core_coord.x != argmax_final_core.x
                            or socket_core.core_coord.y != argmax_final_core.y
                        ):
                            raise ValueError(
                                "socket output active core must match argmax final core and emitting mesh device for lm_head_sampling"
                            )
                    argmax_socket_mode = socket_mode_selected if emit_socket_on_this_device else socket_mode_none

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
                    ("enable_argmax", 1),
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
                    ("matmul_out", matmul_out_cb),
                    ("matmul_k_num_tiles", num_tiles_k),
                    ("matmul_out_w", out_w_per_core),
                    # Argmax sampling
                    ("argmax_num_values", argmax_num_values),
                    ("argmax_winner_page_bytes", argmax_winner_page_bytes),
                    ("argmax_num_senders", argmax_num_senders),
                    ("argmax_expected_remote_incs", argmax_expected_remote_incs),
                    ("argmax_receiver_semaphore_id", argmax_receiver_semaphore_id),
                    ("argmax_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("argmax_mesh_mode", argmax_mesh_mode),
                    ("argmax_stage1_sender", argmax_stage1_sender),
                    ("argmax_stage1_receiver", argmax_stage1_receiver),
                    ("argmax_stage2_sender", argmax_stage2_sender),
                    ("argmax_stage2_receiver", argmax_stage2_receiver),
                    ("argmax_stage1_slot_base_offset", argmax_stage1_slot_base_offset),
                    ("argmax_stage1_num_slots", argmax_stage1_num_slots),
                    ("argmax_stage1_expected_remote_incs", argmax_stage1_expected_remote_incs),
                    ("argmax_stage1_local_slot_offset", argmax_stage1_local_slot_offset),
                    ("argmax_stage2_slot_base_offset", argmax_stage2_slot_base_offset),
                    ("argmax_stage2_num_slots", argmax_stage2_num_slots),
                    ("argmax_stage2_expected_remote_incs", argmax_stage2_expected_remote_incs),
                    ("argmax_stage2_local_slot_offset", argmax_stage2_local_slot_offset),
                    ("argmax_mesh_local_send_slot_offset", argmax_mesh_local_send_slot_offset),
                    ("argmax_gather_cb", argmax_gather_cb),
                    ("argmax_indices_cb", argmax_indices_cb),
                    ("argmax_socket_mode", argmax_socket_mode),
                    ("argmax_socket_cb", argmax_socket_cb if enable_socket_output else 0),
                    ("argmax_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                ]

                # ================================================================
                # BRISC compile-time args
                # ================================================================
                brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
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
                    ("argmax_winner_page_bytes", argmax_winner_page_bytes),
                    ("argmax_local_ready_semaphore_id", argmax_local_ready_semaphore_id),
                    ("argmax_socket_mode", argmax_socket_mode),
                    ("argmax_socket_cb", argmax_socket_cb if enable_socket_output else 0),
                    ("argmax_socket_page_size_bytes", socket_page_size_bytes if enable_socket_output else 0),
                ]

                # ================================================================
                # TRISC compile-time args
                # ================================================================
                trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("enable_argmax", 1),
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
                    final_core_phys = device.worker_core_from_logical_core(argmax_final_core)
                    ncrisc_bcast_common_args = [
                        int(indices_tensor_device.buffer_address()),
                        int(output_index_tensor_device.buffer_address()),
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        0,
                        0,
                        0,
                    ]
                    brisc_bcast_common_args = [
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        0,
                        int(socket_output.get_config_buffer_address()) if enable_socket_output else 0,
                    ]
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

                    final_core_phys = device.worker_core_from_logical_core(argmax_final_core)
                    ncrisc_bcast_common_args = ncrisc_bcast_common_args + [
                        int(indices_tensor_device.buffer_address()),
                        int(output_index_tensor_device.buffer_address()),
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        int(scratch_tensors_per_device[device_idx].buffer_address()),
                        global_sem_addr,
                        global_stage2_sem_addr,
                    ]
                    brisc_bcast_common_args = [
                        int(final_core_phys.x),
                        int(final_core_phys.y),
                        int(scratch_tensors_per_device[device_idx].buffer_address()),
                        int(socket_output.get_config_buffer_address()) if enable_socket_output else 0,
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

                if enable_argmax:
                    # CB 5: Argmax indices — tensor-backed on matmul/argmax cores
                    argmax_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        argmax_indices_cb, indices_tensor_device
                    )

                # CB 16: Matmul output — tensor-backed on matmul cores
                matmul_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_out_cb, output_tensor_device)

                # CB list
                cbs_list = [
                    mcast_src_cb_descriptor,
                    mcast_dst_cb_descriptor,
                    matmul_in1_cb_descriptor,
                    matmul_out_cb_descriptor,
                ]
                if enable_argmax:
                    argmax_winner_cb_descriptor = ttnn.CBDescriptor(
                        total_size=argmax_winner_page_bytes,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=argmax_winner_cb,
                                data_format=ttnn.uint32,
                                page_size=argmax_winner_page_bytes,
                            )
                        ],
                    )
                    argmax_gather_cb_descriptor = ttnn.CBDescriptor(
                        total_size=argmax_winner_page_bytes * argmax_num_senders,
                        core_ranges=argmax_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=argmax_gather_cb,
                                data_format=ttnn.uint32,
                                page_size=argmax_winner_page_bytes,
                            )
                        ],
                    )
                    cbs_list.extend(
                        [
                            argmax_winner_cb_descriptor,
                            argmax_gather_cb_descriptor,
                            argmax_indices_cb_descriptor,
                        ]
                    )
                    if enable_socket_output:
                        argmax_socket_cb_descriptor = ttnn.CBDescriptor(
                            total_size=socket_page_size_bytes,
                            core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(argmax_final_core, argmax_final_core)]),
                            format_descriptors=[
                                ttnn.CBFormatDescriptor(
                                    buffer_index=argmax_socket_cb,
                                    data_format=ttnn.uint32,
                                    page_size=socket_page_size_bytes,
                                )
                            ],
                        )
                        cbs_list.append(argmax_socket_cb_descriptor)

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
                if enable_argmax:
                    semaphore_descriptors.extend(
                        [
                            ttnn.SemaphoreDescriptor(
                                id=argmax_receiver_semaphore_id,
                                core_ranges=argmax_core_grid,
                                initial_value=0,
                            ),
                            ttnn.SemaphoreDescriptor(
                                id=argmax_local_ready_semaphore_id,
                                core_ranges=argmax_core_grid,
                                initial_value=0,
                            ),
                        ]
                    )

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
                    brisc_common_runtime_args=brisc_bcast_common_args,
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
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_argmax_core",
                            core_range=argmax_core_grid,
                            value=1 if enable_argmax else 0,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_argmax_final_core",
                            core_range=argmax_final_core,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_argmax_mesh_sender_core",
                            core_range=argmax_final_core,
                            value=1 if is_argmax_mesh_sender_core else 0,
                            other_value=0,
                        ),
                    ],
                    per_core_compile_time_descriptors=(
                        []
                        if not enable_argmax
                        else [
                            PerCoreCompileTimeDescriptor(
                                named_compile_time_arg="argmax_sender_idx",
                                core_values=[
                                    (core, argmax_sender_idx_lookup[(core.x, core.y)]) for core in argmax_cores_row_wise
                                ],
                                other_value=0,
                            )
                        ]
                    ),
                    # Per-core runtime args: mesh argmax senders get BRISC sender metadata.
                    per_core_runtime_args_descriptor=PerCoreRuntimeArgsDescriptor(
                        ncrisc_args=[(worker_core, [])],
                        brisc_args=[(worker_core, [])] + per_core_brisc_runtime_args,
                    ),
                )

                # ================================================================
                # Program descriptor
                # ================================================================
                kernel_result = unified_kernel.get_kernel_descriptors()
                input_role_cores = set()
                mcast_receiver_role_cores = set()
                matmul_role_cores = set()
                argmax_role_cores = set()
                argmax_final_role_cores = set()
                for group in kernel_result.groups:
                    group_cores = ttnn.corerange_to_cores(group.core_range_set)
                    if group.compile_time_arg_values.get("is_input_core", 0) == 1:
                        input_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_mcast_receiver_core", 0) == 1:
                        mcast_receiver_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_matmul_core", 0) == 1:
                        matmul_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_argmax_core", 0) == 1:
                        argmax_role_cores.update((c.x, c.y) for c in group_cores)
                    if group.compile_time_arg_values.get("is_argmax_final_core", 0) == 1:
                        argmax_final_role_cores.update((c.x, c.y) for c in group_cores)

                expected_input_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(mcast_sender_core_grid)}
                if input_role_cores != expected_input_role_cores:
                    missing = sorted(expected_input_role_cores - input_role_cores)[:16]
                    extra = sorted(input_role_cores - expected_input_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_input_core core-set mismatch. missing={missing}, extra={extra}"
                    )

                expected_mcast_receiver_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(mcast_receiver_grid)}
                if mcast_receiver_role_cores != expected_mcast_receiver_role_cores:
                    missing = sorted(expected_mcast_receiver_role_cores - mcast_receiver_role_cores)[:16]
                    extra = sorted(mcast_receiver_role_cores - expected_mcast_receiver_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_mcast_receiver_core core-set mismatch. missing={missing}, extra={extra}"
                    )

                expected_matmul_role_cores = {(c.x, c.y) for c in ttnn.corerange_to_cores(matmul_core_grid)}
                if matmul_role_cores != expected_matmul_role_cores:
                    missing = sorted(expected_matmul_role_cores - matmul_role_cores)[:16]
                    extra = sorted(matmul_role_cores - expected_matmul_role_cores)[:16]
                    raise RuntimeError(
                        "Unified kernel role mapping mismatch: "
                        f"is_matmul_core core-set mismatch. missing={missing}, extra={extra}"
                    )
                if enable_argmax:
                    if argmax_role_cores != expected_matmul_role_cores:
                        missing = sorted(expected_matmul_role_cores - argmax_role_cores)[:16]
                        extra = sorted(argmax_role_cores - expected_matmul_role_cores)[:16]
                        raise RuntimeError(
                            "Unified kernel role mapping mismatch: "
                            f"is_argmax_core core-set mismatch. missing={missing}, extra={extra}"
                        )
                    if len(argmax_final_role_cores) != 1:
                        raise RuntimeError(
                            "Unified kernel role mapping mismatch: " "is_argmax_final_core must map to exactly one core"
                        )

                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=cbs_list,
                    semaphores=semaphore_descriptors,
                )

                # Append CCL routing args to the broadcast writer kernel (NCRISC in current broadcast split).
                if not skip_ccl and num_connections > 0:
                    ccl_writer_group = None
                    for group in kernel_result.groups:
                        if group.compile_time_arg_values.get("is_input_core", 0) == 1 and group.core_range_set.contains(
                            worker_core
                        ):
                            ccl_writer_group = group
                            break
                    if ccl_writer_group is None:
                        raise RuntimeError("Missing is_input_core kernel group for CCL writer fabric append")
                    writer_kernel_idx = ccl_writer_group.ncrisc_kernel_index
                    writer_rt_args_ref = program.kernels[writer_kernel_idx].runtime_args[worker_core.x][worker_core.y]
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id, dst_nodes, [0], program, writer_kernel_idx, worker_core
                    )
                    writer_rt_args_ref.extend(fabric_args)

                if not skip_ccl and is_argmax_mesh_sender_core:
                    sender_group = kernel_result.get_group_by_arg("is_argmax_mesh_sender_core", 1)
                    if sender_group is None:
                        raise RuntimeError("Missing argmax mesh sender kernel group for BRISC fabric append")
                    sender_kernel_idx = sender_group.brisc_kernel_index
                    fabric_rt_args = ttnn.setup_fabric_connection(
                        src_fabric_node_id=mesh_device.get_fabric_node_id(coord),
                        dst_fabric_node_id=mesh_device.get_fabric_node_id(dest_coord),
                        link_idx=sender_link_idx,
                        program_descriptor=program,
                        worker_core=argmax_final_core,
                    )
                    program.kernels[sender_kernel_idx].runtime_args[argmax_final_core.x][argmax_final_core.y].extend(
                        fabric_rt_args
                    )

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic op
        io_tensors = [input_tensor_mesh, intermediate_tensor_mesh, vocab_tensor, output_tensor]
        io_tensors.extend([indices_tensor, output_index_tensor])
        if not skip_ccl:
            io_tensors.append(fabric_scratch_tensor)
        result = ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return result
