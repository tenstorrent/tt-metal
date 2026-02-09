# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post SDPA fused operation with CCL All-Reduce.

This implements Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 + CCL All-Reduce as a fused execution:
- Matmul1: [1, 512] x [512, 128] -> [1, 128] distributed across 64 cores (8x8 grid)
- Gather1: Collect results from all 64 cores to [1, 8192] on gather core (12, 9)
- Mcast: Broadcast [1, 8192] to 130 cores (13x10 grid, rectangular for efficient mcast)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] on 112 active cores (rows 0-7 full 13 + row 8 cols 0-7)
- Gather2: Collect results from all 112 active cores to [1, 7168] on gather core (12, 9)
- CCL All-Reduce: Exchange [1, 7168] between devices and reduce (local + remote + residual)

The 13x10 mcast grid contains 130 cores, but only 112 are active for matmul2.
The 18 inactive cores (row 8 cols 8-12, row 9 cols 0-12) receive mcast data but skip matmul via is_matmul2_core=false.

CCL All-Reduce uses:
- CCL Receiver core = Gather core (12, 9) - already has local data after Gather2
- CCL Sender core = Adjacent core (11, 9) - reads from gather core, sends via fabric

CB Layout:
- CB 0: matmul1_in0 (8x8 grid)
- CB 1: matmul1_in1 (8x8 grid)
- CB 2: matmul1_out (8x8 grid)
- CB 3: gather1_dst = mcast_src (gather core)
- CB 4: mcast_dst = matmul2_in0 (13x10 grid)
- CB 5: matmul2_in1 (112 active matmul2 cores)
- CB 6: matmul2_out (112 active matmul2 cores)
- CB 7: gather2_dst = ccl_local_data (gather core)
- CB 8: ccl_sender_in (ccl sender core)
- CB 9: ccl_remote_data (gather/receiver core - intermediate tensor)
- CB 10: ccl_residual (gather/receiver core - optional)
- CB 11: ccl_temp (gather/receiver core - for compute)
- CB 12: ccl_output (gather/receiver core - final output)
- CB 13: ccl_packet_header (sender + receiver cores)
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class PostSDPA:
    """
    Post SDPA fused operation implementation using ttnn.generic_op.

    Implements the full post_sdpa fusion with CCL all-reduce:
    - Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 + CCL All-Reduce
    """

    @staticmethod
    def golden(input_tensors, weights1_tensor, weights2_tensor, residual_tensor=None):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensors: List of input tensors (torch.Tensor) [1, 512], one per device
            weights1_tensor: First weights tensor (torch.Tensor) [512, 8192]
            weights2_tensor: Second weights tensor (torch.Tensor) [8192, 7168]
            residual_tensor: Optional residual tensor to add after reduction [1, 7168]

        Returns:
            Output tensor [1, 7168] - result of all-reduce across devices
        """
        # Compute matmul chain for each device
        device_results = []
        for input_tensor in input_tensors:
            intermediate = input_tensor @ weights1_tensor  # [1, 8192]
            result = intermediate @ weights2_tensor  # [1, 7168]
            device_results.append(result)

        # All-reduce: sum across all devices
        reduced = torch.sum(torch.stack(device_results), dim=0)

        # Add residual if provided
        if residual_tensor is not None:
            reduced = reduced + residual_tensor

        return reduced

    @staticmethod
    def op(
        input_tensor_mesh,
        weights1_tensor,
        weights2_tensor,
        gather1_output_tensor,
        gather2_output_tensor,
        intermediate_tensor,
        output_tensor,
        semaphores,
        cluster_axis=0,
        residual_tensor_mesh=None,
        fp32_dest_acc_en=False,
    ):
        """
        Execute post_sdpa fused operation with CCL all-reduce using generic_op.

        Args:
            input_tensor_mesh: Input tensor mesh [1, 512] (height-sharded across 8x8 matmul1 cores)
            weights1_tensor: First weights tensor [512, 8192] (width-sharded across 8x8)
            weights2_tensor: Second weights tensor [8192, 7168] (width-sharded across 112 cores)
            gather1_output_tensor: Intermediate tensor [1, 8192] for gather1/mcast (on gather core)
            gather2_output_tensor: Intermediate tensor mesh [1, 7168] for gather2/CCL (on gather core per device)
            intermediate_tensor: CCL intermediate tensor mesh for receiving remote data (32x32 tiles)
            output_tensor: Final output tensor mesh [1, 7168]
            semaphores: List of two global semaphores for CCL synchronization
            cluster_axis: Axis for all-reduce (default 0)
            residual_tensor_mesh: Optional tensor mesh for residuals [1, 7168]
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor mesh with full fused result including all-reduce
        """
        mesh_device = input_tensor_mesh.device()
        mesh_shape = mesh_device.shape
        mesh_rows = mesh_shape[0]
        mesh_cols = mesh_shape[1]

        # Get per-device tensors
        input_tensors_per_device = ttnn.get_device_tensors(input_tensor_mesh)
        gather2_output_tensors_per_device = ttnn.get_device_tensors(gather2_output_tensor)
        intermediate_tensors_per_device = ttnn.get_device_tensors(intermediate_tensor)
        output_tensors_per_device = ttnn.get_device_tensors(output_tensor)
        if residual_tensor_mesh is not None:
            residual_tensors_per_device = ttnn.get_device_tensors(residual_tensor_mesh)

        # CCL semaphores
        ccl_semaphore1 = semaphores[0]
        ccl_semaphore2 = semaphores[1]
        ccl_semaphore1_addr = ttnn.get_global_semaphore_address(ccl_semaphore1)
        ccl_semaphore2_addr = ttnn.get_global_semaphore_address(ccl_semaphore2)

        # Get tensor properties from first device
        input_tensor_sample = input_tensors_per_device[0]
        data_format = input_tensor_sample.dtype
        element_size = 2  # bfloat16

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        TILE_32x32 = ttnn.Tile((32, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_32x32_size = TILE_32x32.get_tile_size(data_format)

        # CCL intermediate tensor info (32x32 tiles)
        intermediate_tensor_sample = intermediate_tensors_per_device[0]
        intermediate_tile = intermediate_tensor_sample.tile
        intermediate_tile_height, intermediate_tile_width = intermediate_tile.tile_shape
        standard_tile_size_bytes = intermediate_tile_height * intermediate_tile_width * element_size

        # ========================================================================
        # Core grid configuration (same for all devices)
        # ========================================================================
        # Matmul1 grid: 8x8 = 64 cores
        MATMUL1_GRID_START_X = 0
        MATMUL1_GRID_START_Y = 0
        MATMUL1_GRID_END_X = 7  # 8 columns (0-7)
        MATMUL1_GRID_END_Y = 7  # 8 rows (0-7)
        matmul1_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MATMUL1_GRID_START_X, MATMUL1_GRID_START_Y),
            ttnn.CoreCoord(MATMUL1_GRID_END_X, MATMUL1_GRID_END_Y),
        )
        matmul1_core_grid = ttnn.CoreRangeSet([matmul1_grid])
        num_matmul1_cores = matmul1_grid.grid_size().x * matmul1_grid.grid_size().y  # 64

        # Gather/CCL receiver core: (12, 9)
        gather_core = ttnn.CoreCoord(12, 9)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # CCL sender core: (11, 9) - adjacent to gather core
        ccl_sender_core = ttnn.CoreCoord(11, 9)
        ccl_sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ccl_sender_core, ccl_sender_core)])

        # Mcast grid: 13x10 = 130 cores (full rectangular for efficient mcast)
        MCAST_GRID_START_X = 0
        MCAST_GRID_START_Y = 0
        MCAST_GRID_END_X = 12  # 13 columns (0-12)
        MCAST_GRID_END_Y = 9  # 10 rows (0-9)
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )
        mcast_core_grid = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y  # 130

        # Active Matmul2 cores: 112 cores (rows 0-7 full 13 cols + row 8 cols 0-7)
        matmul2_grid_main = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(12, 7),  # 13 columns x 8 rows = 104 cores
        )
        matmul2_grid_extra = ttnn.CoreRange(
            ttnn.CoreCoord(0, 8),
            ttnn.CoreCoord(7, 8),  # 8 columns x 1 row = 8 cores
        )
        matmul2_active_core_grid = ttnn.CoreRangeSet([matmul2_grid_main, matmul2_grid_extra])
        num_matmul2_cores = 112  # 104 + 8 active cores

        # Gather2 sender grid bounds (for offset calculation, use bounding box)
        MATMUL2_GRID_START_X = 0
        MATMUL2_GRID_START_Y = 0
        MATMUL2_GRID_END_X = 12  # Same as mcast grid for offset calculation
        MATMUL2_GRID_END_Y = 8

        # Full grid (union of all cores for semaphore allocation)
        full_grid = matmul1_core_grid.merge(gather_core_grid).merge(mcast_core_grid).merge(ccl_sender_core_grid)

        # ========================================================================
        # Matmul1 parameters: [1, 512] x [512, 128] -> [1, 128]
        # ========================================================================
        matmul1_k_num_tiles = 16  # 512 / 32 = 16 tiles
        matmul1_out_w_per_core = 4  # 128 / 32 = 4 tiles per core

        # ========================================================================
        # Matmul2 parameters: [1, 8192] x [8192, 64] -> [1, 64]
        # ========================================================================
        matmul2_k_num_tiles = 256  # 8192 / 32 = 256 tiles
        matmul2_out_w_per_core = 2  # 64 / 32 = 2 tiles per core

        # ========================================================================
        # CB indices
        # ========================================================================
        matmul1_in0_cb = 0  # Matmul1 input (8x8)
        matmul1_in1_cb = 1  # Matmul1 weights (8x8)
        matmul1_out_cb = 2  # Matmul1 output (8x8)
        gather1_dst_cb = 3  # Gather1 output = Mcast source (gather core)
        matmul2_in0_cb = 4  # Mcast dst = Matmul2 input (13x10 mcast grid)
        matmul2_in1_cb = 5  # Matmul2 weights (112 active cores)
        matmul2_out_cb = 6  # Matmul2 output (112 active cores)
        gather2_dst_cb = 7  # Gather2 output = CCL local data (gather core)
        ccl_sender_in_cb = 8  # CCL sender reads gather2 output (sender core)
        ccl_remote_data_cb = 9  # CCL received remote data (receiver core)
        ccl_residual_cb = 10  # CCL residual (receiver core)
        ccl_temp_cb = 11  # CCL temp for compute (receiver core)
        ccl_output_cb = 12  # CCL output (receiver core)
        ccl_packet_header_cb = 13  # CCL packet headers (sender + receiver cores)

        # ========================================================================
        # Gather1 parameters: 64 cores -> [1, 8192]
        # ========================================================================
        gather1_data_size_bytes = matmul1_out_w_per_core * tile_1x32_size
        gather1_src_num_pages = matmul1_out_w_per_core  # 4 pages per sender
        gather1_dst_num_pages = num_matmul1_cores * matmul1_out_w_per_core  # 64 * 4 = 256 pages
        gather1_noc0_num_senders = num_matmul1_cores
        gather1_noc1_num_senders = 0

        # ========================================================================
        # Mcast parameters: [1, 8192] to 130 cores (13x10 grid)
        # ========================================================================
        mcast_data_size_bytes = gather1_dst_num_pages * tile_1x32_size  # 256 * 64 = 16384 bytes
        mcast_src_num_pages = gather1_dst_num_pages  # 256 pages
        mcast_dst_num_pages = gather1_dst_num_pages  # 256 pages per receiver

        # ========================================================================
        # Gather2 parameters: 112 cores -> [1, 7168]
        # ========================================================================
        gather2_data_size_bytes = matmul2_out_w_per_core * tile_1x32_size
        gather2_src_num_pages = matmul2_out_w_per_core  # 2 pages per sender
        gather2_dst_num_pages = num_matmul2_cores * matmul2_out_w_per_core  # 112 * 2 = 224 pages
        gather2_noc0_num_senders = num_matmul2_cores
        gather2_noc1_num_senders = 0

        # ========================================================================
        # CCL parameters: [1, 7168] all-reduce
        # ========================================================================
        # Using 1x32 tiles to match gather2 output format (for tile-compatible reduction)
        # 7168 elements = 224 tiles of 1x32 (32 elements each)
        ccl_num_tiles = gather2_dst_num_pages  # 224 tiles of 1x32
        ccl_page_size_bytes = tile_1x32_size  # 1x32 tile size
        ccl_num_pages = gather2_dst_num_pages  # 224 pages of 1x32
        ccl_payload_size_bytes = ccl_num_pages * ccl_page_size_bytes  # 224 * 64 = 14336 bytes
        ccl_packet_header_size_bytes = 32
        l1_alignment = 16

        has_residual = 1 if residual_tensor_mesh is not None else 0

        # ========================================================================
        # Semaphore IDs
        # ========================================================================
        gather1_noc0_receiver_semaphore_id = 0
        gather1_noc1_receiver_semaphore_id = 1
        mcast_data_sender_semaphore_id = 2
        mcast_data_receiver_semaphore_id = 3
        gather2_noc0_receiver_semaphore_id = 4
        gather2_noc1_receiver_semaphore_id = 5
        gather2_completion_semaphore_id = 6  # Gather2 signals, CCL sender waits
        # CCL uses global semaphores (passed via runtime args)

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

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
                gather2_output_tensor_device = gather2_output_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]

                device = input_tensor_device.device()

                # Get NOC coordinates for this device
                gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)
                mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid.start)
                mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid.end)
                ccl_sender_noc_core = device.worker_core_from_logical_core(ccl_sender_core)
                ccl_receiver_noc_core = gather_dest_noc_core  # Same as gather core

                # Determine CCL neighbor and semaphores based on position
                ccl_sender_link = 0 if is_first_chip else 1
                ccl_receiver_link = 1 if is_first_chip else 0
                ccl_sender_semaphore_addr = ccl_semaphore1_addr if is_first_chip else ccl_semaphore2_addr
                ccl_receiver_semaphore_addr = ccl_semaphore2_addr if is_first_chip else ccl_semaphore1_addr

                # Calculate neighbor coordinate
                if is_first_chip:
                    neighbor_row = row + 1 if cluster_axis == 0 else row
                    neighbor_col = col if cluster_axis == 0 else col + 1
                else:
                    neighbor_row = row - 1 if cluster_axis == 0 else row
                    neighbor_col = col if cluster_axis == 0 else col - 1

                # Buffer addresses
                gather1_receiver_data_addr = gather1_output_tensor.buffer_address()
                # Gather2 writes to gather2_output_tensor, CCL reads from there and writes to output_tensor
                gather2_receiver_data_addr = gather2_output_tensor_device.buffer_address()

                mcast_is_part_of_receiver_grid = mcast_grid.contains(gather_core)

                # ========================================================================
                # NCRISC compile-time args
                # ========================================================================
                ncrisc_named_compile_time_args = [
                    # Matmul1
                    ("matmul1_in0", matmul1_in0_cb),
                    ("matmul1_in1", matmul1_in1_cb),
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul1_k_num_tiles", matmul1_k_num_tiles),
                    ("matmul1_out_w_per_core", matmul1_out_w_per_core),
                    # Gather1 sender
                    ("gather1_dest_noc_x", gather_dest_noc_core.x),
                    ("gather1_dest_noc_y", gather_dest_noc_core.y),
                    ("gather1_data_size_bytes", gather1_data_size_bytes),
                    ("gather1_receiver_semaphore_id", gather1_noc0_receiver_semaphore_id),
                    ("gather1_src_cb", matmul1_out_cb),
                    ("gather1_src_num_pages", gather1_src_num_pages),
                    ("gather1_sender_grid_start_x", MATMUL1_GRID_START_X),
                    ("gather1_sender_grid_start_y", MATMUL1_GRID_START_Y),
                    ("gather1_sender_grid_end_x", MATMUL1_GRID_END_X),
                    ("gather1_sender_grid_end_y", MATMUL1_GRID_END_Y),
                    ("gather1_row_major", 1),
                    ("gather1_receiver_data_addr", gather1_receiver_data_addr),
                    # Mcast receiver
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_dst_cb", matmul2_in0_cb),
                    ("mcast_dst_num_pages", mcast_dst_num_pages),
                    # Matmul2
                    ("matmul2_in0", matmul2_in0_cb),
                    ("matmul2_in1", matmul2_in1_cb),
                    ("matmul2_out", matmul2_out_cb),
                    ("matmul2_k_num_tiles", matmul2_k_num_tiles),
                    ("matmul2_out_w_per_core", matmul2_out_w_per_core),
                    # Gather2 sender
                    ("gather2_dest_noc_x", gather_dest_noc_core.x),
                    ("gather2_dest_noc_y", gather_dest_noc_core.y),
                    ("gather2_data_size_bytes", gather2_data_size_bytes),
                    ("gather2_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
                    ("gather2_src_cb", matmul2_out_cb),
                    ("gather2_src_num_pages", gather2_src_num_pages),
                    ("gather2_sender_grid_start_x", MATMUL2_GRID_START_X),
                    ("gather2_sender_grid_start_y", MATMUL2_GRID_START_Y),
                    ("gather2_sender_grid_end_x", MATMUL2_GRID_END_X),
                    ("gather2_sender_grid_end_y", MATMUL2_GRID_END_Y),
                    ("gather2_row_major", 1),
                    ("gather2_receiver_data_addr", gather2_receiver_data_addr),
                    # CCL sender (NCRISC reads from gather core)
                    ("ccl_sender_cb0_id", ccl_sender_in_cb),
                    ("ccl_sender_num_tiles", ccl_num_pages),
                    ("ccl_sender_tensor_page_size", ccl_page_size_bytes),
                    ("ccl_sender_data_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_data_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_gather2_completion_semaphore_id", gather2_completion_semaphore_id),
                    # CCL receiver (NCRISC waits for remote data)
                    ("ccl_receiver_packet_header_cb_id", ccl_packet_header_cb),
                    ("ccl_receiver_cb_in1", ccl_remote_data_cb),
                    ("ccl_receiver_l1_alignment", l1_alignment),
                    ("ccl_receiver_cb_in2", gather2_dst_cb),  # Local data from gather2
                    ("ccl_receiver_remote_sender_noc_x", ccl_sender_noc_core.x),
                    ("ccl_receiver_remote_sender_noc_y", ccl_sender_noc_core.y),
                    ("ccl_receiver_num_standard_tiles", ccl_num_tiles),
                    ("ccl_receiver_cb_residual", ccl_residual_cb),
                    ("ccl_receiver_has_residual", has_residual),
                    ("ccl_receiver_skip_local_push", 1),  # Skip local push since gather2 already pushed to CB7
                ]

                # ========================================================================
                # BRISC compile-time args
                # ========================================================================
                brisc_named_compile_time_args = [
                    # Matmul1/2 (no-op)
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul2_out", matmul2_out_cb),
                    # Gather1 receiver
                    ("gather1_noc0_num_senders", gather1_noc0_num_senders),
                    ("gather1_noc1_num_senders", gather1_noc1_num_senders),
                    ("gather1_noc0_receiver_semaphore_id", gather1_noc0_receiver_semaphore_id),
                    ("gather1_noc1_receiver_semaphore_id", gather1_noc1_receiver_semaphore_id),
                    ("gather1_dst_cb", gather1_dst_cb),
                    ("gather1_dst_num_pages", gather1_dst_num_pages),
                    # Mcast sender
                    ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
                    ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
                    ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
                    ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
                    ("mcast_num_cores", num_mcast_cores),
                    ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
                    ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
                    ("mcast_data_size_bytes", mcast_data_size_bytes),
                    ("mcast_src_cb", gather1_dst_cb),
                    ("mcast_src_num_pages", mcast_src_num_pages),
                    ("mcast_dst_cb", matmul2_in0_cb),
                    ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
                    # Gather2 receiver
                    ("gather2_noc0_num_senders", gather2_noc0_num_senders),
                    ("gather2_noc1_num_senders", gather2_noc1_num_senders),
                    ("gather2_noc0_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
                    ("gather2_noc1_receiver_semaphore_id", gather2_noc1_receiver_semaphore_id),
                    ("gather2_dst_cb", gather2_dst_cb),
                    ("gather2_dst_num_pages", gather2_dst_num_pages),
                    # Gather2 completion signal for CCL sender synchronization
                    ("gather2_completion_semaphore_id", gather2_completion_semaphore_id),
                    ("ccl_sender_noc_x", ccl_sender_noc_core.x),
                    ("ccl_sender_noc_y", ccl_sender_noc_core.y),
                    # CCL sender (BRISC sends via fabric)
                    ("ccl_sender_packet_header_cb_id", ccl_packet_header_cb),
                    ("ccl_sender_packet_cb_id", ccl_sender_in_cb),
                    ("ccl_sender_l1_alignment", l1_alignment),
                    ("ccl_sender_input_num_tiles", ccl_num_pages),
                    ("ccl_sender_page_size_bytes", ccl_page_size_bytes),
                    ("ccl_sender_payload_size_bytes", ccl_payload_size_bytes),
                    ("ccl_sender_data_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_data_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_remote_receiver_noc_x", ccl_receiver_noc_core.x),
                    ("ccl_sender_remote_receiver_noc_y", ccl_receiver_noc_core.y),
                    ("ccl_sender_dst_num_hops", 1),
                    ("ccl_sender_num_connections", 1),
                ]

                # ========================================================================
                # TRISC compile-time args
                # ========================================================================
                trisc_named_compile_time_args = [
                    # Matmul1
                    ("matmul1_in0", matmul1_in0_cb),
                    ("matmul1_in1", matmul1_in1_cb),
                    ("matmul1_out", matmul1_out_cb),
                    ("matmul1_k_num_tiles", matmul1_k_num_tiles),
                    ("matmul1_out_w_per_core", matmul1_out_w_per_core),
                    # Matmul2
                    ("matmul2_in0", matmul2_in0_cb),
                    ("matmul2_in1", matmul2_in1_cb),
                    ("matmul2_out", matmul2_out_cb),
                    ("matmul2_k_num_tiles", matmul2_k_num_tiles),
                    ("matmul2_out_w_per_core", matmul2_out_w_per_core),
                    # CCL receiver compute (reduction)
                    ("ccl_receiver_cb_in0", ccl_remote_data_cb),
                    ("ccl_receiver_cb_in1", gather2_dst_cb),  # Local data
                    ("ccl_receiver_cb_out0", ccl_output_cb),
                    ("ccl_receiver_cb_residual", ccl_residual_cb),
                    ("ccl_receiver_cb_temp", ccl_temp_cb),
                    ("ccl_receiver_has_residual", has_residual),
                    ("ccl_receiver_num_tiles", ccl_num_tiles),
                ]

                # ========================================================================
                # Circular buffer descriptors
                # ========================================================================
                # CB 0: Matmul1 input (from sharded tensor, 8x8 grid)
                matmul1_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul1_in0_cb, input_tensor_device)

                # CB 1: Matmul1 weights (from sharded tensor, 8x8 grid)
                matmul1_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul1_in1_cb, weights1_tensor)

                # CB 2: Matmul1 output (4 tiles of 1x32 per core, 8x8 grid)
                matmul1_out_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul1_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul1_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                matmul1_out_cb_descriptor = ttnn.CBDescriptor(
                    total_size=matmul1_out_w_per_core * tile_1x32_size,
                    core_ranges=matmul1_core_grid,
                    format_descriptors=[matmul1_out_cb_format],
                )

                # CB 3: Gather1 output = Mcast source (from sharded tensor, gather core)
                gather1_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather1_dst_cb, gather1_output_tensor
                )

                # CB 4: Mcast destination = Matmul2 input (256 tiles of 1x32 per core)
                matmul2_in0_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul2_in0_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                matmul2_in0_cb_grid = mcast_core_grid.merge(gather_core_grid)
                matmul2_in0_cb_descriptor = ttnn.CBDescriptor(
                    total_size=mcast_dst_num_pages * tile_1x32_size,
                    core_ranges=matmul2_in0_cb_grid,
                    format_descriptors=[matmul2_in0_cb_format],
                )

                # CB 5: Matmul2 weights (from sharded tensor, 112 active matmul2 cores)
                matmul2_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul2_in1_cb, weights2_tensor)

                # CB 6: Matmul2 output (2 tiles of 1x32 per core, 112 active matmul2 cores)
                matmul2_out_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul2_out_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                matmul2_out_cb_descriptor = ttnn.CBDescriptor(
                    total_size=matmul2_out_w_per_core * tile_1x32_size,
                    core_ranges=matmul2_active_core_grid,
                    format_descriptors=[matmul2_out_cb_format],
                )

                # CB 7: Gather2 output = CCL local data (backed by tensor on gather core)
                # CCL sender reads from this tensor via NOC, not from local CB
                gather2_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    gather2_dst_cb, gather2_output_tensor_device
                )

                # CB 8: CCL sender input (reads from gather2 output via NOC)
                ccl_sender_in_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=ccl_sender_in_cb,
                    data_format=data_format,
                    page_size=tile_1x32_size,
                    tile=matmul1_out_tile_descriptor,
                )
                ccl_sender_in_cb_descriptor = ttnn.CBDescriptor(
                    total_size=ccl_num_pages * tile_1x32_size,
                    core_ranges=ccl_sender_core_grid,
                    format_descriptors=[ccl_sender_in_cb_format],
                )

                # CB 9: CCL remote data (backed by intermediate tensor with 1x32 tiles)
                # The intermediate tensor is where the CCL sender writes remote data
                ccl_remote_data_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    ccl_remote_data_cb, intermediate_tensor_device
                )
                ccl_remote_data_cb_descriptor.core_ranges = gather_core_grid

                # CB 10: CCL residual (optional, from sharded tensor)
                cb_list = [
                    matmul1_in0_cb_descriptor,
                    matmul1_in1_cb_descriptor,
                    matmul1_out_cb_descriptor,
                    gather1_dst_cb_descriptor,
                    matmul2_in0_cb_descriptor,
                    matmul2_in1_cb_descriptor,
                    matmul2_out_cb_descriptor,
                    gather2_dst_cb_descriptor,
                    ccl_sender_in_cb_descriptor,
                    ccl_remote_data_cb_descriptor,
                ]

                if residual_tensor_mesh is not None:
                    ccl_residual_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                        ccl_residual_cb, residual_tensors_per_device[device_idx]
                    )
                    ccl_residual_cb_descriptor.core_ranges = gather_core_grid
                    ccl_residual_cb_descriptor.total_size = ccl_num_tiles * tile_1x32_size
                    ccl_residual_cb_descriptor.format_descriptors = [
                        ttnn.CBFormatDescriptor(
                            buffer_index=ccl_residual_cb,
                            data_format=data_format,
                            page_size=tile_1x32_size,
                            tile=matmul1_out_tile_descriptor,  # 1x32 tiles to match gather2
                        )
                    ]
                    cb_list.append(ccl_residual_cb_descriptor)

                    # CB 11: CCL temp scratch buffer (not backed by tensor)
                    ccl_temp_cb_descriptor = ttnn.CBDescriptor(
                        total_size=ccl_num_tiles * tile_1x32_size,
                        core_ranges=gather_core_grid,
                        format_descriptors=[
                            ttnn.CBFormatDescriptor(
                                buffer_index=ccl_temp_cb,
                                data_format=data_format,
                                page_size=tile_1x32_size,
                                tile=matmul1_out_tile_descriptor,  # 1x32 tiles to match gather2
                            )
                        ],
                    )
                    cb_list.append(ccl_temp_cb_descriptor)

                # CB 12: CCL output (from sharded tensor)
                ccl_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(ccl_output_cb, output_tensor_device)
                ccl_output_cb_descriptor.core_ranges = gather_core_grid
                cb_list.append(ccl_output_cb_descriptor)

                # CB 13: CCL packet headers
                ccl_packet_header_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=ccl_packet_header_cb,
                    data_format=ttnn.uint32,
                    page_size=ccl_packet_header_size_bytes,
                )
                ccl_packet_header_cb_descriptor = ttnn.CBDescriptor(
                    total_size=2 * ccl_packet_header_size_bytes,
                    core_ranges=gather_core_grid.merge(ccl_sender_core_grid),
                    format_descriptors=[ccl_packet_header_cb_format],
                )
                cb_list.append(ccl_packet_header_cb_descriptor)

                # ========================================================================
                # Semaphore descriptors
                # ========================================================================
                gather1_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather1_noc0_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather1_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather1_noc1_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_sender_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather2_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather2_noc0_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather2_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather2_noc1_receiver_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )
                gather2_completion_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather2_completion_semaphore_id,
                    core_ranges=full_grid,
                    initial_value=0,
                )

                # ========================================================================
                # Kernel descriptor
                # ========================================================================
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/post_sdpa/kernels/post_sdpa_kernel.cpp",
                    core_ranges=full_grid,
                    ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
                    brisc_named_compile_time_args=brisc_named_compile_time_args,
                    trisc_named_compile_time_args=trisc_named_compile_time_args,
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul1_core",
                            core_range=matmul1_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_gather_receiver_core",
                            core_range=gather_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul2_core",
                            core_range=matmul2_active_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_mcast_receiver_core",
                            core_range=mcast_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_ccl_sender_core",
                            core_range=ccl_sender_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_ccl_receiver_core",
                            core_range=gather_core_grid,  # CCL receiver = gather core
                            value=1,
                            other_value=0,
                        ),
                    ],
                )

                # Get kernel descriptors
                kernel_result = unified_kernel.get_kernel_descriptors()

                # Get kernel indices for runtime args
                ccl_sender_group = kernel_result.get_group_by_arg("is_ccl_sender_core", 1)
                ccl_receiver_group = kernel_result.get_group_by_arg("is_ccl_receiver_core", 1)

                # ========================================================================
                # Program descriptor
                # ========================================================================
                program = ttnn.ProgramDescriptor(
                    kernels=kernel_result.kernels,
                    cbs=cb_list,
                    semaphores=[
                        gather1_noc0_semaphore_descriptor,
                        gather1_noc1_semaphore_descriptor,
                        mcast_sender_semaphore_descriptor,
                        mcast_receiver_semaphore_descriptor,
                        gather2_noc0_semaphore_descriptor,
                        gather2_noc1_semaphore_descriptor,
                        gather2_completion_semaphore_descriptor,
                    ],
                )

                # ========================================================================
                # Runtime args for CCL
                # ========================================================================
                # === COMMON RUNTIME ARGS ===
                # Sender NCRISC common runtime args (arg 0)
                ccl_sender_ncrisc_common_rt_args = [
                    gather2_receiver_data_addr,  # tensor_address
                ]

                # Sender BRISC common runtime args (args 0-1)
                ccl_sender_brisc_common_rt_args = [
                    intermediate_tensor_device.buffer_address(),  # receiver_base_address
                    ccl_sender_semaphore_addr,  # receive_semaphore_addr
                ]

                # Receiver NCRISC common runtime args (arg 0)
                ccl_receiver_ncrisc_common_rt_args = [
                    ccl_receiver_semaphore_addr,  # sender_semaphore_addr
                ]

                # === PER-CORE RUNTIME ARGS (empty, fabric args appended later) ===
                ccl_sender_ncrisc_rt_args = ttnn.RuntimeArgs()
                ccl_sender_ncrisc_rt_args[ccl_sender_core.x][ccl_sender_core.y] = []

                ccl_sender_brisc_rt_args = ttnn.RuntimeArgs()
                ccl_sender_brisc_rt_args[ccl_sender_core.x][ccl_sender_core.y] = []

                ccl_receiver_ncrisc_rt_args = ttnn.RuntimeArgs()
                ccl_receiver_ncrisc_rt_args[gather_core.x][gather_core.y] = []

                # Set runtime args and common runtime args for CCL kernels
                program.kernels[ccl_sender_group.ncrisc_kernel_index].runtime_args = ccl_sender_ncrisc_rt_args
                program.kernels[
                    ccl_sender_group.ncrisc_kernel_index
                ].common_runtime_args = ccl_sender_ncrisc_common_rt_args
                program.kernels[ccl_sender_group.brisc_kernel_index].runtime_args = ccl_sender_brisc_rt_args
                program.kernels[
                    ccl_sender_group.brisc_kernel_index
                ].common_runtime_args = ccl_sender_brisc_common_rt_args
                program.kernels[ccl_receiver_group.ncrisc_kernel_index].runtime_args = ccl_receiver_ncrisc_rt_args
                program.kernels[
                    ccl_receiver_group.ncrisc_kernel_index
                ].common_runtime_args = ccl_receiver_ncrisc_common_rt_args

                # ========================================================================
                # Fabric connection setup
                # ========================================================================
                fabric_node_id = mesh_device.get_fabric_node_id(coord)
                neighbor_coord = ttnn.MeshCoordinate(neighbor_row, neighbor_col)
                neighbor_fabric_node_id = mesh_device.get_fabric_node_id(neighbor_coord)

                # Setup sender fabric connection
                sender_brisc_kernel_idx = ccl_sender_group.brisc_kernel_index
                sender_brisc_rt_args_ref = program.kernels[sender_brisc_kernel_idx].runtime_args[ccl_sender_core.x][
                    ccl_sender_core.y
                ]
                sender_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [ccl_sender_link],
                    program,
                    sender_brisc_kernel_idx,
                    ccl_sender_core,
                )
                sender_brisc_rt_args_ref.extend(sender_fabric_args)

                # Setup receiver fabric connection
                receiver_ncrisc_kernel_idx = ccl_receiver_group.ncrisc_kernel_index
                receiver_ncrisc_rt_args_ref = program.kernels[receiver_ncrisc_kernel_idx].runtime_args[gather_core.x][
                    gather_core.y
                ]
                receiver_fabric_args = ttnn.setup_routing_plane_connection(
                    fabric_node_id,
                    [neighbor_fabric_node_id],
                    [ccl_receiver_link],
                    program,
                    receiver_ncrisc_kernel_idx,
                    gather_core,
                )
                receiver_ncrisc_rt_args_ref.extend(receiver_fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program

        # Execute generic_op
        io_tensors = [
            input_tensor_mesh,
            weights1_tensor,
            weights2_tensor,
            gather1_output_tensor,
            gather2_output_tensor,
            intermediate_tensor,
            output_tensor,
        ]
        if residual_tensor_mesh is not None:
            io_tensors.append(residual_tensor_mesh)

        ttnn.generic_op(io_tensors, mesh_program_descriptor)

        return output_tensor
