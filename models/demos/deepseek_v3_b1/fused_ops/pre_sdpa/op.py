# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class PreSDPA:
    """
    Pre-SDPA fused operation implementation using ttnn.generic_op.

    This class implements the pre-SDPA operations as a fused execution:
    - CCL Broadcast (optional, for mesh devices)
    - RMSNorm on a single core
    - Multicast of the result to a grid of cores
    - Matmul on grid of cores
    - Gather from grid to single core
    - RMSNorm2 on single core
    - Mcast2 to grid of cores
    - Matmul2 on grid of cores
    """

    @staticmethod
    def golden(
        input_tensor,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        epsilon=1e-6,
    ):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, K]
            gamma_tensor: Gamma/weight tensor (torch.Tensor) [1, K]
            matmul_weights_tensor: Matmul weights (torch.Tensor) [K, N]
            rmsnorm2_gamma_tensor: Gamma tensor for second RMSNorm (torch.Tensor) [1, N]
            matmul2_weights_tensor: Matmul2 weights (torch.Tensor) [N, M]
            epsilon: Small value to avoid division by zero

        Returns:
            Output tensor with pre-SDPA operations applied: RMSNorm -> matmul -> RMSNorm2 -> matmul2 [1, M]
        """

        def rmsnorm(x, gamma):
            variance = x.pow(2).mean(-1, keepdim=True)
            normalized = x * torch.rsqrt(variance + epsilon)
            return normalized * gamma

        # RMSNorm -> Matmul: [1, K] @ [K, N] -> [1, N]
        matmul_result = rmsnorm(input_tensor, gamma_tensor) @ matmul_weights_tensor
        # RMSNorm2 -> Matmul2: [1, N] @ [N, M] -> [1, M]
        return rmsnorm(matmul_result, rmsnorm2_gamma_tensor) @ matmul2_weights_tensor

    @staticmethod
    def op(
        input_tensor_mesh,
        intermediate_tensor_mesh,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        output_tensor,
        sender_coord,
        semaphores=None,
        cluster_axis=0,
        secondary_cluster_axis=1,
        num_links=1,
        using_persistent_buffers=True,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
        rsqrt_fast_approx=False,
        skip_ccl=False,
    ):
        """
        Execute pre-SDPA fused operation using generic_op.

        Args:
            input_tensor_mesh: Input mesh tensor (must be sharded on single core per device)
            intermediate_tensor_mesh: Intermediate mesh tensor for CCL broadcast destination
            gamma_tensor: Gamma/weight tensor (must be sharded, same shape as input)
            matmul_weights_tensor: Matmul weights tensor (must be width sharded)
            rmsnorm2_gamma_tensor: Gamma tensor for second RMSNorm (1536 elements = 3 tiles of 16x32)
            matmul2_weights_tensor: Matmul2 weights tensor (width sharded, 4 tiles per core)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            sender_coord: Tuple (row, col) of sender device in mesh
            semaphores: List of global semaphores [out_ready, barrier, secondary_sync] for CCL
            cluster_axis: Primary axis for CCL broadcast (0=row, 1=col)
            secondary_cluster_axis: Secondary axis for CCL broadcast (optional)
            num_links: Number of fabric links for CCL
            using_persistent_buffers: Whether to use persistent buffers for CCL
            epsilon: Small value to avoid division by zero
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel
            skip_ccl: If True, skip CCL broadcast (single-device mode)

        Returns:
            Output tensor with RMSNorm applied
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
        gamma_tensors_per_device = ttnn.get_device_tensors(gamma_tensor)
        matmul_weights_tensors_per_device = ttnn.get_device_tensors(matmul_weights_tensor)
        rmsnorm2_gamma_tensors_per_device = ttnn.get_device_tensors(rmsnorm2_gamma_tensor)
        matmul2_weights_tensors_per_device = ttnn.get_device_tensors(matmul2_weights_tensor)
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

        # Get tensor properties (use a sample device tensor)
        input_tensor_sample = input_tensors_per_device[0]
        input_shape = input_tensor_sample.shape
        print("input shape is: ", input_shape)
        data_format = input_tensor_sample.dtype

        # CCL broadcast page info
        element_size = 2
        tile_id_start = 0
        bcast_page_size_bytes = 32 * 32 * element_size  # interpret as 32x32 tile
        bcast_num_pages = input_shape[0] * input_shape[1] * element_size // bcast_page_size_bytes
        num_pages_per_packet = packet_size_bytes // bcast_page_size_bytes

        # CB indices for CCL broadcast (use separate CBs to avoid conflicts)
        bcast_pkt_cb = 12  # Packet buffer for CCL broadcast

        # Interpret N 1x32 tiles as full 32x32 or 16x32 tiles
        # eg. [1, 7168] = 7 full 32x32 tiles
        # eg. [1, 1536] = 3 half 16x32 tiles
        # eg. [1, 512] = 1 half 16x32 tile
        FULL_32x32_TILE = ttnn.Tile((32, 32))
        HALF_16x32_TILE = ttnn.Tile((16, 32))
        is_16x32_tile = (input_shape[1] // FULL_32x32_TILE.tile_shape[1]) % FULL_32x32_TILE.tile_shape[0] != 0
        interpreted_tile = HALF_16x32_TILE if is_16x32_tile else FULL_32x32_TILE
        tile_height, tile_width = interpreted_tile.tile_shape

        # Calculate single tile size in bytes (bfloat16 = 2 bytes per element)
        tile_size = interpreted_tile.get_tile_size(data_format)

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (tile_height * tile_width)

        # Get number of elements for RMS calculation (use per-device tensor, not mesh)
        numel = input_tensor_sample.logical_volume()

        # Get core grid from input tensor's memory config
        input_memory_config = input_tensor_sample.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        rmsnorm_core = input_core_ranges[0].start
        rmsnorm_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(rmsnorm_core, rmsnorm_core)])

        # Get full device grid (use sample device)
        device = input_tensor_sample.device()
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # Get matmul weights core grid (48 cores for width sharding)
        matmul_weights_sample = matmul_weights_tensors_per_device[0]
        matmul_weights_memory_config = matmul_weights_sample.memory_config()
        matmul_weights_core_grid = matmul_weights_memory_config.shard_spec.grid

        # Calculate per-core width in tiles for matmul1 (from shard spec)
        # Get shard width directly from shard_spec and divide by tile width from tensor
        matmul_weights_tile = matmul_weights_sample.get_tile()
        matmul_weights_shard_shape = matmul_weights_memory_config.shard_spec.shape
        matmul_weights_shard_width = matmul_weights_shard_shape[1]  # Width dimension
        matmul1_out_w = matmul_weights_shard_width // matmul_weights_tile.tile_shape[1]  # Per-core width in tiles

        # Calculate per-core width in tiles for matmul2 (from shard spec)
        matmul2_weights_sample = matmul2_weights_tensors_per_device[0]
        matmul2_weights_memory_config = matmul2_weights_sample.memory_config()
        matmul2_weights_core_grid = matmul2_weights_memory_config.shard_spec.grid
        matmul2_weights_tile = matmul2_weights_sample.get_tile()
        matmul2_weights_shard_shape = matmul2_weights_memory_config.shard_spec.shape
        matmul2_weights_shard_width = matmul2_weights_shard_shape[1]  # Width dimension
        matmul2_out_w = matmul2_weights_shard_width // matmul2_weights_tile.tile_shape[1]  # Per-core width in tiles

        # ========================================================================
        # Mcast grid configuration (decoupled from matmul weights tensor)
        # Mcast to full logical grid; only matmul cores participate in receive
        # P150: (0,0)-(11,7), non-P150: (0,0)-(10,7)
        # ========================================================================
        MCAST_GRID_START_X = 0
        MCAST_GRID_START_Y = 0
        MCAST_GRID_END_X = device_grid_size.x - 1  # 11 for P150, 10 for non-P150
        MCAST_GRID_END_Y = 9
        main_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )

        # Mcast setup: sender core (rmsnorm) -> full mcast grid
        # Only matmul cores (is_matmul_core=true) will actually participate in receive

        # Get NOC coordinates for mcast destination
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(main_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(main_grid.end)

        # Calculate number of mcast cores (full grid)
        mcast_num_cores = main_grid.grid_size().x * main_grid.grid_size().y
        mcast_is_part_of_receiver_grid = main_grid.contains(rmsnorm_core_grid)

        # Semaphore IDs for mcast synchronization
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1

        # Semaphore IDs for gather synchronization
        # Senders on NCRISC use NOC_0, receiver on BRISC uses NOC_1
        # Only use noc0 semaphore since senders are on NOC_0 (default for NCRISC)
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3

        # Calculate mcast data size in bytes (RMSNorm output = num_tiles * tile_size)
        mcast_data_size_bytes = num_tiles * tile_size

        # Calculate runtime args
        epsilon_packed = float_to_uint32(epsilon)

        # Compute 1/sqrt(num_elements) for RMS reduction
        inv_sqrt_numel = 1.0 / math.sqrt(float(numel))
        scalar_packed = float_to_uint32(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

        # CB indices
        input_cb = 0
        gamma_cb = 1
        rmsnorm_output_cb = 2
        matmul_weights_cb = 3
        matmul_output_cb = 4
        matmul_input_cb = 5
        rmsnorm2_gamma_cb = 6  # New gamma for second RMSNorm (1536 elements = 3 tiles of 16x32)
        rmsnorm2_input_cb = 7  # Separate input CB for RMSNorm2
        rmsnorm2_output_cb = 8  # Separate output CB for RMSNorm2
        matmul2_input_cb = 9  # Input CB for second matmul (1x1536 with 1x32 tiles)
        matmul2_weights_cb = 10  # Weights CB for second matmul (width sharded, 4 tiles per core)
        matmul2_output_cb = 11  # Output CB for second matmul

        # RMSNorm2 parameters (for 1536 element input using 16x32 tiles)
        rmsnorm2_numel = 1536
        rmsnorm2_num_tiles = 3  # 3 tiles of 16x32 = 3 * 512 = 1536 elements

        # Compute 1/sqrt(1536) for RMSNorm2 reduction
        inv_sqrt_rmsnorm2_numel = 1.0 / math.sqrt(float(rmsnorm2_numel))
        scalar2_packed = float_to_uint32(inv_sqrt_rmsnorm2_numel)

        # Matmul2 parameters
        # Input: RMSNorm2 output (1x1536 = 48 1x32 tiles)
        # Weights: width sharded with 4 tiles per core on the main grid
        # Grid: 8x12 = 96 cores (P150) or 8x11 = 88 cores (non-P150)
        matmul2_num_tiles_k = 48  # 1536 / 32 = 48 1x32 tiles

        # Mcast2 parameters (broadcasts rmsnorm2 output from input core to all matmul2 cores)
        # Reads from rmsnorm2_output_cb (3 tiles of 16x32), writes to matmul2_in0 (48 1x32 tiles) with loopback
        # Uses same grid and semaphores as first mcast
        mcast2_data_size_bytes = 1536 * 2  # 1536 bfloat16 elements = 3072 bytes
        mcast2_src_num_pages = rmsnorm2_num_tiles  # 3 tiles (rmsnorm2 output in 16x32 format)
        mcast2_dst_num_pages = matmul2_num_tiles_k  # 48 pages (destination uses 1x32 tiles)

        # Calculate mcast page counts for source and destination CBs
        # Source CB (rmsnorm_output): uses RMSNorm tile format (32x32 or 16x32)
        mcast_src_num_pages = num_tiles
        # Destination CB (matmul_input): uses 1x32 tile format
        TILE_1x32 = ttnn.Tile((1, 32))
        matmul_input_page_size = TILE_1x32.get_tile_size(data_format)
        matmul_input_total_size = num_tiles * cb_page_size  # Same total bytes as RMSNorm output
        mcast_dst_num_pages = matmul_input_total_size // matmul_input_page_size

        # RMSNorm reader compile-time args (named args for NCRISC)
        rmsnorm_reader_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_num_tiles", num_tiles),
        ]

        # Mcast sender compile-time args (named args for BRISC)
        mcast_sender_named_compile_time_args = [
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", mcast_num_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", rmsnorm_output_cb),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ]

        # Mcast receiver compile-time args (named args for NCRISC)
        mcast_receiver_named_compile_time_args = [
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", matmul_input_cb),
            ("mcast_dst_num_pages", mcast_dst_num_pages),
        ]

        # Calculate matmul parameters
        # num_tiles_k = number of 1x32 tiles in the input (same as mcast_dst_num_pages)
        matmul_num_tiles_k = mcast_dst_num_pages

        # Matmul compile-time args (different per RISC, only pass what's used)
        # NCRISC: in1, num_tiles
        matmul_ncrisc_named_compile_time_args = [
            ("matmul_in1", matmul_weights_cb),
            ("matmul_k_num_tiles", matmul_num_tiles_k),
            ("matmul_out_w_per_core", matmul1_out_w),
        ]
        # BRISC: out
        matmul_brisc_named_compile_time_args = [
            ("matmul_out", matmul_output_cb),
        ]
        # TRISC: in0, in1, out, num_tiles, out_w_per_core
        matmul_trisc_named_compile_time_args = [
            ("matmul_in0", matmul_input_cb),
            ("matmul_in1", matmul_weights_cb),
            ("matmul_out", matmul_output_cb),
            ("matmul_k_num_tiles", matmul_num_tiles_k),
            ("matmul_out_w_per_core", matmul1_out_w),
        ]

        # Matmul2 compile-time args (different per RISC)
        # NCRISC: in1, num_tiles, rmsnorm2_output_cb (for copy to matmul2_input)
        matmul2_ncrisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_in1", matmul2_weights_cb),
            ("matmul2_out", matmul2_output_cb),
            ("matmul2_k_num_tiles", matmul2_num_tiles_k),
            ("matmul2_out_w_per_core", matmul2_out_w),
        ]
        # BRISC: in0 (for mcast2 receiver), out
        matmul2_brisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_out", matmul2_output_cb),
        ]
        # TRISC: in0, in1, out, num_tiles, out_w_per_core
        matmul2_trisc_named_compile_time_args = [
            ("matmul2_in0", matmul2_input_cb),
            ("matmul2_in1", matmul2_weights_cb),
            ("matmul2_out", matmul2_output_cb),
            ("matmul2_k_num_tiles", matmul2_num_tiles_k),
            ("matmul2_out_w_per_core", matmul2_out_w),
        ]

        # RMSNorm compute compile-time args (named args for TRISC)
        rmsnorm_compute_named_compile_time_args = [
            ("rmsnorm_input_cb", input_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_output_cb", rmsnorm_output_cb),
            ("rmsnorm_fp32_acc", 1 if fp32_dest_acc_en else 0),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_rsqrt_fast_approx", 0),
        ]

        # RMSNorm2 compile-time args (for second RMSNorm on gathered data)
        # Uses separate CBs with exact sizes for testing
        rmsnorm2_ncrisc_named_compile_time_args = [
            ("rmsnorm2_input_cb", rmsnorm2_input_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
        ]
        rmsnorm2_trisc_named_compile_time_args = [
            ("rmsnorm2_input_cb", rmsnorm2_input_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
        ]
        # BRISC needs rmsnorm2 gamma CB for skip_ccl mode (single-device gamma setup)
        rmsnorm2_brisc_named_compile_time_args = [
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
        ]

        # ========================================================================
        # Gather setup: matmul cores (senders) -> rmsnorm core (receiver)
        # Sender runs on NCRISC (NOC_0 default), Receiver runs on BRISC (NOC_1 default)
        # ========================================================================
        gather_receiver_core = rmsnorm_core
        gather_sender_grid = matmul_weights_core_grid

        # Get NOC coordinates for gather destination (receiver core)
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_receiver_core)

        # Calculate gather data size (matmul output size per core = 1 tile of 1x32)
        # Note: matmul_input_page_size == matmul_output_page_size (both are 1x32 tiles)
        gather_data_size_bytes = matmul_input_page_size

        # Get number of sender cores (matmul grid)
        gather_sender_cores_list = ttnn.corerange_to_cores(gather_sender_grid, row_wise=True)
        gather_num_senders = len(gather_sender_cores_list)

        # All senders use NOC_0 (default for NCRISC), so noc0_num_senders = all, noc1_num_senders = 0
        gather_noc0_num_senders = gather_num_senders
        gather_noc1_num_senders = 0

        # Get sender grid dimensions for computing per-core offset in kernel
        # Use logical coordinates since kernel uses UnifiedCoreDescriptor with my_logical_x_/y_
        gather_sender_grid_ranges = list(gather_sender_grid.ranges())
        gather_sender_grid_range = gather_sender_grid_ranges[0]
        gather_sender_grid_start_x = gather_sender_grid_range.start.x
        gather_sender_grid_start_y = gather_sender_grid_range.start.y
        gather_sender_grid_end_x = gather_sender_grid_range.end.x
        gather_sender_grid_end_y = gather_sender_grid_range.end.y

        # Gather sender compile-time args (named args for NCRISC on matmul cores)
        # SenderCTArgs: dest_noc_x, dest_noc_y, data_size_bytes, receiver_semaphore_id
        # Plus grid info for computing per-core offset
        gather_src_num_pages = 1  # Matmul output tiles per core (single 1x32 tile)
        gather_sender_named_compile_time_args = [
            ("gather_dest_noc_x", gather_dest_noc_core.x),
            ("gather_dest_noc_y", gather_dest_noc_core.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", matmul_output_cb),  # Source CB for gather (matmul output)
            ("gather_src_num_pages", gather_src_num_pages),
            ("gather_sender_grid_start_x", gather_sender_grid_start_x),
            ("gather_sender_grid_start_y", gather_sender_grid_start_y),
            ("gather_sender_grid_end_x", gather_sender_grid_end_x),
            ("gather_sender_grid_end_y", gather_sender_grid_end_y),
            ("gather_row_major", 1),  # 1 = row-major linearization
            ("gather_dst_cb", rmsnorm2_input_cb),  # Destination CB: write directly to rmsnorm2_input_cb
        ]

        # Gather receiver compile-time args (named args for BRISC on rmsnorm core)
        # ReceiverCTArgs: noc0_num_senders, noc1_num_senders, noc0_receiver_semaphore_id, noc1_receiver_semaphore_id
        # Plus destination CB info for reserve/push
        # Writes directly to rmsnorm2_input_cb (3 tiles of 16x32 = 3072 bytes)
        gather_receiver_named_compile_time_args = [
            ("gather_noc0_num_senders", gather_noc0_num_senders),
            ("gather_noc1_num_senders", gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", rmsnorm2_input_cb),
            ("gather_dst_num_pages", rmsnorm2_num_tiles),  # 3 pages of 16x32 tiles
        ]

        # Create tile descriptor for proper tile dimensions
        tile_descriptor = ttnn.TileDescriptor(interpreted_tile)

        # RMSNorm2 uses separate CBs with exact sizes (16x32 tiles)
        TILE_16x32 = ttnn.Tile((16, 32))
        rmsnorm2_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
        rmsnorm2_page_size = TILE_16x32.get_tile_size(data_format)

        # Create mesh program descriptor
        mesh_program_descriptor = ttnn.MeshProgramDescriptor()

        # ========================================================================
        # Per-device program creation loop
        # ========================================================================
        print("mesh rows: {}, mesh cols: {}".format(mesh_rows, mesh_cols))
        for row in range(mesh_rows):
            print("start of loop for device row {}".format(row))
            for col in range(mesh_cols):
                print("start of loop for device row {}, col {}".format(row, col))
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

                # Get the device's tensors
                input_tensor_device = input_tensors_per_device[device_idx]
                intermediate_tensor_device = intermediate_tensors_per_device[device_idx]
                gamma_tensor_device = gamma_tensors_per_device[device_idx]
                matmul_weights_tensor_device = matmul_weights_tensors_per_device[device_idx]
                rmsnorm2_gamma_tensor_device = rmsnorm2_gamma_tensors_per_device[device_idx]
                matmul2_weights_tensor_device = matmul2_weights_tensors_per_device[device_idx]
                output_tensor_device = output_tensors_per_device[device_idx]

                # Get worker core from per-device input tensor shard grid
                device_local = input_tensor_device.device()
                device_input_shard_grid = input_tensor_device.memory_config().shard_spec.grid
                device_shard_grid_start = device_input_shard_grid.bounding_box().start
                worker_core = ttnn.CoreCoord(device_shard_grid_start.x, device_shard_grid_start.y)
                worker_core_set = ttnn.CoreRangeSet([ttnn.CoreRange(worker_core, worker_core)])

                # Get physical core for NOC addressing
                data_core_physical = device_local.worker_core_from_logical_core(worker_core)
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

                # ================================================================
                # CCL Broadcast compile-time args (per-device)
                # ================================================================
                bcast_ncrisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_packet_size_in_pages", num_pages_per_packet if not skip_ccl else 0),
                    ("bcast_tensor0_page_size", bcast_page_size_bytes if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    ("bcast_core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("bcast_core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("bcast_is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("bcast_is_active_broadcaster", int(is_sender or is_secondary_sender) if not skip_ccl else 0),
                ]

                bcast_brisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                    ("bcast_cb0_id", bcast_pkt_cb if not skip_ccl else 0),
                    ("bcast_packet_size_in_pages", num_pages_per_packet if not skip_ccl else 0),
                    ("bcast_tensor0_page_size", bcast_page_size_bytes if not skip_ccl else 0),
                    ("bcast_num_targets_forward_direction", num_targets_forward if not skip_ccl else 0),
                    ("bcast_num_targets_backward_direction", num_targets_backward if not skip_ccl else 0),
                    ("bcast_is_sender", int(is_sender) if not skip_ccl else 0),
                    ("bcast_core_noc_x", core_noc_x if not skip_ccl else 0),
                    ("bcast_core_noc_y", core_noc_y if not skip_ccl else 0),
                    ("bcast_is_secondary_sender", int(is_secondary_sender) if not skip_ccl else 0),
                    ("bcast_has_secondary_target", int(has_secondary_target) if not skip_ccl else 0),
                    (
                        "bcast_has_reverse_secondary_connection",
                        int(has_reverse_secondary_connection) if not skip_ccl else 0,
                    ),
                    ("bcast_start_distance_in_hops_forward", start_distance_forward if not skip_ccl else 0),
                    ("bcast_range_hops_forward", range_hops_forward if not skip_ccl else 0),
                    ("bcast_start_distance_in_hops_backward", start_distance_backward if not skip_ccl else 0),
                    ("bcast_range_hops_backward", range_hops_backward if not skip_ccl else 0),
                    ("bcast_using_persistent_buffers", (1 if using_persistent_buffers else 0) if not skip_ccl else 0),
                ]

                bcast_trisc_named_compile_time_args = [
                    ("skip_ccl", 1 if skip_ccl else 0),
                ]

                # ================================================================
                # Create per-device CB descriptors
                # ================================================================
                # CB: Input - in CCL mode backed by intermediate, in single-device backed by input
                cb0_backing_tensor = input_tensor_device if skip_ccl else intermediate_tensor_device
                in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, cb0_backing_tensor)
                # Update the tile descriptor in the format descriptor
                in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB: CCL broadcast packet buffer
                bcast_pkt_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=bcast_pkt_cb,
                    data_format=data_format,
                    page_size=cb_page_size,
                    tile=tile_descriptor,
                )
                bcast_pkt_cb_descriptor = ttnn.CBDescriptor(
                    total_size=num_tiles * cb_page_size,
                    core_ranges=worker_core_set,
                    format_descriptors=[bcast_pkt_cb_format],
                )
                print("after creating bcast pkt")

                # CB: Gamma (created from sharded per-device tensor)
                gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor_device)
                # Update the tile descriptor in the format descriptor
                gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
                gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

                # CB: RMSNorm2 Gamma (created from sharded per-device tensor, 3 tiles of 16x32)
                rmsnorm2_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    rmsnorm2_gamma_cb, rmsnorm2_gamma_tensor_device
                )
                # Update the tile descriptor in the format descriptor to match rmsnorm2 tile shape
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm2_tile_descriptor
                rmsnorm2_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm2_page_size

                # CB: RMSNorm2 input buffer (3 tiles)
                rmsnorm2_input_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=rmsnorm2_input_cb,
                    data_format=data_format,
                    page_size=rmsnorm2_page_size,
                    tile=rmsnorm2_tile_descriptor,
                )
                # Must be allocated on union of matmul cores and rmsnorm core for gather to get write_ptr
                rmsnorm2_input_cb_core_ranges = matmul_weights_core_grid.merge(worker_core_set)
                rmsnorm2_input_cb_descriptor = ttnn.CBDescriptor(
                    total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,  # 3 tiles
                    core_ranges=rmsnorm2_input_cb_core_ranges,
                    format_descriptors=[rmsnorm2_input_cb_format],
                )
                # CB: RMSNorm2 output buffer (3 tiles)
                rmsnorm2_output_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=rmsnorm2_output_cb,
                    data_format=data_format,
                    page_size=rmsnorm2_page_size,
                    tile=rmsnorm2_tile_descriptor,
                )
                rmsnorm2_output_cb_descriptor = ttnn.CBDescriptor(
                    total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,  # 3 tiles
                    core_ranges=worker_core_set,
                    format_descriptors=[rmsnorm2_output_cb_format],
                )

                # CB: RMSNorm output buffer (dynamically created)
                rmsnorm_output_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=rmsnorm_output_cb,
                    data_format=data_format,
                    page_size=cb_page_size,
                    tile=tile_descriptor,
                )
                rmsnorm_output_cb_descriptor = ttnn.CBDescriptor(
                    total_size=num_tiles * cb_page_size,
                    core_ranges=worker_core_set,
                    format_descriptors=[rmsnorm_output_cb_format],
                )

                # CB: Matmul weights (created from sharded tensor)
                matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul_weights_cb, matmul_weights_tensor_device
                )

                # CB: Matmul input buffer (1x32 tiles, receives mcast data)
                # Senders will query the write pointer of this CB to get the receiver address.
                # Constraints on CB creation:
                # - Must be allocated and visible on the union of sender and receiver grids,
                #   even though the sender never uses the space allocated for this CB
                # - Must be single-buffered so senders can use get_write_ptr to get receiver address
                # - Dynamically allocated CB is better because less inputs to OP and technically
                #   uses minimal grid (ie. we can still use the same CB id for cores not in the
                #   union of sender and receiver grid)
                # Note: TILE_1x32, matmul_input_page_size, and matmul_input_total_size
                # were already calculated above for mcast page count calculation
                matmul_input_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul_input_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul_input_cb,
                    data_format=data_format,
                    page_size=matmul_input_page_size,
                    tile=matmul_input_tile_descriptor,
                )
                matmul_input_cb_core_ranges = matmul_weights_core_grid.merge(worker_core_set)
                matmul_input_cb_descriptor = ttnn.CBDescriptor(
                    total_size=matmul_input_total_size,
                    core_ranges=matmul_input_cb_core_ranges,
                    format_descriptors=[matmul_input_cb_format],
                )

                # CB: Matmul output buffer (single tile, on matmul cores only)
                matmul_output_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
                matmul_output_page_size = TILE_1x32.get_tile_size(data_format)
                matmul_output_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul_output_cb,
                    data_format=data_format,
                    page_size=matmul_output_page_size,
                    tile=matmul_output_tile_descriptor,
                )
                matmul_output_cb_descriptor = ttnn.CBDescriptor(
                    total_size=matmul_output_page_size,  # Single tile
                    core_ranges=matmul_weights_core_grid,
                    format_descriptors=[matmul_output_cb_format],
                )

                # CB: Matmul2 input buffer (1x1536 with 1x32 tiles = 48 tiles)
                # Must be allocated on union of sender (rmsnorm input grid) and receiver (matmul2 grid)
                # Similar constraint as gather CB - senders query write_ptr to get receiver address
                matmul2_input_total_size = matmul2_num_tiles_k * matmul_input_page_size  # 48 * 64 bytes
                matmul2_input_cb_format = ttnn.CBFormatDescriptor(
                    buffer_index=matmul2_input_cb,
                    data_format=data_format,
                    page_size=matmul_input_page_size,
                    tile=matmul_input_tile_descriptor,
                )
                matmul2_input_cb_core_ranges = ttnn.CoreRangeSet([main_grid]).merge(worker_core_set)
                matmul2_input_cb_descriptor = ttnn.CBDescriptor(
                    total_size=matmul2_input_total_size,
                    core_ranges=matmul2_input_cb_core_ranges,
                    format_descriptors=[matmul2_input_cb_format],
                )

                # CB: Matmul2 weights (created from sharded per-device tensor, 4 tiles per core)
                matmul2_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul2_weights_cb, matmul2_weights_tensor_device
                )

                # CB: Matmul2 output buffer (width sharded, mapped to per-device output_tensor)
                matmul2_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
                    matmul2_output_cb, output_tensor_device
                )

                # ================================================================
                # Mcast2 compile-time args (uses same grid and semaphores as first mcast)
                # ================================================================
                mcast2_brisc_named_compile_time_args = [
                    ("mcast2_data_size_bytes", mcast2_data_size_bytes),
                    ("mcast2_src_num_pages", mcast2_src_num_pages),
                    ("rmsnorm2_output_cb", rmsnorm2_output_cb),
                ]
                mcast2_ncrisc_named_compile_time_args = [
                    ("mcast2_dst_num_pages", mcast2_dst_num_pages),
                ]

                # ================================================================
                # Semaphore descriptors
                # ================================================================
                mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_sender_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )
                mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=mcast_data_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )
                gather_noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather_noc0_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )
                gather_noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
                    id=gather_noc1_receiver_semaphore_id,
                    core_ranges=full_device_grid,
                    initial_value=0,
                )

                # ================================================================
                # Unified kernel descriptor (per-device with CCL broadcast args)
                # ================================================================
                unified_kernel = UnifiedKernelDescriptor(
                    kernel_source="models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp",
                    core_ranges=full_device_grid,
                    # NCRISC: bcast + rmsnorm reader + mcast receiver + matmul + gather sender + rmsnorm2 + matmul2 + mcast2
                    ncrisc_named_compile_time_args=bcast_ncrisc_named_compile_time_args
                    + rmsnorm_reader_named_compile_time_args
                    + mcast_receiver_named_compile_time_args
                    + matmul_ncrisc_named_compile_time_args
                    + gather_sender_named_compile_time_args
                    + rmsnorm2_ncrisc_named_compile_time_args
                    + matmul2_ncrisc_named_compile_time_args
                    + mcast2_ncrisc_named_compile_time_args,
                    # BRISC: bcast + rmsnorm reader (for gamma setup) + mcast sender + matmul + gather receiver + rmsnorm2 + matmul2 + mcast2
                    brisc_named_compile_time_args=bcast_brisc_named_compile_time_args
                    + rmsnorm_reader_named_compile_time_args
                    + mcast_sender_named_compile_time_args
                    + matmul_brisc_named_compile_time_args
                    + gather_receiver_named_compile_time_args
                    + matmul2_brisc_named_compile_time_args
                    + mcast2_brisc_named_compile_time_args,
                    # TRISC: bcast + rmsnorm compute + matmul + rmsnorm2 + matmul2
                    trisc_named_compile_time_args=bcast_trisc_named_compile_time_args
                    + rmsnorm_compute_named_compile_time_args
                    + matmul_trisc_named_compile_time_args
                    + rmsnorm2_trisc_named_compile_time_args
                    + matmul2_trisc_named_compile_time_args,
                    trisc_common_runtime_args=[
                        epsilon_packed,
                        scalar_packed,
                        scalar2_packed,
                    ],
                    trisc_compute_config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.LoFi,
                        math_approx_mode=False,
                        fp32_dest_acc_en=fp32_dest_acc_en,
                        dst_full_sync_en=fp32_dest_acc_en,
                    ),
                    unified_compile_time_core_descriptors=[
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_input_core",
                            core_range=worker_core,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul_core",
                            core_range=matmul_weights_core_grid,
                            value=1,
                            other_value=0,
                        ),
                        UnifiedCompileTimeCoreDescriptor(
                            named_compile_time_arg="is_matmul2_core",
                            core_range=matmul2_weights_core_grid,
                            value=1,
                            other_value=0,
                        ),
                    ],
                )

                # ================================================================
                # CCL Broadcast runtime args and fabric setup
                # ================================================================
                reader_rt_args = ttnn.RuntimeArgs()
                writer_rt_args = ttnn.RuntimeArgs()

                if skip_ccl:
                    # Single-device mode: no broadcast runtime args
                    reader_rt_args[worker_core.x][worker_core.y] = []
                    writer_rt_args[worker_core.x][worker_core.y] = []
                else:
                    # Multi-device mode: CCL broadcast runtime args
                    reader_rt_args[worker_core.x][worker_core.y] = [
                        int(input_tensor_device.buffer_address()),  # tensor_address0
                        tile_id_start,  # tile_id_start
                        bcast_num_pages,  # tile_id_end
                    ]

                    wait_output_semaphore = is_secondary_sender or is_receiver
                    reset_global_semaphore = is_secondary_sender or is_receiver
                    out_ready_sem_wait_value = 1 * num_links

                    writer_rt_args[worker_core.x][worker_core.y] = [
                        int(intermediate_tensor_device.buffer_address()),  # tensor_address0
                        int(out_ready_sem_addr),  # out_ready_sem_bank_addr
                        tile_id_start,  # tile_id_start
                        bcast_num_pages,  # tile_id_end
                        int(wait_output_semaphore),
                        int(reset_global_semaphore),
                        core_noc_x,  # out_ready_sem_noc0_x
                        core_noc_y,  # out_ready_sem_noc0_y
                        out_ready_sem_wait_value,
                        int(barrier_sem_addr),
                        core_noc_x,  # barrier_sem_noc0_x
                        core_noc_y,  # barrier_sem_noc0_y
                        ring_index,
                        int(secondary_sync_sem_addr),
                    ]

                # Determine fabric connections
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
                        secondary_coord = ttnn.MeshCoordinate(row, 1)
                        dst_nodes.append(mesh_device.get_fabric_node_id(secondary_coord))

                    # Reverse secondary connection
                    if has_reverse_secondary_connection and not using_persistent_buffers:
                        sender_coord_back = ttnn.MeshCoordinate(sender_row, sender_col)
                        dst_nodes.append(mesh_device.get_fabric_node_id(sender_coord_back))

                    num_connections = len(dst_nodes)
                    writer_rt_args[worker_core.x][worker_core.y].append(int(num_connections))

                # ================================================================
                # Create program descriptor
                # ================================================================
                cbs_list = [
                    in_cb_descriptor,
                    gamma_cb_descriptor,
                    rmsnorm_output_cb_descriptor,
                    matmul_weights_cb_descriptor,
                    matmul_output_cb_descriptor,
                    matmul_input_cb_descriptor,
                    rmsnorm2_gamma_cb_descriptor,  # CB: RMSNorm2 gamma
                    rmsnorm2_input_cb_descriptor,  # CB: RMSNorm2 input
                    rmsnorm2_output_cb_descriptor,  # CB: RMSNorm2 output
                    matmul2_input_cb_descriptor,  # CB: Matmul2 input
                    matmul2_weights_cb_descriptor,  # CB: Matmul2 weights
                    matmul2_output_cb_descriptor,  # CB: Matmul2 output
                ]
                if not skip_ccl:
                    cbs_list.append(bcast_pkt_cb_descriptor)

                program = ttnn.ProgramDescriptor(
                    kernels=unified_kernel.get_kernel_descriptors(),
                    cbs=cbs_list,
                    semaphores=[
                        mcast_sender_semaphore_descriptor,  # ID 0
                        mcast_receiver_semaphore_descriptor,  # ID 1
                        gather_noc0_receiver_semaphore_descriptor,  # ID 2
                        gather_noc1_receiver_semaphore_descriptor,  # ID 3
                    ],
                )
                print("Program has {} kernels".format(len(program.kernels)))

                # Set runtime args for reader/writer kernels on worker core
                # With unified_compile_time_core_descriptors, there are multiple kernel groups
                # We need to find the NCRISC (reader) and BRISC (writer) kernels whose core_ranges include worker_core
                worker_writer_kernel_idx = None
                for idx, kernel in enumerate(program.kernels):
                    # Check if this kernel's core_ranges contains the worker_core
                    if kernel.core_ranges.contains(worker_core):
                        # Determine kernel type from config
                        if isinstance(kernel.config, ttnn.ReaderConfigDescriptor):
                            # NCRISC reader kernel
                            kernel.runtime_args = reader_rt_args
                            print(
                                "Set reader runtime args for kernel {} on worker core ({},{})".format(
                                    idx, worker_core.x, worker_core.y
                                )
                            )
                        elif isinstance(kernel.config, ttnn.WriterConfigDescriptor):
                            # BRISC writer kernel
                            kernel.runtime_args = writer_rt_args
                            worker_writer_kernel_idx = idx
                            print(
                                "Set writer runtime args for kernel {} on worker core ({},{})".format(
                                    idx, worker_core.x, worker_core.y
                                )
                            )

                # Append fabric connection args if needed
                if not skip_ccl and num_connections > 0 and worker_writer_kernel_idx is not None:
                    writer_rt_args_ref = program.kernels[worker_writer_kernel_idx].runtime_args[worker_core.x][
                        worker_core.y
                    ]
                    fabric_args = ttnn.setup_routing_plane_connection(
                        fabric_node_id, dst_nodes, [0], program, worker_writer_kernel_idx, worker_core
                    )
                    writer_rt_args_ref.extend(fabric_args)

                mesh_program_descriptor[ttnn.MeshCoordinateRange(coord, coord)] = program
                print("Created program for device at row {}, col {}".format(row, col))

        # Execute generic_op on mesh
        print(
            "About to execute generic_op on mesh with {} tensors".format(
                len(
                    [
                        input_tensor_mesh,
                        intermediate_tensor_mesh,
                        gamma_tensor,
                        matmul_weights_tensor,
                        rmsnorm2_gamma_tensor,
                        matmul2_weights_tensor,
                        output_tensor,
                    ]
                )
            )
        )
        print("mesh_program_descriptor created successfully")
        result = ttnn.generic_op(
            [
                input_tensor_mesh,
                intermediate_tensor_mesh,
                gamma_tensor,
                matmul_weights_tensor,
                rmsnorm2_gamma_tensor,
                matmul2_weights_tensor,
                output_tensor,
            ],
            mesh_program_descriptor,
        )

        return result
