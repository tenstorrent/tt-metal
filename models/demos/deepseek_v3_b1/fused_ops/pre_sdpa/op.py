# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_bfloat16_packed, float_to_uint32


class PreSDPA:
    """
    Pre-SDPA fused operation implementation using ttnn.generic_op.

    This class implements the pre-SDPA operations as a fused execution:
    - RMSNorm on a single core
    - Multicast of the result to a grid of cores
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
        input_tensor,
        gamma_tensor,
        matmul_weights_tensor,
        rmsnorm2_gamma_tensor,
        matmul2_weights_tensor,
        output_tensor,
        epsilon=1e-6,
        fp32_dest_acc_en=False,
    ):
        """
        Execute pre-SDPA fused operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded on single core)
            gamma_tensor: Gamma/weight tensor (must be sharded, same shape as input)
            matmul_weights_tensor: Matmul weights tensor (must be width sharded)
            rmsnorm2_gamma_tensor: Gamma tensor for second RMSNorm (1536 elements = 3 tiles of 16x32)
            matmul2_weights_tensor: Matmul2 weights tensor (width sharded, 4 tiles per core)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            epsilon: Small value to avoid division by zero
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with RMSNorm applied
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        data_format = input_tensor.dtype

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

        # Get number of elements for RMS calculation
        numel = input_tensor.logical_volume()

        # Get core grid from input tensor's memory config
        input_memory_config = input_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        input_core_ranges = list(input_core_grid.ranges())
        rmsnorm_core = input_core_ranges[0].start
        rmsnorm_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(rmsnorm_core, rmsnorm_core)])

        # Get full device grid
        device = input_tensor.device()
        device_grid_size = device.compute_with_storage_grid_size()
        full_device_grid = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(device_grid_size.x - 1, device_grid_size.y - 1))]
        )

        # Get matmul weights core grid (48 cores for width sharding)
        matmul_weights_memory_config = matmul_weights_tensor.memory_config()
        matmul_weights_core_grid = matmul_weights_memory_config.shard_spec.grid

        # Calculate per-core width in tiles for matmul1 (from shard spec)
        # Get shard width directly from shard_spec and divide by tile width from tensor
        matmul_weights_tile = matmul_weights_tensor.get_tile()
        matmul_weights_shard_shape = matmul_weights_memory_config.shard_spec.shape
        matmul_weights_shard_width = matmul_weights_shard_shape[1]  # Width dimension
        matmul1_out_w = matmul_weights_shard_width // matmul_weights_tile.tile_shape[1]  # Per-core width in tiles

        # Calculate per-core width in tiles for matmul2 (from shard spec)
        matmul2_weights_memory_config = matmul2_weights_tensor.memory_config()
        matmul2_weights_core_grid = matmul2_weights_memory_config.shard_spec.grid
        matmul2_weights_tile = matmul2_weights_tensor.get_tile()
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
        scalar_packed = float_to_bfloat16_packed(inv_sqrt_numel)

        # Define circular buffer page size
        cb_page_size = tile_size

        # CB indices
        input_cb = 0
        scalars_cb = 1
        interm_cb = 2
        gamma_cb = 3
        rmsnorm_output_cb = 4
        matmul_weights_cb = 5
        matmul_output_cb = 9
        matmul_input_cb = 8
        rmsnorm2_gamma_cb = 10  # New gamma for second RMSNorm (1536 elements = 3 tiles of 16x32)
        rmsnorm2_input_cb = 11  # Separate input CB for RMSNorm2
        rmsnorm2_interm_cb = 12  # Separate interm CB for RMSNorm2
        rmsnorm2_output_cb = 13  # Separate output CB for RMSNorm2
        matmul2_input_cb = 14  # Input CB for second matmul (1x1536 with 1x32 tiles)
        matmul2_weights_cb = 15  # Weights CB for second matmul (width sharded, 4 tiles per core)
        matmul2_output_cb = 16  # Output CB for second matmul

        # RMSNorm2 parameters (for 1536 element input using 16x32 tiles)
        rmsnorm2_numel = 1536
        rmsnorm2_num_tiles = 3  # 3 tiles of 16x32 = 3 * 512 = 1536 elements
        rmsnorm2_num_faces = 2  # 16x32 tiles have 2 faces

        # Compute 1/sqrt(1536) for RMSNorm2 reduction
        inv_sqrt_rmsnorm2_numel = 1.0 / math.sqrt(float(rmsnorm2_numel))
        scalar2_packed = float_to_bfloat16_packed(inv_sqrt_rmsnorm2_numel)

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
            ("rmsnorm_scalars_cb", scalars_cb),
            ("rmsnorm_gamma_cb", gamma_cb),
            ("rmsnorm_num_tiles", num_tiles),
            ("rmsnorm_num_faces", interpreted_tile.num_faces),
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
            ("rmsnorm_scalars_cb", scalars_cb),
            ("rmsnorm_interm_cb", interm_cb),
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
            ("rmsnorm2_interm_cb", rmsnorm2_interm_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
            ("rmsnorm2_num_tiles", rmsnorm2_num_tiles),
            ("rmsnorm2_num_faces", rmsnorm2_num_faces),
        ]
        rmsnorm2_trisc_named_compile_time_args = [
            ("rmsnorm2_input_cb", rmsnorm2_input_cb),
            ("rmsnorm2_interm_cb", rmsnorm2_interm_cb),
            ("rmsnorm2_gamma_cb", rmsnorm2_gamma_cb),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),
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

        # Create circular buffer descriptors
        # CB 0: Input (created from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        # Update the tile descriptor in the format descriptor
        in_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        in_cb_descriptor.format_descriptors[0].page_size = cb_page_size

        # CB 1: Scalars (reduction scalar only, epsilon passed as runtime arg to compute)
        scalars_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=scalars_cb,
            data_format=data_format,
            page_size=cb_page_size,
            tile=tile_descriptor,
        )
        scalars_cb_descriptor = ttnn.CBDescriptor(
            total_size=cb_page_size,
            core_ranges=rmsnorm_core_grid,
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
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[interm_cb_format],
        )

        # CB 3: Gamma (created from sharded tensor)
        gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gamma_cb, gamma_tensor)
        # Update the tile descriptor in the format descriptor
        gamma_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        gamma_cb_descriptor.format_descriptors[0].page_size = cb_page_size

        # RMSNorm2 uses separate CBs with exact sizes (16x32 tiles)
        TILE_16x32 = ttnn.Tile((16, 32))
        rmsnorm2_tile_descriptor = ttnn.TileDescriptor(TILE_16x32)
        rmsnorm2_page_size = TILE_16x32.get_tile_size(data_format)

        # CB 10: RMSNorm2 Gamma (created from sharded tensor, 3 tiles of 16x32)
        rmsnorm2_gamma_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(rmsnorm2_gamma_cb, rmsnorm2_gamma_tensor)
        # Update the tile descriptor in the format descriptor to match rmsnorm2 tile shape
        rmsnorm2_gamma_cb_descriptor.format_descriptors[0].tile = rmsnorm2_tile_descriptor
        rmsnorm2_gamma_cb_descriptor.format_descriptors[0].page_size = rmsnorm2_page_size

        # CB 11: RMSNorm2 input buffer (3 tiles)
        rmsnorm2_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=rmsnorm2_input_cb,
            data_format=data_format,
            page_size=rmsnorm2_page_size,
            tile=rmsnorm2_tile_descriptor,
        )
        # Must be allocated on union of matmul cores and rmsnorm core for gather to get write_ptr
        rmsnorm2_input_cb_core_ranges = matmul_weights_core_grid.merge(rmsnorm_core_grid)
        rmsnorm2_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,  # 3 tiles
            core_ranges=rmsnorm2_input_cb_core_ranges,
            format_descriptors=[rmsnorm2_input_cb_format],
        )

        # CB 12: RMSNorm2 intermediate buffer (num_tiles = 3 tiles)
        rmsnorm2_interm_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=rmsnorm2_interm_cb,
            data_format=data_format,
            page_size=rmsnorm2_page_size,
            tile=rmsnorm2_tile_descriptor,
        )
        rmsnorm2_interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,  # 3 tiles
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[rmsnorm2_interm_cb_format],
        )

        # CB 13: RMSNorm2 output buffer (3 tiles)
        rmsnorm2_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=rmsnorm2_output_cb,
            data_format=data_format,
            page_size=rmsnorm2_page_size,
            tile=rmsnorm2_tile_descriptor,
        )
        rmsnorm2_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=rmsnorm2_num_tiles * rmsnorm2_page_size,  # 3 tiles
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[rmsnorm2_output_cb_format],
        )

        # CB 4: RMSNorm output buffer (dynamically created)
        rmsnorm_output_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=rmsnorm_output_cb,
            data_format=data_format,
            page_size=cb_page_size,
            tile=tile_descriptor,
        )
        rmsnorm_output_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_tiles * cb_page_size,
            core_ranges=rmsnorm_core_grid,
            format_descriptors=[rmsnorm_output_cb_format],
        )

        # CB 5: Matmul weights (created from sharded tensor) - not used yet
        matmul_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_weights_cb, matmul_weights_tensor)

        # CB 8: Matmul input buffer (1x32 tiles, receives mcast data)
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
        matmul_input_cb_core_ranges = matmul_weights_core_grid.merge(rmsnorm_core_grid)
        matmul_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=matmul_input_total_size,
            core_ranges=matmul_input_cb_core_ranges,
            format_descriptors=[matmul_input_cb_format],
        )

        # CB 9: Matmul output buffer (single tile, on matmul cores only)
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

        # CB 14: Matmul2 input buffer (1x1536 with 1x32 tiles = 48 tiles)
        # Must be allocated on union of sender (rmsnorm input grid) and receiver (matmul2 grid)
        # Similar constraint as gather CB - senders query write_ptr to get receiver address
        matmul2_input_total_size = matmul2_num_tiles_k * matmul_input_page_size  # 48 * 64 bytes
        matmul2_input_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul2_input_cb,
            data_format=data_format,
            page_size=matmul_input_page_size,
            tile=matmul_input_tile_descriptor,
        )
        matmul2_input_cb_core_ranges = ttnn.CoreRangeSet([main_grid]).merge(rmsnorm_core_grid)
        matmul2_input_cb_descriptor = ttnn.CBDescriptor(
            total_size=matmul2_input_total_size,
            core_ranges=matmul2_input_cb_core_ranges,
            format_descriptors=[matmul2_input_cb_format],
        )

        # CB 15: Matmul2 weights (created from sharded tensor, 4 tiles per core)
        matmul2_weights_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            matmul2_weights_cb, matmul2_weights_tensor
        )

        # CB 16: Matmul2 output buffer (width sharded, mapped to output_tensor)
        matmul2_output_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul2_output_cb, output_tensor)

        # ========================================================================
        # Mcast2 compile-time args (uses same grid and semaphores as first mcast)
        # ========================================================================
        # BRISC sender: data_size_bytes, src_num_pages, rmsnorm2_output_cb (grid/semaphores reused from mcast)
        mcast2_brisc_named_compile_time_args = [
            ("mcast2_data_size_bytes", mcast2_data_size_bytes),
            ("mcast2_src_num_pages", mcast2_src_num_pages),
            ("rmsnorm2_output_cb", rmsnorm2_output_cb),  # Source CB for mcast2 sender
        ]
        # NCRISC receiver: dst_num_pages (semaphore reused from mcast)
        mcast2_ncrisc_named_compile_time_args = [
            ("mcast2_dst_num_pages", mcast2_dst_num_pages),
        ]

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================

        # Mcast semaphores (ID 0 and 1)
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

        # Gather semaphores (ID 2 and 3 - two semaphores for NOC0 and NOC1, but only NOC0 is used)
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

        # ========================================================================
        # Kernel descriptors
        # ========================================================================

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/pre_sdpa/kernels/pre_sdpa_kernel.cpp",
            core_ranges=full_device_grid,
            # NCRISC named compile-time args: rmsnorm reader + mcast receiver + matmul + gather sender + rmsnorm2 + matmul2 + mcast2
            ncrisc_named_compile_time_args=rmsnorm_reader_named_compile_time_args
            + mcast_receiver_named_compile_time_args
            + matmul_ncrisc_named_compile_time_args
            + gather_sender_named_compile_time_args
            + rmsnorm2_ncrisc_named_compile_time_args
            + matmul2_ncrisc_named_compile_time_args
            + mcast2_ncrisc_named_compile_time_args,
            # NCRISC common runtime args: scalar + scalar2
            ncrisc_common_runtime_args=[
                scalar_packed,
                scalar2_packed,  # scalar for rmsnorm2 (1/sqrt(1536))
            ],
            # BRISC named compile-time args: mcast sender + matmul + gather receiver + matmul2 + mcast2
            brisc_named_compile_time_args=mcast_sender_named_compile_time_args
            + matmul_brisc_named_compile_time_args
            + gather_receiver_named_compile_time_args
            + matmul2_brisc_named_compile_time_args
            + mcast2_brisc_named_compile_time_args,
            # TRISC named compile-time args: rmsnorm compute + matmul + rmsnorm2 + matmul2
            trisc_named_compile_time_args=rmsnorm_compute_named_compile_time_args
            + matmul_trisc_named_compile_time_args
            + rmsnorm2_trisc_named_compile_time_args
            + matmul2_trisc_named_compile_time_args,
            # TRISC common runtime args: epsilon (used by rmsnorm compute)
            trisc_common_runtime_args=[
                epsilon_packed,
            ],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            # Per-core compile-time role differentiation
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_input_core",
                    core_range=rmsnorm_core,  # First core is the input core
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_weights_core_grid,  # 48 matmul cores
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul2_core",
                    core_range=matmul2_weights_core_grid,  # matmul2 cores
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                in_cb_descriptor,
                scalars_cb_descriptor,
                interm_cb_descriptor,
                gamma_cb_descriptor,
                rmsnorm_output_cb_descriptor,
                matmul_weights_cb_descriptor,
                matmul_output_cb_descriptor,
                matmul_input_cb_descriptor,
                rmsnorm2_gamma_cb_descriptor,  # CB 10: RMSNorm2 gamma
                rmsnorm2_input_cb_descriptor,  # CB 11: RMSNorm2 input
                rmsnorm2_interm_cb_descriptor,  # CB 12: RMSNorm2 interm
                rmsnorm2_output_cb_descriptor,  # CB 13: RMSNorm2 output
                matmul2_input_cb_descriptor,  # CB 14: Matmul2 input
                matmul2_weights_cb_descriptor,  # CB 15: Matmul2 weights
                matmul2_output_cb_descriptor,  # CB 16: Matmul2 output
            ],
            semaphores=[
                mcast_sender_semaphore_descriptor,  # ID 0
                mcast_receiver_semaphore_descriptor,  # ID 1
                gather_noc0_receiver_semaphore_descriptor,  # ID 2
                gather_noc1_receiver_semaphore_descriptor,  # ID 3
            ],
        )

        # Execute generic op
        io_tensors = [
            input_tensor,
            gamma_tensor,
            matmul_weights_tensor,
            rmsnorm2_gamma_tensor,
            matmul2_weights_tensor,
            output_tensor,
        ]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
