# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Post SDPA fused operation.

This implements Matmul1 + Gather1 + Mcast + Matmul2 + Gather2 as a fused execution:
- Matmul1: [1, 512] x [512, 128] -> [1, 128] distributed across 64 cores (8x8 grid)
- Gather1: Collect results from all 64 cores to [1, 8192] on gather core (11, 9)
- Mcast: Broadcast [1, 8192] to 96 cores (12x8 grid)
- Matmul2: [1, 8192] x [8192, 64] -> [1, 64] distributed across 96 cores (12x8 grid)
- Gather2: Collect results from all 96 cores to [1, 6144] on gather core (11, 9)

CB Layout:
- CB 0: matmul1_in0 (8x8 grid)
- CB 1: matmul1_in1 (8x8 grid)
- CB 2: matmul1_out (8x8 grid)
- CB 3: gather1_dst = mcast_src (gather core)
- CB 4: mcast_dst = matmul2_in0 (12x8 grid)
- CB 5: matmul2_in1 (12x8 grid)
- CB 6: matmul2_out (12x8 grid)
- CB 7: gather2_dst (gather core)
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class PostSDPA:
    """
    Post SDPA fused operation implementation using ttnn.generic_op.

    Implements the full post_sdpa fusion:
    - Matmul1 + Gather1 + Mcast + Matmul2 + Gather2
    """

    @staticmethod
    def golden(input_tensor, weights1_tensor, weights2_tensor):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, 512]
            weights1_tensor: First weights tensor (torch.Tensor) [512, 8192]
            weights2_tensor: Second weights tensor (torch.Tensor) [8192, 6144]

        Returns:
            Output tensor [1, 6144]
        """
        intermediate = input_tensor @ weights1_tensor  # [1, 8192]
        return intermediate @ weights2_tensor  # [1, 6144]

    @staticmethod
    def op(
        input_tensor,
        weights1_tensor,
        weights2_tensor,
        gather1_output_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute post_sdpa fused operation using generic_op.

        Args:
            input_tensor: Input tensor [1, 512] (height-sharded across 8x8 matmul1 cores)
            weights1_tensor: First weights tensor [512, 8192] (width-sharded across 8x8)
            weights2_tensor: Second weights tensor [8192, 6144] (width-sharded across 12x8)
            gather1_output_tensor: Intermediate tensor [1, 8192] for gather1/mcast (on gather core)
            output_tensor: Final output tensor [1, 6144] (on gather core)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with full fused result
        """
        # Get device
        device = input_tensor.device()

        # Get tensor properties
        data_format = input_tensor.dtype

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        TILE_32x32 = ttnn.Tile((32, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)
        tile_32x32_size = TILE_32x32.get_tile_size(data_format)

        # ========================================================================
        # Core grid configuration
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

        # Gather receiver core: (11, 9)
        gather_core = ttnn.CoreCoord(11, 9)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # Matmul2/Mcast grid: 12x8 = 96 cores
        MATMUL2_GRID_START_X = 0
        MATMUL2_GRID_START_Y = 0
        MATMUL2_GRID_END_X = 11  # 12 columns (0-11)
        MATMUL2_GRID_END_Y = 7  # 8 rows (0-7)
        matmul2_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MATMUL2_GRID_START_X, MATMUL2_GRID_START_Y),
            ttnn.CoreCoord(MATMUL2_GRID_END_X, MATMUL2_GRID_END_Y),
        )
        matmul2_core_grid = ttnn.CoreRangeSet([matmul2_grid])
        num_matmul2_cores = matmul2_grid.grid_size().x * matmul2_grid.grid_size().y  # 96

        # Full grid (union of all cores for semaphore allocation)
        full_grid = matmul1_core_grid.merge(gather_core_grid).merge(matmul2_core_grid)

        # Get NOC coordinates
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(matmul2_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(matmul2_grid.end)

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
        matmul2_in0_cb = 4  # Mcast dst = Matmul2 input (12x8)
        matmul2_in1_cb = 5  # Matmul2 weights (12x8)
        matmul2_out_cb = 6  # Matmul2 output (12x8)
        gather2_dst_cb = 7  # Gather2 output = final output (gather core)

        # ========================================================================
        # Gather1 parameters: 64 cores -> [1, 8192]
        # ========================================================================
        gather1_data_size_bytes = matmul1_out_w_per_core * tile_1x32_size
        gather1_src_num_pages = matmul1_out_w_per_core  # 4 pages per sender
        gather1_dst_num_pages = num_matmul1_cores * matmul1_out_w_per_core  # 64 * 4 = 256 pages
        gather1_noc0_num_senders = num_matmul1_cores
        gather1_noc1_num_senders = 0

        # ========================================================================
        # Mcast parameters: [1, 8192] to 96 cores
        # ========================================================================
        mcast_data_size_bytes = gather1_dst_num_pages * tile_1x32_size  # 256 * 64 = 16384 bytes
        mcast_src_num_pages = gather1_dst_num_pages  # 256 pages
        mcast_dst_num_pages = gather1_dst_num_pages  # 256 pages per receiver
        mcast_is_part_of_receiver_grid = matmul2_grid.contains(gather_core)

        # ========================================================================
        # Gather2 parameters: 96 cores -> [1, 6144]
        # ========================================================================
        gather2_data_size_bytes = matmul2_out_w_per_core * tile_1x32_size
        gather2_src_num_pages = matmul2_out_w_per_core  # 2 pages per sender
        gather2_dst_num_pages = num_matmul2_cores * matmul2_out_w_per_core  # 96 * 2 = 192 pages
        gather2_noc0_num_senders = num_matmul2_cores
        gather2_noc1_num_senders = 0

        # ========================================================================
        # Semaphore IDs
        # ========================================================================
        gather1_noc0_receiver_semaphore_id = 0
        gather1_noc1_receiver_semaphore_id = 1
        mcast_data_sender_semaphore_id = 2
        mcast_data_receiver_semaphore_id = 3
        gather2_noc0_receiver_semaphore_id = 4
        gather2_noc1_receiver_semaphore_id = 5

        # ========================================================================
        # Buffer addresses
        # ========================================================================
        gather1_receiver_data_addr = gather1_output_tensor.buffer_address()
        mcast_receiver_data_addr = weights2_tensor.buffer_address()  # Actually mcast writes to matmul2_in0
        gather2_receiver_data_addr = output_tensor.buffer_address()

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
            ("mcast_num_cores", num_matmul2_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", gather1_dst_cb),
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_dst_cb", matmul2_in0_cb),  # For get_write_ptr on sender
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
            # Gather2 receiver
            ("gather2_noc0_num_senders", gather2_noc0_num_senders),
            ("gather2_noc1_num_senders", gather2_noc1_num_senders),
            ("gather2_noc0_receiver_semaphore_id", gather2_noc0_receiver_semaphore_id),
            ("gather2_noc1_receiver_semaphore_id", gather2_noc1_receiver_semaphore_id),
            ("gather2_dst_cb", gather2_dst_cb),
            ("gather2_dst_num_pages", gather2_dst_num_pages),
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
        ]

        # ========================================================================
        # Circular buffer descriptors
        # ========================================================================
        # CB 0: Matmul1 input (from sharded tensor, 8x8 grid)
        matmul1_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul1_in0_cb, input_tensor)

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
        gather1_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gather1_dst_cb, gather1_output_tensor)

        # CB 4: Mcast destination = Matmul2 input (256 tiles of 1x32 per core)
        # Allocated on matmul2 grid AND gather core (so sender can use get_write_ptr)
        matmul2_in0_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul2_in0_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=matmul1_out_tile_descriptor,
        )
        matmul2_in0_cb_grid = matmul2_core_grid.merge(gather_core_grid)
        matmul2_in0_cb_descriptor = ttnn.CBDescriptor(
            total_size=mcast_dst_num_pages * tile_1x32_size,
            core_ranges=matmul2_in0_cb_grid,
            format_descriptors=[matmul2_in0_cb_format],
        )

        # CB 5: Matmul2 weights (from sharded tensor, 12x8 grid)
        matmul2_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul2_in1_cb, weights2_tensor)

        # CB 6: Matmul2 output (2 tiles of 1x32 per core, 12x8 grid)
        matmul2_out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul2_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=matmul1_out_tile_descriptor,
        )
        matmul2_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=matmul2_out_w_per_core * tile_1x32_size,
            core_ranges=matmul2_core_grid,
            format_descriptors=[matmul2_out_cb_format],
        )

        # CB 7: Gather2 output = final output (from sharded tensor, gather core)
        gather2_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gather2_dst_cb, output_tensor)

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
                math_fidelity=ttnn.MathFidelity.LoFi,
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
                    core_range=matmul2_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        # ========================================================================
        # Program descriptor
        # ========================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[
                matmul1_in0_cb_descriptor,
                matmul1_in1_cb_descriptor,
                matmul1_out_cb_descriptor,
                gather1_dst_cb_descriptor,
                matmul2_in0_cb_descriptor,
                matmul2_in1_cb_descriptor,
                matmul2_out_cb_descriptor,
                gather2_dst_cb_descriptor,
            ],
            semaphores=[
                gather1_noc0_semaphore_descriptor,
                gather1_noc1_semaphore_descriptor,
                mcast_sender_semaphore_descriptor,
                mcast_receiver_semaphore_descriptor,
                gather2_noc0_semaphore_descriptor,
                gather2_noc1_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, weights1_tensor, weights2_tensor, gather1_output_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
