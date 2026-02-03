# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Output Hidden Post SDPA fused operation.

This implements Matmul + Gather + Mcast as a fused execution:
- Matmul: [1, 512] x [512, 128] -> [1, 128] distributed across 64 cores (8x8 grid)
- Gather: Collect results from all 64 cores to a single output core (11, 9)
- Mcast: Broadcast gathered result [1, 8192] to 8x12 grid of cores

Each matmul core computes: [1, 512] x [512, 128] -> [1, 128] (4 tiles of 1x32)
Total gathered output: 64 cores * 128 = 8192 elements
Mcast broadcasts to: 8x12 = 96 cores
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class OutputHiddenPostSDPA:
    """
    Output Hidden Post SDPA fused operation implementation using ttnn.generic_op.

    This class implements matmul followed by gather and mcast as a fused execution:
    - Input is sharded across 64 cores (8 unique shards, each replicated to 8 cores)
    - Each core gets [1, 512] input
    - Matmul computes [1, 512] x [512, 128] per core
    - Gather collects all results to core (11, 9)
    - Mcast broadcasts [1, 8192] to 8x12 grid
    """

    @staticmethod
    def golden(input_tensor, weights_tensor):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, 512]
            weights_tensor: Weights tensor (torch.Tensor) [512, 8192]

        Returns:
            Output tensor [1, 8192]
        """
        return input_tensor @ weights_tensor

    @staticmethod
    def op(
        input_tensor,
        weights_tensor,
        gather_output_tensor,
        mcast_output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute output_hidden_post_sdpa fused operation using generic_op.

        Args:
            input_tensor: Input tensor [1, 512] (height-sharded across matmul cores)
            weights_tensor: Weights tensor [512, 8192] (width-sharded across 64 cores)
            gather_output_tensor: Intermediate tensor [1, 8192] for gather output (on gather core)
            mcast_output_tensor: Final output tensor [1, 8192] (sharded on mcast grid, 96 cores)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with matmul + gather + mcast result
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
        # Matmul grid: 8x8 = 64 cores
        MATMUL_GRID_START_X = 0
        MATMUL_GRID_START_Y = 0
        MATMUL_GRID_END_X = 7  # 8 columns (0-7)
        MATMUL_GRID_END_Y = 7  # 8 rows (0-7)
        matmul_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MATMUL_GRID_START_X, MATMUL_GRID_START_Y),
            ttnn.CoreCoord(MATMUL_GRID_END_X, MATMUL_GRID_END_Y),
        )
        matmul_core_grid = ttnn.CoreRangeSet([matmul_grid])
        num_matmul_cores = matmul_grid.grid_size().x * matmul_grid.grid_size().y  # 64

        # Gather receiver core: (11, 9)
        gather_core = ttnn.CoreCoord(11, 9)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # Mcast grid: 8x12 = 96 cores (x=0-7, y=0-11)
        MCAST_GRID_START_X = 0
        MCAST_GRID_START_Y = 0
        MCAST_GRID_END_X = 11  # 12 columns (0-11)
        MCAST_GRID_END_Y = 7  # 8 rows (0-7)
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MCAST_GRID_START_X, MCAST_GRID_START_Y),
            ttnn.CoreCoord(MCAST_GRID_END_X, MCAST_GRID_END_Y),
        )
        mcast_core_grid = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = mcast_grid.grid_size().x * mcast_grid.grid_size().y  # 96

        # Full grid (union of all cores for semaphore allocation)
        full_grid = matmul_core_grid.merge(gather_core_grid).merge(mcast_core_grid)

        # Get NOC coordinates for gather destination
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        # Get NOC coordinates for mcast destination grid
        mcast_dest_noc_start_core = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end_core = device.worker_core_from_logical_core(mcast_grid.end)

        # ========================================================================
        # Matmul parameters
        # ========================================================================
        # Input shape: [1, 512] = 16 tiles of 1x32
        # Weights per core: [512, 128] = 16 * 4 tiles of 32x32
        # Output per core: [1, 128] = 4 tiles of 1x32
        matmul_k_num_tiles = 16  # 512 / 32 = 16 tiles
        matmul_out_w_per_core = 4  # 128 / 32 = 4 tiles per core

        # CB indices
        matmul_in0_cb = 0  # Input (1x32 tiles)
        matmul_in1_cb = 1  # Weights (32x32 tiles)
        matmul_out_cb = 2  # Output (1x32 tiles)
        gather_dst_cb = 3  # Gathered output (= mcast source on gather core)
        mcast_dst_cb = 4  # Mcast destination (on mcast grid)

        # ========================================================================
        # Gather parameters
        # ========================================================================
        # Each matmul core sends 4 tiles of 1x32 = 256 bytes (128 bfloat16 elements)
        gather_data_size_bytes = matmul_out_w_per_core * tile_1x32_size
        gather_src_num_pages = matmul_out_w_per_core  # 4 pages per sender
        gather_dst_num_pages = num_matmul_cores * matmul_out_w_per_core  # 64 * 4 = 256 pages

        # All senders use NOC_0 (default for NCRISC)
        gather_noc0_num_senders = num_matmul_cores
        gather_noc1_num_senders = 0

        # ========================================================================
        # Mcast parameters
        # ========================================================================
        # Mcast broadcasts [1, 8192] = 256 tiles of 1x32 to each core in mcast grid
        mcast_data_size_bytes = gather_dst_num_pages * tile_1x32_size  # 256 * 64 = 16384 bytes
        mcast_src_num_pages = gather_dst_num_pages  # 256 pages (from gather output)
        mcast_dst_num_pages = gather_dst_num_pages  # 256 pages per receiver

        # Gather core (11, 9) is NOT in mcast grid (0-7, 0-11), so no loopback needed
        mcast_is_part_of_receiver_grid = mcast_grid.contains(gather_core)

        # ========================================================================
        # Semaphore IDs
        # ========================================================================
        gather_noc0_receiver_semaphore_id = 0
        gather_noc1_receiver_semaphore_id = 1
        mcast_data_sender_semaphore_id = 2
        mcast_data_receiver_semaphore_id = 3

        # ========================================================================
        # Compile-time args
        # ========================================================================
        # Get buffer addresses for data movement
        gather_receiver_data_addr = gather_output_tensor.buffer_address()
        mcast_receiver_data_addr = mcast_output_tensor.buffer_address()

        # NCRISC: matmul reader + gather sender + mcast receiver
        ncrisc_named_compile_time_args = [
            # Matmul
            ("matmul_in0", matmul_in0_cb),
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", matmul_k_num_tiles),
            ("matmul_out_w_per_core", matmul_out_w_per_core),
            # Gather sender
            ("gather_dest_noc_x", gather_dest_noc_core.x),
            ("gather_dest_noc_y", gather_dest_noc_core.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", matmul_out_cb),
            ("gather_src_num_pages", gather_src_num_pages),
            ("gather_sender_grid_start_x", MATMUL_GRID_START_X),
            ("gather_sender_grid_start_y", MATMUL_GRID_START_Y),
            ("gather_sender_grid_end_x", MATMUL_GRID_END_X),
            ("gather_sender_grid_end_y", MATMUL_GRID_END_Y),
            ("gather_row_major", 1),
            ("gather_receiver_data_addr", gather_receiver_data_addr),
            # Mcast receiver
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_dst_num_pages", mcast_dst_num_pages),
        ]

        # BRISC: matmul writer (no-op) + gather receiver + mcast sender
        brisc_named_compile_time_args = [
            # Matmul (no-op, but need the CB index)
            ("matmul_out", matmul_out_cb),
            # Gather receiver
            ("gather_noc0_num_senders", gather_noc0_num_senders),
            ("gather_noc1_num_senders", gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", gather_dst_cb),
            ("gather_dst_num_pages", gather_dst_num_pages),
            # Mcast sender (source = gather_dst_cb)
            ("mcast_dest_noc_start_x", mcast_dest_noc_start_core.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start_core.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end_core.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end_core.y),
            ("mcast_num_cores", num_mcast_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", gather_dst_cb),  # Mcast reads from gather output
            ("mcast_src_num_pages", mcast_src_num_pages),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
        ]

        # TRISC: matmul compute
        trisc_named_compile_time_args = [
            ("matmul_in0", matmul_in0_cb),
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", matmul_k_num_tiles),
            ("matmul_out_w_per_core", matmul_out_w_per_core),
        ]

        # ========================================================================
        # Circular buffer descriptors
        # ========================================================================
        # CB 0: Matmul input (from sharded tensor)
        matmul_in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in0_cb, input_tensor)

        # CB 1: Matmul weights (from sharded tensor)
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, weights_tensor)

        # CB 2: Matmul output (4 tiles of 1x32 per core, on matmul cores)
        matmul_out_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
        matmul_out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=matmul_out_tile_descriptor,
        )
        matmul_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=matmul_out_w_per_core * tile_1x32_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[matmul_out_cb_format],
        )

        # CB 3: Gather output (from sharded tensor, on gather core only)
        # This is also the mcast source
        gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gather_dst_cb, gather_output_tensor)

        # CB 4: Mcast destination (from sharded tensor, on mcast grid)
        mcast_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(mcast_dst_cb, mcast_output_tensor)

        # ========================================================================
        # Semaphore descriptors
        # ========================================================================
        gather_noc0_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc0_receiver_semaphore_id,
            core_ranges=full_grid,
            initial_value=0,
        )

        gather_noc1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=gather_noc1_receiver_semaphore_id,
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

        # ========================================================================
        # Kernel descriptor
        # ========================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/output_hidden_post_sdpa/kernels/output_hidden_post_sdpa_kernel.cpp",
            core_ranges=full_grid,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            # BRISC runtime arg: mcast_receiver_data_addr (output tensor buffer address)
            brisc_common_runtime_args=[mcast_receiver_data_addr],
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_matmul_core",
                    core_range=matmul_core_grid,
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
                    named_compile_time_arg="is_mcast_receiver_core",
                    core_range=mcast_core_grid,
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
                matmul_in0_cb_descriptor,
                matmul_in1_cb_descriptor,
                matmul_out_cb_descriptor,
                gather_dst_cb_descriptor,
                mcast_dst_cb_descriptor,
            ],
            semaphores=[
                gather_noc0_semaphore_descriptor,
                gather_noc1_semaphore_descriptor,
                mcast_sender_semaphore_descriptor,
                mcast_receiver_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, weights_tensor, gather_output_tensor, mcast_output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
