# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Output Hidden fused operation.

This implements Matmul + Gather as a fused execution:
- Matmul: [1, 8192] x [8192, 6144] -> [1, 6144] distributed across 96 cores (8x12 grid)
- Gather: Collect results from all 96 cores to a single output core

Each matmul core computes: [1, 8192] x [8192, 64] -> [1, 64] (2 tiles of 1x32)
Total output: 96 cores * 64 = 6144 elements
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class OutputHidden:
    """
    Output Hidden fused operation implementation using ttnn.generic_op.

    This class implements matmul followed by gather as a fused execution:
    - Input is already sharded/available on all matmul cores
    - Matmul computes [1, 8192] x [8192, 64] per core
    - Gather collects all results to core (9, 11)
    """

    @staticmethod
    def golden(input_tensor, weights_tensor):
        """
        PyTorch reference implementation for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) [1, 8192]
            weights_tensor: Weights tensor (torch.Tensor) [8192, 6144]

        Returns:
            Output tensor [1, 6144]
        """
        return input_tensor @ weights_tensor

    @staticmethod
    def op(
        input_tensor,
        weights_tensor,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute output_hidden fused operation using generic_op.

        Args:
            input_tensor: Input tensor [1, 8192] (height-sharded across matmul cores)
            weights_tensor: Weights tensor [8192, 6144] (width-sharded across 96 cores)
            output_tensor: Pre-allocated output tensor [1, 6144] (height-sharded on gather core)
            fp32_dest_acc_en: Whether to enable FP32 accumulation in compute kernel

        Returns:
            Output tensor with matmul + gather result
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
        # Matmul grid: 8x12 = 96 cores
        MATMUL_GRID_START_X = 0
        MATMUL_GRID_START_Y = 0
        MATMUL_GRID_END_X = 11  # 12 columns (0-11)
        MATMUL_GRID_END_Y = 7  # 8 rows (0-7)
        matmul_grid = ttnn.CoreRange(
            ttnn.CoreCoord(MATMUL_GRID_START_X, MATMUL_GRID_START_Y),
            ttnn.CoreCoord(MATMUL_GRID_END_X, MATMUL_GRID_END_Y),
        )
        matmul_core_grid = ttnn.CoreRangeSet([matmul_grid])
        num_matmul_cores = matmul_grid.grid_size().x * matmul_grid.grid_size().y  # 96

        # Gather receiver core: (9, 11)
        gather_core = ttnn.CoreCoord(9, 11)
        gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(gather_core, gather_core)])

        # Full grid (union of matmul and gather cores for semaphore allocation)
        full_grid = matmul_core_grid.merge(gather_core_grid)

        # Get NOC coordinates for gather destination
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        # ========================================================================
        # Matmul parameters
        # ========================================================================
        # Input shape: [1, 8192] = 256 tiles of 1x32
        # Weights per core: [8192, 64] = 256 * 2 tiles of 32x32
        # Output per core: [1, 64] = 2 tiles of 1x32
        matmul_k_num_tiles = 256  # 8192 / 32 = 256 tiles
        matmul_out_w_per_core = 2  # 64 / 32 = 2 tiles per core

        # CB indices
        matmul_in0_cb = 0  # Input (1x32 tiles)
        matmul_in1_cb = 1  # Weights (32x32 tiles)
        matmul_out_cb = 2  # Output (1x32 tiles)
        gather_dst_cb = 3  # Gathered output

        # ========================================================================
        # Gather parameters
        # ========================================================================
        # Each matmul core sends 2 tiles of 1x32 = 128 bytes (64 bfloat16 elements)
        gather_data_size_bytes = matmul_out_w_per_core * tile_1x32_size
        gather_src_num_pages = matmul_out_w_per_core  # 2 pages per sender
        gather_dst_num_pages = num_matmul_cores * matmul_out_w_per_core  # 96 * 2 = 192 pages

        # All senders use NOC_0 (default for NCRISC)
        gather_noc0_num_senders = num_matmul_cores
        gather_noc1_num_senders = 0

        # Semaphore IDs
        gather_noc0_receiver_semaphore_id = 0
        gather_noc1_receiver_semaphore_id = 1

        # ========================================================================
        # Compile-time args
        # ========================================================================
        # NCRISC: matmul reader + gather sender
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
            ("gather_dst_cb", gather_dst_cb),
        ]

        # BRISC: matmul writer (no-op) + gather receiver
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

        # CB 2: Matmul output (2 tiles of 1x32 per core)
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
        # Must be visible on both sender and receiver grids for get_write_ptr
        gather_dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=gather_dst_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=matmul_out_tile_descriptor,
        )
        gather_dst_cb_core_ranges = matmul_core_grid.merge(gather_core_grid)
        gather_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=gather_dst_num_pages * tile_1x32_size,
            core_ranges=gather_dst_cb_core_ranges,
            format_descriptors=[gather_dst_cb_format],
        )

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

        # ========================================================================
        # Kernel descriptor
        # ========================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/output_hidden/kernels/output_hidden_kernel.cpp",
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
            ],
            semaphores=[
                gather_noc0_semaphore_descriptor,
                gather_noc1_semaphore_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, weights_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
