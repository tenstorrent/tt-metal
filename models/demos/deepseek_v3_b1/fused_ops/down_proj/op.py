# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Down Projection (W_down) fused operation.

Implements Mcast1 + Mcast2 + Matmul + ResidualAdd + Gather as a fused execution on 112 cores:
- Mcast1: Input [1, K] on (12,9) -> broadcast to 130-core grid (13x10)
- Mcast2: Add input [1, N] on (12,9) -> broadcast to 130-core grid (13x10)
- Matmul: [1, K] x [K, N_per_core] -> [1, N_per_core] on 112 matmul cores
- Residual add: matmul_out + shard(residual) -> [1, N_per_core] on 112 matmul cores
- Gather: Collect [1, N_per_core] from 112 scattered cores to [1, N] on (12,9)

Core Layout (13x10 = 130-core mcast grid):
    D = DRAM worker (8 cores) — receives mcast semaphore, skips matmul & gather
    P = Phantom (9 cores, col 12 rows 0-8) — same as DRAM worker
    M = Mcast sender + Gather receiver (12,9)
    R = Matmul core (112 cores)

CB Layout:
- CB 0: mcast_src (input on (12,9), tensor-backed)
- CB 1: mcast_dst / matmul_in0 (all 130 cores, receives mcast data)
- CB 2: matmul_in1 (weights on 112 matmul cores, tensor-backed)
- CB 3: matmul_out (112 matmul cores, intermediate consumed by add)
- CB 4: gather_dst (output on (12,9), tensor-backed)
- CB 5: residual_add_mcast_src (residual input on (12,9), tensor-backed)
- CB 6: residual_add_mcast_dst (all 130 cores, receives mcast2 data)
- CB 7: residual_add_out (112 matmul cores, matmul_out + shard(residual))
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class DownProj:
    """
    Down Projection fused operation using ttnn.generic_op.

    Implements: Mcast1 -> Mcast2 -> Matmul -> ResidualAdd -> Gather on a 13x10 grid with 112 active matmul cores.
    """

    # DRAM worker positions (logical coordinates) — cores that sit on DRAM banks
    DRAM_WORKER_POSITIONS = [(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)]

    # Full mcast grid dimensions
    MCAST_GRID_X = 13
    MCAST_GRID_Y = 10

    # Mcast sender / Gather receiver core
    MCAST_GATHER_CORE = ttnn.CoreCoord(12, 9)

    # Number of active matmul cores
    NUM_MATMUL_CORES = 112

    @staticmethod
    def build_matmul_core_grid():
        """
        Build CoreRangeSet for 112 matmul cores.

        From the 13x10 = 130 core grid, exclude:
        - 8 DRAM workers at known positions
        - 9 phantoms at col 12, rows 0-8
        - 1 mcast/gather core at (12, 9)

        Returns contiguous row segments as CoreRanges for efficiency.
        """
        excluded = set(DownProj.DRAM_WORKER_POSITIONS)
        # Add phantoms (col 12, rows 0-8) and mcast/gather core (12, 9)
        for row in range(10):
            excluded.add((12, row))

        # Collect all non-excluded cores
        all_matmul_cores = []
        for row in range(DownProj.MCAST_GRID_Y):
            for col in range(DownProj.MCAST_GRID_X):
                if (col, row) not in excluded:
                    all_matmul_cores.append((col, row))

        assert (
            len(all_matmul_cores) == DownProj.NUM_MATMUL_CORES
        ), f"Expected {DownProj.NUM_MATMUL_CORES} matmul cores, got {len(all_matmul_cores)}"

        # Build contiguous row segments
        core_ranges = []
        for row in range(DownProj.MCAST_GRID_Y):
            row_cores = [(c, r) for c, r in all_matmul_cores if r == row]
            if not row_cores:
                continue
            row_cores.sort()
            seg_start = row_cores[0][0]
            prev_col = seg_start
            for i in range(1, len(row_cores)):
                col = row_cores[i][0]
                if col != prev_col + 1:
                    core_ranges.append(
                        ttnn.CoreRange(
                            ttnn.CoreCoord(seg_start, row),
                            ttnn.CoreCoord(prev_col, row),
                        )
                    )
                    seg_start = col
                prev_col = col
            core_ranges.append(
                ttnn.CoreRange(
                    ttnn.CoreCoord(seg_start, row),
                    ttnn.CoreCoord(prev_col, row),
                )
            )

        return ttnn.CoreRangeSet(core_ranges)

    @staticmethod
    def build_mcast_receiver_grid():
        """
        Build CoreRangeSet for mcast receivers: all 130 cores minus sender (12,9).
        """
        mcast_gather_core = DownProj.MCAST_GATHER_CORE
        ranges = []
        for row in range(DownProj.MCAST_GRID_Y):
            for col in range(DownProj.MCAST_GRID_X):
                if col == mcast_gather_core.x and row == mcast_gather_core.y:
                    continue
                ranges.append(ttnn.CoreRange(ttnn.CoreCoord(col, row), ttnn.CoreCoord(col, row)))
        return ttnn.CoreRangeSet(ranges)

    @staticmethod
    def golden(input_tensor, weights_tensor, add_input_tensor):
        """
        PyTorch reference implementation.

        Args:
            input_tensor: [1, K] torch.Tensor
            weights_tensor: [K, N] torch.Tensor
            add_input_tensor: [1, N] torch.Tensor

        Returns:
            [1, N] torch.Tensor
        """
        return input_tensor @ weights_tensor + add_input_tensor

    @staticmethod
    def op(input_tensor, weights_tensor, output_tensor, add_input_tensor, fp32_dest_acc_en=False):
        """
        Execute down projection fused operation using generic_op.

        Args:
            input_tensor: Input [1, K] HEIGHT_SHARDED on (12,9)
            weights_tensor: Weights [K, N] WIDTH_SHARDED on 112 matmul cores
            output_tensor: Output [1, N] HEIGHT_SHARDED on (12,9)
            add_input_tensor: Add input [1, N] HEIGHT_SHARDED on (12,9)
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with result
        """
        device = input_tensor.device()
        data_format = input_tensor.dtype

        # Tile definitions
        TILE_1x32 = ttnn.Tile((1, 32))
        tile_1x32_size = TILE_1x32.get_tile_size(data_format)

        # ====================================================================
        # Core grid configuration
        # ====================================================================
        mcast_gather_core = DownProj.MCAST_GATHER_CORE
        mcast_gather_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

        # Full 13x10 mcast grid
        mcast_grid = ttnn.CoreRange(
            ttnn.CoreCoord(0, 0),
            ttnn.CoreCoord(DownProj.MCAST_GRID_X - 1, DownProj.MCAST_GRID_Y - 1),
        )
        mcast_grid_set = ttnn.CoreRangeSet([mcast_grid])
        num_mcast_cores = DownProj.MCAST_GRID_X * DownProj.MCAST_GRID_Y  # 130

        # 112 matmul cores (excludes DRAM workers, phantoms, mcast/gather core)
        matmul_core_grid = DownProj.build_matmul_core_grid()

        # Mcast receiver grid = all 130 minus sender (12,9)
        mcast_receiver_grid = DownProj.build_mcast_receiver_grid()

        # All cores = full mcast grid (sender is already included)
        all_cores = mcast_grid_set

        # Get NOC coordinates
        gather_dest_noc_core = device.worker_core_from_logical_core(mcast_gather_core)
        mcast_dest_noc_start = device.worker_core_from_logical_core(mcast_grid.start)
        mcast_dest_noc_end = device.worker_core_from_logical_core(mcast_grid.end)

        # ====================================================================
        # Dimension parameters
        # ====================================================================
        input_tile = input_tensor.get_tile()
        k_num_tiles = input_tensor.shape[1] // input_tile.tile_shape[1]

        weights_shard_spec = weights_tensor.memory_config().shard_spec
        n_per_core = weights_shard_spec.shape[1]
        out_w_per_core = n_per_core // TILE_1x32.tile_shape[1]

        input_tile_size = input_tile.get_tile_size(data_format)

        # ====================================================================
        # CB indices
        # ====================================================================
        mcast_src_cb = 0  # Input on (12,9)
        mcast_dst_cb = 1  # Mcast destination = matmul in0 (all 130 cores)
        matmul_in1_cb = 2  # Matmul weights (112 matmul cores)
        matmul_out_cb = 3  # Matmul output (112 matmul cores)
        gather_dst_cb = 4  # Gather output (12,9)
        residual_add_mcast_src_cb = 5  # Residual input on (12,9)
        residual_add_mcast_dst_cb = 6  # Residual input after mcast2 (all 130 cores)
        residual_add_out_cb = 7  # Residual add output (112 matmul cores)

        # ====================================================================
        # Mcast parameters
        # ====================================================================
        mcast_data_size_bytes = k_num_tiles * input_tile_size
        mcast_is_part_of_receiver_grid = mcast_grid.contains(mcast_gather_core)  # True

        # ====================================================================
        # Mcast2 parameters (add input)
        # ====================================================================
        total_residual_add_tiles = DownProj.NUM_MATMUL_CORES * out_w_per_core  # = N / 32
        residual_add_mcast_data_size_bytes = total_residual_add_tiles * tile_1x32_size

        # ====================================================================
        # Gather parameters
        # ====================================================================
        gather_data_size_bytes = out_w_per_core * tile_1x32_size
        gather_src_num_pages = out_w_per_core
        gather_dst_num_pages = DownProj.NUM_MATMUL_CORES * out_w_per_core
        gather_noc0_num_senders = DownProj.NUM_MATMUL_CORES
        gather_noc1_num_senders = 0

        # ====================================================================
        # Semaphore IDs
        # ====================================================================
        mcast_data_sender_semaphore_id = 0
        mcast_data_receiver_semaphore_id = 1
        gather_noc0_receiver_semaphore_id = 2
        gather_noc1_receiver_semaphore_id = 3
        mcast2_data_receiver_semaphore_id = 4

        # ====================================================================
        # Buffer addresses
        # ====================================================================
        gather_receiver_data_addr = output_tensor.buffer_address()

        # ====================================================================
        # Per-core gather indices for 112 scattered matmul cores
        # ====================================================================
        matmul_cores_list = ttnn.corerange_to_cores(matmul_core_grid)
        per_core_gather_idx = PerCoreCompileTimeDescriptor(
            named_compile_time_arg="gather_sender_idx",
            core_values=[(core, idx) for idx, core in enumerate(matmul_cores_list)],
            other_value=0,
        )

        # ====================================================================
        # NCRISC compile-time args
        # ====================================================================
        ncrisc_named_compile_time_args = [
            # Mcast1 source (for setup_sharded_buffer on sender core)
            ("mcast_src_cb", mcast_src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            # Mcast1 receiver
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_dst_num_pages", k_num_tiles),
            # Mcast2 source (for setup_sharded_buffer on sender core)
            ("mcast2_src_cb", residual_add_mcast_src_cb),
            ("mcast2_src_num_pages", total_residual_add_tiles),
            # Mcast2 receiver
            ("mcast2_data_receiver_semaphore", mcast2_data_receiver_semaphore_id),
            ("mcast2_dst_cb", residual_add_mcast_dst_cb),
            ("mcast2_dst_num_pages", total_residual_add_tiles),
            # Matmul
            ("matmul_in0", mcast_dst_cb),
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", k_num_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
            # Residual add (needed for ResidualAdd CTArgs template parameter)
            ("residual_add_out_w", out_w_per_core),
            # Gather sender (now reads from residual_add_out_cb instead of matmul_out_cb)
            ("gather_dest_noc_x", gather_dest_noc_core.x),
            ("gather_dest_noc_y", gather_dest_noc_core.y),
            ("gather_data_size_bytes", gather_data_size_bytes),
            ("gather_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_src_cb", residual_add_out_cb),
            ("gather_src_num_pages", gather_src_num_pages),
            ("gather_sender_grid_start_x", 0),
            ("gather_sender_grid_start_y", 0),
            ("gather_sender_grid_end_x", 0),
            ("gather_sender_grid_end_y", 0),
            ("gather_row_major", 1),
            ("gather_receiver_data_addr", gather_receiver_data_addr),
        ]

        # ====================================================================
        # BRISC compile-time args
        # ====================================================================
        brisc_named_compile_time_args = [
            # Mcast1 sender
            ("mcast_dest_noc_start_x", mcast_dest_noc_start.x),
            ("mcast_dest_noc_start_y", mcast_dest_noc_start.y),
            ("mcast_dest_noc_end_x", mcast_dest_noc_end.x),
            ("mcast_dest_noc_end_y", mcast_dest_noc_end.y),
            ("mcast_num_cores", num_mcast_cores),
            ("mcast_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast_data_receiver_semaphore", mcast_data_receiver_semaphore_id),
            ("mcast_data_size_bytes", mcast_data_size_bytes),
            ("mcast_src_cb", mcast_src_cb),
            ("mcast_src_num_pages", k_num_tiles),
            ("mcast_dst_cb", mcast_dst_cb),
            ("mcast_is_part_of_receiver_grid", mcast_is_part_of_receiver_grid),
            # Mcast2 sender
            ("mcast2_data_sender_semaphore", mcast_data_sender_semaphore_id),
            ("mcast2_data_receiver_semaphore", mcast2_data_receiver_semaphore_id),
            ("mcast2_data_size_bytes", residual_add_mcast_data_size_bytes),
            ("mcast2_src_cb", residual_add_mcast_src_cb),
            ("mcast2_src_num_pages", total_residual_add_tiles),
            ("mcast2_dst_cb", residual_add_mcast_dst_cb),
            # Residual add (needed for ResidualAdd CTArgs template parameter)
            ("residual_add_out_w", out_w_per_core),
            # Gather receiver
            ("gather_noc0_num_senders", gather_noc0_num_senders),
            ("gather_noc1_num_senders", gather_noc1_num_senders),
            ("gather_noc0_receiver_semaphore_id", gather_noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", gather_noc1_receiver_semaphore_id),
            ("gather_dst_cb", gather_dst_cb),
            ("gather_dst_num_pages", gather_dst_num_pages),
        ]

        # ====================================================================
        # TRISC compile-time args
        # ====================================================================
        trisc_named_compile_time_args = [
            ("matmul_in0", mcast_dst_cb),
            ("matmul_in1", matmul_in1_cb),
            ("matmul_out", matmul_out_cb),
            ("matmul_k_num_tiles", k_num_tiles),
            ("matmul_out_w_per_core", out_w_per_core),
            # Residual add step
            ("residual_add_in0", matmul_out_cb),
            ("residual_add_in1", residual_add_mcast_dst_cb),
            ("residual_add_out", residual_add_out_cb),
            ("residual_add_out_w", out_w_per_core),
            ("residual_add_total_in1_tiles", total_residual_add_tiles),
        ]

        # ====================================================================
        # Circular buffer descriptors
        # ====================================================================
        # CB 0: Mcast source — input on (12,9), tensor-backed
        mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(mcast_src_cb, input_tensor)

        # CB 1: Mcast destination — on all 130 cores (including sender for get_write_ptr)
        mcast_dst_tile_descriptor = ttnn.TileDescriptor(TILE_1x32)
        mcast_dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=mcast_dst_cb,
            data_format=data_format,
            page_size=input_tile_size,
            tile=mcast_dst_tile_descriptor,
        )
        mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=k_num_tiles * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[mcast_dst_cb_format],
        )

        # CB 2: Matmul weights — tensor-backed on 112 matmul cores
        matmul_in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(matmul_in1_cb, weights_tensor)

        # CB 3: Matmul output — on 112 matmul cores
        matmul_out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=matmul_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=mcast_dst_tile_descriptor,
        )
        matmul_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * tile_1x32_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[matmul_out_cb_format],
        )

        # CB 4: Gather destination — output on (12,9), tensor-backed
        gather_dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(gather_dst_cb, output_tensor)

        # CB 5: Mcast2 source — residual input on (12,9), tensor-backed
        residual_add_mcast_src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            residual_add_mcast_src_cb, add_input_tensor
        )

        # CB 6: Mcast2 destination — on all 130 cores (full [1, N] residual input)
        residual_add_mcast_dst_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=residual_add_mcast_dst_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=mcast_dst_tile_descriptor,
        )
        residual_add_mcast_dst_cb_descriptor = ttnn.CBDescriptor(
            total_size=total_residual_add_tiles * tile_1x32_size,
            core_ranges=all_cores,
            format_descriptors=[residual_add_mcast_dst_cb_format],
        )

        # CB 7: Residual add output — on 112 matmul cores
        residual_add_out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=residual_add_out_cb,
            data_format=data_format,
            page_size=tile_1x32_size,
            tile=mcast_dst_tile_descriptor,
        )
        residual_add_out_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_w_per_core * tile_1x32_size,
            core_ranges=matmul_core_grid,
            format_descriptors=[residual_add_out_cb_format],
        )

        # ====================================================================
        # Semaphore descriptors
        # ====================================================================
        full_device_grid = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        device.compute_with_storage_grid_size().x - 1,
                        device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            ]
        )

        semaphore_descriptors = [
            ttnn.SemaphoreDescriptor(id=mcast_data_sender_semaphore_id, core_ranges=full_device_grid, initial_value=0),
            ttnn.SemaphoreDescriptor(
                id=mcast_data_receiver_semaphore_id, core_ranges=full_device_grid, initial_value=0
            ),
            ttnn.SemaphoreDescriptor(
                id=gather_noc0_receiver_semaphore_id, core_ranges=full_device_grid, initial_value=0
            ),
            ttnn.SemaphoreDescriptor(
                id=gather_noc1_receiver_semaphore_id, core_ranges=full_device_grid, initial_value=0
            ),
            ttnn.SemaphoreDescriptor(
                id=mcast2_data_receiver_semaphore_id, core_ranges=full_device_grid, initial_value=0
            ),
        ]

        # ====================================================================
        # Kernel descriptor
        # ====================================================================
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/down_proj/kernels/down_proj_kernel.cpp",
            core_ranges=all_cores,
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
                    named_compile_time_arg="is_mcast_sender_core",
                    core_range=mcast_gather_core_grid,
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
                    named_compile_time_arg="is_gather_receiver_core",
                    core_range=mcast_gather_core_grid,
                    value=1,
                    other_value=0,
                ),
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="gather_use_per_core_sender_idx",
                    core_range=matmul_core_grid,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[per_core_gather_idx],
        )

        # ====================================================================
        # Program descriptor
        # ====================================================================
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                mcast_src_cb_descriptor,
                mcast_dst_cb_descriptor,
                matmul_in1_cb_descriptor,
                matmul_out_cb_descriptor,
                gather_dst_cb_descriptor,
                residual_add_mcast_src_cb_descriptor,
                residual_add_mcast_dst_cb_descriptor,
                residual_add_out_cb_descriptor,
            ],
            semaphores=semaphore_descriptors,
        )

        # Execute generic op
        # Order must match tensor-backed CBs in cbs list: CB0(input), CB2(weights), CB4(output), CB5(add_input)
        io_tensors = [input_tensor, weights_tensor, output_tensor, add_input_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)
        return output_tensor
