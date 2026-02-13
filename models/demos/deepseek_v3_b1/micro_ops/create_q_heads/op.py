# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import torch

import ttnn


class CreateQHeads:
    """
    Create Q Heads implementation using ttnn.generic_op.

    This class gathers data from 12x8 sender cores to 4x2 receiver cores, then tilizes.
    Each sender row (8 rows total) maps to a different receiver core:
      - Row 0 → (0, 1), Row 1 → (1, 1), Row 2 → (2, 1), Row 3 → (3, 1)
      - Row 4 → (0, 2), Row 5 → (1, 2), Row 6 → (2, 2), Row 7 → (3, 2)

    Memory layout for tilization (max tilize dim = 256):
      - Phase 1: First 8 halves of QNOPE - shape [8, 256] → 8 tiles
      - Phase 2: Second 8 halves of QNOPE - shape [8, 256] → 8 tiles
      - Phase 3: QROPE - shape [8, 64] → 2 tiles

    Sender writes:
      - QNOPE cores (cols 0-7): Split 512 elements into two 256-element halves
        - First half → tight row-major at offset = col * 256 * elem_size
        - Second half → tight row-major at offset = (8*256 + col*256) * elem_size
        - Signals semaphore after each half (2 signals total)
      - QROPE cores (cols 8-11): Write 2 heads × 64 elements = 128 elements
        - Offset = (8*512) * elem_size + 2*qrope_col*64*elem_size
        - Signals semaphore once

    RISC assignment (matching pre_sdpa gather pattern):
      - All senders use NCRISC (RISCV_1)
      - All receivers use BRISC (RISCV_0)
    """

    # Mapping from sender row index to target receiver core
    SENDER_ROW_TO_TARGET_CORE = {
        0: ttnn.CoreCoord(0, 1),
        1: ttnn.CoreCoord(1, 1),
        2: ttnn.CoreCoord(2, 1),
        3: ttnn.CoreCoord(3, 1),
        4: ttnn.CoreCoord(0, 2),
        5: ttnn.CoreCoord(1, 2),
        6: ttnn.CoreCoord(2, 2),
        7: ttnn.CoreCoord(3, 2),
    }

    @staticmethod
    def golden(qnope_input, qrope_input, qnope_grid, qrope_grid, receiver_grid):
        """
        PyTorch reference implementation of create Q heads.

        Args:
            qnope_input: (8, 4096) tensor - BLOCK_SHARDED across 8x8 grid with shard (1, 512)
            qrope_input: (16, 256) tensor - BLOCK_SHARDED across 8x4 grid with shard (2, 64)
            qnope_grid: 8x8 grid
            qrope_grid: 4x8 grid
            receiver_grid: 4x2 grid

        Returns:
            Output tensor matching kernel's phase-based layout (same as kernel writes):
            Phase 1: all first halves  [head0 first 256, head1 first 256, ..., head7 first 256] = 2048 elements
            Phase 2: all second halves [head0 second 256, ..., head7 second 256] = 2048 elements
            Phase 3: all rope          [head0 rope 64, ..., head7 rope 64] = 512 elements
            Total 4608 elements per receiver, reshaped to (8, 576) row-major.
            Output shape: (16, 2304) for 8 receivers = (8, 576) per receiver.
        """
        # Extract dimensions
        qnope_rows = qnope_grid.end.y - qnope_grid.start.y + 1  # 8
        qnope_cols = qnope_grid.end.x - qnope_grid.start.x + 1  # 8
        qrope_rows = qrope_grid.end.y - qrope_grid.start.y + 1  # 8
        qrope_cols = qrope_grid.end.x - qrope_grid.start.x + 1  # 4
        receiver_rows = receiver_grid.end.y - receiver_grid.start.y + 1  # 2
        receiver_cols = receiver_grid.end.x - receiver_grid.start.x + 1  # 4

        # Per-core shard sizes (from tensor shapes and grid)
        qnope_shard_h = qnope_input.shape[0] // qnope_rows  # 1
        qnope_shard_w = qnope_input.shape[1] // qnope_cols  # 512
        qrope_shard_h = qrope_input.shape[0] // qrope_rows  # 2
        qrope_shard_w = qrope_input.shape[1] // qrope_cols  # 64

        half_qnope_size = qnope_shard_w // 2  # 256

        head_elements = qnope_shard_w + qrope_shard_w  # 576
        num_heads_per_receiver = qnope_cols  # 8

        output = torch.zeros(
            receiver_rows * num_heads_per_receiver, receiver_cols * head_elements, dtype=qnope_input.dtype
        )

        for ry_idx in range(receiver_rows):
            for rx_idx in range(receiver_cols):
                sender_row = rx_idx if ry_idx == 0 else (rx_idx + receiver_cols)
                out_row_start = ry_idx * num_heads_per_receiver
                out_col_start = rx_idx * head_elements

                # Phase-based layout to match kernel: Phase 1 (all first halves), Phase 2 (all second halves), Phase 3 (all rope)
                phase1 = []  # head0 first 256, head1 first 256, ..., head7 first 256
                phase2 = []  # head0 second 256, ..., head7 second 256
                phase3 = []  # head0 rope 64, ..., head7 rope 64
                for head_idx in range(num_heads_per_receiver):
                    qnope_col = head_idx
                    qnope_row_start = sender_row * qnope_shard_h
                    qnope_row_end = qnope_row_start + qnope_shard_h
                    qnope_col_start = qnope_col * qnope_shard_w
                    qnope_col_end = qnope_col_start + qnope_shard_w
                    qnope_data = qnope_input[qnope_row_start:qnope_row_end, qnope_col_start:qnope_col_end].flatten()
                    phase1.append(qnope_data[:half_qnope_size])
                    phase2.append(qnope_data[half_qnope_size:])
                    qrope_col_idx = head_idx // qrope_shard_h
                    qrope_head = head_idx % qrope_shard_h
                    qrope_row_start = sender_row * qrope_shard_h + qrope_head
                    qrope_col_start = qrope_col_idx * qrope_shard_w
                    qrope_col_end = qrope_col_start + qrope_shard_w
                    phase3.append(qrope_input[qrope_row_start, qrope_col_start:qrope_col_end])

                # After tilization, each row of the output corresponds to one head.
                # Tilize treats each phase's data as [8, W] row-major, where row i = head i's data.
                # Output tiles are appended sequentially: phase1 tiles (cols 0-255), phase2 tiles (cols 256-511),
                # phase3 tiles (cols 512-575). When untilized, row i = [head_i_first256, head_i_second256, head_i_rope].
                for head_idx in range(num_heads_per_receiver):
                    row_data = torch.cat([phase1[head_idx], phase2[head_idx], phase3[head_idx]])
                    output[out_row_start + head_idx, out_col_start : out_col_start + head_elements] = row_data

        return output.reshape(1, -1)

    @staticmethod
    def op(qnope_tensor, qrope_tensor, interm_tensor, output_tensor):
        """
        Execute gather heads operation using generic_op.

        Args:
            qnope_tensor: Qnope output tensor after matmul3 (sharded across 8x8 cores, [1, 512] per core)
            qrope_tensor: Qrope output tensor after RoPE (sharded across 4x8 cores, [2, 64] per core)
            interm_tensor: Intermediate tensor for row-major data (sharded across 4x2 receiver cores)
            output_tensor: Output tensor (sharded across 4x2 receiver cores, [8, 576] per core)

        Returns:
            Output tensor with input data gathered from all 12x8 cores to 4x2 cores
        """
        device = qnope_tensor.device()

        # Get memory configs and core grids
        qnope_memory_config = qnope_tensor.memory_config()
        qrope_memory_config = qrope_tensor.memory_config()
        qnope_core_grid = qnope_memory_config.shard_spec.grid
        qrope_core_grid = qrope_memory_config.shard_spec.grid
        output_memory_config = output_tensor.memory_config()

        # Sender grid is merged qnope and qrope grids (12x8 = 96 cores)
        sender_core_grid = qrope_core_grid.merge(qnope_core_grid)
        # Receiver grid is the output tensor's grid (4x2 = 8 cores)
        receiver_core_grid = output_memory_config.shard_spec.grid

        # Get sender grid dimensions for offset computation in kernel
        # Compute bounding box across all ranges (merge might produce multiple ranges)
        sender_core_ranges = list(sender_core_grid.ranges())
        sender_grid_start_x = min(r.start.x for r in sender_core_ranges)
        sender_grid_start_y = min(r.start.y for r in sender_core_ranges)
        sender_grid_end_x = max(r.end.x for r in sender_core_ranges)
        sender_grid_width = sender_grid_end_x - sender_grid_start_x + 1  # 12 columns

        # Calculate data sizes from tensor shard shapes
        # Qnope: [1, 512] per core after matmul3
        # Qrope: [2, 64] per core (2 heads of 64 elements each)
        qnope_shard_shape = qnope_memory_config.shard_spec.shape
        qrope_shard_shape = qrope_memory_config.shard_spec.shape

        # Qnope size per core (512 elements)
        qnope_elements = qnope_shard_shape[0] * qnope_shard_shape[1]  # 512
        # Qrope has 2 heads per core, each head is 64 elements
        qrope_head_elements = qrope_shard_shape[1]  # 64

        dtype = qnope_tensor.dtype
        if dtype == ttnn.bfloat16:
            element_size_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # Data sizes in bytes
        qnope_data_size_bytes = qnope_elements * element_size_bytes  # 512 * 2 = 1024 bytes
        qrope_head_size_bytes = qrope_head_elements * element_size_bytes  # 64 * 2 = 128 bytes

        # Head stride: each head is qnope (512) + qrope (64) = 576 elements
        head_elements = qnope_elements + qrope_head_elements  # 576
        head_stride_bytes = head_elements * element_size_bytes  # 576 * 2 = 1152 bytes

        # Compute qnope columns from qnope grid
        qnope_core_ranges = list(qnope_core_grid.ranges())
        qnope_grid_start_x = min(r.start.x for r in qnope_core_ranges)
        qnope_grid_end_x = max(r.end.x for r in qnope_core_ranges)
        qnope_cols = qnope_grid_end_x - qnope_grid_start_x + 1

        # Compute sender and receiver grid dimensions
        sender_grid_end_y = max(r.end.y for r in sender_core_ranges)
        sender_grid_height = sender_grid_end_y - sender_grid_start_y + 1
        receiver_core_ranges = list(receiver_core_grid.ranges())
        receiver_grid_start_x = min(r.start.x for r in receiver_core_ranges)
        receiver_grid_start_y = min(r.start.y for r in receiver_core_ranges)
        receiver_grid_end_x = max(r.end.x for r in receiver_core_ranges)
        receiver_cols = receiver_grid_end_x - receiver_grid_start_x + 1

        # Build sender row to receiver core mapping dynamically
        # Pattern: receiver at (rx, ry) gets data from sender row = rx if ry==0, else rx + receiver_cols
        sender_row_to_target_core = {}
        receiver_cores_list = ttnn.corerange_to_cores(receiver_core_grid, row_wise=True)
        for receiver_core in receiver_cores_list:
            rx = receiver_core.x - receiver_grid_start_x
            ry = receiver_core.y - receiver_grid_start_y
            sender_row = rx if ry == 0 else (rx + receiver_cols)
            if sender_row < sender_grid_height:
                sender_row_to_target_core[sender_row] = receiver_core

        # Convert sender grid to list of cores
        sender_cores_list = ttnn.corerange_to_cores(sender_core_grid, row_wise=True)

        # Get target NOC coordinates for each row (convert logical to physical/NOC coordinates)
        target_noc_coords = {}
        for row, target_logical_core in sender_row_to_target_core.items():
            target_noc_core = device.worker_core_from_logical_core(target_logical_core)
            target_noc_coords[row] = (target_noc_core.x, target_noc_core.y)

        # All senders use NCRISC (matching pre_sdpa gather pattern: NCRISC sender, BRISC receiver)
        # Count senders per receiver row for semaphore initialization
        senders_per_receiver = {row: 0 for row in range(sender_grid_height)}
        for core in sender_cores_list:
            senders_per_receiver[core.y - sender_grid_start_y] += 1

        # All cores for semaphore allocation
        all_cores = sender_core_grid.merge(receiver_core_grid)

        # Semaphore IDs for 3-phase synchronization (race-free)
        # These IDs are chosen to match pre_sdpa semaphore allocation for reuse:
        #   - nope_phase1_semaphore_id = 2 (reuses gather_noc0_receiver_semaphore_id)
        #   - nope_phase2_semaphore_id = 3 (reuses gather_noc1_receiver_semaphore_id)
        #   - rope_semaphore_id = 0 (reuses mcast_data_sender_semaphore_id)
        nope_phase1_semaphore_id = 2  # QNOPE senders signal after first half
        nope_phase2_semaphore_id = 3  # QNOPE senders signal after second half
        rope_semaphore_id = 0  # QROPE senders signal after completion

        # Create semaphore descriptors
        nope_phase1_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=nope_phase1_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )
        nope_phase2_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=nope_phase2_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )
        rope_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=rope_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        # CB indices
        qnope_cb = 0  # Qnope output (input for this op) - sender source
        qrope_cb = 1  # Qrope output (input for this op) - sender source
        receiver_in_cb = 2  # Receiver input CB (row-major data written by senders)
        out_cb = 3  # Output CB (tilized data)

        # Number of pages
        src_num_pages = 1  # Each sender has one page (its shard)
        dst_num_pages = sender_grid_width  # Each receiver gets pages (one per sender in the row)

        # Tile counts for tilization phases
        # [8, 256] → 1 tile row × 8 tile cols = 8 tiles
        nope_tiles = 8
        # [8, 64] → 1 tile row × 2 tile cols = 2 tiles
        rope_tiles = 2

        receiver_data_addr = interm_tensor.buffer_address()

        # Kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/create_q_heads/kernels/create_q_heads_kernel.cpp"

        kernels = []
        semaphores = [
            rope_semaphore_descriptor,  # ID 0
            nope_phase1_semaphore_descriptor,  # ID 2
            nope_phase2_semaphore_descriptor,  # ID 3
        ]

        # ========================================================================
        # Common sender compile-time args (shared by NOC0 and NOC1 senders)
        # Uses 3 separate semaphores for race-free synchronization
        # ========================================================================
        def create_sender_named_compile_time_args():
            args = [
                # Sender role flags (all senders use NCRISC, matching pre_sdpa pattern)
                ("is_sender_core", 1),
                ("is_receiver_core", 0),
                # Sender grid info for offset computation
                ("sender_grid_start_x", sender_grid_start_x),
                ("sender_grid_start_y", sender_grid_start_y),
                # Data sizes
                ("qnope_data_size_bytes", qnope_data_size_bytes),
                ("qrope_head_size_bytes", qrope_head_size_bytes),
                ("head_stride_bytes", head_stride_bytes),
                ("qnope_cols", qnope_cols),
                # CB indices
                ("qnope_cb", qnope_cb),
                ("qrope_cb", qrope_cb),
                ("src_num_pages", src_num_pages),
                # Semaphores (3 separate for race-free synchronization)
                ("nope_phase1_semaphore_id", nope_phase1_semaphore_id),
                ("nope_phase2_semaphore_id", nope_phase2_semaphore_id),
                ("rope_semaphore_id", rope_semaphore_id),
            ]
            # Add target NOC coordinates for each row (packed: x in lower 16 bits, y in upper 16 bits)
            # Kernel expects args for all 8 rows (hardcoded array size), so always pass all 8
            for row in range(8):
                if row in target_noc_coords:
                    noc_x, noc_y = target_noc_coords[row]
                    packed_coords = noc_x | (noc_y << 16)
                    args.append((f"target_noc_coords_row{row}", packed_coords))
                else:
                    # Fill missing rows with dummy coordinates (0,0)
                    args.append((f"target_noc_coords_row{row}", 0))
            return args

        # ========================================================================
        # Sender kernel (NCRISC) - all senders use NCRISC (matching pre_sdpa pattern)
        # ========================================================================
        sender_kernel = ttnn.KernelDescriptor(
            kernel_source=kernel_path,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=sender_core_grid,
            named_compile_time_args=create_sender_named_compile_time_args(),
            common_runtime_args=[receiver_data_addr],
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1,  # NCRISC
                noc=ttnn.NOC.NOC_0,
            ),
        )
        kernels.append(sender_kernel)

        # ========================================================================
        # Receiver kernel (BRISC) - all receivers use BRISC (matching pre_sdpa pattern)
        # ========================================================================
        def create_receiver_named_compile_time_args():
            """Create receiver compile-time args (all receivers use BRISC)."""
            args = [
                # All receiver cores are also sender cores (receiver grid is within sender grid)
                ("is_sender_core", 1),
                ("is_receiver_core", 1),
                # Semaphores (3 separate for race-free synchronization)
                ("nope_phase1_semaphore_id", nope_phase1_semaphore_id),
                ("nope_phase2_semaphore_id", nope_phase2_semaphore_id),
                ("rope_semaphore_id", rope_semaphore_id),
                ("receiver_in_cb", receiver_in_cb),
                ("out_cb", out_cb),
                ("dst_num_pages", dst_num_pages),
                ("nope_tiles", nope_tiles),
                ("rope_tiles", rope_tiles),
                ("sender_grid_width", sender_grid_width),
                ("receiver_grid_start_x", receiver_grid_start_x),
                ("receiver_grid_start_y", receiver_grid_start_y),
                ("receiver_cols", receiver_cols),
                # Number of QNOPE senders (8 cols) and QROPE senders (4 cols) per receiver row
                ("num_nope_senders", qnope_cols),
                ("num_rope_senders", sender_grid_width - qnope_cols),
            ]
            return args

        # All receivers use BRISC (matching pre_sdpa pattern)
        if not receiver_core_grid.empty():
            receiver_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=receiver_core_grid,
                named_compile_time_args=create_receiver_named_compile_time_args(),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,  # BRISC
                    noc=ttnn.NOC.NOC_1,  # Use different NOC than sender
                ),
            )
            kernels.append(receiver_kernel)

        # ========================================================================
        # Compute kernels (TRISC) - tilization on receiver cores
        # ========================================================================
        # Sender-only cores need no-op compute kernel
        sender_only_cores = [c for c in sender_cores_list if not receiver_core_grid.contains(c)]
        if sender_only_cores:
            sender_only_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in sender_only_cores])
            sender_compute_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=sender_only_core_range_set,
                named_compile_time_args=[
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(sender_compute_kernel)

        # Receiver cores need tilize compute kernel
        if not receiver_core_grid.empty():
            receiver_compute_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=receiver_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 1),  # All receiver cores are also senders
                    ("is_receiver_core", 1),
                    ("receiver_in_cb", receiver_in_cb),
                    ("out_cb", out_cb),
                    ("nope_tiles", nope_tiles),
                    ("rope_tiles", rope_tiles),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(receiver_compute_kernel)

        # Create CB descriptors from sharded tensors
        # cb_descriptor_from_sharded_tensor binds the CB buffer to the tensor's L1 buffer,
        # so data written to tensor.buffer_address() is accessible via the CB.
        qnope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qnope_cb, qnope_tensor)
        qrope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qrope_cb, qrope_tensor)

        # Receiver input CB: bound to interm_tensor's buffer where senders write row-major data.
        # interm_tensor must be TILE_LAYOUT so page_size = tile_size (512 bytes for 8x32 bf16),
        # giving 18 tile-sized pages that match the kernel's per-phase tilize_block calls.
        receiver_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(receiver_in_cb, interm_tensor)

        # Output CB: bound to output_tensor's buffer for tilized output
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[qnope_cb_descriptor, qrope_cb_descriptor, receiver_in_cb_descriptor, out_cb_descriptor],
            semaphores=semaphores,
        )

        # Execute generic op
        io_tensors = [qnope_tensor, qrope_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
