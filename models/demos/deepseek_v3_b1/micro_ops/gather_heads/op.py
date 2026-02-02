# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn


class GatherHeads:
    """
    Gather heads implementation using ttnn.generic_op.

    This class implements gather heads from 12x8 sender cores to 4x2 receiver cores.
    Each sender row (8 rows total) maps to a different receiver core:
      - Row 0 → (0, 1), Row 1 → (1, 1), Row 2 → (2, 1), Row 3 → (3, 1)
      - Row 4 → (0, 2), Row 5 → (1, 2), Row 6 → (2, 2), Row 7 → (3, 2)

    Receiver layout (8 heads per core, each head = 576 elements):
      - Head 0: qnope[col=0] (512) + qrope[col=8, head0] (64)
      - Head 1: qnope[col=1] (512) + qrope[col=8, head1] (64)
      - Head 2: qnope[col=2] (512) + qrope[col=9, head0] (64)
      - ...
      - Head 7: qnope[col=7] (512) + qrope[col=11, head1] (64)

    Sender offsets (in elements):
      - Qnope col X (0-7): offset = X * 576 (sends 512 elements)
      - Qrope col X (8-11): sends 2 chunks of 64 elements each:
        - Head 0: offset = 512 + 2*(X-8)*576
        - Head 1: offset = 512 + (2*(X-8)+1)*576

    NOC optimization:
      - For each sender, we determine whether NOC0 or NOC1 has fewer hops to the target core
      - NOC0 senders use NCRISC (RISCV_1)
      - NOC1 senders use BRISC (RISCV_0)
      - Receiver wait runs on the RISC not used by the sender on that core
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
    def op(qnope_tensor, qrope_tensor, output_tensor, noc=None):
        """
        Execute gather heads operation using generic_op.

        Args:
            qnope_tensor: Qnope output tensor after matmul3 (sharded across 8x8 cores, [1, 512] per core)
            qrope_tensor: Qrope output tensor (sharded across 4x8 cores, [2, 64] per core)
            output_tensor: Pre-allocated output tensor (sharded on 4x2 cores)
            noc: NOC to use for gather heads (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1). If None,
                 automatically optimizes NOC routing based on hop distance for each sender core.

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

        # Classify cores into NOC0 vs NOC1 based on hop distance to their target
        # Each sender sends to exactly one target, so a core is either NOC0 or NOC1, not both
        noc0_sender_cores = []
        noc1_sender_cores = []

        # Track per-receiver counts for each NOC (each receiver corresponds to one sender row)
        noc0_senders_per_receiver = {row: 0 for row in range(sender_grid_height)}
        noc1_senders_per_receiver = {row: 0 for row in range(sender_grid_height)}

        if noc is not None:
            # User specified NOC, use it for all cores
            if noc == ttnn.NOC.NOC_0:
                noc0_sender_cores = sender_cores_list
                for core in sender_cores_list:
                    noc0_senders_per_receiver[core.y - sender_grid_start_y] += 1
            else:
                noc1_sender_cores = sender_cores_list
                for core in sender_cores_list:
                    noc1_senders_per_receiver[core.y - sender_grid_start_y] += 1
        else:
            # Auto NOC routing: Choose the NOC with better overall hop distance
            #
            # IMPORTANT: Mixed NOC0/NOC1 routing (where some senders use NOC0 and others use NOC1)
            # causes hangs and is NOT supported. Once a core is configured to use a NOC, it cannot
            # switch during kernel execution. Therefore, we must use a SINGLE NOC for all senders.
            #
            # When fusing into the mega kernel (pre_sdpa_kernel.cpp), the NOC configuration is
            # already determined by which RISC processor runs the sender code:
            # - NCRISC defaults to NOC_0
            # - BRISC defaults to NOC_1
            # The sender logic should use whichever NOC is already configured for that RISC.
            #
            total_noc0_hops = 0
            total_noc1_hops = 0
            for core in sender_cores_list:
                row_idx = core.y - sender_grid_start_y
                if row_idx in sender_row_to_target_core:
                    target_core = sender_row_to_target_core[row_idx]
                    total_noc0_hops += device.get_worker_noc_hop_distance(core, target_core, ttnn.NOC.NOC_0)
                    total_noc1_hops += device.get_worker_noc_hop_distance(core, target_core, ttnn.NOC.NOC_1)

            # Use the NOC with lower total hop count
            if total_noc0_hops <= total_noc1_hops:
                noc0_sender_cores = sender_cores_list
                for core in sender_cores_list:
                    noc0_senders_per_receiver[core.y - sender_grid_start_y] += 1
            else:
                noc1_sender_cores = sender_cores_list
                for core in sender_cores_list:
                    noc1_senders_per_receiver[core.y - sender_grid_start_y] += 1

        # Create CoreRangeSets for NOC0 and NOC1 sender cores (mutually exclusive)
        noc0_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc0_sender_cores])
        noc1_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc1_sender_cores])

        # All cores for semaphore allocation
        all_cores = sender_core_grid.merge(receiver_core_grid)

        # Semaphore IDs (separate for NOC0 and NOC1 to avoid race conditions)
        noc0_receiver_semaphore_id = 0
        noc1_receiver_semaphore_id = 1

        # Create semaphore descriptors
        noc0_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=noc0_receiver_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )
        noc1_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=noc1_receiver_semaphore_id,
            core_ranges=all_cores,
            initial_value=0,
        )

        # CB indices
        qnope_cb = 0  # Qnope output (input for this op)
        qrope_cb = 1  # Qrope output (input for this op)
        out_cb = 2  # Output

        # Number of pages
        src_num_pages = 1  # Each sender has one page (its shard)
        dst_num_pages = sender_grid_width  # Each receiver gets pages (one per sender in the row)

        # Get output tensor's buffer address for receiver data address
        receiver_data_addr = output_tensor.buffer_address()

        # Kernel path
        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/gather_heads/kernels/gather_heads_kernel.cpp"

        kernels = []
        semaphores = [noc0_receiver_semaphore_descriptor, noc1_receiver_semaphore_descriptor]

        # ========================================================================
        # Common sender compile-time args (shared by NOC0 and NOC1 senders)
        # All senders use the SAME semaphore (noc0_receiver_semaphore_id) for simplicity
        # ========================================================================
        # Dummy receiver args (needed for constexpr evaluation even in non-receiver code paths)
        # Kernel expects args for all 8 rows (hardcoded array size), so always pass all 8
        dummy_receiver_args = [
            ("noc0_receiver_semaphore_id", noc0_receiver_semaphore_id),
            ("noc1_receiver_semaphore_id", noc1_receiver_semaphore_id),
            ("out_cb", out_cb),
            ("dst_num_pages", dst_num_pages),
            ("receiver_grid_start_x", receiver_grid_start_x),
            ("receiver_grid_start_y", receiver_grid_start_y),
            ("receiver_cols", receiver_cols),
        ]
        for row in range(8):
            dummy_receiver_args.append((f"noc0_senders_row{row}", 0))
            dummy_receiver_args.append((f"noc1_senders_row{row}", 0))

        def create_sender_named_compile_time_args(is_noc0):
            args = [
                # Sender role flags
                ("is_sender_core", 1),
                ("is_receiver_core", 0),
                ("is_noc0_sender", 1 if is_noc0 else 0),
                ("is_noc1_sender", 0 if is_noc0 else 1),
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
                # Semaphore - ALL senders use the same semaphore for unified synchronization
                ("receiver_semaphore_id", noc0_receiver_semaphore_id),
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
            # Add dummy receiver args for constexpr evaluation
            args.extend(dummy_receiver_args)
            return args

        # Classify receiver cores by their sender type
        # Since receiver grid (4x2 at y=1-2) is within sender grid (12x8),
        # ALL receiver cores are also sender cores - no "pure receivers"
        receiver_cores_list = ttnn.corerange_to_cores(receiver_core_grid, row_wise=True)
        noc0_sender_receiver_cores = [c for c in receiver_cores_list if noc0_core_range_set.contains(c)]
        noc1_sender_receiver_cores = [c for c in receiver_cores_list if noc1_core_range_set.contains(c)]

        # Sender-only cores (not receivers) - these need no-op on the other RISC
        noc0_sender_only_cores = [c for c in noc0_sender_cores if not receiver_core_grid.contains(c)]
        noc1_sender_only_cores = [c for c in noc1_sender_cores if not receiver_core_grid.contains(c)]

        # ========================================================================
        # NOC0 sender kernel (NCRISC) - all NOC0 senders
        # ========================================================================
        if not noc0_core_range_set.empty():
            noc0_sender_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_core_range_set,
                named_compile_time_args=create_sender_named_compile_time_args(is_noc0=True),
                common_runtime_args=[receiver_data_addr],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,  # NCRISC
                    noc=ttnn.NOC.NOC_0,
                ),
            )
            kernels.append(noc0_sender_kernel)

            # BRISC no-op for NOC0 sender-only cores (not receivers)
            # NOC0 sender+receiver cores will have receiver logic on BRISC instead
            if noc0_sender_only_cores:
                noc0_sender_only_core_range_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(c, c) for c in noc0_sender_only_cores]
                )
                noc0_sender_brisc_noop_args = [
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                    ("is_noc0_sender", 1),
                    ("is_noc1_sender", 0),
                ] + dummy_receiver_args
                noc0_sender_brisc_kernel = ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=noc0_sender_only_core_range_set,
                    named_compile_time_args=noc0_sender_brisc_noop_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0,  # BRISC
                        noc=ttnn.NOC.NOC_1,
                    ),
                )
                kernels.append(noc0_sender_brisc_kernel)

        # ========================================================================
        # NOC1 sender kernel (BRISC) - all NOC1 senders
        # ========================================================================
        if not noc1_core_range_set.empty():
            noc1_sender_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_core_range_set,
                named_compile_time_args=create_sender_named_compile_time_args(is_noc0=False),
                common_runtime_args=[receiver_data_addr],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,  # BRISC
                    noc=ttnn.NOC.NOC_1,
                ),
            )
            kernels.append(noc1_sender_kernel)

            # NCRISC no-op for NOC1 sender-only cores (not receivers)
            # NOC1 sender+receiver cores will have receiver logic on NCRISC instead
            if noc1_sender_only_cores:
                noc1_sender_only_core_range_set = ttnn.CoreRangeSet(
                    [ttnn.CoreRange(c, c) for c in noc1_sender_only_cores]
                )
                noc1_sender_ncrisc_noop_args = [
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                    ("is_noc0_sender", 0),
                    ("is_noc1_sender", 1),
                ] + dummy_receiver_args
                noc1_sender_ncrisc_kernel = ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=noc1_sender_only_core_range_set,
                    named_compile_time_args=noc1_sender_ncrisc_noop_args,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1,  # NCRISC
                        noc=ttnn.NOC.NOC_0,
                    ),
                )
                kernels.append(noc1_sender_ncrisc_kernel)

        # ========================================================================
        # Receiver kernels (runs on 4x2 receiver cores)
        # All receiver cores are also sender cores (receiver grid is within sender grid)
        # ========================================================================
        # Receiver logic placement:
        # 1. Receiver cores that are NOC0 senders: sender on NCRISC, receiver on BRISC
        # 2. Receiver cores that are NOC1 senders: sender on BRISC, receiver on NCRISC

        def create_receiver_named_compile_time_args(is_noc0_sender):
            """Create receiver compile-time args with sender type flags."""
            args = [
                # All receiver cores are also sender cores
                ("is_sender_core", 1),
                ("is_receiver_core", 1),
                ("is_noc0_sender", 1 if is_noc0_sender else 0),
                ("is_noc1_sender", 0 if is_noc0_sender else 1),
                ("noc0_receiver_semaphore_id", noc0_receiver_semaphore_id),
                ("noc1_receiver_semaphore_id", noc1_receiver_semaphore_id),
                ("out_cb", out_cb),
                ("dst_num_pages", dst_num_pages),
                ("sender_grid_width", sender_grid_width),
                ("receiver_grid_start_x", receiver_grid_start_x),
                ("receiver_grid_start_y", receiver_grid_start_y),
                ("receiver_cols", receiver_cols),
            ]
            # Add per-row sender counts for NOC0 and NOC1
            # Kernel expects args for all 8 rows (hardcoded array size), so always pass all 8
            for row in range(8):
                args.append((f"noc0_senders_row{row}", noc0_senders_per_receiver.get(row, 0)))
                args.append((f"noc1_senders_row{row}", noc1_senders_per_receiver.get(row, 0)))
            return args

        # NOC0 sender + receiver cores: sender on NCRISC, receiver on BRISC
        if noc0_sender_receiver_cores:
            noc0_sender_receiver_core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(c, c) for c in noc0_sender_receiver_cores]
            )
            noc0_sender_receiver_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc0_sender_receiver_core_range_set,
                named_compile_time_args=create_receiver_named_compile_time_args(is_noc0_sender=True),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0,  # BRISC
                    noc=ttnn.NOC.NOC_1,  # Use different NOC than sender
                ),
            )
            kernels.append(noc0_sender_receiver_kernel)

        # NOC1 sender + receiver cores: sender on BRISC, receiver on NCRISC
        if noc1_sender_receiver_cores:
            noc1_sender_receiver_core_range_set = ttnn.CoreRangeSet(
                [ttnn.CoreRange(c, c) for c in noc1_sender_receiver_cores]
            )
            noc1_sender_receiver_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=noc1_sender_receiver_core_range_set,
                named_compile_time_args=create_receiver_named_compile_time_args(is_noc0_sender=False),
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1,  # NCRISC
                    noc=ttnn.NOC.NOC_0,  # Use different NOC than sender
                ),
            )
            kernels.append(noc1_sender_receiver_kernel)

        # ========================================================================
        # Compute kernels (TRISC) - no-op for gather (dataflow only)
        # ========================================================================
        # All sender cores (which includes all receiver cores) need compute kernel
        if not sender_core_grid.empty():
            sender_compute_kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=sender_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(sender_compute_kernel)

        # Create CB descriptors from sharded tensors
        qnope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qnope_cb, qnope_tensor)
        qrope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qrope_cb, qrope_tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[qnope_cb_descriptor, qrope_cb_descriptor, out_cb_descriptor],
            semaphores=semaphores,
        )

        # Execute generic op
        io_tensors = [qnope_tensor, qrope_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
