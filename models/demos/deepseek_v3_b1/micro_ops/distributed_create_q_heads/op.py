# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn


class DistributedCreateQHeads:
    """
    Distributed Create Q Heads implementation using ttnn.generic_op.

    This class extends :class:`CreateQHeads` by splitting the per-receiver
    tilize work across three core groups:

      - Original receiver cores (``original_grid``): receive the first half of
        QNOPE for each head and tilize tiles 0..7 of the output.
      - NOPE helper cores (``nope_helper_grid``): receive the second half of
        QNOPE for each head, tilize tiles 8..15, and write them directly into
        the original receivers' output backing storage.
      - ROPE helper cores (``rope_helper_grid``): receive the QROPE data,
        tilize tiles 16..17, and write them into the original receivers'
        output backing storage.

    Each group has the same row-count as the original receiver grid, so a
    helper core at row ``r`` writes back to the original core at the same
    logical row ``r``.

    The data movement layout per receiver row is:

      - QNOPE first half [8, 256] -> 8 tiles (original receiver)
      - QNOPE second half [8, 256] -> 8 tiles (nope helper)
      - QROPE [8, 64] -> 2 tiles (rope helper)

    RISC assignment:
      - All data movement (sender + receivers + helpers) uses NCRISC (RISCV_1)
      - BRISC is idle
      - TRISC tilizes on original/helper cores

    Helper grid placement (hardcoded, similar to ``FlashMLAOptimalGridNOC0``):

    * NOPE helpers: 4x2 logical block at ``(0, 5)..(3, 6)``. The kernel's
      ``nope_helper_receiver_row = (y - HELPER_BASE_Y) * cols + x`` formula
      requires the NOPE helpers to start at ``y = NOPE_HELPER_GRID_START_Y``
      and have the same column count as the original receiver grid.
    * ROPE helpers: 1x8 logical column at ``(11, 0)..(11, 7)``. The kernel's
      ``rope_helper_receiver_row = y`` formula requires a single column of
      helpers indexed by ``logical_y``.

    Both ranges are chosen to lie outside the FlashMLA compute grid (cols 0-3
    use rows 0-4 + 7-9 only; cols 7-10 are FlashMLA right side; col 11 is
    unused by FlashMLA). Callers placing helpers alongside FlashMLA can use
    ``ttnn.CoreRangeSet.intersects`` for verification.
    """

    NOPE_HELPER_GRID_START_X = 0
    NOPE_HELPER_GRID_START_Y = 5
    NOPE_HELPER_GRID_END_X = 3
    NOPE_HELPER_GRID_END_Y = 6

    ROPE_HELPER_GRID_X = 11
    ROPE_HELPER_GRID_START_Y = 0
    ROPE_HELPER_GRID_END_Y = 7

    @classmethod
    def nope_helper_grid(cls) -> "ttnn.CoreRangeSet":
        """Hardcoded NOPE helper grid (4x2 block)."""
        return ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(cls.NOPE_HELPER_GRID_START_X, cls.NOPE_HELPER_GRID_START_Y),
                    ttnn.CoreCoord(cls.NOPE_HELPER_GRID_END_X, cls.NOPE_HELPER_GRID_END_Y),
                )
            ]
        )

    @classmethod
    def rope_helper_grid(cls) -> "ttnn.CoreRangeSet":
        """Hardcoded ROPE helper grid (1x8 column)."""
        return ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(
                    ttnn.CoreCoord(cls.ROPE_HELPER_GRID_X, cls.ROPE_HELPER_GRID_START_Y),
                    ttnn.CoreCoord(cls.ROPE_HELPER_GRID_X, cls.ROPE_HELPER_GRID_END_Y),
                )
            ]
        )

    @classmethod
    def nope_helper_cores(cls) -> list:
        """NOPE helper cores in row-major order (matches kernel row formula)."""
        return ttnn.corerange_to_cores(cls.nope_helper_grid(), row_wise=True)

    @classmethod
    def rope_helper_cores(cls) -> list:
        """ROPE helper cores in row-major order (matches kernel row formula)."""
        return ttnn.corerange_to_cores(cls.rope_helper_grid(), row_wise=True)

    @classmethod
    def helper_grid(cls) -> "ttnn.CoreRangeSet":
        """Combined NOPE + ROPE helper grid (union of both helper regions)."""
        return cls.nope_helper_grid().merge(cls.rope_helper_grid())

    @staticmethod
    def golden(qnope_input, qrope_input, qnope_grid, qrope_grid, receiver_grid):
        """PyTorch reference implementation matching the kernel's per-shard layout.

        The output layout is identical to :class:`CreateQHeads.golden`. The
        distributed kernel only changes the data-movement and tilize work
        partitioning; the final 18-tile shard on each original receiver core
        matches the non-distributed implementation.
        """
        from models.demos.deepseek_v3_b1.micro_ops.create_q_heads.op import CreateQHeads

        return CreateQHeads.golden(qnope_input, qrope_input, qnope_grid, qrope_grid, receiver_grid)

    @classmethod
    def op(
        cls,
        qnope_tensor,
        qrope_tensor,
        interm_tensor,
        output_tensor,
        original_grid,
    ):
        """Execute distributed create q heads using generic_op.

        Args:
            qnope_tensor: QNOPE block-sharded across the qnope grid (e.g. 8x8) with [1, 512] per core.
            qrope_tensor: QROPE block-sharded across the qrope grid (e.g. 4x8) with [2, 64] per core.
            interm_tensor: Receiver-side intermediate tensor sharded across
                ``original_grid`` only, with shard size large enough for the
                NOPE phase (8 tiles of 8x32). Helper cores reuse the same L1
                offset (the L1 allocator is in lockstep across the device) so
                no helper-side allocation is needed.
            output_tensor: Output backing tensor sharded across
                ``original_grid`` only with 18 tiles per shard (= [8, 576]).
                Helpers tilize their NOPE/ROPE chunks into the same L1 offset
                locally before NOC-writing the result back to the originals.
            original_grid: ``CoreRangeSet`` for the original receiver cores
                (e.g. 4x2 at ``(0,1)..(3,2)``). The row index for each
                original maps to the corresponding helper row.

        Helper grids are hardcoded inside this op (see class-level constants
        ``NOPE_HELPER_GRID_*`` / ``ROPE_HELPER_GRID_*``) so the layout matches
        the unified attention-block kernel's helper row formulas.

        Returns:
            The output backing tensor; each original-receiver shard contains
            the published 18-tile Q output.
        """
        nope_helper_grid = cls.nope_helper_grid()
        rope_helper_grid = cls.rope_helper_grid()

        device = qnope_tensor.device()

        qnope_memory_config = qnope_tensor.memory_config()
        qrope_memory_config = qrope_tensor.memory_config()
        qnope_core_grid = qnope_memory_config.shard_spec.grid
        qrope_core_grid = qrope_memory_config.shard_spec.grid

        # Sender grid covers qnope + qrope cores
        sender_core_grid = qrope_core_grid.merge(qnope_core_grid)
        # Original receiver grid passed in by the caller
        original_receiver_core_grid = original_grid

        sender_core_ranges = list(sender_core_grid.ranges())
        sender_grid_start_x = min(r.start.x for r in sender_core_ranges)
        sender_grid_start_y = min(r.start.y for r in sender_core_ranges)
        sender_grid_end_x = max(r.end.x for r in sender_core_ranges)
        sender_grid_end_y = max(r.end.y for r in sender_core_ranges)
        sender_grid_width = sender_grid_end_x - sender_grid_start_x + 1
        sender_grid_height = sender_grid_end_y - sender_grid_start_y + 1

        qnope_shard_shape = qnope_memory_config.shard_spec.shape
        qrope_shard_shape = qrope_memory_config.shard_spec.shape

        qnope_elements = qnope_shard_shape[0] * qnope_shard_shape[1]
        qrope_head_elements = qrope_shard_shape[1]

        dtype = qnope_tensor.dtype
        if dtype == ttnn.bfloat16:
            element_size_bytes = 2
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        qnope_data_size_bytes = qnope_elements * element_size_bytes
        qrope_head_size_bytes = qrope_head_elements * element_size_bytes

        qnope_core_ranges = list(qnope_core_grid.ranges())
        qnope_grid_start_x = min(r.start.x for r in qnope_core_ranges)
        qnope_grid_end_x = max(r.end.x for r in qnope_core_ranges)
        qnope_cols = qnope_grid_end_x - qnope_grid_start_x + 1

        # Build sender-row -> per-row helper/original NOC coords.
        # Mapping: row r -> original_core(r % cols, 1 + r // cols), nope_helper at (r % cols, 5 + r // cols),
        # rope_helper at (sender_grid_end_x, r). The convention matches the kernel's helper_receiver_row formulas:
        #   nope_helper_receiver_row = (y - 5) * cols + x
        #   rope_helper_receiver_row = y
        # We compute NOC coords directly from the helper grids by their logical positions.
        original_cores_list = ttnn.corerange_to_cores(original_receiver_core_grid, row_wise=True)
        nope_helper_cores_list = ttnn.corerange_to_cores(nope_helper_grid, row_wise=True)
        rope_helper_cores_list = ttnn.corerange_to_cores(rope_helper_grid, row_wise=True)

        assert (
            len(original_cores_list) == sender_grid_height
        ), f"Expected {sender_grid_height} original receiver cores, got {len(original_cores_list)}"
        assert (
            len(nope_helper_cores_list) == sender_grid_height
        ), f"Expected {sender_grid_height} nope helper cores, got {len(nope_helper_cores_list)}"
        assert (
            len(rope_helper_cores_list) == sender_grid_height
        ), f"Expected {sender_grid_height} rope helper cores, got {len(rope_helper_cores_list)}"

        original_noc_coords = []
        nope_helper_noc_coords = []
        rope_helper_noc_coords = []
        for row in range(sender_grid_height):
            o_core = device.worker_core_from_logical_core(original_cores_list[row])
            n_core = device.worker_core_from_logical_core(nope_helper_cores_list[row])
            r_core = device.worker_core_from_logical_core(rope_helper_cores_list[row])
            original_noc_coords.append((o_core.x, o_core.y))
            nope_helper_noc_coords.append((n_core.x, n_core.y))
            rope_helper_noc_coords.append((r_core.x, r_core.y))

        # Semaphore ids (match the unified_kernels convention)
        nope_phase1_semaphore_id = 2
        nope_phase2_semaphore_id = 3
        rope_semaphore_id = 0

        all_cores = sender_core_grid.merge(original_receiver_core_grid).merge(nope_helper_grid).merge(rope_helper_grid)

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
        qnope_cb = 0
        qrope_cb = 1
        receiver_in_cb = 2
        out_cb = 3

        src_num_pages = 1

        nope_tiles = 8
        rope_tiles = 2

        receiver_data_addr = interm_tensor.buffer_address()

        kernel_path = (
            "models/demos/deepseek_v3_b1/micro_ops/distributed_create_q_heads/kernels/"
            "distributed_create_q_heads_kernel.cpp"
        )

        kernels = []
        semaphores = [
            rope_semaphore_descriptor,
            nope_phase1_semaphore_descriptor,
            nope_phase2_semaphore_descriptor,
        ]

        def base_named_compile_time_args(is_sender, is_original, is_nope_helper, is_rope_helper):
            args = [
                ("is_sender_core", 1 if is_sender else 0),
                ("is_original_receiver_core", 1 if is_original else 0),
                ("is_nope_helper_core", 1 if is_nope_helper else 0),
                ("is_rope_helper_core", 1 if is_rope_helper else 0),
                ("sender_grid_start_x", sender_grid_start_x),
                ("sender_grid_start_y", sender_grid_start_y),
                ("qnope_data_size_bytes", qnope_data_size_bytes),
                ("qrope_head_size_bytes", qrope_head_size_bytes),
                ("qnope_cols", qnope_cols),
                ("qnope_cb", qnope_cb),
                ("qrope_cb", qrope_cb),
                ("src_num_pages", src_num_pages),
                ("nope_phase1_semaphore_id", nope_phase1_semaphore_id),
                ("nope_phase2_semaphore_id", nope_phase2_semaphore_id),
                ("rope_semaphore_id", rope_semaphore_id),
                ("receiver_in_cb", receiver_in_cb),
                ("out_cb", out_cb),
                ("nope_tiles", nope_tiles),
                ("rope_tiles", rope_tiles),
                ("num_nope_senders", qnope_cols),
                ("num_rope_senders", sender_grid_width - qnope_cols),
            ]
            for row in range(8):
                if row < len(original_noc_coords):
                    o = original_noc_coords[row]
                    n = nope_helper_noc_coords[row]
                    r = rope_helper_noc_coords[row]
                else:
                    o = (0, 0)
                    n = (0, 0)
                    r = (0, 0)
                args.append((f"original_noc_coords_row{row}", o[0] | (o[1] << 16)))
                args.append((f"nope_helper_noc_coords_row{row}", n[0] | (n[1] << 16)))
                args.append((f"rope_helper_noc_coords_row{row}", r[0] | (r[1] << 16)))
            return args

        def add_kernel(core_range_set, is_sender, is_original, is_nope_helper, is_rope_helper, processor):
            if core_range_set.empty():
                return
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    named_compile_time_args=base_named_compile_time_args(
                        is_sender, is_original, is_nope_helper, is_rope_helper
                    ),
                    common_runtime_args=[receiver_data_addr],
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=processor,
                        noc=ttnn.NOC.NOC_0,
                    ),
                )
            )

        def add_compute_kernel(core_range_set, is_original, is_nope_helper, is_rope_helper):
            if core_range_set.empty():
                return
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kernel_path,
                    source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                    core_ranges=core_range_set,
                    named_compile_time_args=[
                        ("is_sender_core", 0),
                        ("is_original_receiver_core", 1 if is_original else 0),
                        ("is_nope_helper_core", 1 if is_nope_helper else 0),
                        ("is_rope_helper_core", 1 if is_rope_helper else 0),
                        ("receiver_in_cb", receiver_in_cb),
                        ("out_cb", out_cb),
                        ("nope_tiles", nope_tiles),
                        ("rope_tiles", rope_tiles),
                    ],
                    config=ttnn.ComputeConfigDescriptor(),
                )
            )

        # Partition cores by role. A core may be both a sender and a helper
        # (e.g. a qrope core overlapping with the rope-helper column).
        sender_cores_set = set((c.x, c.y) for c in ttnn.corerange_to_cores(sender_core_grid, row_wise=True))
        original_cores_set = set((c.x, c.y) for c in original_cores_list)
        nope_helper_set = set((c.x, c.y) for c in nope_helper_cores_list)
        rope_helper_set = set((c.x, c.y) for c in rope_helper_cores_list)

        def to_core_range_set(coords):
            if not coords:
                return ttnn.CoreRangeSet([])
            return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for (x, y) in coords])

        # Build disjoint subsets keyed by (is_sender, is_original, is_nope_helper, is_rope_helper).
        role_groups = {}

        def add_role(core, is_sender, is_original, is_nope_helper, is_rope_helper):
            key = (is_sender, is_original, is_nope_helper, is_rope_helper)
            role_groups.setdefault(key, []).append(core)

        all_role_cores = sender_cores_set | original_cores_set | nope_helper_set | rope_helper_set
        for core in all_role_cores:
            add_role(
                core,
                core in sender_cores_set,
                core in original_cores_set,
                core in nope_helper_set,
                core in rope_helper_set,
            )

        for (is_sender, is_original, is_nope_helper, is_rope_helper), cores in role_groups.items():
            crs = to_core_range_set(cores)
            add_kernel(crs, is_sender, is_original, is_nope_helper, is_rope_helper, ttnn.DataMovementProcessor.RISCV_1)
            if is_original or is_nope_helper or is_rope_helper:
                add_compute_kernel(crs, is_original, is_nope_helper, is_rope_helper)
            else:
                # Senders only need a no-op compute kernel for completeness
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=kernel_path,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=crs,
                        named_compile_time_args=[
                            ("is_sender_core", 1),
                            ("is_original_receiver_core", 0),
                            ("is_nope_helper_core", 0),
                            ("is_rope_helper_core", 0),
                            ("receiver_in_cb", receiver_in_cb),
                            ("out_cb", out_cb),
                            ("nope_tiles", nope_tiles),
                            ("rope_tiles", rope_tiles),
                        ],
                        config=ttnn.ComputeConfigDescriptor(),
                    )
                )

        # CB descriptors. ``interm_tensor`` and ``output_tensor`` are sharded
        # only across the original receiver cores, but the L1 allocator is in
        # lockstep across the device, so the same L1 address is "free" on the
        # helper cores too. We can therefore create a single CB per role over
        # the union of receiver + helper cores and let it be backed by the
        # receiver-only tensor: senders address every receiver-class core via
        # ``receiver_data_addr`` (the interm tensor's L1 offset) and helpers
        # write their tilized output back to the originals at ``out_cb``'s L1
        # offset. Helpers only push/pop a sub-range of tiles locally, so the
        # extra L1 "owned" by their CB on helper cores is unused staging
        # space.
        receiver_class_cores = original_receiver_core_grid.merge(nope_helper_grid).merge(rope_helper_grid)

        qnope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qnope_cb, qnope_tensor)
        qrope_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(qrope_cb, qrope_tensor)
        receiver_in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            receiver_in_cb, interm_tensor, core_ranges=receiver_class_cores
        )
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            out_cb, output_tensor, core_ranges=receiver_class_cores
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                qnope_cb_descriptor,
                qrope_cb_descriptor,
                receiver_in_cb_descriptor,
                out_cb_descriptor,
            ],
            semaphores=semaphores,
        )

        io_tensors = [qnope_tensor, qrope_tensor, output_tensor]
        return ttnn.generic_op(io_tensors, program_descriptor)
