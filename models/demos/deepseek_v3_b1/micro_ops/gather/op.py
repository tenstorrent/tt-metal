# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor


def _get_shard_size_bytes(shard_shape, tile, dtype):
    """Get shard size in bytes for a given shard shape, tile, and dtype."""
    tile_h, tile_w = tile.tile_shape
    num_tiles = (shard_shape[0] // tile_h) * (shard_shape[1] // tile_w)
    return num_tiles * tile.get_tile_size(dtype)


def _validate_gather_inputs(input_tensor, output_tensor, sender_cores=None):
    """Validate gather operation inputs."""
    # Validate output has single core
    output_grid = output_tensor.memory_config().shard_spec.grid
    if output_grid.num_cores() != 1:
        raise ValueError(f"Output tensor must be sharded on exactly one core, got {output_grid.num_cores()}")

    # Validate sender cores are within device bounds
    if sender_cores:
        device = input_tensor.device()
        grid_size = device.compute_with_storage_grid_size()
        for core in sender_cores:
            if core.x >= grid_size.x or core.y >= grid_size.y:
                raise ValueError(f"Core ({core.x}, {core.y}) is outside device grid ({grid_size.x}, {grid_size.y})")


class GatherSingleCore:
    """
    Single-core gather implementation using ttnn.generic_op.

    This class implements gather from multiple cores to a single core.
    """

    @staticmethod
    def golden(input_tensor):
        """
        PyTorch reference implementation of gather for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) sharded across multiple cores

        Returns:
            Output tensor with all input shards concatenated on gather_core
        """
        # Simply return the input tensor
        return input_tensor

    @staticmethod
    def _split_cores_by_noc(device, sender_cores, gather_core, noc):
        """
        Split sender cores into NOC0 and NOC1 groups.

        Args:
            device: TT device
            sender_cores: List of CoreCoord
            gather_core: Destination CoreCoord
            noc: Forced NOC (ttnn.NOC.NOC_0/NOC_1) or None for auto-optimization

        Returns:
            Tuple of (noc0_cores, noc1_cores)
        """
        if noc is not None:
            if noc == ttnn.NOC.NOC_0:
                return list(sender_cores), []
            else:
                return [], list(sender_cores)

        # Optimize NOC routing based on hop distance
        noc0_cores = []
        noc1_cores = []
        for core in sender_cores:
            noc0_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_0)
            noc1_hop = device.get_worker_noc_hop_distance(core, gather_core, ttnn.NOC.NOC_1)
            if noc0_hop <= noc1_hop:
                noc0_cores.append(core)
            else:
                noc1_cores.append(core)
        return noc0_cores, noc1_cores

    @staticmethod
    def _build_gather_program(
        input_tensor,
        output_tensor,
        sender_cores,
        noc0_cores,
        noc1_cores,
        gather_core,
        gather_dest_noc_core,
        use_per_core_sender_idx,
        sender_grid_bounds=None,
    ):
        """
        Build the gather program descriptor.

        Args:
            input_tensor: Input tensor
            output_tensor: Output tensor
            sender_cores: List of all sender CoreCoords
            noc0_cores: List of cores using NOC0
            noc1_cores: List of cores using NOC1
            gather_core: Destination core
            gather_dest_noc_core: NOC coordinates of destination
            use_per_core_sender_idx: If True, use per-core sender indices (scattered mode)
            sender_grid_bounds: Tuple of (start_x, start_y, end_x, end_y) for grid-based indexing

        Returns:
            ProgramDescriptor
        """
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()
        output_core_grid = output_memory_config.shard_spec.grid

        # Calculate data size
        shard_spec = input_memory_config.shard_spec
        shard_shape = shard_spec.shape
        total_size = _get_shard_size_bytes(shard_shape, input_tensor.tile, input_tensor.dtype)

        # Create CoreRangeSets
        input_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in sender_cores])
        all_cores = input_core_grid.merge(output_core_grid)
        noc0_core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in noc0_cores])

        # Semaphore setup
        noc0_receiver_semaphore_id = 0
        noc1_receiver_semaphore_id = 1

        # Create semaphores
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
        src_cb = 0  # Input CB (from sharded input tensor)
        dst_cb = 1  # Output CB (from sharded output tensor)

        # Number of pages
        num_senders = len(sender_cores)
        src_num_pages = 1  # Each sender has one page
        dst_num_pages = num_senders  # Receiver gets one page per sender

        kernels = []
        semaphores = [noc0_receiver_semaphore_descriptor, noc1_receiver_semaphore_descriptor]

        kernel_path = "models/demos/deepseek_v3_b1/micro_ops/gather/kernels/gather_kernel.cpp"

        # Get the output tensor's buffer address for receiver data address (runtime arg)
        # The dst CB doesn't exist on sender cores, so we pass the buffer address as runtime arg
        receiver_data_addr = output_tensor.buffer_address()

        # Grid bounds (used only when use_per_core_sender_idx=False)
        if sender_grid_bounds:
            start_x, start_y, end_x, end_y = sender_grid_bounds
        else:
            start_x = start_y = end_x = end_y = 0

        # Build per-core sender index mapping for scattered mode
        core_to_idx = (
            {(core.x, core.y): idx for idx, core in enumerate(sender_cores)} if use_per_core_sender_idx else {}
        )

        # ========================================================================
        # Sender kernels (NCRISC) - separate for NOC0 and NOC1
        # ========================================================================
        def create_sender_kernel(cores, semaphore_id, noc_type):
            if not cores:
                return []

            if use_per_core_sender_idx:
                # Scattered mode: use PerCoreCompileTimeDescriptor for per-core sender_idx
                per_core_desc = PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="gather_sender_idx",
                    core_values=[(core, core_to_idx[(core.x, core.y)]) for core in cores],
                    other_value=0,
                )

                shared_named_args = [
                    ("gather_dest_noc_x", gather_dest_noc_core.x),
                    ("gather_dest_noc_y", gather_dest_noc_core.y),
                    ("gather_data_size_bytes", total_size),
                    ("gather_receiver_semaphore_id", semaphore_id),
                    ("gather_src_cb", src_cb),
                    ("gather_src_num_pages", src_num_pages),
                    ("gather_sender_grid_start_x", 0),
                    ("gather_sender_grid_start_y", 0),
                    ("gather_sender_grid_end_x", 0),
                    ("gather_sender_grid_end_y", 0),
                    ("gather_row_major", 1),
                    ("gather_use_per_core_sender_idx", 1),
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ]

                kernel_list = []
                for core, sender_idx in per_core_desc.core_values:
                    core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
                    named_args = shared_named_args + [
                        (per_core_desc.named_compile_time_arg, sender_idx),
                    ]
                    kernel_list.append(
                        ttnn.KernelDescriptor(
                            kernel_source=kernel_path,
                            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                            core_ranges=core_range_set,
                            named_compile_time_args=named_args,
                            common_runtime_args=[receiver_data_addr],
                            config=ttnn.DataMovementConfigDescriptor(
                                processor=ttnn.DataMovementProcessor.RISCV_1,
                                noc=noc_type,
                                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                            ),
                        )
                    )
                return kernel_list
            else:
                # Grid mode: single kernel for all cores
                core_range_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in cores])
                named_args = [
                    ("gather_dest_noc_x", gather_dest_noc_core.x),
                    ("gather_dest_noc_y", gather_dest_noc_core.y),
                    ("gather_data_size_bytes", total_size),
                    ("gather_receiver_semaphore_id", semaphore_id),
                    ("gather_src_cb", src_cb),
                    ("gather_src_num_pages", src_num_pages),
                    ("gather_sender_grid_start_x", start_x),
                    ("gather_sender_grid_start_y", start_y),
                    ("gather_sender_grid_end_x", end_x),
                    ("gather_sender_grid_end_y", end_y),
                    ("gather_row_major", 1),
                    ("gather_use_per_core_sender_idx", 0),
                    ("gather_sender_idx", 0),
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ]
                return [
                    ttnn.KernelDescriptor(
                        kernel_source=kernel_path,
                        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                        core_ranges=core_range_set,
                        named_compile_time_args=named_args,
                        common_runtime_args=[receiver_data_addr],
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_1,
                            noc=noc_type,
                            noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
                        ),
                    )
                ]

        kernels.extend(create_sender_kernel(noc0_cores, noc0_receiver_semaphore_id, ttnn.NOC.NOC_0))
        kernels.extend(create_sender_kernel(noc1_cores, noc1_receiver_semaphore_id, ttnn.NOC.NOC_1))

        # ========================================================================
        # Receiver kernel (BRISC)
        # ========================================================================
        # Determine receiver NOC - use the NOC that doesn't conflict with senders
        output_in_noc0 = noc0_core_range_set.contains(gather_core)
        if output_in_noc0:
            receiver_noc = ttnn.NOC.NOC_1
        else:
            receiver_noc = ttnn.NOC.NOC_0

        receiver_named_compile_time_args = [
            ("gather_noc0_num_senders", len(noc0_cores)),
            ("gather_noc1_num_senders", len(noc1_cores)),
            ("gather_noc0_receiver_semaphore_id", noc0_receiver_semaphore_id),
            ("gather_noc1_receiver_semaphore_id", noc1_receiver_semaphore_id),
            ("gather_dst_cb", dst_cb),
            ("gather_dst_num_pages", dst_num_pages),
            # Role flags
            ("is_sender_core", 0),
            ("is_receiver_core", 1),
        ]

        receiver_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=kernel_path,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=output_core_grid,
            named_compile_time_args=receiver_named_compile_time_args,
            config=ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0,
                noc=receiver_noc,
                noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC,
            ),
        )
        kernels.append(receiver_kernel_descriptor)

        # ========================================================================
        # Compute kernels (TRISC) - no-op for gather, but needed for all cores
        # ========================================================================
        # Sender cores compute kernel (no-op)
        if not input_core_grid.empty():
            sender_compute_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=input_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 1),
                    ("is_receiver_core", 0),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(sender_compute_kernel_descriptor)

        # Receiver core compute kernel (no-op) - only if not already covered by sender grid
        if not input_core_grid.contains(gather_core):
            receiver_compute_kernel_descriptor = ttnn.KernelDescriptor(
                kernel_source=kernel_path,
                source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                core_ranges=output_core_grid,
                named_compile_time_args=[
                    ("is_sender_core", 0),
                    ("is_receiver_core", 1),
                ],
                config=ttnn.ComputeConfigDescriptor(),
            )
            kernels.append(receiver_compute_kernel_descriptor)

        # Create CB descriptors from sharded tensors
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, input_tensor)
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, output_tensor)

        return ttnn.ProgramDescriptor(
            kernels=kernels,
            cbs=[src_cb_descriptor, dst_cb_descriptor],
            semaphores=semaphores,
        )

    @staticmethod
    def op(input_tensor, output_tensor, noc=None):
        """
        Execute gather operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded across a rectangular grid)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            noc: NOC to use for gather (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1). If None,
                 automatically optimizes NOC routing based on hop distance for each sender core.

        Returns:
            Output tensor with input data gathered from all cores to gather_core
        """
        _validate_gather_inputs(input_tensor, output_tensor)

        device = input_tensor.device()
        input_memory_config = input_tensor.memory_config()
        output_memory_config = output_tensor.memory_config()
        input_core_grid = input_memory_config.shard_spec.grid
        output_core_grid = output_memory_config.shard_spec.grid

        gather_core = output_core_grid.ranges()[0].start
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        # Get sender grid dimensions
        input_core_ranges = list(input_core_grid.ranges())
        sender_grid_range = input_core_ranges[0]
        sender_grid_bounds = (
            sender_grid_range.start.x,
            sender_grid_range.start.y,
            sender_grid_range.end.x,
            sender_grid_range.end.y,
        )

        sender_cores = ttnn.corerange_to_cores(input_core_grid, row_wise=True)
        noc0_cores, noc1_cores = GatherSingleCore._split_cores_by_noc(device, sender_cores, gather_core, noc)

        program_descriptor = GatherSingleCore._build_gather_program(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            sender_cores=sender_cores,
            noc0_cores=noc0_cores,
            noc1_cores=noc1_cores,
            gather_core=gather_core,
            gather_dest_noc_core=gather_dest_noc_core,
            use_per_core_sender_idx=False,
            sender_grid_bounds=sender_grid_bounds,
        )

        return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)

    @staticmethod
    def op_scattered(input_tensor, output_tensor, sender_cores, noc=None):
        """
        Execute gather operation from scattered (non-rectangular) cores using generic_op.

        This variant uses per-core compile-time args to specify unique sender indices,
        enabling gather from non-contiguous cores that don't form a rectangular grid.

        Args:
            input_tensor: Input tensor (must be sharded across the specified scattered cores)
            output_tensor: Pre-allocated output tensor (must be sharded on single core)
            sender_cores: List of CoreCoord specifying which cores participate in gather
            noc: NOC to use for gather (ttnn.NOC.NOC_0 or ttnn.NOC.NOC_1). If None,
                 automatically optimizes NOC routing based on hop distance for each sender core.

        Returns:
            Output tensor with input data gathered from all specified cores to gather_core
        """
        _validate_gather_inputs(input_tensor, output_tensor, sender_cores)

        device = input_tensor.device()
        output_memory_config = output_tensor.memory_config()
        output_core_grid = output_memory_config.shard_spec.grid

        gather_core = output_core_grid.ranges()[0].start
        gather_dest_noc_core = device.worker_core_from_logical_core(gather_core)

        noc0_cores, noc1_cores = GatherSingleCore._split_cores_by_noc(device, sender_cores, gather_core, noc)

        program_descriptor = GatherSingleCore._build_gather_program(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            sender_cores=list(sender_cores),
            noc0_cores=noc0_cores,
            noc1_cores=noc1_cores,
            gather_core=gather_core,
            gather_dest_noc_core=gather_dest_noc_core,
            use_per_core_sender_idx=True,
            sender_grid_bounds=None,
        )

        return ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
