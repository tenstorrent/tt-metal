import torch
from loguru import logger

import ttnn


class FusedResblock:
    class MatmulCoreCBIndex:
        MM1_FULL_CB = 0
        WEIGHT0_CB = 1
        WEIGHT1_CB = 2
        OUT_CB = 3
        INTERMEDIATE_PREGATHER_CB = 4
        MM2_FULL_CB = 5

    class McastCoreCBIndex:
        MCAST_CORE_GATHER_CB = 6

    MCAST_CORE = ttnn.CoreCoord(7, 7)

    @staticmethod
    def golden(input_a, weight0, weight1, num_layers=1):
        def layer(input):
            x = input @ weight0
            x = torch.nn.functional.relu(x)
            x = x @ weight1
            x = x + input
            return x

        x = input_a
        for _ in range(num_layers):
            x = layer(x)

        return x

    @staticmethod
    def create_mcast_kernel(
        all_mcast_cores: ttnn.CoreRangeSet,
        all_sender_cores: ttnn.CoreRangeSet,
        data_format,
        page_size: int,
        num_tiles: int,
        tile: ttnn.TileDescriptor,
        receiver_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        sender_semaphore_descriptor: ttnn.SemaphoreDescriptor,
        device: ttnn.Device,
        debug: bool,
        num_layers: int,
        input_buffer_address: int,
    ) -> tuple[ttnn.KernelDescriptor, ttnn.KernelDescriptor, ttnn.CBDescriptor]:
        logger.debug(f"All mcast cores: {all_mcast_cores}")
        logger.debug(f"Number of mcast cores: {all_mcast_cores.num_cores()}")
        logger.debug(f"All sender cores: {all_sender_cores}")
        logger.debug(f"Number of sender cores: {all_sender_cores.num_cores()}")

        number_of_senders = all_sender_cores.num_cores()

        # Create CBs for gather destination
        mcast_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
            data_format=data_format,
            page_size=page_size,
            tile=tile,
        )
        mcast_cb_descriptor = ttnn.CBDescriptor(
            total_size=page_size * num_tiles,
            core_ranges=all_sender_cores.merge(
                all_mcast_cores
            ),  # Create on all cores since otherwise the senders won't get the correct CB address
            format_descriptors=[mcast_cb_format],
        )

        def get_mcast_sender_noc_coords(all_sender_cores: ttnn.CoreRangeSet):
            all_sender_cores_list = (
                device.worker_core_from_logical_core(all_sender_cores.ranges()[0].start),
                device.worker_core_from_logical_core(all_sender_cores.ranges()[0].end),
            )
            return (
                all_sender_cores_list[0].x,
                all_sender_cores_list[0].y,
                all_sender_cores_list[1].x,
                all_sender_cores_list[1].y,
            )

        (
            mcast_sender_noc_coord_x_start,
            mcast_sender_noc_coord_y_start,
            mcast_sender_noc_coord_x_end,
            mcast_sender_noc_coord_y_end,
        ) = get_mcast_sender_noc_coords(all_sender_cores)
        logger.debug(
            f"Mcast sender NOC coords: {mcast_sender_noc_coord_x_start}, {mcast_sender_noc_coord_y_start}, {mcast_sender_noc_coord_x_end}, {mcast_sender_noc_coord_y_end}"
        )

        mcast_reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/mcast_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_mcast_cores,
            compile_time_args=[
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
                input_buffer_address,  # Base address of input tensor buffer
                number_of_senders,
                receiver_semaphore_descriptor.id,
                sender_semaphore_descriptor.id,
                mcast_sender_noc_coord_x_start,
                mcast_sender_noc_coord_y_start,
                mcast_sender_noc_coord_x_end,
                mcast_sender_noc_coord_y_end,
                num_layers,
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        mcast_writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/mcast_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_mcast_cores,
            compile_time_args=[
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                number_of_senders,
                receiver_semaphore_descriptor.id,
                sender_semaphore_descriptor.id,
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        return mcast_reader_kernel_descriptor, mcast_writer_kernel_descriptor, mcast_cb_descriptor

    @staticmethod
    def op(input_tensor, weight0, weight1, output_tensor, num_layers=1, fp32_dest_acc_en=False, debug=False):
        logger.info(
            f"Running ResBlock operation with shape {input_tensor.shape} x {weight0.shape} x {weight1.shape} -> {output_tensor.shape}"
        )
        logger.debug(f"Input tensor sharding: {input_tensor.memory_config().shard_spec}")
        logger.debug(f"Weight0 tensor sharding: {weight0.memory_config().shard_spec}")
        logger.debug(f"Weight1 tensor sharding: {weight1.memory_config().shard_spec}")
        logger.debug(f"Output tensor sharding: {output_tensor.memory_config().shard_spec}")

        device = input_tensor.device()

        input_shape = input_tensor.shape
        input_tile = input_tensor.get_tile()
        weight0_shape = weight0.shape
        weight0_tile = weight0.get_tile()
        weight1_shape = weight1.shape
        weight1_tile = weight1.get_tile()

        # Validate inputs are ttnn.Tensor and on same device
        assert isinstance(input_tensor, ttnn.Tensor), f"Input tensor must be ttnn.Tensor, got {type(input_tensor)}"
        assert isinstance(weight0, ttnn.Tensor), f"Weight must be ttnn.Tensor, got {type(weight0)}"
        assert isinstance(weight1, ttnn.Tensor), f"Weight must be ttnn.Tensor, got {type(weight1)}"
        assert isinstance(output_tensor, ttnn.Tensor), f"Output tensor must be ttnn.Tensor, got {type(output_tensor)}"
        assert (
            input_tensor.device() == weight0.device() == weight1.device() == output_tensor.device()
        ), "All tensors must be on the same device"

        # Validate rank and matmul shape compatibility
        assert len(input_shape) >= 2, f"Input tensor must have rank >= 2, got {len(input_shape)}"
        assert len(weight0_shape) >= 2, f"Weight0 must have rank >= 2, got {len(weight0_shape)}"
        assert len(weight1_shape) >= 2, f"Weight1 must have rank >= 2, got {len(weight1_shape)}"
        assert len(output_tensor.shape) >= 2, f"Output tensor must have rank >= 2, got {len(output_tensor.shape)}"

        # Validate matmul shape compatibility: input [M, K] @ weight0 [K, K] -> intermediate [M, K]
        # Then intermediate [M, K] @ weight1 [K, K] -> output [M, K]
        input_m, input_k = input_shape[-2], input_shape[-1]
        weight0_k_in, weight0_k_out = weight0_shape[-2], weight0_shape[-1]
        weight1_k_in, weight1_k_out = weight1_shape[-2], weight1_shape[-1]
        output_m, output_k = output_tensor.shape[-2], output_tensor.shape[-1]

        assert input_k == weight0_k_in, f"Input K dimension ({input_k}) must match weight0 K_in ({weight0_k_in})"
        assert (
            weight0_k_out == weight1_k_in
        ), f"Weight0 K_out ({weight0_k_out}) must match weight1 K_in ({weight1_k_in})"
        assert weight1_k_out == input_k, f"Weight1 K_out ({weight1_k_out}) must match input K ({input_k})"

        # Note: input M may be replicated across cores, so input_m can be larger than output_m
        # The actual batch dimension should match after accounting for replication
        assert output_k == input_k, f"Output K dimension ({output_k}) must match input K ({input_k})"

        # Validate tile compatibility and K dimension divisibility
        assert (
            input_shape[1] % input_tile.tile_shape[1] == 0
        ), f"Input K ({input_shape[1]}) must be divisible by tile width ({input_tile.tile_shape[1]})"
        num_tiles_k = input_shape[1] // input_tile.tile_shape[1]

        # Validate dtype compatibility
        input_dtype = input_tensor.dtype
        weight0_dtype = weight0.dtype
        weight1_dtype = weight1.dtype
        out_dtype = output_tensor.dtype
        assert (
            weight0_dtype == weight1_dtype
        ), f"Weight0 dtype ({weight0_dtype}) must match weight1 dtype ({weight1_dtype})"

        # Validate sharding expectations
        assert (
            input_tensor.memory_config().shard_spec is not None
        ), "Input tensor must have shard_spec (must be sharded)"
        assert weight0.memory_config().shard_spec is not None, "Weight0 must have shard_spec (must be sharded)"
        assert weight1.memory_config().shard_spec is not None, "Weight1 must have shard_spec (must be sharded)"
        assert (
            output_tensor.memory_config().shard_spec is not None
        ), "Output tensor must have shard_spec (must be sharded)"

        # Validate matmul cores don't overlap with mcast core
        all_matmul_cores = weight0.memory_config().shard_spec.grid
        assert not all_matmul_cores.contains(
            FusedResblock.MCAST_CORE
        ), f"Matmul cores {all_matmul_cores} must not contain mcast core {FusedResblock.MCAST_CORE}"

        # Validate weight tile is 32x32 and weight shard width is 1 tile
        # This op only supports single output tile in N dimension (we only iterate over K, not multiple output tiles)
        weight0_shard_shape = weight0.memory_config().shard_spec.shape
        weight1_shard_shape = weight1.memory_config().shard_spec.shape
        weight0_shard_tiles_h = weight0_shard_shape[0] // weight0_tile.tile_shape[0]
        weight0_shard_tiles_w = weight0_shard_shape[1] // weight0_tile.tile_shape[1]
        weight1_shard_tiles_h = weight1_shard_shape[0] // weight1_tile.tile_shape[0]
        weight1_shard_tiles_w = weight1_shard_shape[1] // weight1_tile.tile_shape[1]

        assert (
            weight0_tile.tile_shape[0] == 32 and weight0_tile.tile_shape[1] == 32
        ), f"Weight0 tile must be exactly 32x32, got {weight0_tile.tile_shape}"
        assert (
            weight1_tile.tile_shape[0] == 32 and weight1_tile.tile_shape[1] == 32
        ), f"Weight1 tile must be exactly 32x32, got {weight1_tile.tile_shape}"
        assert (
            weight0_shard_tiles_w == 1
        ), f"Weight0 shard width must be exactly 1 tile (got {weight0_shard_tiles_w} tiles). This op only iterates over K dimension and does not support multiple output tiles in N dimension. Shard shape: {weight0_shard_shape} elements, tile: {weight0_tile.tile_shape}"
        assert (
            weight1_shard_tiles_w == 1
        ), f"Weight1 shard width must be exactly 1 tile (got {weight1_shard_tiles_w} tiles). This op only iterates over K dimension and does not support multiple output tiles in N dimension. Shard shape: {weight1_shard_shape} elements, tile: {weight1_tile.tile_shape}"

        # Validate that we only support single output tile per core (M=1 tile, N=1 tile per shard) - we only iterate over K
        # The output is width-sharded, so we check the shard shape, not the global shape
        out_shape = output_tensor.shape
        out_tile = output_tensor.get_tile()
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        out_shard_tiles_m = out_shard_shape[0] // out_tile.tile_shape[0]
        out_shard_tiles_n = out_shard_shape[1] // out_tile.tile_shape[1]
        assert (
            out_shard_tiles_m == 1
        ), f"Output shard M dimension must be exactly 1 tile (got {out_shard_tiles_m} tiles). This op only supports single output tile in M dimension per core and iterates over K dimension only. Shard shape: {out_shard_shape} elements, tile: {out_tile.tile_shape}"
        assert (
            out_shard_tiles_n == 1
        ), f"Output shard N dimension must be exactly 1 tile (got {out_shard_tiles_n} tiles). This op only supports single output tile in N dimension per core and iterates over K dimension only. Shard shape: {out_shard_shape} elements, tile: {out_tile.tile_shape}"

        mm1_full_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB, input_tensor
        )
        weight0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB, weight0
        )
        weight1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB, weight1
        )
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.OUT_CB, output_tensor
        )

        out_dtype = output_tensor.dtype
        out_tile_size = out_tile.get_tile_size(out_dtype)
        out_tile_descriptor = ttnn.TileDescriptor(out_tile)

        logger.debug(f"Placing mcast core at: {FusedResblock.MCAST_CORE}")
        gather_destination_core = device.worker_core_from_logical_core(FusedResblock.MCAST_CORE)
        all_mcast_cores = ttnn.CoreRangeSet({ttnn.CoreRange(FusedResblock.MCAST_CORE, FusedResblock.MCAST_CORE)})

        intermediate_pregather_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        intermediate_pregather_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_matmul_cores,
            format_descriptors=[intermediate_pregather_cb_format],
        )
        mm2_full_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        mm2_full_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size * num_tiles_k,
            core_ranges=all_matmul_cores.merge(
                all_mcast_cores
            ),  # Include mcast cores to ensure all cores can get the correct CB address with get_write_ptr
            format_descriptors=[mm2_full_cb_format],
        )

        logger.debug(
            f"mm1_full_cb_descriptor: {mm1_full_cb_descriptor.total_size}, page_size: {mm1_full_cb_descriptor.format_descriptors[0].page_size}"
        )
        logger.debug(
            f"mm2_full_cb_descriptor: {mm2_full_cb_descriptor.total_size}, page_size: {mm2_full_cb_descriptor.format_descriptors[0].page_size}"
        )
        logger.debug(
            f"intermediate_pregather_cb_descriptor: {intermediate_pregather_cb_descriptor.total_size}, page_size: {intermediate_pregather_cb_descriptor.format_descriptors[0].page_size}"
        )
        logger.debug(
            f"out_cb_descriptor: {out_cb_descriptor.total_size}, page_size: {out_cb_descriptor.format_descriptors[0].page_size}"
        )

        all_cores = all_matmul_cores.merge(all_mcast_cores)
        assert (
            all_cores.size() == all_matmul_cores.size() + all_mcast_cores.size()
        ), "All cores must be the same size as all matmul cores plus one for the mcast core"
        logger.debug(f"All cores: {all_cores}")

        mcast_receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=all_cores,
            initial_value=0,
        )
        mcast_sender_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=1,
            core_ranges=all_cores,
            initial_value=0,
        )

        # Get input tensor buffer address for mcast core to use directly instead of get_write_ptr
        input_buffer_address = input_tensor.buffer_address()

        (
            mcast_reader_kernel_descriptor,
            mcast_writer_kernel_descriptor,
            mcast_cb_descriptor,
        ) = FusedResblock.create_mcast_kernel(
            all_mcast_cores,
            all_matmul_cores,
            out_dtype,
            out_tile_size,
            num_tiles_k,
            out_tile_descriptor,
            mcast_receiver_semaphore_descriptor,
            mcast_sender_semaphore_descriptor,
            device,
            debug,
            num_layers,
            input_buffer_address,
        )

        sender_core_range = all_matmul_cores.ranges()[0]
        sender_logical_x_start = sender_core_range.start.x
        sender_logical_y_start = sender_core_range.start.y
        sender_logical_x_end = sender_core_range.end.x
        sender_logical_y_end = sender_core_range.end.y
        sender_grid_width = sender_logical_x_end - sender_logical_x_start + 1

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
                FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
                FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.OUT_CB,
                num_tiles_k,
                gather_destination_core.x,
                gather_destination_core.y,
                mcast_receiver_semaphore_descriptor.id,
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                mcast_sender_semaphore_descriptor.id,
                sender_logical_x_start,
                sender_logical_y_start,
                sender_grid_width,
                num_layers,
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.OUT_CB,
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.MM1_FULL_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
                FusedResblock.MatmulCoreCBIndex.OUT_CB,
                FusedResblock.MatmulCoreCBIndex.INTERMEDIATE_PREGATHER_CB,
                FusedResblock.MatmulCoreCBIndex.MM2_FULL_CB,
                num_tiles_k,
                1 if fp32_dest_acc_en else 0,
                num_layers,
            ],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,  # Match C++ op behavior
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
        )

        return ttnn.generic_op(
            [input_tensor, weight0, weight1, output_tensor],
            ttnn.ProgramDescriptor(
                kernels=[
                    reader_kernel_descriptor,
                    writer_kernel_descriptor,
                    compute_kernel_descriptor,
                    mcast_reader_kernel_descriptor,
                    mcast_writer_kernel_descriptor,
                ],
                cbs=[
                    mm1_full_cb_descriptor,
                    weight0_cb_descriptor,
                    weight1_cb_descriptor,
                    out_cb_descriptor,
                    intermediate_pregather_cb_descriptor,
                    mm2_full_cb_descriptor,
                    mcast_cb_descriptor,
                ],
                semaphores=[mcast_receiver_semaphore_descriptor, mcast_sender_semaphore_descriptor],
            ),
        )
