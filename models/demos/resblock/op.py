import torch
from loguru import logger

import ttnn


class FusedResblock:
    class MatmulCoreCBIndex:
        ACTIVATION_CB = 0
        WEIGHT0_CB = 1
        WEIGHT1_CB = 2
        OUT_CB = 3
        INTERM_CB = 4
        INTERM_CB2 = 5

    class McastCoreCBIndex:
        MCAST_CORE_GATHER_CB = 6

    MCAST_CORE = ttnn.CoreCoord(8, 8)

    @staticmethod
    def golden(input_a, weight0, weight1):
        x = input_a @ weight0
        x = torch.nn.functional.relu(x)
        x = x @ weight1
        x = x + input_a
        return x

    @staticmethod
    def create_mcast_kernel(
        all_mcast_cores: ttnn.CoreRangeSet,
        all_sender_cores: ttnn.CoreRangeSet,
        data_format,
        page_size: int,
        tile: ttnn.TileDescriptor,
        receiver_semaphore_descriptor: ttnn.SemaphoreDescriptor,
    ) -> tuple[ttnn.KernelDescriptor, ttnn.KernelDescriptor, ttnn.CBDescriptor]:
        logger.debug(f"All mcast cores: {all_mcast_cores}")
        logger.debug(f"Number of mcast cores: {all_mcast_cores.size()}")
        logger.debug(f"All sender cores: {all_sender_cores}")
        logger.debug(f"Number of sender cores: {all_sender_cores.size()}")

        number_of_senders = all_sender_cores.size()

        # Create CBs for gather destination
        mcast_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
            data_format=data_format,
            page_size=page_size,
            tile=tile,
        )
        mcast_cb_descriptor = ttnn.CBDescriptor(
            total_size=page_size,
            core_ranges=all_mcast_cores,
            format_descriptors=[mcast_cb_format],
        )

        mcast_reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/mcast_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_mcast_cores,
            compile_time_args=[
                FusedResblock.McastCoreCBIndex.MCAST_CORE_GATHER_CB,
                number_of_senders,
                receiver_semaphore_descriptor.id,
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
            ],
            config=ttnn.WriterConfigDescriptor(),
        )
        return mcast_reader_kernel_descriptor, mcast_writer_kernel_descriptor, mcast_cb_descriptor

    @staticmethod
    def op(input_tensor, weight0, weight1, output_tensor, fp32_dest_acc_en=False):
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

        assert (
            input_shape[0] // input_tile.tile_shape[0] == 1
        ), f"M ({input_shape[0]}) must be a single tile with height same as tile_height ({input_tile.tile_shape[0]})"
        assert (
            input_shape[1] % input_tile.tile_shape[1] == 0
        ), f"K ({input_shape[1]}) must be divisible by tile_width ({input_tile.tile_shape[1]})"
        assert (
            weight0_shape[1] // weight0_tile.tile_shape[1] == 1
        ), f"N ({weight0_shape[1]}) must be a single tile with width same as tile_width ({weight0_tile.tile_shape[1]})"
        assert (
            weight1_shape[1] // weight1_tile.tile_shape[1] == 1
        ), f"N ({weight1_shape[1]}) must be a single tile with width same as tile_width ({weight1_tile.tile_shape[1]})"
        assert (
            input_shape[1] == weight0_shape[0]
        ), f"input K ({input_shape[1]}) must equal weight0 K ({weight0_shape[0]})"

        num_tiles_k = input_shape[1] // input_tile.tile_shape[1]

        # TODO: Equality comparison checks exact match so we can't use this for now
        # assert weight0.memory_config().shard_spec.grid == weight1.memory_config().shard_spec.grid, f"Weights must have the same core grid"
        all_matmul_cores = (
            weight0.memory_config().shard_spec.grid
        )  # All matmul cores are the same for weight0 and weight1

        activation_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(
            FusedResblock.MatmulCoreCBIndex.ACTIVATION_CB, input_tensor
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

        out_shape = output_tensor.shape
        out_tile = output_tensor.get_tile()
        assert (
            out_shape[0] // out_tile.tile_shape[0] == 1
        ), f"M ({out_shape[0]}) must be a single tile with height same as tile_height ({out_tile.tile_shape[0]})"
        assert (
            out_shape[1] // out_tile.tile_shape[1] == 1
        ), f"N ({out_shape[1]}) must be a single tile with width same as tile_width ({out_tile.tile_shape[1]})"
        out_dtype = output_tensor.dtype
        out_tile_size = out_tile.get_tile_size(out_dtype)
        out_tile_descriptor = ttnn.TileDescriptor(out_tile)

        interm_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.INTERM_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_matmul_cores,
            format_descriptors=[interm_cb_format],
        )
        interm_cb2_format = ttnn.CBFormatDescriptor(
            buffer_index=FusedResblock.MatmulCoreCBIndex.INTERM_CB2,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        interm_cb2_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_matmul_cores,
            format_descriptors=[interm_cb2_format],
        )

        logger.debug(f"Placing mcast core at: {FusedResblock.MCAST_CORE}")
        gather_destination_core = device.worker_core_from_logical_core(FusedResblock.MCAST_CORE)
        all_mcast_cores = ttnn.CoreRangeSet({ttnn.CoreRange(FusedResblock.MCAST_CORE, FusedResblock.MCAST_CORE)})

        all_cores = all_matmul_cores.merge(all_mcast_cores)
        assert (
            all_cores.size() == all_matmul_cores.size() + all_mcast_cores.size()
        ), f"All cores must be the same size as all matmul cores plus one for the mcast core"
        logger.debug(f"All cores: {all_cores}")

        receiver_semaphore_descriptor = ttnn.SemaphoreDescriptor(
            id=0,
            core_ranges=all_cores,
            initial_value=0,
        )

        (
            mcast_reader_kernel_descriptor,
            mcast_writer_kernel_descriptor,
            mcast_cb_descriptor,
        ) = FusedResblock.create_mcast_kernel(
            all_mcast_cores,
            all_matmul_cores,
            out_dtype,
            out_tile_size,
            out_tile_descriptor,
            receiver_semaphore_descriptor,
        )

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.ACTIVATION_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
                FusedResblock.MatmulCoreCBIndex.INTERM_CB,
                FusedResblock.MatmulCoreCBIndex.INTERM_CB2,
                num_tiles_k,
                gather_destination_core.x,
                gather_destination_core.y,
                receiver_semaphore_descriptor.id,
            ],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[FusedResblock.MatmulCoreCBIndex.OUT_CB],
            config=ttnn.WriterConfigDescriptor(),
        )
        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_matmul_cores,
            compile_time_args=[
                FusedResblock.MatmulCoreCBIndex.ACTIVATION_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT0_CB,
                FusedResblock.MatmulCoreCBIndex.WEIGHT1_CB,
                FusedResblock.MatmulCoreCBIndex.OUT_CB,
                FusedResblock.MatmulCoreCBIndex.INTERM_CB,
                FusedResblock.MatmulCoreCBIndex.INTERM_CB2,
                num_tiles_k,
                1 if fp32_dest_acc_en else 0,
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
                    activation_cb_descriptor,
                    weight0_cb_descriptor,
                    weight1_cb_descriptor,
                    out_cb_descriptor,
                    interm_cb_descriptor,
                    interm_cb2_descriptor,
                    mcast_cb_descriptor,
                ],
                semaphores=[receiver_semaphore_descriptor],
            ),
        )
