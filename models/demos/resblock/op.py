import torch
from loguru import logger

import ttnn


class FusedResblock:
    # CB indices
    IN0_CB = 0  # Input A
    WEIGHT0_CB = 1  # Weight0
    WEIGHT1_CB = 2  # Weight1
    OUT_CB = 3  # Output
    INTERM_CB = 4  # Intermediate buffer for accumulation

    @staticmethod
    def golden(input_a, weight0, weight1):
        x = input_a @ weight0
        x = torch.nn.functional.relu(x)
        x = x @ weight1
        return x
        x = x + input_a

    @staticmethod
    def op(input_a, weight0, weight1, output_tensor, fp32_dest_acc_en=False):
        logger.info(f"Running ResBlock operation with shape {input_a.shape} x {weight0.shape} x {weight1.shape}")

        a_shape = input_a.shape
        weight0_shape = weight0.shape
        weight1_shape = weight1.shape
        in0_tile = input_a.get_tile()
        weight0_tile = weight0.get_tile()
        weight1_tile = weight1.get_tile()

        assert (
            a_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"M ({a_shape[0]}) must be a single tile with height same as tile_height ({in0_tile.tile_shape[0]})"
        assert (
            a_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"K ({a_shape[1]}) must be divisible by tile_width ({in0_tile.tile_shape[1]})"
        assert (
            weight0_shape[1] // weight0_tile.tile_shape[1] == 1
        ), f"N ({weight0_shape[1]}) must be a single tile with width same as tile_width ({weight0_tile.tile_shape[1]})"
        assert (
            weight1_shape[1] // weight1_tile.tile_shape[1] == 1
        ), f"N ({weight1_shape[1]}) must be a single tile with width same as tile_width ({weight1_tile.tile_shape[1]})"
        assert a_shape[1] == weight0_shape[0], f"in0 K ({a_shape[1]}) must equal weight0 K ({weight0_shape[0]})"
        num_tiles_k = a_shape[1] // in0_tile.tile_shape[1]

        all_cores = input_a.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, f"Only single core is supported"

        in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(FusedResblock.IN0_CB, input_a)
        weight0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(FusedResblock.WEIGHT0_CB, weight0)
        weight1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(FusedResblock.WEIGHT1_CB, weight1)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(FusedResblock.OUT_CB, output_tensor)

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
            buffer_index=FusedResblock.INTERM_CB,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_cores,
            format_descriptors=[interm_cb_format],
        )

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=[FusedResblock.IN0_CB, FusedResblock.WEIGHT0_CB, FusedResblock.WEIGHT1_CB, num_tiles_k],
            config=ttnn.ReaderConfigDescriptor(),
        )
        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=[FusedResblock.OUT_CB],
            config=ttnn.WriterConfigDescriptor(),
        )

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/resblock/kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=[
                FusedResblock.IN0_CB,
                FusedResblock.WEIGHT0_CB,
                FusedResblock.WEIGHT1_CB,
                FusedResblock.OUT_CB,
                FusedResblock.INTERM_CB,
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

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[
                in0_cb_descriptor,
                weight0_cb_descriptor,
                weight1_cb_descriptor,
                out_cb_descriptor,
                interm_cb_descriptor,
            ],
        )

        return ttnn.generic_op([input_a, weight0, weight1, output_tensor], program_descriptor)
