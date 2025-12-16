# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core matmul operation.

This implements a matmul on a single core where:
- Input A (in0): [1 - 32, K] in L1
- Input B (in1): [K, 32] in L1
- Output: [1 - 32, 32] in L1

The single core computes: output = A @ B
"""

import ttnn


class MatmulSingleCore:
    """
    Single-core matmul implementation using ttnn.generic_op.

    New matmul LLK that only supports outer dims of 1 tile, but uses MOP to loop
    along the inner dim. Hyper optimized for matmuls with shape [1, K] x [K, 32].
    """

    @staticmethod
    def golden(input_a, input_b):
        """
        PyTorch reference implementation of matmul for validation.

        Args:
            input_a: Input tensor A (torch.Tensor) [M, K]
            input_b: Input tensor B (torch.Tensor) [K, N]

        Returns:
            Output tensor [M, N]
        """
        return input_a @ input_b

    @staticmethod
    def op(
        input_a,
        input_b,
        output_tensor,
        fp32_dest_acc_en=False,
    ):
        """
        Execute single-core matmul operation using generic_op.

        Args:
            input_a: Input tensor A [1, K] in L1
            input_b: Input tensor B [K, 32] in L1
            output_tensor: Pre-allocated output tensor [1, 32]
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with matmul result
        """
        # Some basic shape checks on input
        a_shape = input_a.shape
        b_shape = input_b.shape
        in0_tile = input_a.get_tile()
        in1_tile = input_b.get_tile()
        assert (
            a_shape[0] // in0_tile.tile_shape[0] == 1
        ), f"M ({a_shape[0]}) must be a single tile with height same as tile_height ({in0_tile.tile_shape[0]})"
        assert (
            a_shape[1] % in0_tile.tile_shape[1] == 0
        ), f"K ({a_shape[1]}) must be divisible by tile_width ({in0_tile.tile_shape[1]})"
        assert (
            b_shape[1] // in1_tile.tile_shape[1] == 1
        ), f"N ({b_shape[1]}) must be a single tile with width same as tile_width ({in1_tile.tile_shape[1]})"
        assert a_shape[1] == b_shape[0], f"in0 K ({a_shape[1]}) must equal in1 K ({b_shape[0]})"
        num_tiles_k = a_shape[1] // in0_tile.tile_shape[1]

        # Some basic shape checks on output
        out_shape = output_tensor.shape
        out_tile = output_tensor.get_tile()
        assert (
            out_shape[0] // out_tile.tile_shape[0] == 1
        ), f"M ({out_shape[0]}) must be a single tile with height same as tile_height ({out_tile.tile_shape[0]})"
        assert (
            out_shape[1] // out_tile.tile_shape[1] == 1
        ), f"N ({out_shape[1]}) must be a single tile with width same as tile_width ({out_tile.tile_shape[1]})"

        # Get core grid from input tensor (single core)
        all_cores = input_a.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, f"Only single core is supported"

        # Create tile descriptors for intermediate buffer
        out_dtype = output_tensor.dtype
        out_tile_size = out_tile.get_tile_size(out_dtype)
        out_tile_descriptor = ttnn.TileDescriptor(out_tile)

        # CB indices
        in0_cb = 0  # Input A
        in1_cb = 1  # Input B
        out_cb = 2  # Output
        interm_cb = 3  # Intermediate buffer for accumulation

        # CB 0: Input A
        in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_a)

        # CB 1: Input B
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, input_b)

        # CB 2: Output
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # CB 3: Intermediate buffer for partial sums (uses output dtype)
        interm_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=interm_cb,
            data_format=out_dtype,
            page_size=out_tile_size,
            tile=out_tile_descriptor,
        )
        interm_cb_descriptor = ttnn.CBDescriptor(
            total_size=out_tile_size,
            core_ranges=all_cores,
            format_descriptors=[interm_cb_format],
        )

        # Reader kernel - just signals sharded CBs are ready
        reader_compile_args = [
            in0_cb,
            in1_cb,
            num_tiles_k,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        writer_compile_args = [
            out_cb,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel
        compute_compile_args = [
            in0_cb,
            in1_cb,
            out_cb,
            interm_cb,
            num_tiles_k,
            1 if fp32_dest_acc_en else 0,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,  # Match C++ op behavior
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
            ),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[in0_cb_descriptor, in1_cb_descriptor, out_cb_descriptor, interm_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_a, input_b, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
