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
    along the inner dim. Hyper optimized for matmuls with shape [1 - 32, K] x [K, 32].
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
        core_grid,
        tile_height=1,
        tile_width=32,
        fp32_dest_acc_en=False,
    ):
        """
        Execute single-core matmul operation using generic_op.

        Args:
            input_a: Input tensor A [1 - 32, K] in L1
            input_b: Input tensor B [K, 32] in L1
            output_tensor: Pre-allocated output tensor [1 - 32, 32]
            core_grid: Core grid for the operation (single core)
            tile_height: Height of tiles (default 1 for tiny tiles)
            tile_width: Width of tiles (default 32)
            fp32_dest_acc_en: Whether to enable FP32 accumulation

        Returns:
            Output tensor with matmul result
        """
        device = input_a.device()

        # Get tensor shapes
        a_shape = input_a.shape
        b_shape = input_b.shape

        M = a_shape[0]  # Usually 1 for tiny tiles
        K = a_shape[1]
        N = b_shape[1]

        # Calculate number of tiles
        num_tiles_m = M // tile_height
        num_tiles_k = K // tile_width
        num_tiles_n = N // tile_width

        # Get number of cores
        num_cores = core_grid.num_cores

        # Single core processes one output tile
        assert num_tiles_n == num_cores, f"num_tiles_n ({num_tiles_n}) must equal num_cores ({num_cores})"

        # Get data format info - each tensor may have different dtype
        in0_dtype = input_a.dtype
        in1_dtype = input_b.dtype
        out_dtype = output_tensor.dtype

        in0_tile = ttnn.Tile((tile_height, tile_width))
        in1_tile = ttnn.Tile((tile_width, tile_width))  # B has standard tiles for K dimension
        out_tile = ttnn.Tile((tile_height, tile_width))

        in0_tile_size = in0_tile.get_tile_size(in0_dtype)
        in1_tile_size = in1_tile.get_tile_size(in1_dtype)
        out_tile_size = out_tile.get_tile_size(out_dtype)

        # Create tile descriptors
        in0_tile_descriptor = ttnn.TileDescriptor(in0_tile)
        in1_tile_descriptor = ttnn.TileDescriptor(in1_tile)
        out_tile_descriptor = ttnn.TileDescriptor(out_tile)

        # CB indices
        in0_cb = 0  # Input A
        in1_cb = 1  # Input B
        out_cb = 2  # Output
        interm_cb = 3  # Intermediate buffer for accumulation

        # Create core range for all cores
        core_range = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(core_grid.x - 1, core_grid.y - 1))
        all_cores = ttnn.CoreRangeSet([core_range])

        # CB 0: Input A
        in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_a)
        in0_cb_descriptor.format_descriptors[0].tile = in0_tile_descriptor
        in0_cb_descriptor.format_descriptors[0].page_size = in0_tile_size

        # CB 1: Input B
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, input_b)
        in1_cb_descriptor.format_descriptors[0].tile = in1_tile_descriptor
        in1_cb_descriptor.format_descriptors[0].page_size = in1_tile_size

        # CB 2: Output
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)
        out_cb_descriptor.format_descriptors[0].tile = out_tile_descriptor
        out_cb_descriptor.format_descriptors[0].page_size = out_tile_size

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

        # Empty runtime args (no DRAM reading needed)
        reader_runtime_args = [[[] for _ in range(core_grid.y)] for _ in range(core_grid.x)]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_args,
            runtime_args=reader_runtime_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        writer_compile_args = [
            out_cb,
            1,  # num_output_tiles per core
        ]

        # Empty runtime args for writer
        writer_runtime_args = [[[] for _ in range(core_grid.y)] for _ in range(core_grid.x)]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_args,
            runtime_args=writer_runtime_args,
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

        # Empty runtime args for compute
        compute_runtime_args = [[[] for _ in range(core_grid.y)] for _ in range(core_grid.x)]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_args,
            runtime_args=compute_runtime_args,
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
            semaphores=[],
            cbs=[in0_cb_descriptor, in1_cb_descriptor, out_cb_descriptor, interm_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_a, input_b, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
