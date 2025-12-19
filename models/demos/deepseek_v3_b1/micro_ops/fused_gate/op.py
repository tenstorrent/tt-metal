# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


class FusedGateSingleCore:
    """
    Single-core fused gate (sigmoid) implementation using ttnn.generic_op.

    This class implements sigmoid as a static operation for single-core execution.
    """

    @staticmethod
    def golden(input_tensor, bias_tensor):
        """
        PyTorch reference implementation of sigmoid for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor)
            bias_tensor: Bias tensor (torch.Tensor)
        Returns:
            Output tensor with sigmoid applied and bias added
        """
        sigmoid_scores = torch.sigmoid(input_tensor)
        return torch.cat([sigmoid_scores, sigmoid_scores + bias_tensor], dim=1)

    @staticmethod
    def op(input_tensor, bias_tensor, output_tensor, fast_and_approx=False):
        """
        Execute sigmoid operation using generic_op.

        Args:
            input_tensor: Input tensor (must be sharded)
            bias_tensor: Bias tensor (must be sharded)
            output_tensor: Pre-allocated output tensor (must be sharded, same shape as input)
            fast_and_approx: Whether to use fast approximation for sigmoid

        Returns:
            Output tensor with sigmoid applied
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        output_shape = output_tensor.shape

        # Get tile info from input tensor
        input_tile = input_tensor.tile
        input_tile_height, input_tile_width = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(input_tensor.dtype)

        # Get tile info from bias tensor
        bias_tile = bias_tensor.tile
        bias_tile_size = bias_tile.get_tile_size(bias_tensor.dtype)

        assert input_tensor.dtype == bias_tensor.dtype, "Input and bias tensors must have the same data type"
        assert input_tensor.layout == bias_tensor.layout, "Input and bias tensors must have the same layout"
        assert input_tile == bias_tile, "Input and bias tiles must be the same"

        # Get tile info from output tensor
        output_tile = output_tensor.tile
        output_tile_height, output_tile_width = output_tile.tile_shape
        output_tile_size = output_tile.get_tile_size(output_tensor.dtype)

        assert output_tensor.dtype == input_tensor.dtype, "Output tensor must have the same data type as input tensor"
        assert output_tensor.layout == input_tensor.layout, "Output tensor must have the same layout as input tensor"
        assert input_tile == output_tile, "Input and output tiles must be the same"
        tiny_tile = input_tile != ttnn.Tile([32, 32])

        # Calculate num_tiles from tensor shape
        num_tiles = (input_shape[0] * input_shape[1]) // (input_tile_height * input_tile_width)
        num_output_tiles = (output_shape[0] * output_shape[1]) // (output_tile_height * output_tile_width)
        assert num_output_tiles == num_tiles * 2, "Output tensor must have twice the number of tiles as input tensor"

        # Get core grid from input tensor (single core)
        all_cores = input_tensor.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, f"Only single core is supported"

        # CB indices
        cb_index = 0
        input_cb = cb_index
        cb_index += 1
        bias_cb = cb_index
        cb_index += 1
        sigmoid_cb = cb_index
        cb_index += 1
        output_cb = cb_index
        cb_index += 1

        # Create tile descriptors for proper tile dimensions
        input_tile_descriptor = ttnn.TileDescriptor(input_tile)
        bias_tile_descriptor = ttnn.TileDescriptor(bias_tile)
        output_tile_descriptor = ttnn.TileDescriptor(output_tile)

        # CB: Input (created from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        # Update the tile descriptor in the format descriptor
        in_cb_descriptor.format_descriptors[0].tile = input_tile_descriptor
        in_cb_descriptor.format_descriptors[0].page_size = input_tile_size

        # CB: Bias (created from sharded tensor)
        bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
        # Update the tile descriptor in the format descriptor
        bias_cb_descriptor.format_descriptors[0].tile = bias_tile_descriptor
        bias_cb_descriptor.format_descriptors[0].page_size = bias_tile_size

        # CB: Sigmoid (created from sharded tensor)
        sigmoid_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=sigmoid_cb,
            data_format=input_tensor.dtype,
            page_size=input_tile_size,
            tile=input_tile_descriptor,
        )
        sigmoid_cb_descriptor = ttnn.CBDescriptor(
            total_size=num_tiles * input_tile_size,
            core_ranges=all_cores,
            format_descriptors=[sigmoid_cb_format],
        )

        # CB: Output (created from sharded tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
        # Update the tile descriptor in the format descriptor
        out_cb_descriptor.format_descriptors[0].tile = output_tile_descriptor
        out_cb_descriptor.format_descriptors[0].page_size = output_tile_size

        # Reader kernel
        # Note: input_cb is backed by sharded tensor
        reader_compile_time_args = [
            input_cb,
            bias_cb,
            num_tiles,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/fused_gate/kernels/fused_gate_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            runtime_args=[[[]]],
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        # Note: output_cb is backed by sharded tensor
        writer_compile_time_args = [
            output_cb,
            num_tiles,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/fused_gate/kernels/fused_gate_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel
        compute_compile_time_args = [
            input_cb,
            bias_cb,
            sigmoid_cb,
            output_cb,
            num_tiles,
            tiny_tile,
            fast_and_approx,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/fused_gate/kernels/fused_gate_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            runtime_args=[[[]]],
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )
        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[in_cb_descriptor, bias_cb_descriptor, sigmoid_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_tensor, bias_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
