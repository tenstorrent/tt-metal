# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class DeepseekMoeGateSingleCore:
    """
    Single-core Deepseek Moe Gate implementation using ttnn.generic_op.

    This class implements the Deepseek Moe Gate operation on a single face (16x16) of data.
    The operation is a combination of sigmoid activation, element-wise addition with bias,
    and sorting of top 8 values per group.
    """

    @staticmethod
    def golden(input_tensor, bias_tensor, eps=1e-20, scaling_factor=2.5, enable_sigmoid=False):
        """
        PyTorch reference implementation of Deepseek Moe Gate for validation.

        Args:
            input_tensor: Input tensor (torch.Tensor) of shape [16, 16]
            bias_tensor: Bias tensor (torch.Tensor) of shape [16, 16]
            eps: Epsilon value for normalization
            scaling_factor: Scaling factor for normalization
            enable_sigmoid: Whether to enable sigmoid activation

        Returns:
            Top8 normalized scores tensor (torch.Tensor) of shape [1, 8]
            Top8 indices tensor (torch.Tensor) of shape [1, 8]
        """
        scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
        bias_scores = scores + bias_tensor
        sorted_bias, sorted_indices = torch.sort(bias_scores, dim=-1, descending=True)
        sorted_scores = scores.clone()
        for i in range(sorted_indices.shape[0]):
            sorted_scores[i] = scores[i][sorted_indices[i]]
            sorted_indices[i] += i * sorted_indices.shape[1]
        top2_sum = sorted_bias[:, 0] + sorted_bias[:, 1]
        sorted_top2_sum, sorted_top2_indices = torch.sort(top2_sum, descending=True)
        top4_values = sorted_bias[sorted_top2_indices[:4]].flatten()
        sorted_scores = sorted_scores[sorted_top2_indices[:4]].flatten()
        sorted_indices = sorted_indices[sorted_top2_indices[:4]].flatten()
        top8_values, top8_indices = torch.topk(top4_values, 8, dim=-1, sorted=True)
        top8_values = sorted_scores[top8_indices]
        top8_indices = sorted_indices[top8_indices]
        denominator = torch.sum(top8_values, dim=-1) + eps
        normalized_scores = top8_values / denominator * scaling_factor
        return normalized_scores, top8_indices

    @staticmethod
    def op(
        input_tensor,
        bias_tensor,
        output_tensor,
        input_indices_tensor,
        output_indices_tensor,
        eps=1e-20,
        scaling_factor=2.5,
        enable_sigmoid=False,
    ):
        """
        Execute Deepseek Moe Gate operation using generic_op.

        Args:
            input_tensor: Input tensor with values to sort (must be sharded, shape [16, 16])
            bias_tensor: Transposed bias tensor with values to add (must be sharded, shape [16, 16])
            output_tensor: Pre-allocated output tensor for top8 normalized scores (must be sharded, shape [16, 16])
            input_indices_tensor: Input tensor with transposed indices to sort (must be sharded, shape [16, 16])
            output_indices_tensor: Pre-allocated output tensor for top8 indices (must be sharded, shape [16, 16])
            eps: Epsilon value for normalization
            scaling_factor: Scaling factor for normalization
            enable_sigmoid: Whether to enable sigmoid activation

        Returns:
            output_tensor with top8 normalized scores (must be sharded, shape [16, 16])
            output_indices_tensor with top8 indices (must be sharded, shape [16, 16])
            Note: Only the first 8 values are relevant
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        bias_shape = bias_tensor.shape
        output_shape = output_tensor.shape
        input_indices_shape = input_indices_tensor.shape
        output_indices_shape = output_indices_tensor.shape

        assert input_shape == output_shape, "Input and output tensors must have the same shape"
        assert bias_shape == input_shape, "Bias and input tensors must have the same shape"
        assert input_indices_shape == input_shape, "Input indices and input tensors must have the same shape"
        assert output_indices_shape == input_shape, "Output indices and input tensors must have the same shape"
        assert input_shape[0] == 16, f"Height must be 16 for Deepseek Moe Gate, got {input_shape[0]}"
        assert input_shape[1] == 16, f"Width must be 16 for Deepseek Moe Gate, got {input_shape[1]}"

        # Get tile info from input tensor
        input_tile = input_tensor.tile
        input_tile_height, input_tile_width = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(input_tensor.dtype)
        assert input_tile_height == 16, f"Height must be 16 for Deepseek Moe Gate, got {input_tile_height}"
        assert input_tile_width == 16, f"Width must be 16 for Deepseek Moe Gate, got {input_tile_width}"

        # Get tile info from bias tensor
        bias_tile = bias_tensor.tile
        bias_tile_size = bias_tile.get_tile_size(bias_tensor.dtype)

        # Get tile info from output tensor
        output_tile = output_tensor.tile
        output_tile_size = output_tile.get_tile_size(output_tensor.dtype)

        # Get tile info from input indices tensor
        input_indices_tile = input_indices_tensor.tile
        input_indices_tile_size = input_indices_tile.get_tile_size(input_indices_tensor.dtype)

        # Get tile info from output indices tensor
        output_indices_tile = output_indices_tensor.tile
        output_indices_tile_size = output_indices_tile.get_tile_size(output_indices_tensor.dtype)

        # Get core grid from input tensor (single core)
        all_cores = input_tensor.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, f"Only single core is supported"
        assert all_cores == bias_tensor.memory_config().shard_spec.grid
        assert all_cores == output_tensor.memory_config().shard_spec.grid
        assert all_cores == input_indices_tensor.memory_config().shard_spec.grid
        assert all_cores == output_indices_tensor.memory_config().shard_spec.grid

        # CB indices
        cb_index = 0
        input_cb = cb_index
        cb_index += 1
        bias_cb = cb_index
        cb_index += 1
        output_cb = cb_index
        cb_index += 1
        input_indices_cb = cb_index
        cb_index += 1
        output_indices_cb = cb_index
        cb_index += 1

        # Create tile descriptors
        input_tile_descriptor = ttnn.TileDescriptor(input_tile)
        bias_tile_descriptor = ttnn.TileDescriptor(bias_tile)
        output_tile_descriptor = ttnn.TileDescriptor(output_tile)
        input_indices_tile_descriptor = ttnn.TileDescriptor(input_indices_tile)
        output_indices_tile_descriptor = ttnn.TileDescriptor(output_indices_tile)

        # CB: Input values (created from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        in_cb_descriptor.format_descriptors[0].tile = input_tile_descriptor
        in_cb_descriptor.format_descriptors[0].page_size = input_tile_size

        # CB: Bias values (created from sharded tensor)
        bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
        bias_cb_descriptor.format_descriptors[0].tile = bias_tile_descriptor
        bias_cb_descriptor.format_descriptors[0].page_size = bias_tile_size

        # CB: Output values (created from sharded tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
        out_cb_descriptor.format_descriptors[0].tile = output_tile_descriptor
        out_cb_descriptor.format_descriptors[0].page_size = output_tile_size

        # CB: Input indices (created from sharded tensor)
        in_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_indices_cb, input_indices_tensor)
        in_indices_cb_descriptor.format_descriptors[0].tile = input_indices_tile_descriptor
        in_indices_cb_descriptor.format_descriptors[0].page_size = input_indices_tile_size

        # CB: Output indices (created from sharded tensor)
        out_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor)
        out_indices_cb_descriptor.format_descriptors[0].tile = output_indices_tile_descriptor
        out_indices_cb_descriptor.format_descriptors[0].page_size = output_indices_tile_size

        # Reader kernel
        reader_compile_time_args = [
            input_cb,
            bias_cb,
            input_indices_cb,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/kernels/deepseek_moe_gate_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        writer_compile_time_args = [
            output_cb,
            output_indices_cb,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/kernels/deepseek_moe_gate_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel
        compute_compile_time_args = [
            input_cb,
            bias_cb,
            input_indices_cb,
            output_cb,
            output_indices_cb,
            float_to_uint32(eps),
            float_to_uint32(scaling_factor),
            enable_sigmoid,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/kernels/deepseek_moe_gate_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,  # Makes no difference for this operation
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[
                in_cb_descriptor,
                bias_cb_descriptor,
                out_cb_descriptor,
                in_indices_cb_descriptor,
                out_indices_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [input_tensor, bias_tensor, input_indices_tensor, output_tensor, output_indices_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor, output_indices_tensor
