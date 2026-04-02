# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct

import torch

import ttnn
from models.demos.deepseek_v3.tt.deepseek_moe_gate.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


def float_to_uint32(value):
    """Convert float to uint32"""
    return int.from_bytes(struct.pack("f", value), byteorder="little")


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
        row_offsets = torch.arange(input_tensor.shape[-2]) * input_tensor.shape[-1]
        batch_idx = torch.arange(input_tensor.shape[0]).unsqueeze(-1)

        scores = torch.sigmoid(input_tensor) if enable_sigmoid else input_tensor
        bias_scores = scores + bias_tensor
        sorted_bias, sorted_indices = torch.sort(bias_scores, dim=-1, descending=True)
        sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
        sorted_indices = sorted_indices + row_offsets.view(1, -1, 1)

        top2_sum = sorted_bias[:, :, 0] + sorted_bias[:, :, 1]
        sorted_top2_sum, sorted_top2_indices = torch.sort(top2_sum, dim=-1, descending=True)
        top4_values = sorted_bias[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top4_scores = sorted_scores[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top4_indices = sorted_indices[batch_idx, sorted_top2_indices[:, :4]].flatten(1)
        top8_values, top8_indices = torch.topk(top4_values, 8, dim=-1, sorted=True)
        top8_values = torch.gather(top4_scores, dim=-1, index=top8_indices)
        top8_indices = torch.gather(top4_indices, dim=-1, index=top8_indices)
        denominator = torch.sum(top8_values, dim=-1, keepdim=True) + eps
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
            output_tensor: Pre-allocated output tensor for top8 normalized scores (must be sharded, shape [32, 32])
            input_indices_tensor: Input tensor with transposed indices to sort (must be sharded, shape [16, 16])
            output_indices_tensor: Pre-allocated output tensor for top8 indices (must be sharded, shape [32, 32])
            eps: Epsilon value for normalization
            scaling_factor: Scaling factor for normalization
            enable_sigmoid: Whether to enable sigmoid activation

        Returns:
            output_tensor with top8 normalized scores (shape [32, 32], but only [0, :8] is valid)
            output_indices_tensor with top8 indices (shape [32, 32], but only [0, :8] is valid)

        Note:
            For the output_tensor and output_indices_tensor, it should be 32x32.
            And we will just return the first 8 values from the 32x32 tensor.
        """
        # Get tensor properties
        input_shape = input_tensor.shape
        bias_shape = bias_tensor.shape
        output_shape = output_tensor.shape
        input_indices_shape = input_indices_tensor.shape
        output_indices_shape = output_indices_tensor.shape

        assert bias_shape == input_shape, "Bias and input tensors must have the same shape"
        assert input_indices_shape == input_shape, "Input indices and input tensors must have the same shape"
        assert output_indices_shape == output_shape, "Output indices and output tensors must have the same shape"

        # Get core grid from input tensor
        input_shard_spec = input_tensor.memory_config().shard_spec
        output_shard_spec = output_tensor.memory_config().shard_spec
        all_cores = input_shard_spec.grid
        # In deepseek_v3_b1, the input and input indices are tiled with a size of 16x16.
        # Since there is currently no function to change the tile size, we use a tile size of 32x32 here.
        # Note that the input is still 16x16 from the projection before the gate, and its layout matches the 16x16 tiling.
        # For the output, the op in deepseek_v3_b1 expects a 1x16 tiling, but we use 32x32 tiling.
        # However, the first 8 elements of the output are consistent with the 1x16 tiled layout.
        assert input_shape[-1] * input_shape[-2] == 256, "Input tensor must have 256 elements"
        assert input_indices_shape[-1] * input_indices_shape[-2] == 256, "Input indices tensor must have 256 elements"
        assert input_shard_spec == bias_tensor.memory_config().shard_spec
        assert input_shard_spec == input_indices_tensor.memory_config().shard_spec
        assert output_shard_spec == output_indices_tensor.memory_config().shard_spec
        assert all_cores == output_shard_spec.grid

        # Get tile info from input tensor
        input_tile = input_tensor.tile
        input_tile_height, input_tile_width = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(input_tensor.dtype)
        expected_input_tile_size = (32, 32)
        output_tile = output_tensor.tile
        output_tile_height, output_tile_width = output_tile.tile_shape
        output_tile_size = output_tile.get_tile_size(output_tensor.dtype)
        expected_output_tile_size = (32, 32)
        assert input_tile == bias_tensor.tile
        assert input_tile == input_indices_tensor.tile
        assert output_tile == output_indices_tensor.tile
        assert input_tile_height == expected_input_tile_size[0]
        assert input_tile_width == expected_input_tile_size[1]
        assert output_tile_height == expected_output_tile_size[0]
        assert output_tile_width == expected_output_tile_size[1]
        assert input_shard_spec.shape[0] == expected_input_tile_size[0]
        assert input_shard_spec.shape[1] == expected_input_tile_size[1]
        assert output_shard_spec.shape[0] == expected_output_tile_size[0]
        assert output_shard_spec.shape[1] == expected_output_tile_size[1]

        # Get tile info from bias tensor
        bias_tile = bias_tensor.tile
        bias_tile_size = bias_tile.get_tile_size(bias_tensor.dtype)

        # Get tile info from input indices tensor
        input_indices_tile = input_indices_tensor.tile
        input_indices_tile_size = input_indices_tile.get_tile_size(input_indices_tensor.dtype)

        # Get tile info from output indices tensor
        output_indices_tile = output_indices_tensor.tile
        output_indices_tile_size = output_indices_tile.get_tile_size(output_indices_tensor.dtype)

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

        # Page size must divide total_size (circular_buffer_config constraint).
        # total_size cannot exceed the L1 buffer (e.g. 2048 B). Use page_size = total_size to satisfy both.
        def _set_page_size(desc, tile_descriptor, tile_size):
            desc.format_descriptors[0].tile = tile_descriptor
            desc.format_descriptors[0].page_size = tile_size if desc.total_size % tile_size == 0 else desc.total_size

        # CB: Input values (created from sharded tensor)
        in_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_cb, input_tensor)
        _set_page_size(in_cb_descriptor, input_tile_descriptor, input_tile_size)

        # CB: Bias values (created from sharded tensor)
        bias_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(bias_cb, bias_tensor)
        _set_page_size(bias_cb_descriptor, bias_tile_descriptor, bias_tile_size)

        # CB: Output values (created from sharded tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_cb, output_tensor)
        _set_page_size(out_cb_descriptor, output_tile_descriptor, output_tile_size)

        # CB: Input indices (created from sharded tensor)
        in_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(input_indices_cb, input_indices_tensor)
        _set_page_size(in_indices_cb_descriptor, input_indices_tile_descriptor, input_indices_tile_size)

        # CB: Output indices (created from sharded tensor)
        out_indices_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(output_indices_cb, output_indices_tensor)
        _set_page_size(out_indices_cb_descriptor, output_indices_tile_descriptor, output_indices_tile_size)

        # ========== UNIFIED KERNEL DESCRIPTOR ==========
        # Core logic is in unified_kernels/deepseek_moe_gate.hpp
        KERNEL_PATH = "models/demos/deepseek_v3/tt/deepseek_moe_gate/kernels/deepseek_moe_gate_kernel.cpp"

        # Named compile-time args for NCRISC (reader - signals tensor-backed CBs ready)
        ncrisc_named_compile_time_args = [
            ("moe_gate_input_cb", input_cb),
            ("moe_gate_bias_cb", bias_cb),
            ("moe_gate_input_indices_cb", input_indices_cb),
        ]

        # Named compile-time args for BRISC (writer - waits for output CBs)
        brisc_named_compile_time_args = [
            ("moe_gate_output_cb", output_cb),
            ("moe_gate_output_indices_cb", output_indices_cb),
        ]

        # Named compile-time args for TRISC (compute - gate logic)
        trisc_named_compile_time_args = [
            ("moe_gate_input_cb", input_cb),
            ("moe_gate_bias_cb", bias_cb),
            ("moe_gate_input_indices_cb", input_indices_cb),
            ("moe_gate_output_cb", output_cb),
            ("moe_gate_output_indices_cb", output_indices_cb),
            ("moe_gate_eps", float_to_uint32(eps)),
            ("moe_gate_scaling_factor", float_to_uint32(scaling_factor)),
            ("moe_gate_enable_sigmoid", 1 if enable_sigmoid else 0),
        ]

        # Unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL_PATH,
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=brisc_named_compile_time_args,
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,  # Makes no difference for this operation
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="moe_gate_is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        kernel_descriptors = unified_kernel.get_kernel_descriptors()

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors.kernels,
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
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor, output_indices_tensor
