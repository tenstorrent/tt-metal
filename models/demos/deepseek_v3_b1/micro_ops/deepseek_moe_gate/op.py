# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class DeepseekMoeGateSingleCore:
    """
    Single-core Deepseek Moe Gate implementation using ttnn.generic_op.

    This class implements the Deepseek Moe Gate operation on a single face (16x16) of data.
    The operation is a combination of sigmoid activation, element-wise addition with bias,
    and sorting of top 8 values per group.

    .. note:: Kimi K2.6 compatibility

       This kernel is **structurally specific to DeepSeek V3 / R1's grouped routing**:
       16 groups × 16 experts (=256), top-2-per-group → top-4-groups → top-8 overall.
       The ``(16, 16)`` tile shape is asserted at line ~114 (``expected_input_tile_size``).
       The hardcode also lives in two layers below this one:
       ``ttnn.experimental.deepseek_prefill.moe_grouped_topk`` has
       ``TT_FATAL(n_groups == 8, topk_groups == 4, n_experts == 256)``, and the
       underlying SFPU bitonic-topk LLK is a 16-element-face primitive.

       Kimi K2.6 cannot use this kernel as-is — its routing config is
       1 group × 384 experts, plain ``topk(sigmoid(scores) + bias, k=8)``, i.e.
       the degenerate one-group case of grouped routing.

       Three reframed options for whoever picks this up (full breakdown in
       ``KIMI_K26_PORT_NOTES`` at the end of
       ``models/demos/deepseek_v3_b1/tests/unit_tests/host_io_decoder_harness.py``):

       - **C1' (host-fallback gate, ~1-2 days, RECOMMENDED for validation infra).**
         Port ``kimi26_d_p``'s ``GateComputeMode.HOST_GROUPED_GATE`` recipe: keep
         the gate matmul on device, pull logits to host, run grouped-topk in
         torch (handles ``n_groups=1`` trivially), push results back. Reference
         algorithm is ``DeepseekMoeGateSingleCore.golden`` above, just
         parameterized over ``(n_groups, topk_groups, n_experts, top_k,
         routed_scaling_factor)``. This is what shipping kimi26_d_p does today.
       - **C2 (device kernel generalization, weeks of LLK work).** Lift the
         three TT_FATALs in ``moe_grouped_topk_device_operation.cpp`` and teach
         the SFPU bitonic-topk primitive to handle 384 experts (the 16-element
         face primitive doesn't fit 384 directly — likely a multi-tile path).
         **No active upstream branch is doing this work**;
         ``origin/gchoudhary/41826-generalize-moe_compute-shape-support`` and
         ``origin/dchen/moe_compute*`` generalize the *expert* side, not the
         gate. Don't block on them.
       - **C3 (new ungrouped kernel, ~3-4 weeks).** Write
         ``UngroupedTopKGateSingleCore`` alongside this one (e.g. ``(16, 24)``
         or ``(32, 16)`` padded with –∞ dummies). Likely superseded by C2.

       The ``weight_key_prefix`` knob for Kimi's ``language_model.`` HF prefix
       already exists on ``CacheWeightProvider`` (added 2026-05-15); the gate
       routing above is the last blocker for Kimi MoE-layer on-device
       validation. **C1' is the right next step** if the goal is correctness
       validation against bit_sculpt reference traces rather than production
       throughput.
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

        assert bias_shape == input_shape, "Bias and input tensors must have the same shape"
        assert input_indices_shape == input_shape, "Input indices and input tensors must have the same shape"
        assert output_indices_shape == output_shape, "Output indices and output tensors must have the same shape"

        # Get core grid from input tensor
        input_shard_spec = input_tensor.memory_config().shard_spec
        output_shard_spec = output_tensor.memory_config().shard_spec
        all_cores = input_shard_spec.grid
        assert input_shard_spec == bias_tensor.memory_config().shard_spec
        assert input_shard_spec == input_indices_tensor.memory_config().shard_spec
        assert output_shard_spec == output_indices_tensor.memory_config().shard_spec
        assert all_cores == output_shard_spec.grid

        # Get tile info from input tensor
        input_tile = input_tensor.tile
        input_tile_height, input_tile_width = input_tile.tile_shape
        input_tile_size = input_tile.get_tile_size(input_tensor.dtype)
        expected_input_tile_size = (16, 16)
        output_tile = output_tensor.tile
        output_tile_height, output_tile_width = output_tile.tile_shape
        output_tile_size = output_tile.get_tile_size(output_tensor.dtype)
        expected_output_tile_size = (1, 16)
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

        # ========== UNIFIED KERNEL DESCRIPTOR ==========
        # Core logic is in unified_kernels/deepseek_moe_gate.hpp
        KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/kernels/deepseek_moe_gate_kernel.cpp"

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
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor, output_indices_tensor
