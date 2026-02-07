# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Gated Local Reduce fused operation.

Composes three LocalReduceSilu operations:
    Phase 1: reduce(group1) + SiLU -> intermed[0]  (ADD)
    Phase 2: reduce(group2)        -> intermed[1]  (ADD)
    Phase 3: intermed[0] * intermed[1] -> out      (MUL)

This is useful for gated MLP patterns where:
  - Group 1 is the "gate" path (with SiLU activation)
  - Group 2 is the "up" path (no activation)
  - Final result is element-wise product (gate * up)
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class GatedLocalReduceOp:
    """
    Two reductions (group1 with SiLU, group2 without), then multiply.

    Computes: SiLU(reduce(group1)) * reduce(group2)
    """

    @staticmethod
    def golden(group1_inputs, group2_inputs):
        """
        PyTorch reference implementation.

        Args:
            group1_inputs: List of tensors for group 1 (SiLU applied)
            group2_inputs: List of tensors for group 2 (no SiLU)

        Returns:
            SiLU(reduce(group1)) * reduce(group2)
        """
        # Sum group 1 and apply SiLU
        group1_sum = group1_inputs[0]
        for inp in group1_inputs[1:]:
            group1_sum = group1_sum + inp
        group1_sum = torch.nn.functional.silu(group1_sum)

        # Sum group 2 (no SiLU)
        group2_sum = group2_inputs[0]
        for inp in group2_inputs[1:]:
            group2_sum = group2_sum + inp

        # Combine with element-wise multiply
        return group1_sum * group2_sum

    @staticmethod
    def op(
        input_tensor_group1,
        input_tensor_group2,
        output_tensor,
        group1_num_tiles,
        group2_num_tiles,
    ):
        """
        Execute gated local reduce: SiLU on group1, no SiLU on group2, then multiply.

        Args:
            input_tensor_group1: Input tensor for group 1 (shape [N*tile_h, tile_w])
            input_tensor_group2: Input tensor for group 2 (shape [M*tile_h, tile_w])
            output_tensor: Pre-allocated output tensor (shape [tile_h, tile_w])
            group1_num_tiles: Number of tiles in group 1
            group2_num_tiles: Number of tiles in group 2

        Returns:
            Output tensor with SiLU(reduce(group1)) * reduce(group2)
        """
        all_cores = input_tensor_group1.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, "Only single core is supported"
        assert group1_num_tiles >= 2, "Need at least 2 tiles in group 1"
        assert group1_num_tiles % 2 == 0, "Group 1 tile count must be even"
        assert group2_num_tiles >= 2, "Need at least 2 tiles in group 2"
        assert group2_num_tiles % 2 == 0, "Group 2 tile count must be even"

        # CB indices (4 CBs total)
        in0_cb = 0  # Group 1 input
        in1_cb = 1  # Group 2 input
        intermed_cb = 2  # Intermediate: holds 2 tiles (group1 result + group2 result)
        out_cb = 3  # Final output

        # GatedReduce requires all CBs to share the same data format (binary_op_init_common called once)
        assert input_tensor_group1.dtype == input_tensor_group2.dtype == output_tensor.dtype, (
            f"GatedReduce requires matching dtypes: group1={input_tensor_group1.dtype}, "
            f"group2={input_tensor_group2.dtype}, output={output_tensor.dtype}"
        )

        # Get tile info from output tensor for intermediate buffer
        output_tile = output_tensor.tile
        tile_h, tile_w = output_tile.tile_shape
        data_format = output_tensor.dtype
        tile_size = output_tile.get_tile_size(data_format)
        tile_descriptor = ttnn.TileDescriptor(tile_h, tile_w, False)

        # CB descriptors for inputs (backed by tensors)
        in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_tensor_group1)
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, input_tensor_group2)

        # Intermediate CB: 2 tiles (group1 result + group2 result)
        intermed_format = ttnn.CBFormatDescriptor(
            buffer_index=intermed_cb,
            data_format=data_format,
            page_size=tile_size,
            tile=tile_descriptor,
        )
        intermed_cb_descriptor = ttnn.CBDescriptor(
            total_size=2 * tile_size,  # 2 tiles
            core_ranges=all_cores,
            format_descriptors=[intermed_format],
        )

        # Output CB (backed by output tensor)
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # Named compile-time args for NCRISC
        ncrisc_named_compile_time_args = [
            ("gated_local_reduce_in0_cb", in0_cb),
            ("gated_local_reduce_in1_cb", in1_cb),
            ("gated_local_reduce_group1_num_tiles", group1_num_tiles),
            ("gated_local_reduce_group2_num_tiles", group2_num_tiles),
        ]

        # Named compile-time args for TRISC
        trisc_named_compile_time_args = [
            ("gated_local_reduce_in0_cb", in0_cb),
            ("gated_local_reduce_in1_cb", in1_cb),
            ("gated_local_reduce_intermed_cb", intermed_cb),
            ("gated_local_reduce_out_cb", out_cb),
            ("gated_local_reduce_group1_num_tiles", group1_num_tiles),
            ("gated_local_reduce_group2_num_tiles", group2_num_tiles),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/fused_ops/gated_local_reduce/kernels/gated_local_reduce_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=True,  # SiLU uses approximation
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=all_cores,
                    value=1,
                    other_value=0,
                ),
            ],
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[in0_cb_descriptor, in1_cb_descriptor, intermed_cb_descriptor, out_cb_descriptor],
        )

        io_tensors = [input_tensor_group1, input_tensor_group2, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
