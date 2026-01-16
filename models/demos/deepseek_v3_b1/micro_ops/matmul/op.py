# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core matmul operation.

This implements a matmul on a single core where:
- Input A (in0): [1 - 32, K] in L1
- Input B (in1): [K, N] in L1 (N up to 4 tiles)
- Output: [1 - 32, N] in L1

The single core computes: output = A @ B
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)


class MatmulSingleCore:
    """
    Single-core matmul implementation using ttnn.generic_op.

    New matmul LLK that only supports outer dims of 1 tile, but uses MOP to loop
    along the inner dim. Hyper optimized for matmuls with shape [1, K] x [K, N]
    where N is up to 4 tiles (128 elements).
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
            input_b: Input tensor B [K, N] in L1 (N up to 4 tiles)
            output_tensor: Pre-allocated output tensor [1, N]
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
        assert a_shape[1] == b_shape[0], f"in0 K ({a_shape[1]}) must equal in1 K ({b_shape[0]})"
        num_tiles_k = a_shape[1] // in0_tile.tile_shape[1]

        # Some basic shape checks on output
        out_shape = output_tensor.shape
        out_tile = output_tensor.get_tile()
        assert (
            out_shape[0] // out_tile.tile_shape[0] == 1
        ), f"M ({out_shape[0]}) must be a single tile with height same as tile_height ({out_tile.tile_shape[0]})"

        # Calculate output width in tiles
        out_w = out_shape[1] // out_tile.tile_shape[1]

        # Get core grid from input tensor (single core)
        all_cores = input_a.memory_config().shard_spec.grid
        assert all_cores.num_cores() == 1, f"Only single core is supported"

        # CB indices
        in0_cb = 0  # Input A
        in1_cb = 1  # Input B
        out_cb = 2  # Output

        # CB 0: Input A
        in0_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in0_cb, input_a)

        # CB 1: Input B
        in1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(in1_cb, input_b)

        # CB 2: Output
        out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor)

        # Named compile-time args for NCRISC
        ncrisc_named_compile_time_args = [
            ("matmul_in0", in0_cb),
            ("matmul_in1", in1_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", out_w),
        ]

        # Named compile-time args for TRISC
        trisc_named_compile_time_args = [
            ("matmul_in0", in0_cb),
            ("matmul_in1", in1_cb),
            ("matmul_out", out_cb),
            ("matmul_k_num_tiles", num_tiles_k),
            ("matmul_out_w", out_w),
        ]

        # Unified kernel descriptor
        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul/kernels/matmul_kernel.cpp",
            core_ranges=all_cores,
            ncrisc_named_compile_time_args=ncrisc_named_compile_time_args,
            brisc_named_compile_time_args=[],
            trisc_named_compile_time_args=trisc_named_compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=fp32_dest_acc_en,
                dst_full_sync_en=fp32_dest_acc_en,
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

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors(),
            cbs=[in0_cb_descriptor, in1_cb_descriptor, out_cb_descriptor],
        )

        # Execute generic op
        io_tensors = [input_a, input_b, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
