# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
KNSlicedMatmul standalone micro-op.

Computes: output[1, out_w] = act[k_offset..k_offset+k_per_core] @ weights[k_per_core, out_w]

Each core takes a K-slice of the shared activation buffer (via k_offset) and
multiplies it against its local weight shard, producing out_w output tiles.
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor


class KNSlicedMatmulOp:
    """
    KN-sliced matmul micro-op.

    Each core computes: act[k_offset..k_offset+k_per_core] @ weights[k_per_core, out_w]
    """

    KERNEL_SOURCE = "models/demos/deepseek_v3_b1/micro_ops/kn_sliced_matmul/kernels/kn_sliced_matmul_kernel.cpp"

    @staticmethod
    def op(
        activation_tensor,
        weights_tensor,
        output_tensor,
        k_per_core,
        out_w,
        core_k_offsets=None,
    ):
        """
        Execute KNSlicedMatmul.

        Args:
            activation_tensor: [M, K] HEIGHT_SHARDED (full K on each core)
            weights_tensor: [k_per_core*32, out_w*32] HEIGHT_SHARDED or WIDTH_SHARDED per core
            output_tensor: [M, out_w*32] HEIGHT_SHARDED per core
            k_per_core: Number of K tiles each core processes
            out_w: Number of output tiles per core
            core_k_offsets: List of (CoreCoord, k_offset) tuples for per-core offsets.
                            If None, all cores use k_offset=0.

        Returns:
            Output tensor with results
        """
        act_shard_spec = activation_tensor.memory_config().shard_spec
        act_total_tiles = act_shard_spec.shape[1] // activation_tensor.get_tile().tile_shape[1]

        # CB indices
        act_cb = 0
        weights_cb = 1
        out_cb = 2

        core_grid = output_tensor.memory_config().shard_spec.grid

        # Compile-time args
        ncrisc_args = [
            ("act_cb", act_cb),
            ("act_num_pages", act_total_tiles),
            ("weights_cb", weights_cb),
            ("weights_num_pages", k_per_core * out_w),
        ]
        brisc_args = []
        trisc_args = [
            ("act_cb", act_cb),
            ("weights_cb", weights_cb),
            ("out_cb", out_cb),
            ("k_per_core", k_per_core),
            ("act_total_tiles", act_total_tiles),
            ("out_w", out_w),
        ]

        per_core_descs = []
        if core_k_offsets is not None:
            per_core_descs.append(
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="k_offset",
                    core_values=core_k_offsets,
                    other_value=0,
                )
            )
        else:
            trisc_args.append(("k_offset", 0))

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=KNSlicedMatmulOp.KERNEL_SOURCE,
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_args,
            brisc_named_compile_time_args=brisc_args,
            trisc_named_compile_time_args=trisc_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            per_core_compile_time_descriptors=per_core_descs,
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[
                ttnn.cb_descriptor_from_sharded_tensor(act_cb, activation_tensor),
                ttnn.cb_descriptor_from_sharded_tensor(weights_cb, weights_tensor),
                ttnn.cb_descriptor_from_sharded_tensor(out_cb, output_tensor),
            ],
            semaphores=[],
        )

        ttnn.generic_op([activation_tensor, weights_tensor, output_tensor], program_descriptor)
        return output_tensor
