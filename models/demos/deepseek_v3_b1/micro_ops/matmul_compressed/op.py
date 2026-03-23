# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Matmul with Compressed Weights.

Computes: output = A @ decompress(B_compressed)

A: bf16 HEIGHT_SHARDED, [M, K] per core
B: CompressedTensor (mixed bfp8/bfp4/bfp2/bfp0), WIDTH_SHARDED, [K, N] per core
output: bf16 WIDTH_SHARDED, [M, N] per core

CB in1 constexpr is bfp8 (for valid HW init).
The kernel reconfigures srcA per K-tile based on the assignment.
"""

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class MatmulCompressed:
    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, N]) = output [M, N].
        """
        core_grid = a_tensor.memory_config().shard_spec.grid
        data_tensor = ct.get_data_tensor()

        # CB indices
        cb_in0 = 0  # A (bf16, srcB in matmul)
        cb_in1 = 1  # B compressed (srcA in matmul)
        cb_out = 2  # output (bf16)

        # Shapes
        a_shard_shape = a_tensor.memory_config().shard_spec.shape
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        num_tiles_k = a_shard_shape[1] // 32
        out_w = out_shard_shape[1] // 32

        # CB0: A tensor — standard
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_tensor)

        # CB1: compressed data, override format to bfp8 for HW init
        tile_32x32 = ttnn.Tile([32, 32])

        cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb_in1,
            data_tensor,
            total_size=ct.max_shard_size,
        )
        cb1_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_in1,
            data_format=ttnn.bfloat8_b,
            page_size=ct.max_shard_size,
            tile=ttnn.TileDescriptor(tile_32x32),
        )
        cb1_desc.format_descriptors = [cb1_fmt]

        # CB2: output — standard
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # Number of pages for sharded buffer setup
        cb_in0_num_pages = num_tiles_k
        cb_in1_num_pages = 1  # whole compressed shard as 1 page

        assign_l1_addr = ct.get_assignment_l1_address()

        compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", num_tiles_k),
            ("out_w", out_w),
            ("cb_in0_num_pages", cb_in0_num_pages),
            ("cb_in1_num_pages", cb_in1_num_pages),
            ("assign_l1_addr", assign_l1_addr),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul_compressed/kernels/matmul_compressed_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=compile_time_args,
            brisc_named_compile_time_args=compile_time_args,
            trisc_named_compile_time_args=compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[cb0_desc, cb1_desc, cb2_desc],
            semaphores=[],
        )

        io_tensors = [a_tensor, data_tensor, output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
