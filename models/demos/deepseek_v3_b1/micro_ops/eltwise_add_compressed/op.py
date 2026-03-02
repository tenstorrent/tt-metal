# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Eltwise Add with Compressed Tensor.

Computes: out = A + decompress(B_compressed)
"""


import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class EltwiseAddCompressed:
    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
    ) -> ttnn.Tensor:
        """
        A (bf16) + B (bfp8) = C (bf16) using standard add_tiles.
        """
        core_grid = a_tensor.memory_config().shard_spec.grid
        data_tensor = ct.get_data_tensor()

        # CB indices
        cb_in0 = 0
        cb_in1 = 1
        cb_out = 2

        num_tiles = ct.num_tiles

        # CB0: A tensor — standard
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_tensor)

        # CB1: backed by compressed data tensor, override format to bfp8
        tile_32x32 = ttnn.Tile([32, 32])
        bfp8_page_size = tile_32x32.get_tile_size(ttnn.bfloat8_b)  # 1088
        cb1_total_size = bfp8_page_size * num_tiles

        cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(
            cb_in1,
            data_tensor,
            total_size=cb1_total_size,
        )
        cb1_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_in1,
            data_format=ttnn.bfloat8_b,
            page_size=bfp8_page_size,
            tile=ttnn.TileDescriptor(tile_32x32),
        )
        cb1_desc.format_descriptors = [cb1_fmt]

        # CB2: output tensor — standard
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # Number of pages for sharded buffer setup
        cb_in0_num_pages = num_tiles
        cb_in1_num_pages = num_tiles

        assign_l1_addr = ct.get_assignment_l1_address()

        compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles", num_tiles),
            ("cb_in0_num_pages", cb_in0_num_pages),
            ("cb_in1_num_pages", cb_in1_num_pages),
            ("assign_l1_addr", assign_l1_addr),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/eltwise_add_compressed/kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=compile_time_args,
            brisc_named_compile_time_args=compile_time_args,
            trisc_named_compile_time_args=compile_time_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
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
