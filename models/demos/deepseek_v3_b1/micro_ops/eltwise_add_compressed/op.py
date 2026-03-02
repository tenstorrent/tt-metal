# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Eltwise Add with Compressed Tensor.

Computes: out = A + decompress(B_compressed)

A: bf16 TILE_LAYOUT tensor
B: CompressedTensor (mixed bfp8/bfp4/bfp2/bfp0)
out: bf16 TILE_LAYOUT tensor

The compressed tensor's data and assignment are already in L1.
The kernel reconfigures the unpacker per tile based on the assignment.
"""

from loguru import logger

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
        Execute A + decompress(B_compressed).

        Args:
            a_tensor: bf16 TILE_LAYOUT, HEIGHT_SHARDED in L1
            ct: CompressedTensor with data + assignment in L1
            output_tensor: bf16 TILE_LAYOUT, HEIGHT_SHARDED in L1 (pre-allocated)

        Returns:
            Output tensor with A + decompressed(B)
        """
        num_tiles = ct.num_tiles
        assign_l1_addr = ct.get_assignment_l1_address()
        data_l1_addr = ct.get_data_l1_address()

        core_grid = a_tensor.memory_config().shard_spec.grid

        # CB indices
        cb_in0 = 0  # A (bf16, tile layout)
        cb_in1 = 1  # compressed data (backed by ct.data)
        cb_out = 2  # output (bf16)

        # CB0: A tensor — standard tile CB
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_tensor)

        # CB1: compressed data — backed by ct.data tensor
        # Set tile to 32x32 so constexpr arrays get correct tile dims (face_r=16, num_faces=4).
        # The kernel overrides src_format per tile via reconfig_unpack_srca,
        # and bypasses fifo_page_size via explicit address computation.
        tile_32x32 = ttnn.Tile([32, 32])
        cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in1, ct.get_data_tensor())
        cb1_desc.format_descriptors[0].tile = ttnn.TileDescriptor(tile_32x32)

        # CB2: output tensor
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # Number of pages for sharded buffer setup
        a_shard_shape = a_tensor.memory_config().shard_spec.shape
        a_page_size = cb0_desc.format_descriptors[0].page_size
        a_total_size = cb0_desc.total_size
        cb_in0_num_pages = a_total_size // a_page_size if a_page_size > 0 else 1

        # Compressed data: treat as 1 big page (we manage read pointer manually)
        cb_in1_num_pages = 1

        compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles", num_tiles),
            ("assign_l1_addr", assign_l1_addr),
            ("cb_in0_num_pages", cb_in0_num_pages),
            ("cb_in1_num_pages", cb_in1_num_pages),
        ]

        logger.debug(f"EltwiseAddCompressed: num_tiles={num_tiles}, assign_addr={assign_l1_addr:#x}")
        logger.debug(f"  cb_in0_num_pages={cb_in0_num_pages}, cb_in1_num_pages={cb_in1_num_pages}")

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

        io_tensors = [a_tensor, ct.get_data_tensor(), ct.get_assignment_tensor(), output_tensor]
        output = ttnn.generic_op(io_tensors, program_descriptor)

        return output
