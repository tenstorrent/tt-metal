# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Extract sub-tensor from an overlapped (fused) sharded tensor.

The NCRISC kernel on each core copies a contiguous byte range from the
input shard (at a given tile offset) to the output shard.
"""

import ttnn
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import UnifiedKernelDescriptor


class CopyToOutput:
    """Copy a sub-tensor from a fused/overlapped sharded tensor to an output tensor.

    On each core where the output tensor has a shard, the NCRISC kernel
    copies the output shard's worth of bytes (starting at an optional
    byte offset) from the fused tensor's shard into the output shard.

    The byte size per tile is derived from the output tensor's dtype.
    """

    @staticmethod
    def op(
        fused_tensor: ttnn.Tensor,
        output_tensor: ttnn.Tensor,
        byte_offset: int = 0,
    ) -> ttnn.Tensor:
        """Execute the copy.

        Args:
            fused_tensor: The fused WIDTH_SHARDED tensor.
            output_tensor: Pre-allocated output tensor with the correct shard
                spec (same dtype/layout, subset of the fused tensor's cores).
            byte_offset: Byte offset from the start of the fused shard
                (default 0).

        Returns:
            The output tensor populated with the extracted data.
        """
        src_shard_shape = fused_tensor.memory_config().shard_spec.shape
        dst_shard_shape = output_tensor.memory_config().shard_spec.shape
        core_grid = output_tensor.memory_config().shard_spec.grid

        # Derive source page count from the fused tensor's own format so that
        # setup_sharded_buffer does not try to push more pages than the CB holds.
        if fused_tensor.layout == ttnn.ROW_MAJOR_LAYOUT:
            src_num_tiles = src_shard_shape[0]  # 1 page per row for ROW_MAJOR
        else:
            src_tile = fused_tensor.get_tile()
            src_num_tiles = (src_shard_shape[0] // src_tile.tile_shape[0]) * (
                src_shard_shape[1] // src_tile.tile_shape[1]
            )

        dst_tile = output_tensor.get_tile()
        dst_tile_h, dst_tile_w = dst_tile.tile_shape
        dst_tile_bytes = dst_tile.get_tile_size(output_tensor.dtype)
        dst_num_tiles = (dst_shard_shape[0] // dst_tile_h) * (dst_shard_shape[1] // dst_tile_w)

        copy_size_bytes = dst_num_tiles * dst_tile_bytes

        # CB indices
        src_cb = 0
        dst_cb = 16

        # CB descriptors aliased to the tensor L1 buffers
        src_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(src_cb, fused_tensor)
        dst_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(dst_cb, output_tensor)

        # Compile-time args for NCRISC (BRISC and TRISC are no-ops)
        ncrisc_args = [
            ("src_cb", src_cb),
            ("dst_cb", dst_cb),
            ("src_num_tiles", src_num_tiles),
            ("dst_num_tiles", dst_num_tiles),
            ("byte_offset", byte_offset),
            ("copy_size_bytes", copy_size_bytes),
        ]

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/tests/blitz_weights_tests/kernels/extract_shard_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=ncrisc_args,
        )

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=unified_kernel.get_kernel_descriptors().kernels,
            cbs=[src_cb_descriptor, dst_cb_descriptor],
        )

        io_tensors = [fused_tensor, output_tensor]
        return ttnn.generic_op(io_tensors, program_descriptor)
