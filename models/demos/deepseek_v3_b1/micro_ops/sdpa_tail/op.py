# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Single-core SDPA tail operation.

Computes the SDPA tail reduction:
    m = max(m1, m2)
    P1 = exp((m1 - m) * scale)
    P2 = exp((m2 - m) * scale)
    s = s1 * P1 + s2 * P2
    l = l1 * P1 + l2 * P2
"""

import torch

import ttnn
from models.demos.deepseek_v3_b1.utils import float_to_uint32


class SdpaTailSingleCore:
    """
    Single-core SDPA tail reduction using ttnn.generic_op.

    Computes the correction block and l tensor operations for SDPA.
    """

    @staticmethod
    def golden(l1, l2, m1, m2, s1, s2, scale=1.0, final_reduction=False):
        """
        PyTorch reference implementation for validation.

        Args:
            l1: First accumulator tensor [height, width]
            l2: Second accumulator tensor [height, width]
            m1: First max tensor [height, tile_width] (column 0 contains values)
            m2: Second max tensor [height, tile_width]
            s1: First sum tensor [height, tile_width]
            s2: Second sum tensor [height, tile_width]
            scale: Scale factor for exponent

        Returns:
            Tuple of (l_out, m_out, s_out)
        """
        # m = max(m1, m2)
        m_out = torch.maximum(m1, m2)

        # P1 = exp((m1 - m) * scale)
        P1 = torch.exp((m1 - m_out) * scale)

        # P2 = exp((m2 - m) * scale)
        P2 = torch.exp((m2 - m_out) * scale)

        # s = s1 * P1 + s2 * P2
        s_out = s1 * P1 + s2 * P2

        # l = l1 * P1 + l2 * P2 (P1/P2 broadcast across columns)
        l_out = l1 * P1 + l2 * P2
        if final_reduction:
            l_out = l_out / s_out

        return l_out, m_out, s_out

    @staticmethod
    def op(
        l1_tensor,
        l2_tensor,
        ms1_tensor,
        ms2_tensor,
        l_out_tensor,
        ms_out_tensor,
        scale=1.0,
        block_size=1,
        num_blocks=1,
        final_reduction=False,
    ):
        """
        Execute SDPA tail reduction using generic_op.

        Args:
            l1_tensor: First accumulator input tensor (must be sharded)
            l2_tensor: Second accumulator input tensor (must be sharded)
            ms1_tensor: First max/sum input tensor (must be sharded)
            ms2_tensor: Second max/sum input tensor (must be sharded)
            l_out_tensor: Pre-allocated output tensor for l (must be sharded)
            ms_out_tensor: Pre-allocated output tensor for m/s (must be sharded)
            scale: Scale factor for exponent
            block_size: Number of tiles per block
            num_blocks: Number of blocks
            final_reduction: Whether to apply final normalization (l /= s)

        Returns:
            Tuple of (l_out, m_out, s_out) tensors
        """
        data_format = l1_tensor.dtype

        # Tile configuration
        TILE = l1_tensor.tile
        tile_size = TILE.get_tile_size(data_format)

        # Calculate number of tiles
        num_l_tiles = block_size * num_blocks
        num_ms_tiles = 1

        all_cores = l1_tensor.memory_config().shard_spec.grid

        # Convert scale to fp32 bits for compile-time arg
        scale_fp32 = float_to_uint32(scale)

        # CB indices (simplified - no temp buffers needed)
        cb_l1 = 0
        cb_l2 = 1
        cb_ms1 = 2
        cb_ms2 = 3
        cb_l_out = 4
        cb_ms_out = 5

        tile_descriptor = ttnn.TileDescriptor(TILE)

        # Create CB descriptors for sharded input tensors
        l1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_l1, l1_tensor)
        l1_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        l1_cb_descriptor.format_descriptors[0].page_size = tile_size

        l2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_l2, l2_tensor)
        l2_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        l2_cb_descriptor.format_descriptors[0].page_size = tile_size

        ms1_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_ms1, ms1_tensor)
        ms1_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        ms1_cb_descriptor.format_descriptors[0].page_size = tile_size

        ms2_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_ms2, ms2_tensor)
        ms2_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        ms2_cb_descriptor.format_descriptors[0].page_size = tile_size

        # Create CB descriptors for sharded output tensors
        l_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_l_out, l_out_tensor)
        l_out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        l_out_cb_descriptor.format_descriptors[0].page_size = tile_size

        ms_out_cb_descriptor = ttnn.cb_descriptor_from_sharded_tensor(cb_ms_out, ms_out_tensor)
        ms_out_cb_descriptor.format_descriptors[0].tile = tile_descriptor
        ms_out_cb_descriptor.format_descriptors[0].page_size = tile_size

        # Reader kernel
        reader_compile_time_args = [
            cb_l1,
            cb_l2,
            cb_ms1,
            cb_ms2,
            num_l_tiles,
            num_ms_tiles,
        ]

        reader_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_tail/kernels/sdpa_tail_reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=reader_compile_time_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Writer kernel
        writer_compile_time_args = [
            cb_l_out,
            cb_ms_out,
            block_size,
            num_blocks,
            final_reduction,
        ]

        writer_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_tail/kernels/sdpa_tail_writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=writer_compile_time_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        # Compute kernel - simplified args
        compute_compile_time_args = [
            cb_l1,
            cb_l2,
            cb_ms1,
            cb_ms2,
            cb_l_out,
            cb_ms_out,
            scale_fp32,
            block_size,
            num_blocks,
            final_reduction,
        ]

        compute_kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/sdpa_tail/kernels/sdpa_tail_compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=all_cores,
            compile_time_args=compute_compile_time_args,
            config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
        )

        # Create program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[reader_kernel_descriptor, writer_kernel_descriptor, compute_kernel_descriptor],
            cbs=[
                l1_cb_descriptor,
                l2_cb_descriptor,
                ms1_cb_descriptor,
                ms2_cb_descriptor,
                l_out_cb_descriptor,
                ms_out_cb_descriptor,
            ],
        )

        # Execute generic op
        io_tensors = [
            l1_tensor,
            l2_tensor,
            ms1_tensor,
            ms2_tensor,
            l_out_tensor,
            ms_out_tensor,
        ]
        ttnn.generic_op(io_tensors, program_descriptor)

        return l_out_tensor, ms_out_tensor
