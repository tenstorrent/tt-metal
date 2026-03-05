# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Custom Matmul with Compressed Weights.

Computes: output = A @ decompress(B_compressed)

Uses custom_mm_block init/uninit with a custom _run_ that does
per-tile format reconfig + variable address increment.
ct_dim=1 (N=32 only). kt_dim must be even.
"""

import numpy as np

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCorePositionalCTADescriptor,
    UnifiedKernelDescriptor,
)

# Must match compressed::ASSIGN_BITS / TILES_PER_UINT32 in llk_unpack_compressed.h
_ASSIGN_BITS = 2
_ASSIGN_MASK = (1 << _ASSIGN_BITS) - 1
_TILES_PER_UINT32 = 32 // _ASSIGN_BITS  # 16


def pack_formats_as_ctas(assignment_flat: np.ndarray) -> list[int]:
    """Pack flat format indices into uint32 CTAs.

    Each uint32 holds TILES_PER_UINT32 format indices (ASSIGN_BITS bits each).

    Args:
        assignment_flat: 1D array of format indices.

    Returns:
        List of uint32 values with packed format indices.
    """
    total_tiles = len(assignment_flat)
    num_packed = (total_tiles + _TILES_PER_UINT32 - 1) // _TILES_PER_UINT32
    result = []
    for word_idx in range(num_packed):
        packed = 0
        for bit_idx in range(_TILES_PER_UINT32):
            flat_idx = word_idx * _TILES_PER_UINT32 + bit_idx
            if flat_idx < total_tiles:
                fmt = int(assignment_flat[flat_idx]) & _ASSIGN_MASK
                packed |= fmt << (bit_idx * _ASSIGN_BITS)
        result.append(packed)
    return result


class MatmulCustomCompressed:
    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
        use_constexpr_unroll: bool = True,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, 32]) = output [M, 32].

        Args:
            use_constexpr_unroll: If True, bake per-tile formats into code as constexpr
                (zero runtime lookup, but O(K) code size). If False, read formats from
                assignment tensor in L1 at runtime (compact code, ~3us overhead).
        """
        core_grid = a_tensor.memory_config().shard_spec.grid
        data_tensor = ct.get_data_tensor()

        # Shapes
        a_shard_shape = a_tensor.memory_config().shard_spec.shape
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        num_tiles_k = a_shard_shape[1] // 32

        out_w = out_shard_shape[1] // 32
        assert (num_tiles_k * out_w) % 2 == 0, f"total tiles (K*N) must be even, got {num_tiles_k * out_w}"

        # CB indices
        cb_in0 = 0
        cb_in1 = 1
        cb_out = 2

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

        fmt_cta_base = 0  # positional CTAs start at index 0

        named_compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", num_tiles_k),
            ("out_w", out_w),
            ("cb_in0_num_pages", num_tiles_k),
            ("cb_in1_num_pages", 1),
            ("fmt_cta_base", fmt_cta_base),
        ]

        defines = []
        if use_constexpr_unroll:
            defines.append(("USE_CONSTEXPR_UNROLL", "1"))

        # Build per-core positional CTAs
        all_cores = ttnn.corerange_to_cores(core_grid)
        core_values = []
        for core_coord in all_cores:
            shard_assignment = ct.get_assignment_per_shard(core_coord)
            if use_constexpr_unroll:
                # Packed: multiple formats per uint32 (for template-unrolled path)
                ctas = pack_formats_as_ctas(shard_assignment)
            else:
                # Flat: one format index per CTA element (for runtime loop path)
                ctas = [int(x) for x in shard_assignment]
            core_values.append((core_coord, ctas))
        per_core_pos_cta = PerCorePositionalCTADescriptor(core_values=core_values)

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul_custom_compressed/kernels/matmul_custom_compressed_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=named_compile_time_args,
            brisc_named_compile_time_args=named_compile_time_args,
            trisc_named_compile_time_args=named_compile_time_args,
            trisc_compile_time_args=[],
            per_core_positional_cta_descriptor=per_core_pos_cta,
            defines=defines,
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
        return ttnn.generic_op(io_tensors, program_descriptor)
