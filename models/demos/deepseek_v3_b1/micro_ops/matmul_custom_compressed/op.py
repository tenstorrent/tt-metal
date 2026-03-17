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
import torch

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

# Pre-resolved values (must match tensix_types.h and llk_unpack_compressed.h)
_DATA_FORMATS = [6, 7, 15, 0]  # bfp8=Bfp8_b(6), bfp4=Bfp4_b(7), bfp2=Bfp2_b(15), bfp0=0
_TILE_SIZES_SHIFTED = [68, 36, 20, 0]  # tile_bytes >> 4 (cb_addr_shift)
_IMPL_RUNTIME = 0
_IMPL_CONSTEXPR_COMPACT = 1
_IMPL_CONSTEXPR_UNROLL = 2
_IMPL_TO_DEFINE = {
    "runtime": _IMPL_RUNTIME,
    "constexpr_compact": _IMPL_CONSTEXPR_COMPACT,
    "constexpr_unroll": _IMPL_CONSTEXPR_UNROLL,
}


_CB_ADDR_SHIFT = 4  # CIRCULAR_BUFFER_COMPUTE_ADDR_SHIFT for TRISC on Blackhole
# Must match MEM_ZEROS_BASE from dev_mem_map.h: ((MEM_MAILBOX_END + 31) & ~31)
# MEM_MAILBOX_END = MEM_MAILBOX_BASE(96) + MEM_MAILBOX_SIZE(12896) = 12992
_MEM_ZEROS_BASE = 12992
_ZEROS_ADDR_SHIFTED = _MEM_ZEROS_BASE >> _CB_ADDR_SHIFT  # 812


def pack_tile_pairs(assignment_flat: np.ndarray, base_addr_shifted: int) -> list[int]:
    """Pack per-pair metadata as two uint32s: lo=[addr0:24|fmt0:8], hi=[addr1:24|fmt1:8].

    Kernel loads both uint32s per pair (adjacent in memory). Each uint32 has the
    absolute THCON-shifted address and DataFormat value — zero per-tile arithmetic.

    Args:
        assignment_flat: 1D array of format indices (0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0).
        base_addr_shifted: THCON-shifted base address of the weight shard (buffer_address >> 4).

    Returns:
        List of uint32, two per pair (interleaved: info0, info1, info0, info1, ...).
    """
    assert len(assignment_flat) % 2 == 0, f"Need even tile count, got {len(assignment_flat)}"
    result = []
    cum_offset_shifted = 0
    for i in range(len(assignment_flat)):
        a = int(assignment_flat[i])
        if a == 3:  # bfp0 / zero tile
            fmt = _DATA_FORMATS[2]  # bfp2 format for zero-tile decode
            addr = _ZEROS_ADDR_SHIFTED
        else:
            fmt = _DATA_FORMATS[a]
            addr = base_addr_shifted + cum_offset_shifted
            cum_offset_shifted += _TILE_SIZES_SHIFTED[a]
        assert addr <= 0xFFFFFF, f"Address {addr} exceeds 24 bits"
        result.append((addr << 8) | fmt)
    return result


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
        impl: str = "constexpr_compact",
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, 32]) = output [M, 32].

        Args:
            impl: One of "runtime", "constexpr_compact", or "constexpr_unroll".
                - "runtime": read packed pair metadata from L1 tensor at runtime.
                - "constexpr_compact": constexpr formats with compact run-detection.
                - "constexpr_unroll": fully template-unrolled constexpr formats.
        """
        core_grid = a_tensor.memory_config().shard_spec.grid
        data_tensor = ct.get_data_tensor()

        # Shapes
        a_shard_shape = a_tensor.memory_config().shard_spec.shape
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        num_tiles_k = a_shard_shape[1] // 32

        out_w = out_shard_shape[1] // 32
        assert (num_tiles_k * out_w) % 2 == 0, f"total tiles (K*N) must be even, got {num_tiles_k * out_w}"
        assert out_w == 1 or out_w % 2 == 0, f"out_w must be 1 or even, got {out_w}"

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

        named_compile_time_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", num_tiles_k),
            ("out_w", out_w),
            ("cb_in0_num_pages", num_tiles_k),
            ("cb_in1_num_pages", 1),
        ]

        defines = []
        per_core_pos_cta = None
        fmt_tensor = None

        if impl not in _IMPL_TO_DEFINE:
            valid = ", ".join(sorted(_IMPL_TO_DEFINE.keys()))
            raise ValueError(f"Unsupported impl '{impl}'. Expected one of: {valid}")

        impl_define = _IMPL_TO_DEFINE[impl]
        defines.append(("COMPRESSED_MM_IMPL", str(impl_define)))

        if impl in ("constexpr_compact", "constexpr_unroll"):
            named_compile_time_args.append(("fmt_cta_base", 0))

            all_cores = ttnn.corerange_to_cores(core_grid)
            core_values = []
            for core_coord in all_cores:
                shard_assignment = ct.get_assignment_per_shard(core_coord)
                ctas = pack_formats_as_ctas(shard_assignment)
                core_values.append((core_coord, ctas))
            per_core_pos_cta = PerCorePositionalCTADescriptor(trisc_core_values=core_values)
        else:
            # Runtime path: create per-tile metadata tensor in L1
            # Each tile gets one uint32: [abs_addr:24 | fmt:8], precomputed with absolute addresses.
            all_cores = ttnn.corerange_to_cores(core_grid)
            num_tiles = num_tiles_k * out_w
            # fifo_rd_ptr - 1: the -1 is a HW convention for THCON address registers
            shard_data = []
            for core_coord in all_cores:
                base_addr_shifted = (ct.get_data_l1_address_per_core(core_coord) >> _CB_ADDR_SHIFT) - 1
                shard_assignment = ct.get_assignment_per_shard(core_coord)
                tiles = pack_tile_pairs(shard_assignment, base_addr_shifted)
                shard_data.extend(tiles)

            num_cores = len(all_cores)
            fmt_torch = torch.tensor(shard_data, dtype=torch.int32).reshape(num_cores, num_tiles)
            fmt_shard_spec = ttnn.ShardSpec(core_grid, [1, num_tiles * 4], ttnn.ShardOrientation.ROW_MAJOR)
            fmt_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fmt_shard_spec
            )
            fmt_tensor = ttnn.from_torch(
                fmt_torch.view(torch.uint8).reshape(num_cores, num_tiles * 4),
                dtype=ttnn.uint8,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=a_tensor.device(),
                memory_config=fmt_mem_config,
            )
            fmt_l1_addr = fmt_tensor.buffer_address()
            named_compile_time_args.append(("fmt_l1_addr", fmt_l1_addr))

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
