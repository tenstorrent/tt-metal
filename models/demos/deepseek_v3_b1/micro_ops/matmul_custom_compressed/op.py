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
    PerCoreCompileTimeDescriptor,
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


def pack_tile_pairs(
    assignment_flat: np.ndarray, base_addr_shifted: int, zero_tile_addr: int = _ZEROS_ADDR_SHIFTED
) -> list[int]:
    """Pack per-pair metadata as two uint32s: lo=[addr0:24|fmt0:8], hi=[addr1:24|fmt1:8].

    Kernel loads both uint32s per pair (adjacent in memory). Each uint32 has the
    absolute THCON-shifted address and DataFormat value — zero per-tile arithmetic.

    Args:
        assignment_flat: 1D array of format indices (0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0).
        base_addr_shifted: THCON-shifted base address of the weight shard (buffer_address >> 4).
        zero_tile_addr: Address to use for zero tiles (bfp0). Defaults to _ZEROS_ADDR_SHIFTED.
            Use 0xFFFFFF for relative-address mode (DRAM streaming).

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
            addr = zero_tile_addr
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


def _create_constexpr_ctas(ct, all_cores):
    """Build per-core positional CTA descriptor for constexpr paths."""
    core_values = []
    for core_coord in all_cores:
        shard_assignment = ct.get_assignment_per_shard(core_coord)
        ctas = pack_formats_as_ctas(shard_assignment)
        core_values.append((core_coord, ctas))
    return PerCorePositionalCTADescriptor(trisc_core_values=core_values)


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def create_runtime_fmt_tensors(ct, core_grid, num_tiles, device, all_cores):
    """Create per-core per-tile metadata tensors for runtime path.

    Each tile gets one uint32: [abs_addr:24 | fmt:8], precomputed with absolute addresses.
    Returns dict {core_idx: ttnn.Tensor} with one tensor per core.
    """
    raw_size = num_tiles * 4
    # The L1 allocator's min_allocation_size is DRAM alignment (not L1 alignment),
    # because L1 buffers must be DRAM-aligned for L1<->DRAM transfers.
    # Per-core tensors must be padded to at least this size.
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)

    tensors = {}
    # fifo_rd_ptr - 1: the -1 is a HW convention for THCON address registers
    for core_idx, core_coord in enumerate(all_cores):
        base_addr_shifted = (ct.get_data_l1_address_per_core(core_coord) >> _CB_ADDR_SHIFT) - 1
        shard_assignment = ct.get_assignment_per_shard(core_coord)
        tiles = pack_tile_pairs(shard_assignment, base_addr_shifted)

        data_np = np.array(tiles, dtype=np.uint32).view(np.uint8)
        pad = aligned_size - raw_size
        if pad > 0:
            data_np = np.concatenate([data_np, np.zeros(pad, dtype=np.uint8)])
        core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
        core_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)]),
            [1, aligned_size],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            core_shard_spec,
            per_core_shard_sizes=[aligned_size],
        )
        tensors[core_idx] = ttnn.from_torch(
            core_torch,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=core_mem_config,
        )
    return tensors


class MatmulCustomCompressed:
    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
        impl: str = "constexpr_compact",
        fmt_tensors: dict = None,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, 32]) = output [M, 32].

        Args:
            impl: One of "runtime", "constexpr_compact", or "constexpr_unroll".
                - "runtime": read packed pair metadata from L1 tensor at runtime.
                  Requires fmt_tensors (created via create_runtime_fmt_tensors).
                - "constexpr_compact": constexpr formats with compact run-detection.
                - "constexpr_unroll": fully template-unrolled constexpr formats.
            fmt_tensors: Per-core format metadata tensors for runtime path.
                dict {core_idx: ttnn.Tensor} from create_runtime_fmt_tensors().
        """
        core_grid = a_tensor.memory_config().shard_spec.grid

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

        # CB1: compressed data — per-core or lockstep depending on ct mode
        cb1_descs = ct.cb_descriptor_from_compressed_tensor(cb_in1)

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
        per_core_descriptors = []

        if impl not in _IMPL_TO_DEFINE:
            valid = ", ".join(sorted(_IMPL_TO_DEFINE.keys()))
            raise ValueError(f"Unsupported impl '{impl}'. Expected one of: {valid}")

        impl_define = _IMPL_TO_DEFINE[impl]
        defines.append(("COMPRESSED_MM_IMPL", str(impl_define)))

        all_cores = ttnn.corerange_to_cores(core_grid)

        if impl in ("constexpr_compact", "constexpr_unroll"):
            named_compile_time_args.append(("fmt_cta_base", 0))
            per_core_pos_cta = _create_constexpr_ctas(ct, all_cores)
        else:
            assert fmt_tensors is not None, "runtime impl requires fmt_tensors (use create_runtime_fmt_tensors)"
            per_core_descriptors.append(
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="fmt_l1_addr",
                    core_values=[(all_cores[i], fmt_tensors[i].buffer_address()) for i in range(len(all_cores))],
                    other_value=0,
                )
            )

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source="models/demos/deepseek_v3_b1/micro_ops/matmul_custom_compressed/kernels/matmul_custom_compressed_kernel.cpp",
            core_ranges=core_grid,
            ncrisc_named_compile_time_args=named_compile_time_args,
            brisc_named_compile_time_args=named_compile_time_args,
            trisc_named_compile_time_args=named_compile_time_args,
            trisc_compile_time_args=[],
            per_core_positional_cta_descriptor=per_core_pos_cta,
            per_core_compile_time_descriptors=per_core_descriptors,
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
            cbs=[cb0_desc, *cb1_descs, cb2_desc],
            semaphores=[],
        )

        # io_tensors: include per-core data and fmt tensors for lifetime management
        fmt_io = list(fmt_tensors.values()) if fmt_tensors else []
        io_tensors = [a_tensor, *ct.get_io_tensors(), output_tensor, *fmt_io]
        ttnn.generic_op(io_tensors, program_descriptor)
        return output_tensor
