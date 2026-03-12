# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DRAM Streaming Matmul with Compressed Weights.

Computes: output = A @ decompress(B_compressed)

B is WIDTH_SHARDED in DRAM with compressed tiles (mixed bfp8/bfp4/bfp2).
Tiles are streamed from DRAM in variable-size subblocks.
An L1 metadata tensor stores per-subblock byte sizes and an L1 format
tensor stores per-tile packed pair info for compute-side format reconfig.

Key differences from MatmulCustomCompressed (L1 version):
  - B lives in DRAM, streamed in subblocks via NCRISC
  - Variable-size DRAM reads: metadata tensor tells NCRISC how many bytes per subblock
  - Subblock-K partial accumulation with finalize on last subblock
  - Uses DRAM bank-to-worker core mapping
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.compressed_tensor.compressed_tensor import CompressedTensor
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _ZERO_TILE_SENTINEL, pack_tile_pairs
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import (
    PerCoreCompileTimeDescriptor,
    UnifiedCompileTimeCoreDescriptor,
    UnifiedKernelDescriptor,
)

# Must match compressed::TILE_SIZES in llk_unpack_compressed.h
_TILE_SIZES = [1088, 576, 320, 0]  # bfp8, bfp4, bfp2, bfp0


def _compute_subblock_metadata(
    device,
    shard_assignment: np.ndarray,
    subblock_k: int,
    per_core_n: int,
    num_subblocks_k: int,
) -> tuple[list[int], list[int]]:
    """Compute per-subblock NOC read metadata and per-tile format metadata.

    The assignment is in column-major order within the shard:
    for each N column, K tiles are contiguous.

    Returns:
        block_sizes: list of uint32, one per subblock (block_size_bytes).
            Total entries: num_subblocks_k * per_core_n
        tile_infos: list of uint32, per-tile [relative_offset:24 | fmt:8].
            Relative offsets are within each subblock (base=0).
            Zero tiles use _ZERO_TILE_SENTINEL as address.
            Total entries: subblock_k * num_subblocks_k * per_core_n
    """
    num_tiles_k = subblock_k * num_subblocks_k
    assert len(shard_assignment) == num_tiles_k * per_core_n

    block_sizes = []
    tile_infos = []

    tile_idx = 0
    for _n in range(per_core_n):
        for sb_k in range(num_subblocks_k):
            # Compute block size for this subblock
            block_bytes = 0
            subblock_start = tile_idx
            for t in range(subblock_k):
                fmt_idx = int(shard_assignment[tile_idx])
                block_bytes += _TILE_SIZES[fmt_idx]
                tile_idx += 1

            block_sizes.append(block_bytes)

            # Pack per-tile info for this subblock using pack_tile_pairs.
            # base_addr_shifted=0 gives relative offsets within the subblock.
            # The kernel adds addr_in1 (CB read pointer) at runtime.
            subblock_slice = shard_assignment[subblock_start : subblock_start + subblock_k]
            tiles = pack_tile_pairs(subblock_slice, base_addr_shifted=0, zero_tile_addr=_ZERO_TILE_SENTINEL)
            tile_infos.extend(tiles)

    return block_sizes, tile_infos


class DRAMStreamingMatmulCompressed:
    @staticmethod
    def op(
        input_a: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
        subblock_k: int = None,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, N]) = output [M, N].

        B is WIDTH_SHARDED in DRAM. Tiles are streamed in variable-size subblocks.

        Args:
            input_a: Activation tensor, HEIGHT_SHARDED (replicated) on compute cores.
            ct: CompressedTensor with data in DRAM.
            output_tensor: Output tensor, WIDTH_SHARDED on compute cores.
            subblock_k: K subblock size in tiles. None means full K.
        """
        device = input_a.device()
        data_tensor = ct.get_data_tensor()

        # Get compute cores from DRAM bank assignment
        in1_noc = ttnn.NOC.NOC_0
        all_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_cores = len(all_worker_cores)

        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_worker_cores]
        )

        # Shapes from shard specs
        a_shard_shape = input_a.memory_config().shard_spec.shape
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        M = a_shard_shape[0]
        K = a_shard_shape[1]
        Kt = K // 32
        per_core_N = out_shard_shape[1] // 32

        if subblock_k is None:
            subblock_k = Kt
        assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
        num_subblocks_k = Kt // subblock_k
        assert subblock_k % 2 == 0, f"subblock_k ({subblock_k}) must be even"

        logger.debug(
            f"DRAMStreamingMatmulCompressed: M={M}, K={K}, Kt={Kt}, per_core_N={per_core_N}, "
            f"subblock_k={subblock_k}, num_subblocks_k={num_subblocks_k}, num_cores={num_cores}"
        )

        # CB indices
        cb_in0 = 0  # A tensor (HEIGHT_SHARDED, replicated)
        cb_in1 = 1  # Working buffer for DRAM streaming
        cb_out = 2  # Output tensor

        # CB0: A tensor — tensor-backed
        cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, input_a)

        # CB1: Working buffer for streaming — NOT tensor-backed.
        # Size: 3 buffers * max possible subblock size (all bfp8 = subblock_k * 1088)
        max_tile_size = _TILE_SIZES[0]  # bfp8 = 1088
        num_in1_buffers = 3
        max_subblock_bytes = subblock_k * max_tile_size
        cb_in1_total_bytes = num_in1_buffers * max_subblock_bytes

        tile_32x32 = ttnn.Tile([32, 32])
        cb1_fmt = ttnn.CBFormatDescriptor(
            buffer_index=cb_in1,
            data_format=ttnn.bfloat8_b,
            page_size=max_tile_size,
            tile=ttnn.TileDescriptor(tile_32x32),
        )
        cb1_desc = ttnn.CBDescriptor(
            total_size=cb_in1_total_bytes,
            core_ranges=compute_cores,
            format_descriptors=[cb1_fmt],
        )

        # CB2: Output tensor — tensor-backed
        cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, output_tensor)

        # Get DRAM buffer address
        in1_buffer_addr = data_tensor.buffer_address()

        # Build per-core metadata (block sizes + format pairs) and upload to L1
        num_dram_banks = len(all_worker_cores)
        bank_id_core_values = []
        vc_core_values = []
        bank_ids = []

        # DRAM cores from B tensor's shard grid (matches CompressedTensor's assignment keys)
        dram_cores = ttnn.corerange_to_cores(data_tensor.memory_config().shard_spec.grid)

        # Metadata: block_sizes and fmt_pairs per core
        all_block_sizes = []
        all_fmt_pairs = []

        for idx, core in enumerate(all_worker_cores):
            # Bank ID and VC
            bank_id = idx % num_dram_banks
            vc = bank_id & 0x3
            for j in range(idx):
                prev_core = all_worker_cores[j]
                if prev_core.y == core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)
            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))

            # Get this bank's shard assignment using DRAM core coords.
            # The data was pre-shuffled to column-major by shuffle_tensor_tiles before
            # CompressedTensor creation, so the assignment from get_assignment_per_shard
            # is already in the physical DRAM read order (column-major: n-major, k-minor).
            shard_assignment = ct.get_assignment_per_shard(dram_cores[idx])

            # Compute metadata
            block_sizes, fmt_pairs = _compute_subblock_metadata(
                device, shard_assignment, subblock_k, per_core_N, num_subblocks_k
            )
            all_block_sizes.append(block_sizes)
            all_fmt_pairs.append(fmt_pairs)

        # Upload block size metadata to L1 (HEIGHT_SHARDED, one shard per compute core)
        # Each subblock has 1 uint32 entry: block_size_bytes
        num_meta_entries = num_subblocks_k * per_core_N
        meta_flat = []
        for core_sizes in all_block_sizes:
            meta_flat.extend(core_sizes)
        meta_torch = torch.tensor(meta_flat, dtype=torch.int32).reshape(num_cores, num_meta_entries)
        meta_shard_spec = ttnn.ShardSpec(compute_cores, [1, num_meta_entries * 4], ttnn.ShardOrientation.ROW_MAJOR)
        meta_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, meta_shard_spec)
        meta_tensor = ttnn.from_torch(
            meta_torch.view(torch.uint8).reshape(num_cores, num_meta_entries * 4),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=meta_mem_config,
        )
        meta_l1_addr = meta_tensor.buffer_address()

        # Upload per-tile format metadata to L1 (HEIGHT_SHARDED)
        # Each tile has one uint32: [relative_offset:24 | fmt:8]
        num_tile_entries = subblock_k * num_subblocks_k * per_core_N
        fmt_flat = []
        for core_pairs in all_fmt_pairs:
            fmt_flat.extend(core_pairs)
        fmt_torch = torch.tensor(fmt_flat, dtype=torch.int32).reshape(num_cores, num_tile_entries)
        fmt_shard_spec = ttnn.ShardSpec(compute_cores, [1, num_tile_entries * 4], ttnn.ShardOrientation.ROW_MAJOR)
        fmt_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fmt_shard_spec)
        fmt_tensor = ttnn.from_torch(
            fmt_torch.view(torch.uint8).reshape(num_cores, num_tile_entries * 4),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=fmt_mem_config,
        )
        fmt_l1_addr = fmt_tensor.buffer_address()

        # NOC max page size (arch-dependent)
        arch = device.arch()
        if arch == ttnn.device.Arch.WORMHOLE_B0:
            noc_max_page_size = 8192
        elif arch == ttnn.device.Arch.BLACKHOLE:
            noc_max_page_size = 16384
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        # Compile-time args
        ncrisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
            ("in1_tensor_addr", in1_buffer_addr),
            ("subblock_k", subblock_k),
            ("per_core_n", per_core_N),
            ("out_num_tiles", per_core_N),
            ("num_subblocks_k", num_subblocks_k),
            ("meta_l1_addr", meta_l1_addr),
            ("cb_in1_size_bytes", cb_in1_total_bytes),
            ("noc_max_page_size", noc_max_page_size),
        ]

        brisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
            ("in1_tensor_addr", in1_buffer_addr),
            ("subblock_k", subblock_k),
            ("per_core_n", per_core_N),
            ("out_num_tiles", per_core_N),
            ("num_subblocks_k", num_subblocks_k),
            ("meta_l1_addr", meta_l1_addr),
            ("cb_in1_size_bytes", cb_in1_total_bytes),
            ("fmt_l1_addr", fmt_l1_addr),
        ]

        trisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
            ("subblock_k", subblock_k),
            ("per_core_n", per_core_N),
            ("num_subblocks_k", num_subblocks_k),
            ("fmt_l1_addr", fmt_l1_addr),
        ]

        KERNEL_PATH = "models/demos/deepseek_v3_b1/micro_ops/dram_streaming_matmul_compressed/kernels/dram_streaming_matmul_compressed_kernel.cpp"

        unified_kernel = UnifiedKernelDescriptor(
            kernel_source=KERNEL_PATH,
            core_ranges=compute_cores,
            ncrisc_named_compile_time_args=ncrisc_named_args,
            brisc_named_compile_time_args=brisc_named_args,
            trisc_named_compile_time_args=trisc_named_args,
            trisc_compute_config=ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                dst_full_sync_en=False,
            ),
            unified_compile_time_core_descriptors=[
                UnifiedCompileTimeCoreDescriptor(
                    named_compile_time_arg="is_active_core",
                    core_range=compute_cores,
                    value=1,
                    other_value=0,
                ),
            ],
            per_core_compile_time_descriptors=[
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="bank_id",
                    core_values=bank_id_core_values,
                    other_value=0,
                ),
                PerCoreCompileTimeDescriptor(
                    named_compile_time_arg="vc",
                    core_values=vc_core_values,
                    other_value=0,
                ),
            ],
        )

        kernel_descriptors = unified_kernel.get_kernel_descriptors()

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors.kernels,
            cbs=[cb0_desc, cb1_desc, cb2_desc],
            semaphores=[],
        )

        io_tensors = [input_a, data_tensor, output_tensor, meta_tensor, fmt_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor
