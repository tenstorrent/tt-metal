# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
DRAM Streaming Matmul with Compressed Weights.

Computes: output = A @ decompress(B_compressed)

B is WIDTH_SHARDED in DRAM with compressed tiles (mixed bfp8/bfp4/bfp2).
Tiles are streamed from DRAM in variable-size subblocks.
An L1 metadata tensor stores per-subblock byte sizes and an L1 format
tensor stores per-tile packed pair info for compute-side format reconfig.

Supports pipelined multi-core-per-bank mode (cores_per_bank > 1):
  Each core reads its own portion of N columns directly from DRAM.
  Cores sharing a bank read sequentially with semaphore handoff:
  core 0 reads first, signals core 1 after last request is sent,
  core 1 waits on semaphore then reads from its offset, etc.
  BRISC: no-op. TRISC: compressed matmul (unchanged).
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


def _compute_dram_start_offset(
    shard_assignment: np.ndarray,
    subblock_k: int,
    num_subblocks_k: int,
    col_start: int,
) -> int:
    """Compute the byte offset into the DRAM shard where col_start begins.

    Tiles are stored column-major: for each N column, K tiles are contiguous.
    We sum up the byte sizes of all tiles in columns [0, col_start).
    """
    num_tiles_k = subblock_k * num_subblocks_k
    offset = 0
    for col in range(col_start):
        for t in range(num_tiles_k):
            fmt_idx = int(shard_assignment[col * num_tiles_k + t])
            offset += _TILE_SIZES[fmt_idx]
    return offset


class DRAMStreamingMatmulCompressed:
    @staticmethod
    def op(
        input_a: ttnn.Tensor,
        ct: CompressedTensor,
        output_tensor: ttnn.Tensor,
        subblock_k: int = None,
        cores_per_bank: int = 1,
    ) -> ttnn.Tensor:
        """
        A [M, K] @ decompress(B_compressed [K, N]) = output [M, N].

        B is WIDTH_SHARDED in DRAM. Tiles are streamed in variable-size subblocks.

        Args:
            input_a: Activation tensor, HEIGHT_SHARDED (replicated) on compute cores.
            ct: CompressedTensor with data in DRAM.
            output_tensor: Output tensor, WIDTH_SHARDED on compute cores.
            subblock_k: K subblock size in tiles. None means full K.
            cores_per_bank: Number of compute cores per DRAM bank (1, 2, or 4).
                When > 1, uses pipelined DRAM reads: each core reads its own
                N columns directly, with semaphore handoff between cores sharing a bank.
        """
        assert cores_per_bank in (1, 2, 4), f"cores_per_bank must be 1, 2, or 4, got {cores_per_bank}"

        device = input_a.device()
        data_tensor = ct.get_data_tensor()

        # Get primary cores from DRAM bank assignment (1 per bank)
        in1_noc = ttnn.NOC.NOC_0
        primary_worker_cores = device.get_optimal_dram_bank_to_logical_worker_assignment(in1_noc)
        num_banks = len(primary_worker_cores)

        # Shapes from shard specs
        a_shard_shape = input_a.memory_config().shard_spec.shape
        out_shard_shape = output_tensor.memory_config().shard_spec.shape
        M = a_shard_shape[0]
        K = a_shard_shape[1]
        Kt = K // 32
        # per_core_N is the N columns each core computes (from output shard)
        per_core_N = out_shard_shape[1] // 32
        # total_N_per_bank is the total N columns per DRAM bank shard
        total_N_per_bank = per_core_N * cores_per_bank

        assert total_N_per_bank % cores_per_bank == 0

        if subblock_k is None:
            subblock_k = Kt
        assert Kt % subblock_k == 0, f"Kt ({Kt}) must be divisible by subblock_k ({subblock_k})"
        num_subblocks_k = Kt // subblock_k
        assert subblock_k % 2 == 0, f"subblock_k ({subblock_k}) must be even"

        # Build expanded core grid: primary + partner cores
        # Partner cores are at (primary.x + offset, primary.y)
        all_compute_cores = []  # flat list: [primary0, partner0_1, ..., primary1, partner1_1, ...]
        for bank_idx, primary_core in enumerate(primary_worker_cores):
            for offset in range(cores_per_bank):
                core = ttnn.CoreCoord(primary_core.x + offset, primary_core.y)
                all_compute_cores.append(core)

        num_total_cores = len(all_compute_cores)
        compute_cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in all_compute_cores]
        )

        logger.debug(
            f"DRAMStreamingMatmulCompressed: M={M}, K={K}, Kt={Kt}, per_core_N={per_core_N}, "
            f"total_N_per_bank={total_N_per_bank}, cores_per_bank={cores_per_bank}, "
            f"subblock_k={subblock_k}, num_subblocks_k={num_subblocks_k}, "
            f"num_banks={num_banks}, num_total_cores={num_total_cores}"
        )

        # CB indices
        cb_in0 = 0  # A tensor (HEIGHT_SHARDED, replicated)
        cb_in1 = 1  # Working buffer for streaming
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

        cbs = [cb0_desc, cb1_desc, cb2_desc]

        # Get DRAM buffer address
        in1_buffer_addr = data_tensor.buffer_address()

        # DRAM cores from B tensor's shard grid
        dram_cores = ttnn.corerange_to_cores(data_tensor.memory_config().shard_spec.grid)

        # Semaphore for pipeline synchronization between cores sharing a bank
        pipeline_sem_id = 0
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=pipeline_sem_id,
                core_ranges=compute_cores,
                initial_value=0,
            )
        ]

        # ====================================================================
        # Build per-core metadata
        # ====================================================================
        per_core_block_sizes = {}  # core_idx → list of block sizes
        per_core_fmt_pairs = {}  # core_idx → list of tile info uint32s

        # Per-core compile-time arg values
        bank_id_core_values = []
        vc_core_values = []
        per_core_n_core_values = []
        dram_start_offset_core_values = []
        core_in_bank_idx_core_values = []
        next_core_noc_x_core_values = []
        next_core_noc_y_core_values = []

        bank_ids = []

        for bank_idx, primary_core in enumerate(primary_worker_cores):
            # Bank ID and VC for this bank
            bank_id = bank_idx % num_banks
            vc = bank_id & 0x3
            for j in range(bank_idx):
                prev_core = primary_worker_cores[j]
                if prev_core.y == primary_core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                    vc = (vc + 1) & 0x3
                    break
            bank_ids.append(bank_id)

            # Get full shard assignment (all N columns for this bank)
            shard_assignment = ct.get_assignment_per_shard(dram_cores[bank_idx])

            num_tiles_k = subblock_k * num_subblocks_k

            for offset in range(cores_per_bank):
                core_flat_idx = bank_idx * cores_per_bank + offset
                core = all_compute_cores[core_flat_idx]
                is_last = offset == cores_per_bank - 1

                col_start = offset * per_core_N
                col_end = col_start + per_core_N

                # This core's slice of the shard assignment
                core_assignment = np.concatenate(
                    [shard_assignment[col * num_tiles_k : (col + 1) * num_tiles_k] for col in range(col_start, col_end)]
                )

                # Compute metadata for this core's columns
                block_sizes, tile_infos = _compute_subblock_metadata(
                    device, core_assignment, subblock_k, per_core_N, num_subblocks_k
                )
                per_core_block_sizes[core_flat_idx] = block_sizes
                per_core_fmt_pairs[core_flat_idx] = tile_infos

                # Compute DRAM byte offset to this core's first column
                dram_offset = _compute_dram_start_offset(shard_assignment, subblock_k, num_subblocks_k, col_start)

                bank_id_core_values.append((core, bank_id))
                vc_core_values.append((core, vc))
                per_core_n_core_values.append((core, per_core_N))
                dram_start_offset_core_values.append((core, dram_offset))
                core_in_bank_idx_core_values.append((core, offset))

                # Next core NOC coords (for semaphore signal)
                # Last core wraps back to first core in the bank group
                if not is_last:
                    next_core = all_compute_cores[core_flat_idx + 1]
                else:
                    next_core = all_compute_cores[bank_idx * cores_per_bank]
                next_noc = device.worker_core_from_logical_core(next_core)
                next_core_noc_x_core_values.append((core, next_noc.x))
                next_core_noc_y_core_values.append((core, next_noc.y))

        # ====================================================================
        # Upload block size metadata to L1
        # ====================================================================
        meta_entries = per_core_N * num_subblocks_k  # same for all cores

        meta_flat = []
        for core_idx in range(num_total_cores):
            meta_flat.extend(per_core_block_sizes[core_idx])

        meta_torch = torch.tensor(meta_flat, dtype=torch.int32).reshape(num_total_cores, meta_entries)
        meta_shard_spec = ttnn.ShardSpec(compute_cores, [1, meta_entries * 4], ttnn.ShardOrientation.ROW_MAJOR)
        meta_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, meta_shard_spec)
        meta_tensor = ttnn.from_torch(
            meta_torch.view(torch.uint8).reshape(num_total_cores, meta_entries * 4),
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=meta_mem_config,
        )
        meta_l1_addr = meta_tensor.buffer_address()

        # ====================================================================
        # Upload per-tile format metadata to L1
        # ====================================================================
        num_tile_entries = subblock_k * num_subblocks_k * per_core_N
        fmt_flat = []
        for core_idx in range(num_total_cores):
            fmt_flat.extend(per_core_fmt_pairs[core_idx])

        fmt_torch = torch.tensor(fmt_flat, dtype=torch.int32).reshape(num_total_cores, num_tile_entries)
        fmt_shard_spec = ttnn.ShardSpec(compute_cores, [1, num_tile_entries * 4], ttnn.ShardOrientation.ROW_MAJOR)
        fmt_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fmt_shard_spec)
        fmt_tensor = ttnn.from_torch(
            fmt_torch.view(torch.uint8).reshape(num_total_cores, num_tile_entries * 4),
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

        # ====================================================================
        # Compile-time args
        # ====================================================================
        # NCRISC args
        ncrisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
            ("in1_tensor_addr", in1_buffer_addr),
            ("subblock_k", subblock_k),
            ("out_num_tiles", per_core_N),
            ("num_subblocks_k", num_subblocks_k),
            ("meta_l1_addr", meta_l1_addr),
            ("cb_in1_size_bytes", cb_in1_total_bytes),
            ("noc_max_page_size", noc_max_page_size),
            ("pipeline_sem_id", pipeline_sem_id),
        ]

        # BRISC args — no-op, minimal
        brisc_named_args = [
            ("cb_in0", cb_in0),
            ("cb_in1", cb_in1),
            ("cb_out", cb_out),
            ("num_tiles_k", Kt),
        ]

        # TRISC args
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

        per_core_descriptors = [
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
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="per_core_n",
                core_values=per_core_n_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="dram_start_offset",
                core_values=dram_start_offset_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="core_in_bank_idx",
                core_values=core_in_bank_idx_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="next_core_noc_x",
                core_values=next_core_noc_x_core_values,
                other_value=0,
            ),
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="next_core_noc_y",
                core_values=next_core_noc_y_core_values,
                other_value=0,
            ),
        ]

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
            per_core_compile_time_descriptors=per_core_descriptors,
        )

        kernel_descriptors = unified_kernel.get_kernel_descriptors()

        program_descriptor = ttnn.ProgramDescriptor(
            kernels=kernel_descriptors.kernels,
            cbs=cbs,
            semaphores=semaphores,
        )

        io_tensors = [input_a, data_tensor, output_tensor, meta_tensor, fmt_tensor]
        ttnn.generic_op(io_tensors, program_descriptor)

        return output_tensor
