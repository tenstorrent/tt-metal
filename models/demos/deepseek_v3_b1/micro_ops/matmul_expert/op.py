# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Expert Kernel: unified SRAM, DRAM, and hybrid matmul with compressed weights.

ExpertKernel handles all three cases via sram_cts / dram_cts arguments:
  - SRAM-only: pass sram_cts, leave dram_cts empty.
  - DRAM-only: pass dram_cts, leave sram_cts empty. Caller must pre-build
    dram_meta_tensors via create_dram_expert_tensors_multi_device().
  - Hybrid: pass both. Caller provides sram_core_grid and dram_core_grid explicitly.
    For SRAM-only or DRAM-only, core grids are auto-derived from a_tensor.
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import _TILE_SIZES
from models.demos.deepseek_v3_b1.micro_ops.host_io.utils import dtype_size
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _CB_ADDR_SHIFT
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor

_KERNEL_SOURCE = "models/demos/deepseek_v3_b1/micro_ops/matmul_expert/kernels/matmul_expert_kernel.cpp"


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


def _assignment_to_llk_fmt(idx: int) -> int:
    """Map CompressedTensor assignment index to new LLK format code."""
    _ASSIGNMENT_TO_LLK_FMT = [3, 2, 1, 0]  # bfp8→3, bfp4→2, bfp2→1, bfp0→0
    return _ASSIGNMENT_TO_LLK_FMT[idx]


def _meta_words_for_tiles(num_tiles: int) -> int:
    """Number of uint32 words needed for packed 3-bit metadata (10 tiles per word)."""
    return (num_tiles + 9) // 10


def _compute_expert_subblock_metadata(
    shard_assignment: np.ndarray,
    subblock_k: int,
    per_core_n: int,
    num_subblocks_k: int,
    subblock_n: int = 1,
) -> tuple[list[int], list[int]]:
    """Compute per-subblock block_sizes and packed 3-bit fmt metadata.

    Returns:
        block_sizes: list of uint32, byte size per subblock.
        tile_infos: list of uint32, packed 3-bit metadata per block (word-aligned).
            Each block gets ceil(tiles_per_block / 10) words.
    """
    num_tiles_k = subblock_k * num_subblocks_k
    assert len(shard_assignment) == num_tiles_k * per_core_n
    assert per_core_n % subblock_n == 0

    block_sizes = []
    tile_infos = []

    tiles_per_block = subblock_k * subblock_n
    tile_idx = 0
    prev_fmt_carry = _assignment_to_llk_fmt(3)  # sentinel: assignment idx 3 (bfp0) → fmt 0
    for _ng in range(per_core_n // subblock_n):
        for _sb_k in range(num_subblocks_k):
            block_start = tile_idx
            iter_bytes = 0
            for _t in range(tiles_per_block):
                fmt_idx = int(shard_assignment[tile_idx])
                iter_bytes += _TILE_SIZES[fmt_idx]
                tile_idx += 1

            # Pack this block's metadata (word-aligned).
            block_slice = shard_assignment[block_start : block_start + tiles_per_block]
            block_words, prev_fmt_carry = _pack_tile_metadata(block_slice, subblock_n, prev_fmt_carry)
            tile_infos.extend(block_words)
            block_sizes.append(iter_bytes)

    return block_sizes, tile_infos


def upload_per_core_uint32_tensor(device, all_cores, per_core_data, entries_per_core):
    """Create one single-core HEIGHT_SHARDED L1 tensor per core from uint32 data.

    Args:
        all_cores: flat list of CoreCoord, index-aligned with per_core_data keys.
        per_core_data: dict {core_idx: list[int]} of uint32 values per core.
        entries_per_core: number of uint32 entries per core.
    Returns:
        dict {core_idx: ttnn.Tensor}: per-core device tensors.
    """
    raw_size = entries_per_core * 4
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)
    tensors = {}
    for core_idx, core in enumerate(all_cores):
        data_np = np.array(per_core_data[core_idx], dtype=np.uint32).view(np.uint8)
        pad = aligned_size - raw_size
        if pad > 0:
            data_np = np.concatenate([data_np, np.zeros(pad, dtype=np.uint8)])
        core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
        core_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            [1, aligned_size],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            core_shard_spec,
        )
        core_mem_config.experimental_set_per_core_allocation(True)
        tensors[core_idx] = ttnn.from_torch(
            core_torch,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=core_mem_config,
        )
    return tensors


def upload_per_core_uint16_tensor(device, all_cores, per_core_data, entries_per_core):
    """Create one single-core HEIGHT_SHARDED L1 tensor per core from uint16 data."""
    raw_size = entries_per_core * 2
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)
    tensors = {}
    for core_idx, core in enumerate(all_cores):
        data_np = np.array(per_core_data[core_idx], dtype=np.uint16).view(np.uint8)
        pad = aligned_size - raw_size
        if pad > 0:
            data_np = np.concatenate([data_np, np.zeros(pad, dtype=np.uint8)])
        core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
        core_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            [1, aligned_size],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            core_shard_spec,
        )
        core_mem_config.experimental_set_per_core_allocation(True)
        tensors[core_idx] = ttnn.from_torch(
            core_torch,
            dtype=ttnn.uint8,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=core_mem_config,
        )
    return tensors


def _pack_tile_metadata(
    assignment: np.ndarray, ct_dim: int, prev_fmt: int = _assignment_to_llk_fmt(3)
) -> tuple[list[int], int]:
    """Pack tile formats into 3-bit-per-tile metadata for optimized compressed LLK.

    Encoding per uint32 (10 tiles):
      bits 0-1: prev_fmt (format of last tile in previous word)
      bits 2-4: tile 0 {use_b:1, fmt_lo:1, fmt_hi:1}
      ...

    use_b=1 when column index (tile_idx % ct_dim) == 0.

    Args:
        assignment: 1D array of format indices for this block/expert.
        ct_dim: number of output columns (subblock_n or out_w).
        prev_fmt: previous tile's LLK format (for first word's prefix).

    Returns:
        (packed_words, last_fmt): list of uint32 words and the last tile's LLK format.
    """
    num_tiles = len(assignment)
    meta_words = _meta_words_for_tiles(num_tiles)
    result = []

    tile_idx = 0
    for _ in range(meta_words):
        word = prev_fmt & 0x3
        bit_pos = 2
        tiles_in_word = min(10, num_tiles - tile_idx)
        for t in range(tiles_in_word):
            fmt = _assignment_to_llk_fmt(int(assignment[tile_idx]))
            use_b = 1 if (tile_idx % ct_dim) == 0 else 0
            word |= use_b << bit_pos
            word |= fmt << (bit_pos + 1)
            bit_pos += 3
            prev_fmt = fmt
            tile_idx += 1
        result.append(word)

    return result, prev_fmt


def create_expert_fmt_tensors(cts: list, mesh_device, core_grid, num_tiles_k: int, out_w: int):
    """Create per-device per-core format metadata and base address tensors.

    Returns two L1 tables per core:
      - fmt_tensors: packed 3-bit metadata [num_experts * meta_words] uint32s
      - base_addr_tensors: [num_experts] uint32s (weight base byte addr per expert)

    Args:
        cts: List of CompressedTensor, one per SRAM expert.
        mesh_device: The mesh device.
        core_grid: CoreRangeSet used for B and output tensors.
        num_tiles_k: K tiles per core per expert.
        out_w: N tiles per core per expert.

    Returns:
        (fmt_dict, base_addr_dict) where each is {MeshCoordinate: {core_idx: ttnn.Tensor}}
    """
    mesh_shape = mesh_device.shape
    all_cores = ttnn.corerange_to_cores(core_grid)
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    num_tiles = num_tiles_k * out_w
    meta_words_per_expert = (num_tiles + 9) // 10
    num_experts = len(cts)

    fmt_result = {}
    base_result = {}
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            fmt_tensors = {}
            base_tensors = {}
            for core_idx, core_coord in enumerate(all_cores):
                all_meta = []
                base_addrs = []
                for ct in cts:
                    base_addrs.append(ct.get_data_l1_address_per_core(core_coord, device_coord=coord))
                    shard_assignment = ct.get_assignment_per_shard(core_coord, device_coord=coord)
                    words, _ = _pack_tile_metadata(shard_assignment, out_w)
                    all_meta.extend(words)

                # Upload fmt metadata.
                fmt_np = np.array(all_meta, dtype=np.uint32).view(np.uint8)
                fmt_raw = len(fmt_np)
                fmt_aligned = _align(max(fmt_raw, dram_alignment), dram_alignment)
                if fmt_aligned > fmt_raw:
                    fmt_np = np.concatenate([fmt_np, np.zeros(fmt_aligned - fmt_raw, dtype=np.uint8)])
                fmt_torch = torch.from_numpy(fmt_np.copy()).reshape(1, fmt_aligned)
                fmt_shard = ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)]),
                    [1, fmt_aligned],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                fmt_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, fmt_shard)
                fmt_mem.experimental_set_per_core_allocation(True)
                fmt_tensors[core_idx] = ttnn._ttnn.tensor.experimental_to_single_device(
                    ttnn.from_torch(fmt_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT),
                    mesh_device,
                    coord,
                    fmt_mem,
                )

                # Upload base addresses.
                base_np = np.array(base_addrs, dtype=np.uint32).view(np.uint8)
                base_raw = len(base_np)
                base_aligned = _align(max(base_raw, dram_alignment), dram_alignment)
                if base_aligned > base_raw:
                    base_np = np.concatenate([base_np, np.zeros(base_aligned - base_raw, dtype=np.uint8)])
                base_torch = torch.from_numpy(base_np.copy()).reshape(1, base_aligned)
                base_shard = ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)]),
                    [1, base_aligned],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                base_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, base_shard)
                base_mem.experimental_set_per_core_allocation(True)
                base_tensors[core_idx] = ttnn._ttnn.tensor.experimental_to_single_device(
                    ttnn.from_torch(base_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT),
                    mesh_device,
                    coord,
                    base_mem,
                )

            fmt_result[coord] = fmt_tensors
            base_result[coord] = base_tensors
    return fmt_result, base_result


def _build_program_for_device(
    a_tensor,
    out_tensor,
    index_tensor,
    num_active_experts: int = 1,
    # SRAM ingredients (optional).
    sram_cts: list = None,
    sram_fmt_addrs: list = None,
    sram_base_addrs: list = None,
    sram_active_flags: list = None,
    dram_active_flags: list = None,
    coord=None,
    # DRAM ingredients (optional).
    in1_backing_tensor=None,
    dram_expert_offsets_addrs: list = None,
    dram_block_sizes_addrs: list = None,
    dram_fmt_layout: dict = None,
    dram_core_params: dict = None,
    subblock_k: int = 0,
    subblock_n: int = 1,
    cores_per_dram_bank: int = 1,
    num_in1_buffers: int = 3,
    accum_experts: bool = False,
    sram_per_core_n: int = 0,
    dram_per_core_n: int = 0,
    sram_k_per_core: int = 0,
    sram_k_offsets: list = None,
    sram_out_tensor=None,
    dram_fuse_silu: bool = False,
    index_offset: int = 0,
    fmt_cb_l1_addr: int = 0,
    fmt_sem_addr_0: int = 0,
    fmt_sem_addr_1: int = 0,
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for one device — handles SRAM-only, DRAM-only, and hybrid.

    Pass SRAM ingredients (sram_cts, sram_fmt_addrs) for SRAM path.
    Pass DRAM ingredients (in1_backing_tensor, dram_*) for DRAM path.
    Pass both for hybrid. The builder derives all CT args from the provided ingredients.

    sram_per_core_n / dram_per_core_n: output tiles per core for each path.
    When 0, derived from the output tensor shard spec.
    """

    core_grid = a_tensor.memory_config().shard_spec.grid
    K_total = a_tensor.memory_config().shard_spec.shape[1]
    # When accum_experts, activation holds per-expert activations concatenated
    # in index-tensor order; per-expert K = total width / num_active_experts.
    K = K_total // num_active_experts if accum_experts else K_total
    Kt = K // 32

    # CB indices: always separate — SRAM B data = 1, DRAM streaming = 4, fmt metadata = 5, SRAM output = 6.
    cb_in0 = 0
    cb_in1 = 1
    cb_out = 2
    cb_index = 3
    cb_in1_dram = 4
    cb_fmt_dram = 5
    cb_out_sram = 6

    # CB descriptors.
    cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_tensor)
    cb1_descs = sram_cts[0].cb_descriptor_from_compressed_tensor(cb_in1, device_coord=coord) if sram_cts else []
    cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_tensor)
    cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, index_tensor)
    cbs = [cb0_desc, *cb1_descs, cb2_desc, cb3_desc]
    if sram_out_tensor is not None:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(cb_out_sram, sram_out_tensor))

    # cb_in1_dram and cb_fmt_dram share one backing tensor (dram_backing_tensor).
    # in1 region: [0, in1_region_bytes), fmt region: [in1_region_bytes, total).
    num_subblocks_k = Kt // subblock_k
    max_tile_size = _TILE_SIZES[0]
    max_subblock_bytes = subblock_k * subblock_n * max_tile_size
    in1_region_bytes = subblock_k * subblock_n * num_in1_buffers * max_tile_size
    cb_in1_dram_total_bytes = in1_region_bytes

    cb4_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_in1_dram,
        in1_backing_tensor,
        address_offset=0,
        total_size=in1_region_bytes,
    )
    cb4_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_in1_dram,
            data_format=ttnn.bfloat8_b,
            page_size=max_tile_size,
        ),
    ]
    cbs.append(cb4_desc)

    fmt_per_expert_bytes = dram_fmt_layout["fmt_per_expert_bytes"]
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    cb_fmt_dram_page_size = _align(max(fmt_per_expert_bytes, dram_alignment), dram_alignment)
    cb_fmt_desc = ttnn.cb_descriptor_from_sharded_tensor(
        cb_fmt_dram,
        in1_backing_tensor,
        address_offset=in1_region_bytes,
        total_size=cb_fmt_dram_page_size,
    )
    cb_fmt_desc.format_descriptors = [
        ttnn.CBFormatDescriptor(
            buffer_index=cb_fmt_dram,
            data_format=ttnn.uint8,
            page_size=cb_fmt_dram_page_size,
        ),
    ]
    cbs.append(cb_fmt_desc)

    # NOC max page size.
    device = a_tensor.device()
    arch = device.arch()
    if arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    elif arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Semaphores (always needed — DRAM infrastructure always present).
    pipeline_sem_id = 0
    semaphores = [ttnn.SemaphoreDescriptor(id=pipeline_sem_id, core_ranges=core_grid, initial_value=0)]

    # Activation tile size in shifted units (byte >> cb_addr_shift=4).
    # bf16 tile: face_r_dim * 32 * 2 bytes per face, 4 faces if full tile.
    # For M=1: tile shape [1, 32], page_size from CB.
    # Activation tile page size in bytes (override_cb_rd_ptr handles shifting).
    tile_h, tile_w = a_tensor.get_tile().tile_shape
    datum_size = dtype_size(a_tensor.dtype)
    in0_page_size = tile_h * tile_w * datum_size

    # Metadata words per expert/block for new packed LLK format.
    sram_meta_words_per_expert = _meta_words_for_tiles(sram_k_per_core * sram_per_core_n)
    dram_meta_words_per_block = _meta_words_for_tiles(subblock_k * subblock_n)

    # Named CT args — shared across all RISCs.
    named_ct_args = [
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_in1_dram", cb_in1_dram),
        ("cb_out", cb_out),
        ("cb_index", cb_index),
        ("num_tiles_k", Kt),
        ("num_active_experts", num_active_experts),
        ("out_w", sram_per_core_n),
        ("cb_in0_num_pages", Kt),
        ("subblock_k", subblock_k),
        ("subblock_n", subblock_n),
        ("num_subblocks_k", num_subblocks_k),
        ("per_core_n", dram_per_core_n),
        ("cb_in1_dram_size_bytes", cb_in1_dram_total_bytes),
        ("noc_max_page_size", noc_max_page_size),
        ("pipeline_sem_id", pipeline_sem_id),
        ("cores_per_dram_bank", cores_per_dram_bank),
        ("index_l1_addr", index_tensor.buffer_address()),
        ("cb_fmt_dram", cb_fmt_dram),
        ("fmt_dram_addr", dram_fmt_layout["fmt_dram_addr"]),
        ("fmt_per_expert_bytes", dram_fmt_layout["fmt_per_expert_bytes"]),
        ("fmt_per_core_bytes", dram_fmt_layout["fmt_per_core_bytes"]),
        ("in0_page_size", in0_page_size),
        ("sram_meta_words_per_expert", sram_meta_words_per_expert),
        ("dram_meta_words_per_block", dram_meta_words_per_block),
        ("fmt_cb_l1_addr", fmt_cb_l1_addr),
        ("fmt_cb_page_size", cb_fmt_dram_page_size),
        ("fmt_sem_addr_0", fmt_sem_addr_0),
        ("fmt_sem_addr_1", fmt_sem_addr_1),
        ("accum_experts", 1 if accum_experts else 0),
        ("sram_k_per_core", sram_k_per_core),
        ("cb_out_sram", cb_out_sram),
        ("dram_fuse_silu", 1 if dram_fuse_silu else 0),
        ("index_offset", index_offset),
    ]

    # Per-core descriptors.
    per_core_descriptors = [
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_active",
            core_values=sram_active_flags or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_active",
            core_values=dram_active_flags or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_fmt_l1_addr",
            core_values=sram_fmt_addrs or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_base_addrs_l1_addr",
            core_values=sram_base_addrs or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_fmt_l1_addr",
            core_values=[],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="expert_offsets_l1_addr",
            core_values=dram_expert_offsets_addrs or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="block_sizes_l1_addr",
            core_values=dram_block_sizes_addrs or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_k_offset",
            core_values=sram_k_offsets or [],
            other_value=0,
        ),
    ]
    dram_pcv = dram_core_params or {}
    for name in ("bank_id", "vc", "core_in_bank_idx", "next_core_noc_x", "next_core_noc_y"):
        per_core_descriptors.append(
            PerCoreCompileTimeDescriptor(named_compile_time_arg=name, core_values=dram_pcv.get(name, []), other_value=0)
        )

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_SOURCE,
        core_ranges=core_grid,
        ncrisc_named_compile_time_args=named_ct_args,
        brisc_named_compile_time_args=named_ct_args,
        trisc_named_compile_time_args=named_ct_args,
        trisc_compile_time_args=[],
        per_core_compile_time_descriptors=per_core_descriptors,
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=cbs,
        semaphores=semaphores,
    )


def create_dram_expert_metadata(
    device,
    cts: list,
    compute_cores_list: list,
    compute_core_grid,
    primary_worker_cores: list,
    subblock_k: int,
    subblock_n: int,
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_dram_bank: int,
    cb_in1_base_shifted: int,
    max_subblock_bytes_shifted: int,
    num_in1_buffers: int,
    num_total_experts: int,
    is_dram_flags: list,
    device_coord=None,
) -> tuple:
    """Build per-core meta and fmt tensors for ExpertKernel.

    Tables are indexed by global expert ID (num_total_experts entries).
    SRAM experts get zero-filled dummy entries; DRAM experts get real metadata.

    Meta table layout per core (num_total_experts × meta_stride uint32s):
      meta_stride = 2 + num_subblocks_k × per_core_N
      For expert e:
        meta[e * meta_stride + 0] = in1_tensor_addr  (bank-relative DRAM buffer addr)
        meta[e * meta_stride + 1] = dram_col_offset  (byte offset to this core's col start)
        meta[e * meta_stride + 2..] = block_sizes[num_subblocks_k × per_core_N]

    Fmt table layout per core (num_total_experts × fmt_words_per_expert uint32s):
      fmt_words_per_expert = num_iterations × meta_words_per_block (packed 3-bit format)
      Slot offset = (global_expert_id × num_iterations) % num_buffers.

    Returns:
        ((offset_tensors, block_size_tensors), per_core_fmt,
         (expert_offsets_l1_addrs, block_sizes_l1_addrs), per_core_values)
    """
    num_experts = num_total_experts
    num_banks = len(primary_worker_cores)
    num_total_cores = len(compute_cores_list)
    num_iterations = num_subblocks_k * (per_core_N // subblock_n)
    tiles_per_block = subblock_k * subblock_n
    meta_words_per_block = _meta_words_for_tiles(tiles_per_block)
    fmt_words_per_expert = num_iterations * meta_words_per_block
    BLOCK_SIZE_UNIT = 64

    per_core_expert_offsets = {i: [] for i in range(num_total_cores)}
    per_core_block_sizes = {i: [] for i in range(num_total_cores)}
    per_core_fmt = {i: [] for i in range(num_total_cores)}

    bank_id_core_values = []
    vc_core_values = []
    core_in_bank_idx_core_values = []
    next_core_noc_x_core_values = []
    next_core_noc_y_core_values = []

    bank_ids = []

    for bank_idx, primary_core in enumerate(primary_worker_cores):
        bank_id = bank_idx % num_banks
        vc = bank_id & 0x3
        for j in range(bank_idx):
            prev_core = primary_worker_cores[j]
            if prev_core.y == primary_core.y and (bank_ids[j] & 0x3) == (bank_id & 0x3):
                vc = (vc + 1) & 0x3
                break
        bank_ids.append(bank_id)

        for offset in range(cores_per_dram_bank):
            core_flat_idx = bank_idx * cores_per_dram_bank + offset
            core = compute_cores_list[core_flat_idx]
            is_last = offset == cores_per_dram_bank - 1
            col_start = offset * per_core_N

            core_offsets = []
            core_block_sizes = []
            core_fmt = []

            num_tiles_k = subblock_k * num_subblocks_k
            dram_ct_idx = 0
            for global_expert_idx in range(num_total_experts):
                if not is_dram_flags[global_expert_idx]:
                    # SRAM expert — zero-filled placeholder so global indexing works.
                    core_offsets.append(0)
                    core_block_sizes.extend([0] * num_iterations)
                    core_fmt.extend([0] * fmt_words_per_expert)
                    continue

                ct = cts[dram_ct_idx]
                dram_ct_idx += 1
                data_tensor = ct.get_data_tensors()[0]
                dram_cores = ttnn.corerange_to_cores(data_tensor.memory_config().shard_spec.grid)
                shard_assignment = ct.get_assignment_per_shard(dram_cores[bank_idx], device_coord=device_coord)

                # Tiles per n-group in shard layout: num_tiles_k × subblock_n.
                # subblock_n=1 (column-major): one full column per group.
                # subblock_n>1 (row-major blocks, num_subblocks_k must be 1): subblock_k×subblock_n.
                # In both cases subblock_k * num_subblocks_k = num_tiles_k.
                block_size_tiles = num_tiles_k * subblock_n
                ng_start = col_start // subblock_n
                num_core_ng = per_core_N // subblock_n
                tile_start = ng_start * block_size_tiles
                tile_end = tile_start + num_core_ng * block_size_tiles
                core_assignment = shard_assignment[tile_start:tile_end]
                dram_col_offset = sum(_TILE_SIZES[int(shard_assignment[t])] for t in range(tile_start))

                block_sizes, tile_infos = _compute_expert_subblock_metadata(
                    core_assignment,
                    subblock_k,
                    per_core_N,
                    num_subblocks_k,
                    subblock_n,
                )
                expert_in1_addr = data_tensor.buffer_address()

                # Fuse tensor addr + col offset into one value.
                core_offsets.append(expert_in1_addr + dram_col_offset)
                # Block sizes in units of BLOCK_SIZE_UNIT.
                core_block_sizes.extend([bs // BLOCK_SIZE_UNIT for bs in block_sizes])
                core_fmt.extend(tile_infos)

            per_core_expert_offsets[core_flat_idx] = core_offsets
            per_core_block_sizes[core_flat_idx] = core_block_sizes
            per_core_fmt[core_flat_idx] = core_fmt

            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            core_in_bank_idx_core_values.append((core, offset))

            if not is_last:
                next_core = compute_cores_list[core_flat_idx + 1]
            else:
                next_core = compute_cores_list[bank_idx * cores_per_dram_bank]
            next_noc = device.worker_core_from_logical_core(next_core)
            next_core_noc_x_core_values.append((core, next_noc.x))
            next_core_noc_y_core_values.append((core, next_noc.y))

    # Upload expert offsets (uint32) and block sizes (uint16) as separate L1 tensors.
    offset_tensors = upload_per_core_uint32_tensor(device, compute_cores_list, per_core_expert_offsets, num_experts)
    block_size_tensors = upload_per_core_uint16_tensor(
        device, compute_cores_list, per_core_block_sizes, num_experts * num_iterations
    )

    expert_offsets_l1_addr_core_values = [
        (
            compute_cores_list[i],
            offset_tensors[i].experimental_per_core_buffer_address(compute_cores_list[i]),
        )
        for i in range(num_total_cores)
    ]
    block_sizes_l1_addr_core_values = [
        (
            compute_cores_list[i],
            block_size_tensors[i].experimental_per_core_buffer_address(compute_cores_list[i]),
        )
        for i in range(num_total_cores)
    ]

    per_core_values = {
        "bank_id": bank_id_core_values,
        "vc": vc_core_values,
        "core_in_bank_idx": core_in_bank_idx_core_values,
        "next_core_noc_x": next_core_noc_x_core_values,
        "next_core_noc_y": next_core_noc_y_core_values,
    }

    return (
        (offset_tensors, block_size_tensors),
        per_core_fmt,
        (expert_offsets_l1_addr_core_values, block_sizes_l1_addr_core_values),
        per_core_values,
    )


def _pack_fmt_bank_data(
    per_core_fmt, num_banks, cores_per_dram_bank, num_total_experts, fmt_words_per_expert, fmt_bytes_per_expert
):
    """Pack per-core fmt uint32 arrays into per-bank byte arrays with 64B-aligned expert padding."""
    bank_data_list = []
    for bank_idx in range(num_banks):
        bank_bytes = []
        for offset in range(cores_per_dram_bank):
            core_flat_idx = bank_idx * cores_per_dram_bank + offset
            raw_uint32s = per_core_fmt[core_flat_idx]
            core_bytes = bytearray()
            for eidx in range(num_total_experts):
                start = eidx * fmt_words_per_expert
                end = start + fmt_words_per_expert
                expert_data = np.array(raw_uint32s[start:end], dtype=np.uint32).view(np.uint8)
                pad = fmt_bytes_per_expert - len(expert_data)
                if pad > 0:
                    expert_data = np.concatenate([expert_data, np.zeros(pad, dtype=np.uint8)])
                core_bytes.extend(expert_data)
            bank_bytes.append(np.frombuffer(bytes(core_bytes), dtype=np.uint8))
        bank_data_list.append(np.concatenate(bank_bytes))
    return bank_data_list


def _create_fmt_dram_tensor(mesh_device, mesh_shape, per_device_fmt_bank_data, fmt_sizes, primary_cores_list):
    """Phase 2: create fmt DRAM tensor on the mesh device."""
    num_devices = mesh_device.get_num_devices()
    num_banks = len(primary_cores_list)
    fmt_bytes_per_bank = fmt_sizes["fmt_bytes_per_bank"]

    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1))]
    )

    all_device_data = []
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            all_device_data.append(np.concatenate(per_device_fmt_bank_data[coord]))

    all_fmt_np = np.stack(all_device_data)
    fmt_shard_spec = ttnn.ShardSpec(dram_grid, [1, fmt_bytes_per_bank], ttnn.ShardOrientation.ROW_MAJOR)
    fmt_dram_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, fmt_shard_spec)

    fmt_dram_tensor = ttnn.from_torch(
        torch.from_numpy(all_fmt_np.copy()).reshape(num_devices, num_banks * fmt_bytes_per_bank),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=fmt_dram_mem,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    fmt_dram_addr = fmt_dram_tensor.buffer_address()
    logger.info(f"  fmt DRAM tensor created, addr={fmt_dram_addr}, per_expert={fmt_sizes['fmt_bytes_per_expert']}B")

    return {
        "fmt_dram_tensor": fmt_dram_tensor,
        "fmt_dram_addr": fmt_dram_addr,
        "fmt_per_expert_bytes": fmt_sizes["fmt_bytes_per_expert"],
        "fmt_per_core_bytes": fmt_sizes["fmt_bytes_per_core"],
    }


def _assemble_dram_results(
    mesh_shape,
    per_device_results,
    dram_backing_tensor,
    fmt_dram_info,
    num_in1_buffers,
    fmt_cb_l1_addr,
    fmt_sem_addr_0,
    fmt_sem_addr_1,
    fmt_sem_0,
    fmt_sem_1,
):
    """Phase 3: assemble per-device result tuples."""
    result = {}
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            meta_tensors, l1_addrs, per_core_values = per_device_results[coord]
            result[coord] = (
                dram_backing_tensor,
                meta_tensors,
                fmt_dram_info,
                l1_addrs,
                per_core_values,
                num_in1_buffers,
                fmt_cb_l1_addr,
                fmt_sem_addr_0,
                fmt_sem_addr_1,
                fmt_sem_0,
                fmt_sem_1,
            )
    logger.info("  All device metadata created")
    return result


def create_dram_expert_tensors_multi_device(
    mesh_device,
    cts: list,
    subblock_k: int,
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_dram_bank: int,
    num_total_experts: int,
    is_dram_flags: list,
    num_in1_buffers: int = 3,
    subblock_n: int = 1,
) -> dict:
    """Create per-device tensors for ExpertKernel.mesh_op.

    Calls create_dram_expert_metadata once per device in the mesh.
    Assumes homogeneous DRAM bank topology across all devices.

    Returns:
        {MeshCoordinate: (in1_backing, meta_tensors, fmt_tensors,
                          l1_addrs, per_core_values, num_in1_buffers)}
    """
    mesh_shape = mesh_device.shape
    num_devices = mesh_device.get_num_devices()
    logger.info(
        f"create_dram_expert_tensors_multi_device: {num_devices} devices, {num_total_experts} total experts, {len(cts)} DRAM CTs"
    )
    primary_cores_list = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_cores_list = []
    for primary_core in primary_cores_list:
        for offset in range(cores_per_dram_bank):
            compute_cores_list.append(ttnn.CoreCoord(primary_core.x + offset, primary_core.y))
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores_list]
    )

    logger.info("  Creating dram_backing_tensor (in1 + fmt, replicated mesh tensor)...")
    num_cores = len(compute_cores_list)
    max_tile_size = _TILE_SIZES[0]
    in1_cb_tiles = subblock_k * subblock_n * num_in1_buffers
    in1_region_bytes = in1_cb_tiles * max_tile_size

    # fmt double-buffer region appended after in1 region.
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    tiles_per_block = subblock_k * subblock_n
    num_iterations_local = num_subblocks_k * (per_core_N // subblock_n)
    fmt_words_per_expert = num_iterations_local * _meta_words_for_tiles(tiles_per_block)
    fmt_bytes_per_expert_raw = fmt_words_per_expert * 4
    fmt_bytes_per_expert = _align(fmt_bytes_per_expert_raw, dram_alignment)
    cb_fmt_dram_page_size = _align(fmt_bytes_per_expert, dram_alignment)
    # Two data slots for pipelined double-buffering (NCRISC can run 1 expert ahead).
    fmt_region_bytes = 2 * cb_fmt_dram_page_size

    total_shard_bytes = in1_region_bytes + fmt_region_bytes
    backing_shard_spec = ttnn.ShardSpec(compute_core_grid, [1, total_shard_bytes], ttnn.ShardOrientation.ROW_MAJOR)
    backing_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, backing_shard_spec
    )
    dram_backing_tensor = ttnn.from_torch(
        torch.zeros((num_cores, total_shard_bytes), dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=backing_mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cb_in1_base_shifted = (dram_backing_tensor.buffer_address() >> _CB_ADDR_SHIFT) - 1
    max_subblock_bytes_shifted = (subblock_k * subblock_n * max_tile_size) >> _CB_ADDR_SHIFT

    # fmt metadata sync: 2 global sems as atomic counters (0..2).
    #   sem_0: UNPACK-side counter. NCRISC atomic_add(+1) per expert; UNPACK atomic_sub(-1) after consume.
    #   sem_1: MATH-side counter.   Same protocol for MATH.
    # NCRISC waits sem_{0,1} < 2 before writing next fmt, allowing 1-expert-ahead pipelining.
    fmt_sem_0 = ttnn.create_global_semaphore(mesh_device, compute_core_grid, 0)
    fmt_sem_1 = ttnn.create_global_semaphore(mesh_device, compute_core_grid, 0)
    fmt_sem_addr_0 = ttnn.get_global_semaphore_address(fmt_sem_0)
    fmt_sem_addr_1 = ttnn.get_global_semaphore_address(fmt_sem_1)
    fmt_cb_l1_addr = dram_backing_tensor.buffer_address() + in1_region_bytes

    logger.info(
        f"  dram_backing created, addr={dram_backing_tensor.buffer_address()}, "
        f"in1={in1_region_bytes}B, fmt_offset={in1_region_bytes}, fmt={fmt_region_bytes}B"
    )

    # --- Phase 1: compute per-device metadata and pack fmt bank data ---
    tiles_per_block = subblock_k * subblock_n
    num_iterations = num_subblocks_k * (per_core_N // subblock_n)
    fmt_words_per_expert = num_iterations * _meta_words_for_tiles(tiles_per_block)
    fmt_bytes_per_expert_raw = fmt_words_per_expert * 4
    fmt_bytes_per_expert = _align(fmt_bytes_per_expert_raw, dram_alignment)
    fmt_bytes_per_core = num_total_experts * fmt_bytes_per_expert
    fmt_bytes_per_bank = cores_per_dram_bank * fmt_bytes_per_core
    num_banks = len(primary_cores_list)

    per_device_results = {}
    per_device_fmt_bank_data = {}

    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            logger.info(f"  Creating metadata for device ({row},{col})...")
            meta_tensors, per_core_fmt, l1_addrs, per_core_values = create_dram_expert_metadata(
                mesh_device,
                cts,
                compute_cores_list,
                compute_core_grid,
                primary_cores_list,
                subblock_k,
                subblock_n,
                num_subblocks_k,
                per_core_N,
                cores_per_dram_bank,
                cb_in1_base_shifted,
                max_subblock_bytes_shifted,
                num_in1_buffers,
                num_total_experts=num_total_experts,
                is_dram_flags=is_dram_flags,
                device_coord=coord,
            )
            per_device_results[coord] = (meta_tensors, l1_addrs, per_core_values)

            per_device_fmt_bank_data[coord] = _pack_fmt_bank_data(
                per_core_fmt,
                num_banks,
                cores_per_dram_bank,
                num_total_experts,
                fmt_words_per_expert,
                fmt_bytes_per_expert,
            )

    fmt_sizes = dict(
        fmt_bytes_per_expert=fmt_bytes_per_expert,
        fmt_bytes_per_core=fmt_bytes_per_core,
        fmt_bytes_per_bank=fmt_bytes_per_bank,
    )

    fmt_dram_info = _create_fmt_dram_tensor(
        mesh_device,
        mesh_shape,
        per_device_fmt_bank_data,
        fmt_sizes,
        primary_cores_list,
    )

    return _assemble_dram_results(
        mesh_shape,
        per_device_results,
        dram_backing_tensor,
        fmt_dram_info,
        num_in1_buffers,
        fmt_cb_l1_addr,
        fmt_sem_addr_0,
        fmt_sem_addr_1,
        fmt_sem_0,
        fmt_sem_1,
    )


def encode_expert_indices(expert_ids, is_dram_flags):
    """Encode SRAM/DRAM routing into expert index values via bit 15.

    SRAM experts get bit 15 set + compact slot index in bits 0-14: 0x8000 | slot_idx.
    DRAM experts keep their global expert ID unchanged (bit 15 = 0).

    Args:
        expert_ids: list/tensor of global expert IDs (the active experts to encode).
        is_dram_flags: list of 0/1, length num_experts. 0=SRAM, 1=DRAM.

    Returns:
        list of encoded uint16 index values.
    """
    encoded = []
    sram_slot = 0
    # Build global-to-slot mapping for SRAM experts.
    sram_slot_map = {}
    for i, flag in enumerate(is_dram_flags):
        if not flag:
            sram_slot_map[i] = sram_slot
            sram_slot += 1
    for eid in expert_ids:
        eid = int(eid)
        if not is_dram_flags[eid]:
            encoded.append(0x8000 | sram_slot_map[eid])
        else:
            encoded.append(eid)
    return encoded


class ExpertKernel:
    """Unified SRAM, DRAM, and hybrid expert matmul with compressed weights.

    Dispatches to SRAM-only, DRAM-only, or hybrid based on sram_cts / dram_cts:
      - SRAM-only: pass sram_cts, leave dram_cts empty.
      - DRAM-only: pass dram_cts, leave sram_cts empty.
      - Hybrid: pass both; provide sram_core_grid and dram_core_grid explicitly.

    Supports both single-device (1×1 mesh) and multi-device meshes. For
    multi-device, SRAM and DRAM CTs must be distributed via PlacementShard so
    each device holds its own N-slice. The caller provides CTs, is_dram flags,
    and separate core grids.
    """

    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        sram_cts: list,
        dram_cts: list,
        output_tensor: ttnn.Tensor,
        index_tensor: ttnn.Tensor,
        num_active_experts: int,
        subblock_k: int,
        dram_core_grid,  # CoreRangeSet for DRAM expert cores (always required).
        dram_meta_tensors: dict,  # from create_dram_expert_tensors_multi_device() (always required).
        dram_per_core_n: int,
        has_sram: bool = False,
        sram_per_core_n: int = 0,
        sram_k_per_core: int = 0,
        sram_core_grid=None,  # CoreRangeSet for SRAM cores, or None if no SRAM.
        sram_fmt_tensors: dict = None,  # from create_expert_fmt_tensors(), keyed by MeshCoordinate.
        sram_base_addr_tensors: dict = None,  # from create_expert_fmt_tensors(), keyed by MeshCoordinate.
        sram_k_offsets: list = None,  # [(CoreCoord, k_offset_tiles), ...] for K-sliced SRAM.
        cores_per_dram_bank: int = 1,
        accum_experts: bool = False,
        sram_output_tensor: ttnn.Tensor = None,
        dram_fuse_silu: bool = False,
        tp_expert: bool = True,
        subblock_n: int = 1,
    ) -> ttnn.Tensor:
        """
        Args:
            a_tensor: A [M, K], HEIGHT_SHARDED on the union of sram + dram cores.
            sram_cts: List of CompressedTensors for SRAM experts (per-core L1).
            dram_cts: List of CompressedTensors for DRAM experts (DRAM WIDTH_SHARDED).
            output_tensor: Pre-allocated DRAM output, WIDTH_SHARDED on dram_core_grid.
            index_tensor: HEIGHT_SHARDED [1, 16] uint16, active expert IDs per core.
                          SRAM experts must have bit 15 set (see encode_expert_indices).
            num_active_experts: Number of entries in the index tensor (loop count).
            subblock_k: K subblock size in tiles.
            dram_core_grid: CoreRangeSet for DRAM expert cores (always required).
            dram_meta_tensors: {MeshCoordinate: tuple} from create_dram_expert_tensors_multi_device().
            dram_per_core_n: Number of N tiles per DRAM core.
            has_sram: Whether SRAM expert path is active.
            sram_core_grid: CoreRangeSet for SRAM expert cores (required when has_sram).
            cores_per_dram_bank: Compute cores per DRAM bank.
            sram_output_tensor: Separate SRAM output tensor on sram_core_grid (required when has_sram).
        """
        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        if has_sram:
            assert sram_per_core_n > 0, "sram_per_core_n must be set when has_sram=True"
            assert sram_k_per_core > 0, "sram_k_per_core must be set when has_sram=True"
            assert sram_core_grid is not None, "sram_core_grid must be set when has_sram=True"
            assert sram_fmt_tensors is not None, "sram_fmt_tensors must be set when has_sram=True"
            assert sram_output_tensor is not None, "sram_output_tensor must be set when has_sram=True"

        if not tp_expert:
            assert not has_sram, "Expert parallel (tp_expert=False) only supports DRAM matmul"
            assert (
                not accum_experts
            ), "Expert parallel (tp_expert=False) processes 1 expert per device, accum not applicable"

        a_per_device = ttnn.get_device_tensors(a_tensor)
        out_per_device = ttnn.get_device_tensors(output_tensor)
        index_per_device = ttnn.get_device_tensors(index_tensor)
        sram_out_per_device = (
            ttnn.get_device_tensors(sram_output_tensor) if sram_output_tensor else [None] * len(a_per_device)
        )

        # Precompute core sets for per-core active flags.
        sram_core_set = set((c.x, c.y) for c in ttnn.corerange_to_cores(sram_core_grid)) if has_sram else set()
        dram_core_set = set((c.x, c.y) for c in ttnn.corerange_to_cores(dram_core_grid))

        mesh_program = ttnn.MeshProgramDescriptor()
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                dev_idx = row * mesh_cols + col
                a_dev = a_per_device[dev_idx]
                out_dev = out_per_device[dev_idx]
                sram_out_dev = sram_out_per_device[dev_idx]
                idx_dev = index_per_device[dev_idx]

                # SRAM fmt + base addrs for this device.
                sram_fmt_l1 = []
                sram_base_addrs_l1 = []
                sram_fmt_tensors_dev = {}
                sram_base_addr_tensors_dev = {}
                if has_sram:
                    sram_fmt_tensors_dev = sram_fmt_tensors[coord]
                    sram_base_addr_tensors_dev = sram_base_addr_tensors[coord]
                    sram_cores_list = ttnn.corerange_to_cores(sram_core_grid)
                    sram_fmt_l1 = [
                        (
                            sram_cores_list[i],
                            sram_fmt_tensors_dev[i].experimental_per_core_buffer_address(sram_cores_list[i]),
                        )
                        for i in range(len(sram_cores_list))
                    ]
                    sram_base_addrs_l1 = [
                        (
                            sram_cores_list[i],
                            sram_base_addr_tensors_dev[i].experimental_per_core_buffer_address(sram_cores_list[i]),
                        )
                        for i in range(len(sram_cores_list))
                    ]

                # DRAM for this device.
                (
                    in1_backing,
                    dram_meta_tensors_dev,
                    dram_fmt_layout,
                    (dram_expert_offsets_l1, dram_block_sizes_l1),
                    per_core_vals,
                    num_in1_buffers,
                    fmt_cb_l1_addr,
                    fmt_sem_addr_0,
                    fmt_sem_addr_1,
                    _fmt_sem_0,
                    _fmt_sem_1,
                ) = dram_meta_tensors[coord]
                # Per-core active flags — each core runs only the path it belongs to.
                all_cores_dev = ttnn.corerange_to_cores(a_dev.memory_config().shard_spec.grid)
                sram_active_cv = [(c, 1) for c in all_cores_dev if (c.x, c.y) in sram_core_set]
                dram_active_cv = [(c, 1) for c in all_cores_dev if (c.x, c.y) in dram_core_set]

                dev_num_active = 1 if not tp_expert else num_active_experts
                program = _build_program_for_device(
                    a_dev,
                    out_dev,
                    idx_dev,
                    num_active_experts=dev_num_active,
                    sram_cts=sram_cts if has_sram else None,
                    sram_fmt_addrs=sram_fmt_l1,
                    sram_base_addrs=sram_base_addrs_l1,
                    sram_active_flags=sram_active_cv,
                    dram_active_flags=dram_active_cv,
                    coord=coord,
                    in1_backing_tensor=in1_backing,
                    dram_expert_offsets_addrs=dram_expert_offsets_l1,
                    dram_block_sizes_addrs=dram_block_sizes_l1,
                    dram_fmt_layout=dram_fmt_layout,
                    dram_core_params=per_core_vals,
                    subblock_k=subblock_k,
                    subblock_n=subblock_n,
                    cores_per_dram_bank=cores_per_dram_bank,
                    num_in1_buffers=num_in1_buffers,
                    accum_experts=accum_experts,
                    sram_per_core_n=sram_per_core_n,
                    dram_per_core_n=dram_per_core_n,
                    sram_k_per_core=sram_k_per_core,
                    sram_k_offsets=sram_k_offsets,  # computed above
                    sram_out_tensor=sram_out_dev,
                    dram_fuse_silu=dram_fuse_silu,
                    index_offset=dev_idx if not tp_expert else 0,
                    fmt_cb_l1_addr=fmt_cb_l1_addr,
                    fmt_sem_addr_0=fmt_sem_addr_0,
                    fmt_sem_addr_1=fmt_sem_addr_1,
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        # --- Collect all live tensors ---
        all_ct_data = [t for ct in (sram_cts + dram_cts) for t in ct.get_data_tensors()]
        all_sram_fmt = [t for per_dev in sram_fmt_tensors.values() for t in per_dev.values()]
        all_sram_base = [t for per_dev in (sram_base_addr_tensors or {}).values() for t in per_dev.values()]
        per_device_dram = []
        for in1_backing, (offset_t, bsize_t), fmt_info, *_ in dram_meta_tensors.values():
            per_device_dram.extend([in1_backing, *offset_t.values(), *bsize_t.values(), fmt_info["fmt_dram_tensor"]])
        io_tensors = [
            a_tensor,
            *all_ct_data,
            output_tensor,
            *([sram_output_tensor] if sram_output_tensor else []),
            index_tensor,
            *all_sram_fmt,
            *all_sram_base,
            *per_device_dram,
        ]

        logger.info("ExpertKernel: running kernel...")
        ttnn.generic_op(io_tensors, mesh_program)
        return output_tensor
