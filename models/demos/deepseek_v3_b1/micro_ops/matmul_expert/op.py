# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Expert Kernel: unified SRAM, DRAM, and hybrid matmul with compressed weights.

ExpertKernel handles all three cases via sram_cts / dram_cts arguments:
  - SRAM-only: pass sram_cts, leave dram_cts empty.
  - DRAM-only: pass dram_cts, leave sram_cts empty. Caller must pre-build
    dram_device_data via create_dram_expert_tensors_multi_device().
  - Hybrid: pass both. Caller provides sram_core_grid and dram_core_grid explicitly.
    For SRAM-only or DRAM-only, core grids are auto-derived from a_tensor.
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
    _TILE_SIZES,
    _compute_dram_start_offset,
    _compute_subblock_metadata,
)
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _CB_ADDR_SHIFT, pack_tile_pairs
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor

_KERNEL_SOURCE = "models/demos/deepseek_v3_b1/micro_ops/matmul_expert/kernels/matmul_expert_kernel.cpp"


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


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


def create_expert_fmt_tensors(cts: list, mesh_device, core_grid, num_tiles: int) -> dict:
    """Create per-device per-core format metadata tensors for ExpertKernel.

    Builds a 2D table [num_experts, num_tiles] of packed (addr:24|fmt:8) uint32 entries
    per core. The kernel indexes into this table via expert_idx to select the active expert.

    Must be called before ExpertKernel.op() and the returned tensors must be
    kept alive for the duration of the op call.

    Args:
        cts: List of CompressedTensor, one per expert (all with per-core allocation,
             same K×N tile dimensions but potentially different compressed sizes).
        mesh_device: The mesh device.
        core_grid: CoreRangeSet used for B and output tensors.
        num_tiles: Total tiles per core per expert (K // 32 * N_per_device // 32).

    Returns:
        dict {MeshCoordinate: {core_idx: ttnn.Tensor}}
    """
    num_experts = len(cts)
    mesh_shape = mesh_device.shape
    all_cores = ttnn.corerange_to_cores(core_grid)
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()

    result = {}
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            tensors = {}
            for core_idx, core_coord in enumerate(all_cores):
                # Pack all experts consecutively: [expert0_tile0..tileN, expert1_tile0..tileN, ...]
                all_tiles = []
                for ct in cts:
                    base_addr_shifted = (
                        ct.get_data_l1_address_per_core(core_coord, device_coord=coord) >> _CB_ADDR_SHIFT
                    ) - 1
                    shard_assignment = ct.get_assignment_per_shard(core_coord, device_coord=coord)
                    all_tiles.extend(pack_tile_pairs(shard_assignment, base_addr_shifted))

                data_np = np.array(all_tiles, dtype=np.uint32).view(np.uint8)
                raw_size = len(data_np)
                aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)
                pad = aligned_size - raw_size
                if pad > 0:
                    data_np = np.concatenate([data_np, np.zeros(pad, dtype=np.uint8)])
                core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
                core_shard_spec = ttnn.ShardSpec(
                    ttnn.CoreRangeSet([ttnn.CoreRange(core_coord, core_coord)]),
                    [1, aligned_size],
                    ttnn.ShardOrientation.ROW_MAJOR,
                )
                core_mem_config = ttnn.per_core_allocation.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    core_shard_spec,
                )
                host_tensor = ttnn.from_torch(core_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
                tensors[core_idx] = ttnn._ttnn.per_core_allocation.to_single_device(
                    host_tensor,
                    mesh_device,
                    coord,
                    core_mem_config,
                )
            result[coord] = tensors
    return result


def _build_program_for_device(
    a_device,
    out_device,
    index_device,
    num_active_experts: int = 1,
    # SRAM ingredients (optional).
    sram_cts: list = None,
    sram_fmt_l1_addr_core_values: list = None,
    sram_active_core_values: list = None,
    coord=None,
    # DRAM ingredients (optional).
    in1_backing_tensor=None,
    dram_meta_l1_addr_core_values: list = None,
    dram_fmt_l1_addr_core_values: list = None,
    dram_active_core_values: list = None,
    dram_per_core_values: dict = None,
    subblock_k: int = 0,
    cores_per_bank: int = 1,
    num_in1_buffers: int = 3,
    # Routing (required).
    is_dram_l1_addr_core_values: list = None,
    table_idx_l1_addr_core_values: list = None,
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for one device — handles SRAM-only, DRAM-only, and hybrid.

    Pass SRAM ingredients (sram_cts, sram_fmt_l1_addr_core_values) for SRAM path.
    Pass DRAM ingredients (in1_backing_tensor, dram_*) for DRAM path.
    Pass both for hybrid. The builder derives all CT args from the provided ingredients.
    """
    has_sram = sram_cts is not None and len(sram_cts) > 0
    has_dram = in1_backing_tensor is not None

    core_grid = a_device.memory_config().shard_spec.grid
    K = a_device.memory_config().shard_spec.shape[1]
    Kt = K // 32
    out_w = out_device.memory_config().shard_spec.shape[1] // 32 // num_active_experts

    assert (Kt * out_w) % 2 == 0, f"total tiles K*N={Kt * out_w} must be even"
    assert out_w == 1 or out_w % 2 == 0, f"out_w must be 1 or even, got {out_w}"

    # CB indices: SRAM B data = 1, DRAM streaming = 4 (separate in hybrid, aliased in standalone).
    cb_in0, cb_in1, cb_out, cb_index = 0, 1, 2, 3
    cb_in1_dram = 4 if (has_sram and has_dram) else cb_in1

    # CB descriptors.
    cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_device)
    cb1_descs = sram_cts[0].cb_descriptor_from_compressed_tensor(cb_in1, device_coord=coord) if has_sram else []
    cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_device)
    cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, index_device)
    cbs = [cb0_desc, *cb1_descs, cb2_desc, cb3_desc]

    if has_dram:
        cb4_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in1_dram, in1_backing_tensor)
        cbs.append(cb4_desc)

    # DRAM streaming parameters.
    num_subblocks_k = Kt // subblock_k if subblock_k > 0 else 0
    max_tile_size = _TILE_SIZES[0]
    max_subblock_bytes = subblock_k * max_tile_size
    cb_in1_dram_total_bytes = num_in1_buffers * max_subblock_bytes

    # NOC max page size.
    noc_max_page_size = 0
    if has_dram:
        device = a_device.device()
        arch = device.arch()
        if arch == ttnn.device.Arch.BLACKHOLE:
            noc_max_page_size = 16384
        elif arch == ttnn.device.Arch.WORMHOLE_B0:
            noc_max_page_size = 8192
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

    # Semaphores (needed for DRAM pipeline).
    pipeline_sem_id = 0
    semaphores = (
        [ttnn.SemaphoreDescriptor(id=pipeline_sem_id, core_ranges=core_grid, initial_value=0)] if has_dram else []
    )

    # Named CT args — shared across all RISCs.
    named_ct_args = [
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_in1_dram", cb_in1_dram),
        ("cb_out", cb_out),
        ("cb_index", cb_index),
        ("num_tiles_k", Kt),
        ("num_active_experts", num_active_experts),
        ("out_w", out_w),
        ("cb_in0_num_pages", Kt),
        ("subblock_k", subblock_k),
        ("num_subblocks_k", num_subblocks_k),
        ("per_core_n", out_w),
        ("cb_in1_dram_size_bytes", cb_in1_dram_total_bytes),
        ("noc_max_page_size", noc_max_page_size),
        ("pipeline_sem_id", pipeline_sem_id),
        ("cores_per_bank", cores_per_bank),
        ("index_l1_addr", index_device.buffer_address()),
    ]

    # Per-core descriptors — combine SRAM and DRAM ingredients.
    per_core_descriptors = [
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_fmt_l1_addr",
            core_values=sram_fmt_l1_addr_core_values or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="sram_active",
            core_values=sram_active_core_values or [],
            other_value=1 if has_sram and not has_dram else 0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_active",
            core_values=dram_active_core_values or [],
            other_value=1 if has_dram and not has_sram else 0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="dram_fmt_l1_addr",
            core_values=dram_fmt_l1_addr_core_values or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="meta_l1_addr",
            core_values=dram_meta_l1_addr_core_values or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="is_dram_l1_addr",
            core_values=is_dram_l1_addr_core_values or [],
            other_value=0,
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="table_idx_l1_addr",
            core_values=table_idx_l1_addr_core_values or [],
            other_value=0,
        ),
    ]
    dram_pcv = dram_per_core_values or {}
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
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_bank: int,
    cb_in1_base_shifted: int,
    max_subblock_bytes_shifted: int,
    num_in1_buffers: int,
    device_coord=None,
) -> tuple:
    """Build per-core meta and fmt tensors for ExpertKernel.

    Meta table layout per core (num_experts × meta_stride uint32s):
      meta_stride = 2 + num_subblocks_k × per_core_N
      For expert e:
        meta[e * meta_stride + 0] = in1_tensor_addr  (bank-relative DRAM buffer addr)
        meta[e * meta_stride + 1] = dram_col_offset  (byte offset to this core's col start)
        meta[e * meta_stride + 2..] = block_sizes[num_subblocks_k × per_core_N]

    Fmt table layout per core (num_experts × tiles_per_expert uint32s):
      tiles_per_expert = subblock_k × num_subblocks_k × per_core_N
      CB slot addresses are identical for all experts (same triple-buffer rotation);
      only the format byte varies per expert.

    Returns:
        (meta_tensors, fmt_tensors, meta_l1_addr_core_values,
         fmt_l1_addr_core_values, per_core_compile_time_values)
    """
    num_experts = len(cts)
    num_banks = len(primary_worker_cores)
    num_total_cores = len(compute_cores_list)
    num_iterations = num_subblocks_k * per_core_N
    meta_stride = 2 + num_iterations
    tiles_per_expert = subblock_k * num_subblocks_k * per_core_N

    per_core_meta = {i: [] for i in range(num_total_cores)}
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

        for offset in range(cores_per_bank):
            core_flat_idx = bank_idx * cores_per_bank + offset
            core = compute_cores_list[core_flat_idx]
            is_last = offset == cores_per_bank - 1
            col_start = offset * per_core_N

            core_meta = []
            core_fmt = []

            num_iterations = num_subblocks_k * per_core_N
            for expert_idx, ct in enumerate(cts):
                data_tensor = ct.get_data_tensors()[0]
                dram_cores = ttnn.corerange_to_cores(data_tensor.memory_config().shard_spec.grid)
                shard_assignment = ct.get_assignment_per_shard(dram_cores[bank_idx], device_coord=device_coord)
                num_tiles_k = subblock_k * num_subblocks_k

                core_assignment = np.concatenate(
                    [
                        shard_assignment[col * num_tiles_k : (col + 1) * num_tiles_k]
                        for col in range(col_start, col_start + per_core_N)
                    ]
                )

                block_sizes, tile_infos = _compute_subblock_metadata(
                    device,
                    core_assignment,
                    subblock_k,
                    per_core_N,
                    num_subblocks_k,
                    cb_in1_base_shifted,
                    max_subblock_bytes_shifted,
                    num_in1_buffers,
                    iteration_offset=expert_idx * num_iterations,
                )
                dram_col_offset = _compute_dram_start_offset(shard_assignment, subblock_k, num_subblocks_k, col_start)
                expert_in1_addr = data_tensor.buffer_address()

                core_meta.append(expert_in1_addr)
                core_meta.append(dram_col_offset)
                core_meta.extend(block_sizes)
                core_fmt.extend(tile_infos)

            per_core_meta[core_flat_idx] = core_meta
            per_core_fmt[core_flat_idx] = core_fmt

            bank_id_core_values.append((core, bank_id))
            vc_core_values.append((core, vc))
            core_in_bank_idx_core_values.append((core, offset))

            if not is_last:
                next_core = compute_cores_list[core_flat_idx + 1]
            else:
                next_core = compute_cores_list[bank_idx * cores_per_bank]
            next_noc = device.worker_core_from_logical_core(next_core)
            next_core_noc_x_core_values.append((core, next_noc.x))
            next_core_noc_y_core_values.append((core, next_noc.y))

    meta_entries_per_core = num_experts * meta_stride
    meta_tensors = upload_per_core_uint32_tensor(device, compute_cores_list, per_core_meta, meta_entries_per_core)
    meta_l1_addr_core_values = [
        (
            compute_cores_list[i],
            ttnn.per_core_allocation.per_core_buffer_address(meta_tensors[i], compute_cores_list[i]),
        )
        for i in range(num_total_cores)
    ]

    fmt_entries_per_core = num_experts * tiles_per_expert
    fmt_tensors = upload_per_core_uint32_tensor(device, compute_cores_list, per_core_fmt, fmt_entries_per_core)
    fmt_l1_addr_core_values = [
        (
            compute_cores_list[i],
            ttnn.per_core_allocation.per_core_buffer_address(fmt_tensors[i], compute_cores_list[i]),
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

    return meta_tensors, fmt_tensors, meta_l1_addr_core_values, fmt_l1_addr_core_values, per_core_values


def create_dram_expert_tensors_multi_device(
    mesh_device,
    cts: list,
    subblock_k: int,
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_bank: int,
    num_in1_buffers: int = 3,
) -> dict:
    """Create per-device tensors for ExpertKernel.mesh_op.

    Calls create_dram_expert_metadata once per device in the mesh.
    Assumes homogeneous DRAM bank topology across all devices.

    Returns:
        {MeshCoordinate: (in1_backing, meta_tensors, fmt_tensors,
                          meta_l1_addr, fmt_l1_addr, per_core_values, num_in1_buffers)}
    """
    mesh_shape = mesh_device.shape
    num_devices = mesh_device.get_num_devices()
    logger.info(f"create_dram_expert_tensors_multi_device: {num_devices} devices, {len(cts)} experts")
    dummy = ttnn.from_torch(
        torch.zeros(num_devices, 1, dtype=torch.uint8),
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    per_device_tensors = ttnn.get_device_tensors(dummy)

    dev0 = per_device_tensors[0].device()
    primary_cores_list = dev0.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    compute_cores_list = []
    for primary_core in primary_cores_list:
        for offset in range(cores_per_bank):
            compute_cores_list.append(ttnn.CoreCoord(primary_core.x + offset, primary_core.y))
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in compute_cores_list]
    )

    logger.info("  Creating in1_backing_tensor (replicated mesh tensor)...")
    num_cores = len(compute_cores_list)
    max_tile_size = _TILE_SIZES[0]
    in1_cb_tiles = subblock_k * num_in1_buffers
    in1_backing_shard_shape = (32, in1_cb_tiles * 32)
    in1_backing_total_width = in1_cb_tiles * 32 * num_cores
    in1_backing_shard_spec = ttnn.ShardSpec(compute_core_grid, in1_backing_shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    in1_backing_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, in1_backing_shard_spec
    )
    in1_backing_tensor = ttnn.from_torch(
        torch.zeros([1, 1, 32, in1_backing_total_width]).bfloat16().float(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=in1_backing_mem_config,
        tile=ttnn.Tile([32, 32]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cb_in1_base_shifted = (in1_backing_tensor.buffer_address() >> _CB_ADDR_SHIFT) - 1
    max_subblock_bytes_shifted = (subblock_k * max_tile_size) >> _CB_ADDR_SHIFT

    logger.info(f"  in1_backing created, addr={in1_backing_tensor.buffer_address()}")
    result = {}
    for row in range(mesh_shape[0]):
        for col in range(mesh_shape[1]):
            coord = ttnn.MeshCoordinate(row, col)
            dev_idx = row * mesh_shape[1] + col
            logger.info(f"  Creating metadata for device ({row},{col})...")
            dev = per_device_tensors[dev_idx].device()
            meta_tensors, fmt_tensors, meta_l1_addr, fmt_l1_addr, per_core_values = create_dram_expert_metadata(
                dev,
                cts,
                compute_cores_list,
                compute_core_grid,
                primary_cores_list,
                subblock_k,
                num_subblocks_k,
                per_core_N,
                cores_per_bank,
                cb_in1_base_shifted,
                max_subblock_bytes_shifted,
                num_in1_buffers,
                device_coord=coord,
            )
            result[coord] = (
                in1_backing_tensor,
                meta_tensors,
                fmt_tensors,
                meta_l1_addr,
                fmt_l1_addr,
                per_core_values,
                num_in1_buffers,
            )
    logger.info("  All device metadata created")
    return result


def _upload_per_core_uint8_array(mesh_device, compute_cores_list, data_np, device_coord=None):
    """Upload identical uint8 array to each core. Returns (tensors_dict, l1_addr_core_values)."""
    if device_coord is None:
        device_coord = ttnn.MeshCoordinate(0, 0)
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    raw_size = len(data_np)
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)

    if aligned_size > raw_size:
        data_np = np.concatenate([data_np, np.zeros(aligned_size - raw_size, dtype=np.uint8)])

    tensors = {}
    l1_addr_core_values = []
    for i, core in enumerate(compute_cores_list):
        core_torch = torch.from_numpy(data_np.copy()).reshape(1, aligned_size)
        core_shard_spec = ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]),
            [1, aligned_size],
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        core_mem_config = ttnn.per_core_allocation.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            core_shard_spec,
        )
        host_tensor = ttnn.from_torch(core_torch, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)
        tensors[i] = ttnn._ttnn.per_core_allocation.to_single_device(
            host_tensor, mesh_device, device_coord, core_mem_config
        )
        l1_addr_core_values.append((core, ttnn.per_core_allocation.per_core_buffer_address(tensors[i], core)))

    return tensors, l1_addr_core_values


def create_expert_routing_tensors(mesh_device, compute_cores_list, is_dram_flags, num_experts, device_coord=None):
    """Create per-core is_dram and table_idx L1 arrays for expert routing.

    Two separate uint8 arrays indexed by expert_id:
      - is_dram[eid]: 0=SRAM, 1=DRAM
      - table_idx[eid]: packed index within that type's fmt/meta table (full byte, 0-255)

    Args:
        is_dram_flags: list of 0/1, length num_experts. 0=SRAM, 1=DRAM.

    Returns:
        (is_dram_tensors, is_dram_l1_addr, table_idx_tensors, table_idx_l1_addr)
    """
    is_dram_arr = np.array(is_dram_flags, dtype=np.uint8)

    # Build packed table indices: sequential within each type.
    sram_idx = 0
    dram_idx = 0
    table_idx_entries = []
    for flag in is_dram_flags:
        if flag:
            table_idx_entries.append(dram_idx)
            dram_idx += 1
        else:
            table_idx_entries.append(sram_idx)
            sram_idx += 1
    table_idx_arr = np.array(table_idx_entries, dtype=np.uint8)

    is_dram_tensors, is_dram_l1_addr = _upload_per_core_uint8_array(
        mesh_device, compute_cores_list, is_dram_arr, device_coord
    )
    table_idx_tensors, table_idx_l1_addr = _upload_per_core_uint8_array(
        mesh_device, compute_cores_list, table_idx_arr, device_coord
    )

    return is_dram_tensors, is_dram_l1_addr, table_idx_tensors, table_idx_l1_addr


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
        is_dram_flags: list,
        num_active_experts: int,
        subblock_k: int = 0,
        sram_core_grid=None,
        dram_core_grid=None,
        dram_device_data: dict = None,
        cores_per_bank: int = 1,
    ) -> ttnn.Tensor:
        """
        Args:
            a_tensor: A [M, K], HEIGHT_SHARDED on the union of sram + dram cores.
                      ReplicateTensorToMesh for multi-device.
            sram_cts: List of CompressedTensors for SRAM experts (per-core L1).
                      Use PlacementShard for multi-device.
            dram_cts: List of CompressedTensors for DRAM experts (DRAM WIDTH_SHARDED).
                      Use PlacementShard for multi-device.
            output_tensor: Pre-allocated output, WIDTH_SHARDED on the union core grid.
                           ShardTensorToMesh along N for multi-device.
            index_tensor: HEIGHT_SHARDED [1, 16] uint16, active expert IDs per core.
            is_dram_flags: list of 0/1, length num_experts (indexed by expert_id).
            num_active_experts: Number of entries in the index tensor (loop count).
            subblock_k: K subblock size in tiles.
            sram_core_grid: CoreRangeSet for SRAM expert cores. sram_mm active on these.
            dram_core_grid: CoreRangeSet for DRAM expert cores. dram_mm active on these.
            dram_device_data: {MeshCoordinate: tuple} from create_dram_expert_tensors_multi_device().
                              Required when dram_cts is non-empty.
            cores_per_bank: Compute cores per DRAM bank.
        """
        has_sram_cts = bool(sram_cts)
        has_dram_cts = bool(dram_cts)
        if not has_sram_cts and not has_dram_cts:
            raise ValueError("ExpertKernel requires at least one of sram_cts or dram_cts")
        if has_dram_cts and not dram_device_data:
            raise ValueError("dram_device_data is required when dram_cts is non-empty")

        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        a_per_device = ttnn.get_device_tensors(a_tensor)
        out_per_device = ttnn.get_device_tensors(output_tensor)
        index_per_device = ttnn.get_device_tensors(index_tensor)

        # Auto-derive core grids for single-path cases (hybrid must provide both explicitly).
        if has_sram_cts and not has_dram_cts and not sram_core_grid:
            sram_core_grid = a_per_device[0].memory_config().shard_spec.grid
        if has_dram_cts and not has_sram_cts and not dram_core_grid:
            dram_core_grid = a_per_device[0].memory_config().shard_spec.grid

        has_sram = bool(sram_cts and sram_core_grid)
        has_dram = bool(dram_cts and dram_core_grid)

        K = a_per_device[0].memory_config().shard_spec.shape[1]
        Kt = K // 32
        per_core_N_tiles = out_per_device[0].memory_config().shard_spec.shape[1] // 32 // num_active_experts
        num_tiles = Kt * per_core_N_tiles

        # --- SRAM setup (multi-device aware via create_expert_fmt_tensors) ---
        sram_fmt_per_device = {}
        if has_sram:
            sram_cores = ttnn.corerange_to_cores(sram_core_grid)
            logger.info(
                f"ExpertKernel: creating SRAM fmt tensors for {len(sram_cts)} experts on {len(sram_cores)} cores..."
            )
            sram_fmt_per_device = create_expert_fmt_tensors(sram_cts, mesh_device, sram_core_grid, num_tiles)

        # --- Routing tensors per device ---
        all_routing = {}
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                dev_idx = row * mesh_cols + col
                core_grid = a_per_device[dev_idx].memory_config().shard_spec.grid
                all_cores_dev = ttnn.corerange_to_cores(core_grid)
                is_dram_t, is_dram_l1, table_idx_t, table_idx_l1 = create_expert_routing_tensors(
                    mesh_device, all_cores_dev, is_dram_flags, len(is_dram_flags), device_coord=coord
                )
                all_routing[coord] = (is_dram_t, is_dram_l1, table_idx_t, table_idx_l1)

        # Precompute core sets for per-core active flags (hybrid only).
        sram_core_set = set((c.x, c.y) for c in ttnn.corerange_to_cores(sram_core_grid)) if has_sram else set()
        dram_core_set = set((c.x, c.y) for c in ttnn.corerange_to_cores(dram_core_grid)) if has_dram else set()

        mesh_program = ttnn.MeshProgramDescriptor()
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                dev_idx = row * mesh_cols + col
                a_dev = a_per_device[dev_idx]
                out_dev = out_per_device[dev_idx]
                idx_dev = index_per_device[dev_idx]
                _, is_dram_l1, _, table_idx_l1 = all_routing[coord]

                # SRAM fmt for this device.
                sram_fmt_l1 = []
                sram_fmt_tensors_dev = {}
                if has_sram:
                    sram_fmt_tensors_dev = sram_fmt_per_device[coord]
                    sram_cores_list = ttnn.corerange_to_cores(sram_core_grid)
                    sram_fmt_l1 = [
                        (
                            sram_cores_list[i],
                            ttnn.per_core_allocation.per_core_buffer_address(
                                sram_fmt_tensors_dev[i], sram_cores_list[i]
                            ),
                        )
                        for i in range(len(sram_cores_list))
                    ]

                # DRAM for this device.
                in1_backing = None
                dram_meta_tensors_dev = {}
                dram_fmt_tensors_dev = {}
                dram_meta_l1 = []
                dram_fmt_l1 = []
                per_core_vals = {
                    k: [] for k in ("bank_id", "vc", "core_in_bank_idx", "next_core_noc_x", "next_core_noc_y")
                }
                num_in1_buffers = 3
                if has_dram:
                    (
                        in1_backing,
                        dram_meta_tensors_dev,
                        dram_fmt_tensors_dev,
                        dram_meta_l1,
                        dram_fmt_l1,
                        per_core_vals,
                        num_in1_buffers,
                    ) = dram_device_data[coord]

                # Per-core active flags (only when both paths present on this device).
                sram_active_cv = None
                dram_active_cv = None
                if has_sram and has_dram:
                    all_cores_dev = ttnn.corerange_to_cores(a_dev.memory_config().shard_spec.grid)
                    sram_active_cv = [(c, 1) for c in all_cores_dev if (c.x, c.y) in sram_core_set]
                    dram_active_cv = [(c, 1) for c in all_cores_dev if (c.x, c.y) in dram_core_set]

                program = _build_program_for_device(
                    a_dev,
                    out_dev,
                    idx_dev,
                    num_active_experts=num_active_experts,
                    sram_cts=sram_cts if has_sram else None,
                    sram_fmt_l1_addr_core_values=sram_fmt_l1,
                    sram_active_core_values=sram_active_cv,
                    coord=coord,
                    in1_backing_tensor=in1_backing,
                    dram_meta_l1_addr_core_values=dram_meta_l1,
                    dram_fmt_l1_addr_core_values=dram_fmt_l1,
                    dram_active_core_values=dram_active_cv,
                    dram_per_core_values=per_core_vals,
                    subblock_k=subblock_k,
                    cores_per_bank=cores_per_bank,
                    num_in1_buffers=num_in1_buffers,
                    is_dram_l1_addr_core_values=is_dram_l1,
                    table_idx_l1_addr_core_values=table_idx_l1,
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        # --- Collect all live tensors ---
        all_ct_data = [t for ct in (sram_cts + dram_cts) for t in ct.get_data_tensors()]
        all_sram_fmt = [t for per_dev in sram_fmt_per_device.values() for t in per_dev.values()]
        per_device_dram = []
        if dram_device_data:
            for in1_backing, meta_t, fmt_t, *_ in dram_device_data.values():
                per_device_dram.extend([in1_backing, *meta_t.values(), *fmt_t.values()])
        all_routing_tensors = [t for rt in all_routing.values() for t in list(rt[0].values()) + list(rt[2].values())]
        io_tensors = [
            a_tensor,
            *all_ct_data,
            output_tensor,
            index_tensor,
            *all_sram_fmt,
            *per_device_dram,
            *all_routing_tensors,
        ]

        logger.info("ExpertKernel: running kernel...")
        ttnn.generic_op(io_tensors, mesh_program)
        return output_tensor
