# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Expert Kernel: SRAM and DRAM matmul with compressed weights.

SRAM path (ExpertKernel):
  Unified single-device and multi-device implementation. Single device is the
  degenerate case of a 1×1 mesh — no special handling needed. For each device in
  the mesh, a separate ProgramDescriptor is built with per-device per-core runtime
  metadata tensors derived from that device's compressed format assignment.
  Callers must create fmt tensors via create_expert_fmt_tensors() first.

DRAM path (ExpertKernelDRAM):
  Single-device only. B is WIDTH_SHARDED in DRAM and streamed subblock by subblock.
  Callers must create metadata via create_dram_expert_tensors() first.
"""

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.micro_ops.dram_streaming_matmul_compressed.op import (
    _TILE_SIZES,
    _compute_dram_start_offset,
    _compute_subblock_metadata,
    upload_per_core_uint32_tensor,
)
from models.demos.deepseek_v3_b1.micro_ops.matmul_custom_compressed.op import _CB_ADDR_SHIFT, pack_tile_pairs
from models.demos.deepseek_v3_b1.unified_kernel_descriptor import PerCoreCompileTimeDescriptor, UnifiedKernelDescriptor

_KERNEL_SOURCE = "models/demos/deepseek_v3_b1/micro_ops/matmul_expert/kernels/matmul_expert_kernel.cpp"


def _align(n: int, alignment: int) -> int:
    return ((n + alignment - 1) // alignment) * alignment


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
    raw_size = num_experts * num_tiles * 4
    dram_alignment = ttnn._ttnn.bfp_utils.get_dram_alignment()
    aligned_size = _align(max(raw_size, dram_alignment), dram_alignment)

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
    cts: list, a_device, out_device, index_device, fmt_tensors: dict, coord
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for one device in the mesh.

    Args:
        cts: List of CompressedTensors (one per expert). The CT with the largest
             per-core buffer is used for cb_in1 setup.
        index_device: Per-device index tensor ([1, 16] HEIGHT_SHARDED, uint16). The first
                      element is the expert index; NCRISC sets up its CB so TRISC can read it.
    """
    core_grid = a_device.memory_config().shard_spec.grid
    a_shard_shape = a_device.memory_config().shard_spec.shape
    out_shard_shape = out_device.memory_config().shard_spec.shape
    num_tiles_k = a_shard_shape[1] // 32
    out_w = out_shard_shape[1] // 32

    assert (num_tiles_k * out_w) % 2 == 0, f"total tiles K*N={num_tiles_k * out_w} must be even"
    assert out_w == 1 or out_w % 2 == 0, f"out_w must be 1 or even, got {out_w}"

    cb_in0, cb_in1, cb_out, cb_index = 0, 1, 2, 3

    cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_device)
    # cb_in1 is used for initial format config only — actual per-tile addresses
    # come from the fmt table. Any CT works; cts[0] is simplest.
    cb1_descs = cts[0].cb_descriptor_from_compressed_tensor(cb_in1, device_coord=coord)
    cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_device)
    cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, index_device)

    all_cores = ttnn.corerange_to_cores(core_grid)

    named_ct_args = [
        ("is_dram", 0),
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_out", cb_out),
        ("cb_index", cb_index),
        ("num_tiles_k", num_tiles_k),
        ("out_w", out_w),
        ("cb_in0_num_pages", num_tiles_k),
        ("cb_in1_num_pages", 1),
        # DRAM-path args — unused when is_dram=0, required in the arg map.
        ("subblock_k", 0),
        ("num_subblocks_k", 0),
        ("per_core_n", 0),
        ("cb_in1_size_bytes", 0),
        ("noc_max_page_size", 0),
        ("pipeline_sem_id", 0),
    ]
    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_SOURCE,
        core_ranges=core_grid,
        ncrisc_named_compile_time_args=named_ct_args,
        brisc_named_compile_time_args=named_ct_args,
        trisc_named_compile_time_args=named_ct_args,
        trisc_compile_time_args=[],
        per_core_compile_time_descriptors=[
            PerCoreCompileTimeDescriptor(
                named_compile_time_arg="fmt_l1_addr",
                core_values=[
                    (all_cores[i], ttnn.per_core_allocation.per_core_buffer_address(fmt_tensors[i], all_cores[i]))
                    for i in range(len(all_cores))
                ],
                other_value=0,
            ),
            # DRAM per-core args — unused when is_dram=0, required in the arg map.
            PerCoreCompileTimeDescriptor(named_compile_time_arg="bank_id", core_values=[], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="vc", core_values=[], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="meta_l1_addr", core_values=[], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="core_in_bank_idx", core_values=[], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="next_core_noc_x", core_values=[], other_value=0),
            PerCoreCompileTimeDescriptor(named_compile_time_arg="next_core_noc_y", core_values=[], other_value=0),
        ],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[cb0_desc, *cb1_descs, cb2_desc, cb3_desc],
        semaphores=[],
    )


class ExpertKernel:
    """SRAM matmul with compressed weights — single-device and multi-device unified."""

    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        cts: list,
        output_tensor: ttnn.Tensor,
        fmt_tensors_per_device: dict,
        index_tensor: ttnn.Tensor = None,
    ) -> ttnn.Tensor:
        """
        Execute expert kernel across all devices in the mesh.

        Args:
            a_tensor: A [M, K] mesh tensor (HEIGHT_SHARDED, one core per device).
            cts: List of CompressedTensor for B [K, N], per-core allocation, multi-device.
                 Pass a single-element list for single-expert use (index_tensor with value 0).
            output_tensor: Pre-allocated output [M, N] mesh tensor, WIDTH_SHARDED.
            fmt_tensors_per_device: {MeshCoordinate: {core_idx: ttnn.Tensor}} from
                create_expert_fmt_tensors(). Tensors must stay alive through this call.
            index_tensor: HEIGHT_SHARDED mesh tensor, [1, 16] per core, uint16. The first
                          element is the expert index to run (selects row in the fmt table).
                          Same format as DRAMStreamingExpertsMatmul index_tensor.

        Returns:
            output_tensor (modified in-place by generic_op).
        """
        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        a_per_device = ttnn.get_device_tensors(a_tensor)
        out_per_device = ttnn.get_device_tensors(output_tensor)
        index_per_device = ttnn.get_device_tensors(index_tensor)

        mesh_program = ttnn.MeshProgramDescriptor()

        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                device_idx = row * mesh_cols + col

                a_device = a_per_device[device_idx]
                out_device = out_per_device[device_idx]
                index_device = index_per_device[device_idx]

                program = _build_program_for_device(
                    cts, a_device, out_device, index_device, fmt_tensors_per_device[coord], coord
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        all_fmt_tensors = [t for per_device in fmt_tensors_per_device.values() for t in per_device.values()]
        all_ct_data = [t for ct in cts for t in ct.get_data_tensors()]
        io_tensors = [a_tensor, *all_ct_data, output_tensor, index_tensor, *all_fmt_tensors]
        ttnn.generic_op(io_tensors, mesh_program)
        return output_tensor


# =============================================================================
# DRAM expert path
# =============================================================================


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
    """Build per-core meta and fmt tensors for ExpertKernelDRAM.

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

            for ct in cts:
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


def create_dram_expert_tensors(
    device,
    cts: list,
    compute_cores_list: list,
    compute_core_grid,
    primary_cores_list: list,
    subblock_k: int,
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_bank: int,
    num_in1_buffers: int = 3,
    device_coord=None,
) -> tuple:
    """Create all device tensors needed for ExpertKernelDRAM.

    Creates the in1 working buffer (triple-buffered CB backing tensor), then builds
    and uploads per-core meta and fmt metadata for all experts.

    Returns:
        (in1_backing_tensor, meta_tensors, fmt_tensors,
         meta_l1_addr_core_values, fmt_l1_addr_core_values, per_core_values, num_in1_buffers)
    """
    num_cores = len(compute_cores_list)
    max_tile_size = _TILE_SIZES[0]  # bfp8 = 1088 bytes

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
        device=device,
        memory_config=in1_backing_mem_config,
        tile=ttnn.Tile([32, 32]),
    )
    cb_in1_base_shifted = (in1_backing_tensor.buffer_address() >> _CB_ADDR_SHIFT) - 1
    max_subblock_bytes_shifted = (subblock_k * max_tile_size) >> _CB_ADDR_SHIFT

    (
        meta_tensors,
        fmt_tensors,
        meta_l1_addr_core_values,
        fmt_l1_addr_core_values,
        per_core_values,
    ) = create_dram_expert_metadata(
        device,
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
        device_coord=device_coord,
    )

    return (
        in1_backing_tensor,
        meta_tensors,
        fmt_tensors,
        meta_l1_addr_core_values,
        fmt_l1_addr_core_values,
        per_core_values,
        num_in1_buffers,
    )


def create_dram_expert_tensors_multi_device(
    mesh_device,
    cts: list,
    subblock_k: int,
    num_subblocks_k: int,
    per_core_N: int,
    cores_per_bank: int,
    num_in1_buffers: int = 3,
) -> dict:
    """Create per-device tensors for ExpertKernelDRAM.mesh_op.

    Calls create_dram_expert_tensors once per device in the mesh.
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


def _build_dram_program_for_device(
    a_device: ttnn.Tensor,
    out_device: ttnn.Tensor,
    index_device: ttnn.Tensor,
    in1_backing_tensor: ttnn.Tensor,
    meta_tensors: dict,
    fmt_tensors: dict,
    meta_l1_addr_core_values: list,
    fmt_l1_addr_core_values: list,
    per_core_values: dict,
    subblock_k: int,
    cores_per_bank: int,
    num_in1_buffers: int,
) -> ttnn.ProgramDescriptor:
    """Build a ProgramDescriptor for one device in the mesh (DRAM expert path)."""
    a_shard_shape = a_device.memory_config().shard_spec.shape
    out_shard_shape = out_device.memory_config().shard_spec.shape
    K = a_shard_shape[1]
    Kt = K // 32
    per_core_N = out_shard_shape[1] // 32

    num_subblocks_k = Kt // subblock_k
    max_tile_size = _TILE_SIZES[0]
    max_subblock_bytes = subblock_k * max_tile_size
    cb_in1_total_bytes = num_in1_buffers * max_subblock_bytes

    # Determine NOC max page size from the device
    device = a_device.device()
    arch = device.arch()
    if arch == ttnn.device.Arch.WORMHOLE_B0:
        noc_max_page_size = 8192
    elif arch == ttnn.device.Arch.BLACKHOLE:
        noc_max_page_size = 16384
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    cb_in0, cb_in1, cb_out, cb_index = 0, 1, 2, 3
    cb0_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in0, a_device)
    cb1_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_in1, in1_backing_tensor)
    cb2_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_out, out_device)
    cb3_desc = ttnn.cb_descriptor_from_sharded_tensor(cb_index, index_device)

    compute_cores = a_device.memory_config().shard_spec.grid
    pipeline_sem_id = 0
    semaphores = [ttnn.SemaphoreDescriptor(id=pipeline_sem_id, core_ranges=compute_cores, initial_value=0)]

    ncrisc_named_args = [
        ("is_dram", 1),
        ("cb_in0", cb_in0),
        ("cb_in1", cb_in1),
        ("cb_out", cb_out),
        ("cb_index", cb_index),
        ("num_tiles_k", Kt),
        ("subblock_k", subblock_k),
        ("num_subblocks_k", num_subblocks_k),
        ("per_core_n", per_core_N),
        ("cb_in1_size_bytes", cb_in1_total_bytes),
        ("noc_max_page_size", noc_max_page_size),
        ("pipeline_sem_id", pipeline_sem_id),
        ("out_w", 0),
        ("cb_in0_num_pages", 0),
        ("cb_in1_num_pages", 0),
    ]

    per_core_descriptors = [
        PerCoreCompileTimeDescriptor(named_compile_time_arg=name, core_values=per_core_values[name], other_value=0)
        for name in ("bank_id", "vc", "core_in_bank_idx", "next_core_noc_x", "next_core_noc_y")
    ] + [
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="meta_l1_addr", core_values=meta_l1_addr_core_values, other_value=0
        ),
        PerCoreCompileTimeDescriptor(
            named_compile_time_arg="fmt_l1_addr", core_values=fmt_l1_addr_core_values, other_value=0
        ),
    ]

    unified_kernel = UnifiedKernelDescriptor(
        kernel_source=_KERNEL_SOURCE,
        core_ranges=compute_cores,
        ncrisc_named_compile_time_args=ncrisc_named_args,
        brisc_named_compile_time_args=ncrisc_named_args,
        trisc_named_compile_time_args=ncrisc_named_args,
        trisc_compile_time_args=[],
        trisc_compute_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            dst_full_sync_en=False,
        ),
        per_core_compile_time_descriptors=per_core_descriptors,
    )
    return ttnn.ProgramDescriptor(
        kernels=unified_kernel.get_kernel_descriptors().kernels,
        cbs=[cb0_desc, cb1_desc, cb2_desc, cb3_desc],
        semaphores=semaphores,
    )


class ExpertKernelDRAM:
    """DRAM streaming compressed matmul for expert selection.

    Works for both single device (1×1 mesh) and multi-device meshes.
    Callers always use create_dram_expert_tensors_multi_device() which returns
    a {MeshCoordinate: tuple} dict; the op iterates over it to build one
    ProgramDescriptor per device and assembles a MeshProgramDescriptor.
    """

    @staticmethod
    def op(
        a_tensor: ttnn.Tensor,
        cts: list,
        output_tensor: ttnn.Tensor,
        device_data: dict,
        index_tensor: ttnn.Tensor,
        subblock_k: int,
        cores_per_bank: int = 1,
    ) -> ttnn.Tensor:
        """
        Execute DRAM expert kernel on a single device or across a mesh.

        Args:
            a_tensor: A [M, K] (mesh) tensor, HEIGHT_SHARDED.
                      ReplicateTensorToMesh for multi-device, plain tensor for single.
            cts: List of CompressedTensors (one per expert), DRAM, WIDTH_SHARDED.
                 Use PlacementShard for multi-device, plain from_torch for single.
            output_tensor: Pre-allocated output, WIDTH_SHARDED.
                           ShardTensorToMesh along N for multi-device.
            device_data: {MeshCoordinate: (in1_backing, meta_tensors, fmt_tensors,
                           meta_l1_addr_core_values, fmt_l1_addr_core_values,
                           per_core_values, num_in1_buffers)}
                         from create_dram_expert_tensors_multi_device().
            index_tensor: HEIGHT_SHARDED [1, 16] uint16 tensor on compute cores.
            subblock_k: K subblock size in tiles (must be even).
            cores_per_bank: Number of compute cores per DRAM bank (1, 2, or 4).
        """
        assert subblock_k % 2 == 0, f"subblock_k ({subblock_k}) must be even"
        mesh_device = a_tensor.device()
        mesh_shape = mesh_device.shape
        mesh_rows, mesh_cols = mesh_shape[0], mesh_shape[1]

        a_per_device = ttnn.get_device_tensors(a_tensor)
        out_per_device = ttnn.get_device_tensors(output_tensor)
        index_per_device = ttnn.get_device_tensors(index_tensor)

        mesh_program = ttnn.MeshProgramDescriptor()
        for row in range(mesh_rows):
            for col in range(mesh_cols):
                coord = ttnn.MeshCoordinate(row, col)
                dev_idx = row * mesh_cols + col
                a_dev = a_per_device[dev_idx]
                out_dev = out_per_device[dev_idx]
                idx_dev = index_per_device[dev_idx]
                (
                    in1_backing,
                    meta_tensors,
                    fmt_tensors,
                    meta_l1_addr,
                    fmt_l1_addr,
                    per_core_values,
                    num_in1_buffers,
                ) = device_data[coord]
                program = _build_dram_program_for_device(
                    a_dev,
                    out_dev,
                    idx_dev,
                    in1_backing,
                    meta_tensors,
                    fmt_tensors,
                    meta_l1_addr,
                    fmt_l1_addr,
                    per_core_values,
                    subblock_k,
                    cores_per_bank,
                    num_in1_buffers,
                )
                mesh_program[ttnn.MeshCoordinateRange(coord, coord)] = program

        all_ct_data = [t for ct in cts for t in ct.get_data_tensors()]
        per_device_tensors = []
        for in1_backing, meta_tensors, fmt_tensors, *_ in device_data.values():
            per_device_tensors.extend([in1_backing, *meta_tensors.values(), *fmt_tensors.values()])
        io_tensors = [a_tensor, *all_ct_data, output_tensor, index_tensor, *per_device_tensors]
        ttnn.generic_op(io_tensors, mesh_program)
        return output_tensor
