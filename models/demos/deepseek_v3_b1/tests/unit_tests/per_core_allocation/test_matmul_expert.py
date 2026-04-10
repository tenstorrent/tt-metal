# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for ExpertKernel (unified SRAM, DRAM, and hybrid):
  SRAM: multi-device, multi-core matmul with compressed weights pre-loaded in L1.
  DRAM (single-device): B WIDTH_SHARDED in DRAM, streamed subblock by subblock.
  DRAM (multi-device): each device gets its own N-slice of B in DRAM.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.compressed_tensor import CompressedTensor, CompressedTensorAssigner
from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import (
    ExpertKernel,
    create_dram_expert_tensors_multi_device,
    create_expert_selection_meta,
)
from models.demos.deepseek_v3_b1.tests.unit_tests.test_dram_streaming_matmul import shuffle_tensor_tiles


def _build_down_grid(device):
    """
    Build 112-core down-proj grid, adapting to device grid size.

    On 13x10: exclude 8 DRAM workers, 9 phantoms (col 12 rows 0-8), 1 mcast core (12,9).
    On 12x10: exclude 8 DRAM workers only (col 12 doesn't exist).
    """
    grid = device.compute_with_storage_grid_size()
    num_cols = grid.x
    assert num_cols >= 12 and grid.y >= 10, f"Need at least 12x10 grid, got {num_cols}x{grid.y}"

    dram_workers = {(0, 0), (0, 3), (0, 7), (0, 9), (7, 1), (7, 4), (7, 6), (7, 9)}
    cores = []
    max_col = min(num_cols, 13)
    for row in range(10):
        for col in range(max_col):
            if col == 12:
                continue  # phantom / mcast cores
            if (col, row) in dram_workers:
                continue
            cores.append(ttnn.CoreCoord(col, row))
    assert len(cores) == 112, f"Expected 112 down cores, got {len(cores)}"
    return cores


def _build_ab_grids(device):
    """
    Build A (gate) / B (up) 64-core irregular grids, adapting to device grid size.

    Uses the same layout as SharedExpertOp.build_ab_grids() (13x10).
    On 12x10 devices, B-core columns are shifted left by 1 to fit.
    """
    grid = device.compute_with_storage_grid_size()
    num_cols = grid.x
    assert num_cols >= 12 and grid.y >= 10, f"Need at least 12x10 grid, got {num_cols}x{grid.y}"
    b_col_shift = 0 if num_cols >= 13 else 1

    a_cores = []
    b_cores = []
    for row in range(10):
        for col in range(13):
            if col == 12 and row >= 8:
                continue
            if row < 4:
                is_a = col in {0, 1, 2, 3, 7, 8, 9}
            else:
                is_a = col in {0, 1, 2, 7, 8, 9}
            if is_a:
                a_cores.append(ttnn.CoreCoord(col, row))
            else:
                b_cores.append(ttnn.CoreCoord(col - b_col_shift, row))
    assert len(a_cores) == 64, f"Expected 64 A cores, got {len(a_cores)}"
    assert len(b_cores) == 64, f"Expected 64 B cores, got {len(b_cores)}"
    return a_cores, b_cores


def _scale_tiles_random_formats(b_torch, formats):
    """Randomly assign formats to tiles so the assigner picks a mix.

    Unlike scale_tiles_for_mixed_formats (which uses deterministic round-robin
    and can cause entire tile columns to share one format), this randomly assigns
    each tile a format. This avoids alignment issues with WIDTH_SHARDED cores.
    """
    if len(formats) <= 1:
        return

    M, N = b_torch.shape
    tiles_h, tiles_w = M // 32, N // 32

    for tr in range(tiles_h):
        for tc in range(tiles_w):
            fmt = formats[torch.randint(0, len(formats), (1,)).item()]
            r0, r1 = tr * 32, (tr + 1) * 32
            c0, c1 = tc * 32, (tc + 1) * 32
            if fmt == "bfp8":
                for r in range(32):
                    b_torch[r0 + r, c0:c1] *= 2.0 ** (r % 16)
            elif fmt == "bfp2":
                for r in range(32):
                    exp = torch.randint(-3, 4, (1,)).float()
                    signs = torch.sign(torch.randn(32))
                    signs[signs == 0] = 1.0
                    b_torch[r0 + r, c0:c1] = signs * (2.0**exp)
            elif fmt == "bfp0":
                b_torch[r0:r1, c0:c1] = torch.randn(32, 32) * 1e-3
            # bfp4: keep randn as-is


def _pad_to_dram_banks(n, tile_w, lcm):
    remainder = n % lcm
    return n if remainder == 0 else n + (lcm - remainder)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------


def _build_sram_cts_standard(
    sram_expert_ids,
    torch_b_all,
    assigner,
    mesh_device,
    sram_core_grid,
    K,
    N_sram_per_device,
    sram_per_core_N,
    num_devices,
    mesh_rows,
    mesh_cols,
    tile_w,
):
    """WIDTH_SHARDED SRAM CTs — each core holds full K, N/num_cores columns."""
    sram_b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sram_core_grid, [K, sram_per_core_N * tile_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    sram_cts = []
    for eidx in sram_expert_ids:
        b_4d = torch.stack(torch_b_all[eidx]).reshape(mesh_rows, mesh_cols, K, N_sram_per_device)
        ct = CompressedTensor.from_torch(
            b_4d,
            assigner,
            device=mesh_device,
            memory_config=sram_b_mem,
            per_core_allocation=True,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        sram_cts.append(ct)
        logger.info(f"  SRAM expert {eidx} uploaded (packed idx {len(sram_cts) - 1})")
    return sram_cts


def _build_sram_cts_slice_k(
    sram_expert_ids,
    torch_b_all,
    assigner,
    mesh_device,
    sram_core_grid,
    sram_k_per_core,
    sram_n_parallel,
    num_sram_cores,
    num_devices,
    mesh_rows,
    mesh_cols,
    tile_w,
):
    """HEIGHT_SHARDED SRAM CTs with K-slicing — each core holds (k_per_core*32, 32)."""
    sram_b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sram_core_grid, [sram_k_per_core * tile_w, tile_w], ttnn.ShardOrientation.ROW_MAJOR),
    )
    sram_cts = []
    for eidx in sram_expert_ids:
        per_dev_shards = []
        for dev_idx in range(num_devices):
            b_full = torch_b_all[eidx][dev_idx]
            shards = []
            for i in range(num_sram_cores):
                k_idx = i // sram_n_parallel
                n_idx = i % sram_n_parallel
                k_start = k_idx * sram_k_per_core * tile_w
                k_end = k_start + sram_k_per_core * tile_w
                n_start = n_idx * tile_w
                n_end = n_start + tile_w
                shards.append(b_full[k_start:k_end, n_start:n_end])
            per_dev_shards.append(torch.cat(shards, dim=0))
        b_4d = torch.stack(per_dev_shards).reshape(
            mesh_rows, mesh_cols, num_sram_cores * sram_k_per_core * tile_w, tile_w
        )
        ct = CompressedTensor.from_torch(
            b_4d,
            assigner,
            device=mesh_device,
            memory_config=sram_b_mem,
            per_core_allocation=True,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        sram_cts.append(ct)
        logger.info(f"  SRAM expert {eidx} uploaded K-sliced (packed idx {len(sram_cts) - 1})")
    return sram_cts


def _build_sram_output(
    mesh_device,
    M,
    sram_per_core_N,
    num_active_sram,
    num_sram_cores,
    num_devices,
    sram_core_grid,
    tile_w,
):
    """SRAM output tensor on sram_core_grid."""
    sram_out_per_core = sram_per_core_N * tile_w * num_active_sram
    sram_out_total = sram_out_per_core * num_sram_cores * num_devices
    sram_out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(sram_core_grid, [M, sram_out_per_core], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.zeros((M, sram_out_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=sram_out_mem,
        tile=ttnn.Tile([M, tile_w]),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


def _build_dram_output(
    mesh_device,
    M,
    dram_per_core_N,
    num_active_dram,
    num_dram_cores_active,
    num_devices,
    dram_core_grid,
    tile_w,
):
    """DRAM output tensor on dram_core_grid."""
    dram_out_per_core = dram_per_core_N * tile_w * num_active_dram
    dram_out_total = dram_out_per_core * num_dram_cores_active * num_devices
    out_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(dram_core_grid, [M, dram_out_per_core], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch.zeros((M, dram_out_total), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=out_mem,
        tile=ttnn.Tile([M, tile_w]),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )


# ---------------------------------------------------------------------------
# Validate helpers
# ---------------------------------------------------------------------------


def _validate_sram_output(
    sram_out_tensor,
    torch_a,
    torch_b_all,
    active_sram,
    sram_per_core_N,
    num_sram_cores_active,
    num_active_sram,
    pcc_threshold,
    tile_w,
):
    """Validate SRAM expert output from separate sram_out_tensor (no K-reduction)."""
    sram_core_width = sram_per_core_N * tile_w
    sram_out_shard = sram_core_width * num_active_sram

    for dev_idx, sram_dev in enumerate(ttnn.get_device_tensors(sram_out_tensor)):
        sram_output_dev = ttnn.to_torch(sram_dev)
        for exp_offset, eidx in enumerate(active_sram):
            slices = []
            for ci in range(num_sram_cores_active):
                start = ci * sram_out_shard + exp_offset * sram_core_width
                slices.append(sram_output_dev[..., start : start + sram_core_width])
            expert_output = torch.cat(slices, dim=-1)
            torch_expected = (torch_a.float() @ torch_b_all[eidx][dev_idx].float()).bfloat16()
            passing, msg = comp_pcc(torch_expected, expert_output, pcc_threshold)
            logger.info(f"Device {dev_idx} expert {eidx} (SRAM) PCC: {msg}")
            assert passing, f"Device {dev_idx} expert {eidx} (SRAM) failed: {msg}"


def _validate_sram_output_accum(
    sram_out_tensor,
    torch_a,
    torch_b_all,
    active_sram,
    sram_per_core_N,
    num_sram_cores_active,
    pcc_threshold,
    tile_w,
):
    """Validate accumulated SRAM expert output from separate sram_out_tensor."""
    sram_core_width = sram_per_core_N * tile_w

    for dev_idx, sram_dev in enumerate(ttnn.get_device_tensors(sram_out_tensor)):
        sram_output_dev = ttnn.to_torch(sram_dev)
        slices = []
        for ci in range(num_sram_cores_active):
            start = ci * sram_core_width
            slices.append(sram_output_dev[..., start : start + sram_core_width])
        accum_output = torch.cat(slices, dim=-1)
        torch_expected = sum(torch_a.float() @ torch_b_all[eidx][dev_idx].float() for eidx in active_sram).bfloat16()
        passing, msg = comp_pcc(torch_expected, accum_output, pcc_threshold)
        logger.info(f"Device {dev_idx} accum (SRAM) PCC: {msg}")
        assert passing, f"Device {dev_idx} accum (SRAM) failed: {msg}"


def _validate_dram_output(
    result,
    torch_a,
    torch_b_all,
    active_dram,
    dram_per_core_N,
    num_dram_cores_active,
    pcc_threshold,
    dram_fuse_silu,
    tile_w,
):
    """Validate per-expert DRAM output from dram_core_grid output tensor."""
    dram_core_width = dram_per_core_N * tile_w
    num_active_dram = len(active_dram)

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result)):
        output_dev = ttnn.to_torch(out_dev)
        for exp_offset, eidx in enumerate(active_dram):
            slices = []
            for ci in range(num_dram_cores_active):
                start = ci * dram_core_width * num_active_dram + exp_offset * dram_core_width
                slices.append(output_dev[..., start : start + dram_core_width])
            expert_output = torch.cat(slices, dim=-1)
            mm_result = torch_a.float() @ torch_b_all[eidx][dev_idx].float()
            if dram_fuse_silu:
                mm_result = torch.nn.functional.silu(mm_result)
            torch_expected = mm_result.bfloat16()
            passing, msg = comp_pcc(torch_expected, expert_output, pcc_threshold)
            logger.info(f"Device {dev_idx} expert {eidx} (DRAM) PCC: {msg}")
            assert passing, f"Device {dev_idx} expert {eidx} (DRAM) failed: {msg}"


def _validate_dram_output_accum(
    result,
    torch_a,
    torch_b_all,
    active_dram,
    dram_per_core_N,
    num_dram_cores_active,
    pcc_threshold,
    tile_w,
):
    """Validate accumulated DRAM output from dram_core_grid output tensor."""
    dram_core_width = dram_per_core_N * tile_w

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result)):
        output_dev = ttnn.to_torch(out_dev)
        slices = []
        for ci in range(num_dram_cores_active):
            start = ci * dram_core_width
            slices.append(output_dev[..., start : start + dram_core_width])
        accum_output = torch.cat(slices, dim=-1)
        torch_expected = sum(torch_a.float() @ torch_b_all[eidx][dev_idx].float() for eidx in active_dram).bfloat16()
        passing, msg = comp_pcc(torch_expected, accum_output, pcc_threshold)
        logger.info(f"Device {dev_idx} accum (DRAM) PCC: {msg}")
        assert passing, f"Device {dev_idx} accum (DRAM) failed: {msg}"


def _validate_sram_output_slice_k(
    sram_out_tensor,
    torch_a,
    torch_b_all,
    active_sram,
    sram_per_core_N,
    num_active_sram,
    sram_cores_list,
    sram_k_parallel,
    sram_n_parallel,
    M,
    N_sram_per_device,
    pcc_threshold,
    tile_w,
):
    """K-sliced SRAM verification — reduce K-partials from separate sram_out_tensor."""
    for dev_idx, sram_dev in enumerate(ttnn.get_device_tensors(sram_out_tensor)):
        sram_output_dev = ttnn.to_torch(sram_dev)
        sram_out_shard = sram_per_core_N * tile_w * num_active_sram
        for exp_offset, eidx in enumerate(active_sram):
            partials = {}
            for si, core in enumerate(sram_cores_list):
                k_idx = si // sram_n_parallel
                n_idx = si % sram_n_parallel
                start = si * sram_out_shard + exp_offset * sram_per_core_N * tile_w
                partials.setdefault((k_idx, n_idx), []).append(
                    sram_output_dev[..., start : start + sram_per_core_N * tile_w]
                )
            reduced = torch.zeros((M, N_sram_per_device), dtype=sram_output_dev.dtype)
            for n_idx in range(sram_n_parallel):
                col_sum = sum(partials[(k_idx, n_idx)][0] for k_idx in range(sram_k_parallel))
                reduced[..., n_idx * tile_w : (n_idx + 1) * tile_w] = col_sum
            torch_expected = (torch_a.float() @ torch_b_all[eidx][dev_idx].float()).bfloat16()
            passing, msg = comp_pcc(torch_expected, reduced, pcc_threshold)
            logger.info(f"Device {dev_idx} expert {eidx} (SRAM-Ksliced) PCC: {msg}")
            assert passing, f"Device {dev_idx} expert {eidx} (SRAM-Ksliced) failed: {msg}"


# ---------------------------------------------------------------------------
# Hybrid SRAM + DRAM expert tests
# ---------------------------------------------------------------------------


def _setup_core_grids(mesh_device, cores_per_dram_bank, num_sram_cores, sram_cores_override, has_sram):
    """Build SRAM / DRAM / compute core grids from device topology.

    DRAM cores are always included. SRAM cores are included only when has_sram is True.
    """
    primary_cores_list = mesh_device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)
    dram_cores_list = []
    for pc in primary_cores_list:
        for offset in range(cores_per_dram_bank):
            dram_cores_list.append(ttnn.CoreCoord(pc.x + offset, pc.y))
    dram_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in dram_cores_list]
    )

    dram_core_set = set((c.x, c.y) for c in dram_cores_list)
    compute_grid = mesh_device.compute_with_storage_grid_size()
    available_cores = [
        ttnn.CoreCoord(x, y)
        for y in range(compute_grid.y)
        for x in range(compute_grid.x)
        if (x, y) not in dram_core_set
    ]
    if sram_cores_override is not None:
        sram_cores_list = list(sram_cores_override)
        num_sram_cores = len(sram_cores_list)
        num_overlap = sum(1 for c in sram_cores_list if (c.x, c.y) in dram_core_set)
        logger.info(
            f"Using {num_sram_cores} SRAM cores from override "
            f"({num_overlap} overlap with DRAM cores — both paths active on those cores)"
        )
    else:
        assert (
            len(available_cores) >= num_sram_cores
        ), f"Need {num_sram_cores} SRAM cores but only {len(available_cores)} non-DRAM cores available"
        sram_cores_list = available_cores[:num_sram_cores]
    sram_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in sram_cores_list]
    )

    # Combine SRAM + DRAM, deduplicate (may overlap with sram_cores_override).
    all_cores_list = list(dram_cores_list)
    if has_sram:
        all_cores_list = list(sram_cores_list) + all_cores_list
    else:
        sram_core_grid = None
    seen = set()
    unique_cores_list = []
    for c in all_cores_list:
        key = (c.x, c.y)
        if key not in seen:
            seen.add(key)
            unique_cores_list.append(c)
    compute_core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(c.x, c.y), ttnn.CoreCoord(c.x, c.y)) for c in unique_cores_list]
    )

    return dict(
        sram_cores_list=sram_cores_list,
        sram_core_grid=sram_core_grid,
        dram_cores_list=dram_cores_list,
        dram_core_grid=dram_core_grid,
        compute_core_grid=compute_core_grid,
    )


def _build_dram_experts(
    dram_expert_ids,
    torch_b_all,
    assigner,
    mesh_device,
    K,
    dram_per_core_N,
    N_dram_per_device,
    num_banks,
    cores_per_dram_bank,
    num_experts,
    dram_meta_flags,
    mesh_rows,
    mesh_cols,
    subblock_k,
    Kt,
    tile_w,
):
    """Build DRAM CompressedTensors and dram_meta_tensors for the kernel."""
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        ]
    )
    dram_b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [K, dram_per_core_N * tile_w * cores_per_dram_bank], ttnn.ShardOrientation.ROW_MAJOR),
    )
    dram_cts = []
    for eidx in dram_expert_ids:
        slices_shuffled = [shuffle_tensor_tiles(b, tile_w, num_banks) for b in torch_b_all[eidx]]
        b_4d = torch.stack(slices_shuffled).reshape(mesh_rows, mesh_cols, K, N_dram_per_device)
        ct = CompressedTensor.from_torch(
            b_4d,
            assigner,
            device=mesh_device,
            memory_config=dram_b_mem,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        dram_cts.append(ct)
        logger.info(f"  DRAM expert {eidx} uploaded (packed idx {len(dram_cts) - 1})")

    num_subblocks_k = Kt // subblock_k
    dram_meta_tensors = create_dram_expert_tensors_multi_device(
        mesh_device,
        dram_cts,
        subblock_k,
        num_subblocks_k,
        dram_per_core_N,
        cores_per_dram_bank=cores_per_dram_bank,
        num_total_experts=num_experts,
        is_dram_flags=dram_meta_flags,
    )
    return dram_cts, dram_meta_tensors


def _compute_dram_matmul_params(
    K, N, tile_w, num_banks, num_dram_cores, num_dram_cores_active, cores_per_dram_bank, subblock_k
):
    """Compute DRAM per-core tiling and subblock_k."""
    Kt = K // tile_w
    n_dram_padded = _pad_to_dram_banks(N, tile_w, tile_w * num_banks * cores_per_dram_bank)
    dram_per_core_N = n_dram_padded // num_dram_cores // tile_w
    if subblock_k is None:
        subblock_k = Kt // 4 if Kt > 8 else Kt
    if subblock_k % 2 != 0:
        subblock_k = max(2, subblock_k - 1)
    assert Kt % subblock_k == 0
    N_dram_per_device = dram_per_core_N * tile_w * num_dram_cores_active
    return Kt, dram_per_core_N, subblock_k, N_dram_per_device


def _build_assigner(formats_per_device):
    """Create CompressedTensorAssigner from per-device format lists."""
    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    bfp0_mae = 1e-3 if "bfp0" in all_formats else 0.01
    return CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=all_formats, bfp0_mae_threshold=bfp0_mae)


def _build_weight_tensors(num_experts, K, N, N_dram_per_device, sram_id_set, formats_per_device, num_devices):
    """Build per-expert, per-device random B weight tensors."""
    torch_b_all = {}
    for eidx in range(num_experts):
        torch.manual_seed(eidx * 1000 + 42)
        N_per_dev = N if eidx in sram_id_set else N_dram_per_device
        per_dev = []
        for dev_idx in range(num_devices):
            b = torch.randn((K, N_per_dev)).float()
            _scale_tiles_random_formats(b, formats_per_device[dev_idx])
            per_dev.append(b)
        torch_b_all[eidx] = per_dev
        logger.info(f"  torch_b expert {eidx}/{num_experts} created")
    return torch_b_all


def _build_expert_flags(num_experts, sram_expert_ids, dram_expert_ids):
    """Build dram_meta_flags (build-time) and is_dram_flags (run-time)."""
    dram_meta_flags = [0] * num_experts
    for eidx in dram_expert_ids:
        dram_meta_flags[eidx] = 1
    is_dram_flags = list(dram_meta_flags)
    for eidx in sram_expert_ids:
        is_dram_flags[eidx] = 0
    return dram_meta_flags, is_dram_flags


def _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w):
    """Create HEIGHT_SHARDED A tensor, replicated across devices."""
    a_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, [M, K], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        torch_a.repeat(num_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=a_mem,
        tile=ttnn.Tile([M, tile_w]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores):
    """Create HEIGHT_SHARDED uint16 index tensor, replicated across devices."""
    index_torch = torch.zeros(num_cores, 16, dtype=torch.int32)
    for i, expert_id in enumerate(active_expert_ids):
        index_torch[:, i] = expert_id
    index_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(compute_core_grid, [1, 16], ttnn.ShardOrientation.ROW_MAJOR),
    )
    return ttnn.from_torch(
        index_torch.to(torch.uint16),
        dtype=ttnn.uint16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=index_mem,
        tile=ttnn.Tile([1, 16]),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _build_expert_selection_meta(mesh_device, a_tensor, is_dram_flags):
    """Build per-device expert selection metadata (is_dram + table_idx) for expert dispatch."""
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    a_per_device = ttnn.get_device_tensors(a_tensor)
    expert_selection_meta = {}
    for row in range(mesh_rows):
        for col in range(mesh_cols):
            coord = ttnn.MeshCoordinate(row, col)
            dev_idx = row * mesh_cols + col
            core_grid = a_per_device[dev_idx].memory_config().shard_spec.grid
            all_cores_dev = ttnn.corerange_to_cores(core_grid)
            expert_selection_meta[coord] = create_expert_selection_meta(
                mesh_device,
                all_cores_dev,
                is_dram_flags,
                len(is_dram_flags),
                device_coord=coord,
            )
    return expert_selection_meta


def _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, sram_k_per_core, sram_per_core_n, Kt):
    """Build SRAM format tensors and K-offset core values."""
    from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import create_expert_fmt_tensors

    num_tiles = sram_k_per_core * sram_per_core_n
    sram_fmt_tensors = create_expert_fmt_tensors(sram_cts, mesh_device, sram_core_grid, num_tiles)

    sram_k_offsets = None
    if sram_k_per_core < Kt:
        sram_cores = ttnn.corerange_to_cores(sram_core_grid)
        n_parallel = len(sram_cores) * sram_k_per_core // Kt
        sram_k_offsets = [(sram_cores[i], (i // n_parallel) * sram_k_per_core) for i in range(len(sram_cores))]

    return sram_fmt_tensors, sram_k_offsets


# ---------------------------------------------------------------------------
# Variant runners — each does common setup, builds SRAM CTs + output, runs
# the kernel, and validates.
# ---------------------------------------------------------------------------


def _run_standard(
    mesh_device,
    M,
    K,
    N,
    num_experts,
    sram_expert_ids,
    dram_expert_ids,
    active_expert_ids,
    formats_per_device,
    subblock_k,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
):
    """Standard path: WIDTH_SHARDED SRAM, per-expert output slices on compute_core_grid."""
    tile_w = 32
    sram_id_set = set(sram_expert_ids)
    has_sram = bool(sram_expert_ids)
    num_sram_cores = sram_k_parallel * sram_n_parallel

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, num_sram_cores, sram_cores_override, has_sram)
    sram_cores_list = grids["sram_cores_list"]
    sram_core_grid = grids["sram_core_grid"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(grids["dram_cores_list"])

    num_devices = mesh_device.get_num_devices()
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
    )

    assert len(formats_per_device) == num_devices
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    assigner = _build_assigner(formats_per_device)
    torch_b_all = _build_weight_tensors(
        num_experts,
        K,
        N,
        N_dram_per_device,
        sram_id_set,
        formats_per_device,
        num_devices,
    )
    dram_meta_flags, is_dram_flags = _build_expert_flags(num_experts, sram_expert_ids, dram_expert_ids)
    dram_cts, dram_meta_tensors = _build_dram_experts(
        dram_expert_ids,
        torch_b_all,
        assigner,
        mesh_device,
        K,
        dram_per_core_N,
        N_dram_per_device,
        num_banks,
        cores_per_dram_bank,
        num_experts,
        dram_meta_flags,
        mesh_device.shape[0],
        mesh_device.shape[1],
        subblock_k,
        Kt,
        tile_w,
    )
    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores)

    active_sram = [eid for eid in active_expert_ids if eid in sram_id_set]
    active_dram = [eid for eid in active_expert_ids if eid not in sram_id_set]
    num_active_experts = len(active_sram) + len(active_dram)
    num_sram_cores_active = len(sram_cores_list) if sram_expert_ids else 0
    num_cores = compute_core_grid.num_cores()
    sram_per_core_N = N // num_sram_cores_active // tile_w if num_sram_cores_active else 0

    sram_cts = (
        _build_sram_cts_standard(
            sram_expert_ids,
            torch_b_all,
            assigner,
            mesh_device,
            sram_core_grid,
            K,
            N,
            sram_per_core_N,
            num_devices,
            mesh_device.shape[0],
            mesh_device.shape[1],
            tile_w,
        )
        if has_sram
        else []
    )

    num_active_sram = len(active_sram)
    num_active_dram = len(active_dram)

    sram_out_tensor = (
        _build_sram_output(
            mesh_device,
            M,
            sram_per_core_N,
            num_active_sram,
            num_sram_cores_active,
            num_devices,
            sram_core_grid,
            tile_w,
        )
        if has_sram
        else None
    )
    out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        num_active_dram,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
    )
    expert_selection_meta = _build_expert_selection_meta(mesh_device, a_tensor, is_dram_flags)
    sram_fmt_tensors, sram_k_offsets = (
        _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, Kt, sram_per_core_N, Kt) if has_sram else ({}, None)
    )
    result = ExpertKernel.op(
        a_tensor,
        sram_cts,
        dram_cts,
        out_tensor,
        index_tensor,
        is_dram_flags,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        expert_selection_meta=expert_selection_meta,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=Kt,
        sram_output_tensor=sram_out_tensor,
        dram_fuse_silu=dram_fuse_silu,
    )
    if active_sram:
        _validate_sram_output(
            sram_out_tensor,
            torch_a,
            torch_b_all,
            active_sram,
            sram_per_core_N,
            num_sram_cores_active,
            num_active_sram,
            pcc_threshold,
            tile_w,
        )
    if active_dram:
        _validate_dram_output(
            result,
            torch_a,
            torch_b_all,
            active_dram,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            dram_fuse_silu,
            tile_w,
        )


def _run_accum(
    mesh_device,
    M,
    K,
    N,
    num_experts,
    sram_expert_ids,
    dram_expert_ids,
    active_expert_ids,
    formats_per_device,
    subblock_k,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
):
    """Accumulation path: WIDTH_SHARDED SRAM, expert outputs summed in-place."""
    tile_w = 32
    sram_id_set = set(sram_expert_ids)
    has_sram = bool(sram_expert_ids)
    num_sram_cores = sram_k_parallel * sram_n_parallel

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, num_sram_cores, sram_cores_override, has_sram)
    sram_cores_list = grids["sram_cores_list"]
    sram_core_grid = grids["sram_core_grid"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(grids["dram_cores_list"])

    num_devices = mesh_device.get_num_devices()
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
    )

    assert len(formats_per_device) == num_devices
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    assigner = _build_assigner(formats_per_device)
    torch_b_all = _build_weight_tensors(
        num_experts,
        K,
        N,
        N_dram_per_device,
        sram_id_set,
        formats_per_device,
        num_devices,
    )
    dram_meta_flags, is_dram_flags = _build_expert_flags(num_experts, sram_expert_ids, dram_expert_ids)
    dram_cts, dram_meta_tensors = _build_dram_experts(
        dram_expert_ids,
        torch_b_all,
        assigner,
        mesh_device,
        K,
        dram_per_core_N,
        N_dram_per_device,
        num_banks,
        cores_per_dram_bank,
        num_experts,
        dram_meta_flags,
        mesh_device.shape[0],
        mesh_device.shape[1],
        subblock_k,
        Kt,
        tile_w,
    )
    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores)

    active_sram = [eid for eid in active_expert_ids if eid in sram_id_set]
    active_dram = [eid for eid in active_expert_ids if eid not in sram_id_set]
    num_active_experts = len(active_sram) + len(active_dram)
    num_sram_cores_active = len(sram_cores_list) if sram_expert_ids else 0
    num_cores = compute_core_grid.num_cores()
    sram_per_core_N = N // num_sram_cores_active // tile_w if num_sram_cores_active else 0

    sram_cts = (
        _build_sram_cts_standard(
            sram_expert_ids,
            torch_b_all,
            assigner,
            mesh_device,
            sram_core_grid,
            K,
            N,
            sram_per_core_N,
            num_devices,
            mesh_device.shape[0],
            mesh_device.shape[1],
            tile_w,
        )
        if has_sram
        else []
    )

    sram_out_tensor = (
        _build_sram_output(
            mesh_device,
            M,
            sram_per_core_N,
            1,
            num_sram_cores_active,
            num_devices,
            sram_core_grid,
            tile_w,
        )
        if has_sram
        else None
    )
    out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        1,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
    )
    expert_selection_meta = _build_expert_selection_meta(mesh_device, a_tensor, is_dram_flags)
    sram_fmt_tensors, sram_k_offsets = (
        _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, Kt, sram_per_core_N, Kt) if has_sram else ({}, None)
    )
    result = ExpertKernel.op(
        a_tensor,
        sram_cts,
        dram_cts,
        out_tensor,
        index_tensor,
        is_dram_flags,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        expert_selection_meta=expert_selection_meta,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        accum_experts=True,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=Kt,
        sram_output_tensor=sram_out_tensor,
    )
    if active_sram:
        _validate_sram_output_accum(
            sram_out_tensor,
            torch_a,
            torch_b_all,
            active_sram,
            sram_per_core_N,
            num_sram_cores_active,
            pcc_threshold,
            tile_w,
        )
    if active_dram:
        _validate_dram_output_accum(
            result,
            torch_a,
            torch_b_all,
            active_dram,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            tile_w,
        )


def _run_slice_k(
    mesh_device,
    M,
    K,
    N,
    num_experts,
    sram_expert_ids,
    dram_expert_ids,
    active_expert_ids,
    formats_per_device,
    subblock_k,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
):
    """K-sliced path: HEIGHT_SHARDED SRAM, separate output grids."""
    tile_w = 32
    sram_id_set = set(sram_expert_ids)
    has_sram = bool(sram_expert_ids)
    num_sram_cores = sram_k_parallel * sram_n_parallel

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, num_sram_cores, sram_cores_override, has_sram)
    sram_cores_list = grids["sram_cores_list"]
    sram_core_grid = grids["sram_core_grid"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(grids["dram_cores_list"])

    num_devices = mesh_device.get_num_devices()
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
    )

    assert len(formats_per_device) == num_devices
    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    assigner = _build_assigner(formats_per_device)
    torch_b_all = _build_weight_tensors(
        num_experts,
        K,
        N,
        N_dram_per_device,
        sram_id_set,
        formats_per_device,
        num_devices,
    )
    dram_meta_flags, is_dram_flags = _build_expert_flags(num_experts, sram_expert_ids, dram_expert_ids)
    dram_cts, dram_meta_tensors = _build_dram_experts(
        dram_expert_ids,
        torch_b_all,
        assigner,
        mesh_device,
        K,
        dram_per_core_N,
        N_dram_per_device,
        num_banks,
        cores_per_dram_bank,
        num_experts,
        dram_meta_flags,
        mesh_device.shape[0],
        mesh_device.shape[1],
        subblock_k,
        Kt,
        tile_w,
    )
    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores)

    active_sram = [eid for eid in active_expert_ids if eid in sram_id_set]
    active_dram = [eid for eid in active_expert_ids if eid not in sram_id_set]
    num_active_experts = len(active_sram) + len(active_dram)
    num_sram_cores_active = len(sram_cores_list)
    num_active_sram = len(active_sram)
    num_active_dram = len(active_dram)

    sram_k_per_core = Kt // sram_k_parallel
    sram_per_core_N = N // sram_n_parallel // tile_w

    sram_cts = _build_sram_cts_slice_k(
        sram_expert_ids,
        torch_b_all,
        assigner,
        mesh_device,
        sram_core_grid,
        sram_k_per_core,
        sram_n_parallel,
        num_sram_cores_active,
        num_devices,
        mesh_device.shape[0],
        mesh_device.shape[1],
        tile_w,
    )

    sram_out_tensor = _build_sram_output(
        mesh_device,
        M,
        sram_per_core_N,
        num_active_sram,
        num_sram_cores_active,
        num_devices,
        sram_core_grid,
        tile_w,
    )
    out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        num_active_dram,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
    )
    expert_selection_meta = _build_expert_selection_meta(mesh_device, a_tensor, is_dram_flags)
    sram_fmt_tensors, sram_k_offsets = _build_sram_fmt_data(
        sram_cts,
        mesh_device,
        sram_core_grid,
        sram_k_per_core,
        sram_per_core_N,
        Kt,
    )
    result = ExpertKernel.op(
        a_tensor,
        sram_cts,
        dram_cts,
        out_tensor,
        index_tensor,
        is_dram_flags,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        expert_selection_meta=expert_selection_meta,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=sram_k_per_core,
        sram_output_tensor=sram_out_tensor,
        dram_fuse_silu=dram_fuse_silu,
    )
    if active_sram:
        _validate_sram_output_slice_k(
            sram_out_tensor,
            torch_a,
            torch_b_all,
            active_sram,
            sram_per_core_N,
            num_active_sram,
            sram_cores_list,
            sram_k_parallel,
            sram_n_parallel,
            M,
            N,
            pcc_threshold,
            tile_w,
        )
    if active_dram:
        _validate_dram_output(
            result,
            torch_a,
            torch_b_all,
            active_dram,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            dram_fuse_silu,
            tile_w,
        )


def _run_hybrid_expert_multi_device(
    mesh_device,
    M,
    K,
    N,
    num_experts,
    sram_expert_ids,
    dram_expert_ids,
    active_expert_ids,
    formats_per_device,
    subblock_k=None,
    cores_per_dram_bank=1,
    pcc_threshold=0.97,
    accum_experts=False,
    sram_cores_override=None,
    sram_k_parallel=1,
    sram_n_parallel=1,
    dram_fuse_silu=False,
):
    """Dispatcher: delegate to the appropriate variant."""
    assert dram_expert_ids, "DRAM expert path is always required"
    slice_k = sram_k_parallel > 1
    if slice_k:
        _run_slice_k(
            mesh_device,
            M,
            K,
            N,
            num_experts,
            sram_expert_ids,
            dram_expert_ids,
            active_expert_ids,
            formats_per_device,
            subblock_k,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
        )
    elif accum_experts:
        _run_accum(
            mesh_device,
            M,
            K,
            N,
            num_experts,
            sram_expert_ids,
            dram_expert_ids,
            active_expert_ids,
            formats_per_device,
            subblock_k,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
        )
    else:
        _run_standard(
            mesh_device,
            M,
            K,
            N,
            num_experts,
            sram_expert_ids,
            dram_expert_ids,
            active_expert_ids,
            formats_per_device,
            subblock_k,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
        )


# ---------------------------------------------------------------------------
# Hybrid expert test variants — each exercises different SRAM/DRAM
# configurations, formats, and router selections.
# ---------------------------------------------------------------------------


def test_hybrid_expert_1sram_1dram(device):
    """1 SRAM expert + 1 DRAM expert, bfp8 only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=128,
        N=256,
        num_experts=2,
        sram_expert_ids=[0],
        dram_expert_ids=[1],
        active_expert_ids=[0, 1],
        formats_per_device=[["bfp8"]],
        sram_n_parallel=4,
        subblock_k=2,
    )


def test_hybrid_expert_0sram_2dram(device):
    """0 SRAM experts + 2 DRAM experts, bfp8 only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=128,
        N=256,
        num_experts=2,
        sram_expert_ids=[],
        dram_expert_ids=[0, 1],
        active_expert_ids=[0, 1],
        formats_per_device=[["bfp8"]],
        sram_n_parallel=4,
        subblock_k=2,
    )


def test_hybrid_expert_2sram_2dram_mixed(device):
    """4 experts, 2 SRAM + 2 DRAM, bfp8+bfp4, router selects all 4."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=4,
        sram_expert_ids=[0, 2],
        dram_expert_ids=[1, 3],
        active_expert_ids=[0, 1, 2, 3],
        formats_per_device=[["bfp8", "bfp4"]],
        sram_n_parallel=8,
    )


def test_hybrid_expert_sparse_activation(device):
    """8 experts (3 SRAM, 5 DRAM), router selects 4 out of 8."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[1, 4, 6],
        dram_expert_ids=[0, 2, 3, 5, 7],
        active_expert_ids=[0, 1, 4, 7],
        formats_per_device=[["bfp8", "bfp4"]],
        sram_n_parallel=8,
    )


def test_hybrid_expert_multi_device_2sram_2dram(bh_2d_mesh_device):
    """Multi-device hybrid: 2 SRAM + 2 DRAM experts, alternating formats, all 4 active, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=256,
        num_experts=4,
        sram_expert_ids=[0, 2],
        dram_expert_ids=[1, 3],
        active_expert_ids=[0, 1, 2, 3],
        formats_per_device=[["bfp8", "bfp4"], ["bfp8", "bfp2"]] * 4,
        sram_n_parallel=8,
    )


def test_hybrid_expert_multi_device_sparse(bh_2d_mesh_device):
    """Multi-device hybrid: 8 experts (3 SRAM + 5 DRAM), router selects 4, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[1, 4, 6],
        dram_expert_ids=[0, 2, 3, 5, 7],
        active_expert_ids=[0, 1, 4, 7],
        formats_per_device=[
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp4", "bfp2", "bfp0"],
            ["bfp4", "bfp0"],
            ["bfp2", "bfp0"],
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp2"],
        ],
        sram_n_parallel=8,
    )


def test_hybrid_expert_multi_device_sparse_2cores_per_bank(bh_2d_mesh_device):
    """Multi-device hybrid: 8 experts (3 SRAM + 5 DRAM), router selects 4, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[1, 4, 6],
        dram_expert_ids=[0, 2, 3, 5, 7],
        active_expert_ids=[0, 1, 4, 7],
        formats_per_device=[
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp4", "bfp2", "bfp0"],
            ["bfp4", "bfp2"],
            ["bfp2", "bfp0"],
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp2"],
        ],
        sram_n_parallel=8,
        cores_per_dram_bank=2,
    )


# ---------------------------------------------------------------------------
# Accumulation tests — accum_experts=True sums expert outputs in-place
# ---------------------------------------------------------------------------


def test_hybrid_expert_single_device_sparse_accum_experts(device):
    """Single-device hybrid: 8 experts (3 SRAM + 5 DRAM), router selects 4, 1 device."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=256,
        N=7168,
        num_experts=8,
        sram_expert_ids=[1, 4, 6],
        dram_expert_ids=[0, 2, 3, 5, 7],
        active_expert_ids=[0, 1, 4, 7],
        formats_per_device=[
            ["bfp4", "bfp2", "bfp0"],
        ],
        sram_n_parallel=56,
        subblock_k=8,  # Kt=8, num_subblocks_k=1
        accum_experts=True,
    )


def test_hybrid_expert_multi_device_sparse_accum_experts(bh_2d_mesh_device):
    """Multi-device hybrid: 8 experts (3 SRAM + 5 DRAM), router selects 4, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=256,
        N=7168,
        num_experts=8,
        sram_expert_ids=[1, 4, 6],
        dram_expert_ids=[0, 2, 3, 5, 7],
        active_expert_ids=[0, 1, 4, 7],
        formats_per_device=[
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp4", "bfp2", "bfp0"],
            ["bfp4", "bfp2"],
            ["bfp2", "bfp0"],
            ["bfp4", "bfp2"],
            ["bfp4"],
            ["bfp2"],
        ],
        sram_n_parallel=56,
        subblock_k=8,  # Kt=8, num_subblocks_k=1
        accum_experts=True,
    )


# ---------------------------------------------------------------------------
# Irregular SRAM core grid tests — uses SharedExpertOp A/B core layouts
# ---------------------------------------------------------------------------


@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_gate_grid(device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=7168, N=256, gate-proj A-cores (64 irregular)."""
    a_cores, _ = _build_ab_grids(device)
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        sram_cores_override=a_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
        dram_fuse_silu=True,
    )


@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_up_grid(device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=7168, N=256, up-proj B-cores (64 irregular)."""
    _, b_cores = _build_ab_grids(device)
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        sram_cores_override=b_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
    )


@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_down_grid(device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=256, N=7168, down-proj 112-core grid."""
    down_cores = _build_down_grid(device)
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=256,
        N=7168,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        sram_cores_override=down_cores,
        accum_experts=True,
    )
