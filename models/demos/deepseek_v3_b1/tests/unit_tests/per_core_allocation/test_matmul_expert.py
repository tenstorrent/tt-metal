# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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
    encode_expert_indices,
)


def shuffle_tensor_tiles(tensor, tile_size, num_banks, subblock_k=None, subblock_n=None):
    """Tile shuffle for WIDTH_SHARDED DRAM layout with subblock support.

    Reorders tiles within each bank's shard for DRAM streaming matmul.
    When subblock_n>1, tiles are grouped into [subblock_k, subblock_n] blocks,
    row-major within each block, column-major at block level.
    When subblock_n=1 (default): fully column-major within each shard.
    """
    orig_shape = tensor.shape
    K = orig_shape[-2]
    N = orig_shape[-1]

    lcm = tile_size * num_banks
    n_padded = ((N + lcm - 1) // lcm) * lcm
    needs_padding = n_padded != N

    tensor = tensor.reshape(-1, K, N)
    batch_size = tensor.shape[0]

    if needs_padding:
        tensor = torch.nn.functional.pad(tensor, (0, n_padded - N))

    K_tiles = K // tile_size
    per_N = n_padded // num_banks
    per_N_tiles = per_N // tile_size
    num_tiles_per_shard = K_tiles * per_N_tiles

    if subblock_k is None:
        subblock_k = K_tiles
    if subblock_n is None:
        subblock_n = 1

    assert K_tiles % subblock_k == 0, f"K_tiles ({K_tiles}) must be divisible by subblock_k ({subblock_k})"
    assert per_N_tiles % subblock_n == 0, f"per_N_tiles ({per_N_tiles}) must be divisible by subblock_n ({subblock_n})"

    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    num_subblocks_k = K_tiles // subblock_k
    num_n_groups = per_N_tiles // subblock_n
    block_size = subblock_k * subblock_n

    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    block_idx = i // block_size
    pos_in_block = i % block_size
    local_k = pos_in_block // subblock_n
    local_n = pos_in_block % subblock_n

    n_group = block_idx % num_n_groups
    k_sub = block_idx // num_n_groups

    global_k = k_sub * subblock_k + local_k
    global_n = n_group * subblock_n + local_n
    source_idx = global_k * per_N_tiles + global_n

    shuffled_tiles = tiles[:, source_idx, :, :]

    shuffled_tiles = shuffled_tiles.reshape(-1, K_tiles, per_N_tiles, tile_size, tile_size)
    shuffled_tiles = shuffled_tiles.permute(0, 1, 3, 2, 4).contiguous()
    shuffled_shards = shuffled_tiles.reshape(-1, K, per_N)

    shuffled = shuffled_shards.reshape(batch_size, num_banks, K, per_N)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch_size, K, n_padded)

    if needs_padding:
        shuffled = shuffled[:, :, :N]

    shuffled = shuffled.reshape(*orig_shape)
    return shuffled


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


def _scale_tiles_random_formats(b_torch, formats, distribution="random", weights=None):
    """Assign formats to tiles and scale values to be representable.

    Args:
        b_torch: weight tensor to modify in-place.
        formats: list of format names (e.g. ["bfp8", "bfp2", "bfp4"]).
        distribution: "random" — each tile picks a format independently (default).
                      "uniform" — deterministic 2D interleave so any row-slice or
                      column-slice sees all formats in proportion.
        weights: optional dict mapping format name to relative proportion,
                 e.g. {"bfp8": 3, "bfp2": 1}. None means equal distribution.
    """
    assert formats, "formats must not be empty"

    M, N = b_torch.shape
    tiles_h, tiles_w = M // 32, N // 32
    num_tiles = tiles_h * tiles_w

    if weights is not None:
        probs = torch.tensor([weights.get(f, 1) for f in formats], dtype=torch.float)
    else:
        probs = torch.ones(len(formats))
    probs = probs / probs.sum()

    if distribution == "random":
        fmt_indices = torch.multinomial(probs.expand(num_tiles, -1), 1).squeeze(1)
        fmt_indices = fmt_indices.reshape(tiles_h, tiles_w)
    elif distribution == "uniform":
        if weights is not None:
            int_weights = [int(weights.get(f, 1)) for f in formats]
        else:
            int_weights = [1] * len(formats)
        pattern = torch.cat([torch.full((w,), i, dtype=torch.long) for i, w in enumerate(int_weights)])
        reps = (tiles_h + len(pattern) - 1) // len(pattern)
        col = pattern.repeat(reps)[:tiles_h]
        fmt_indices = col.unsqueeze(1).expand(tiles_h, tiles_w).contiguous()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    # bfp8: scale each row by 2^(row%16) — build a (32, 1) multiplier column
    if "bfp8" in formats:
        bfp8_idx = formats.index("bfp8")
        bfp8_scale = (2.0 ** (torch.arange(32).float() % 16)).unsqueeze(1)  # (32, 1)
        mask = fmt_indices == bfp8_idx
        for tr in range(tiles_h):
            for tc in range(tiles_w):
                if mask[tr, tc]:
                    r0, c0 = tr * 32, tc * 32
                    b_torch[r0 : r0 + 32, c0 : c0 + 32] *= bfp8_scale

    # bfp2: random sign × 2^exp per row — generate all at once
    if "bfp2" in formats:
        bfp2_idx = formats.index("bfp2")
        mask = fmt_indices == bfp2_idx
        num_bfp2 = mask.sum().item()
        if num_bfp2 > 0:
            all_exps = torch.randint(-3, 4, (num_bfp2, 32)).float()
            all_signs = torch.sign(torch.randn(num_bfp2, 32, 32))
            all_signs[all_signs == 0] = 1.0
            all_vals = all_signs * (2.0 ** all_exps.unsqueeze(2))
            idx = 0
            for tr in range(tiles_h):
                for tc in range(tiles_w):
                    if mask[tr, tc]:
                        r0, c0 = tr * 32, tc * 32
                        b_torch[r0 : r0 + 32, c0 : c0 + 32] = all_vals[idx]
                        idx += 1

    # bfp0: small noise
    if "bfp0" in formats:
        bfp0_idx = formats.index("bfp0")
        mask = fmt_indices == bfp0_idx
        num_bfp0 = mask.sum().item()
        if num_bfp0 > 0:
            all_noise = torch.randn(num_bfp0, 32, 32) * 1e-3
            idx = 0
            for tr in range(tiles_h):
                for tc in range(tiles_w):
                    if mask[tr, tc]:
                        r0, c0 = tr * 32, tc * 32
                        b_torch[r0 : r0 + 32, c0 : c0 + 32] = all_noise[idx]
                        idx += 1
    # bfp4: keep randn as-is (no action needed)


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
    torch_a_per_expert,
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
        torch_expected = sum(
            torch_a_per_expert[eidx].float() @ torch_b_all[eidx][dev_idx].float() for eidx in active_sram
        ).bfloat16()
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
    tp_expert=True,
):
    """Validate per-expert DRAM output from dram_core_grid output tensor."""
    dram_core_width = dram_per_core_N * tile_w

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result)):
        output_dev = ttnn.to_torch(out_dev)
        # Expert parallel: each device processes only its own expert.
        dev_active_dram = active_dram if tp_expert else [active_dram[dev_idx]]
        num_active_dram = len(dev_active_dram)
        for exp_offset, eidx in enumerate(dev_active_dram):
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
    torch_a_per_expert,
    torch_b_all,
    active_dram,
    dram_per_core_N,
    num_dram_cores_active,
    pcc_threshold,
    tile_w,
    tp_expert=True,
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
        # Expert parallel: each device accumulates only its own expert.
        dev_active_dram = active_dram if tp_expert else [active_dram[dev_idx]]
        torch_expected = sum(
            torch_a_per_expert[eidx].float() @ torch_b_all[eidx][dev_idx].float() for eidx in dev_active_dram
        ).bfloat16()
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
    subblock_n,
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
        slices_shuffled = [shuffle_tensor_tiles(b, tile_w, num_banks, subblock_n=subblock_n) for b in torch_b_all[eidx]]
        b_4d = torch.stack(slices_shuffled).reshape(mesh_rows, mesh_cols, K, N_dram_per_device)
        ct = CompressedTensor.from_torch(
            b_4d,
            assigner,
            device=mesh_device,
            memory_config=dram_b_mem,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        logger.info(f"DRAM expert {eidx} tile counts: {ct.tile_counts}")
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
        subblock_n=subblock_n,
    )
    return dram_cts, dram_meta_tensors


def _compute_dram_matmul_params(
    K, N, tile_w, num_banks, num_dram_cores, num_dram_cores_active, cores_per_dram_bank, subblock_k, subblock_n=None
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
    if subblock_n is None:
        subblock_n = 1
    assert (
        dram_per_core_N % subblock_n == 0
    ), f"dram_per_core_N ({dram_per_core_N}) must be divisible by subblock_n ({subblock_n})"
    N_dram_per_device = dram_per_core_N * tile_w * num_dram_cores_active
    logger.info(
        f"DRAM matmul params: Kt={Kt}, dram_per_core_N={dram_per_core_N}, subblock_k={subblock_k}, subblock_n={subblock_n}, N_dram_per_device={N_dram_per_device}"
    )
    return Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device


def _build_assigner(formats_per_device):
    """Create CompressedTensorAssigner from per-device format lists."""
    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    bfp0_mae = 1e-3 if "bfp0" in all_formats else 0.01
    return CompressedTensorAssigner(metric="pcc", threshold=0.993, formats=all_formats, bfp0_mae_threshold=bfp0_mae)


def _build_weight_tensors(
    num_experts,
    K,
    N,
    N_dram_per_device,
    sram_id_set,
    formats_per_device,
    num_devices,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """Build per-expert, per-device random B weight tensors."""
    torch_b_all = {}
    for eidx in range(num_experts):
        torch.manual_seed(eidx * 1000 + 42)
        N_per_dev = N if eidx in sram_id_set else N_dram_per_device
        per_dev = []
        for dev_idx in range(num_devices):
            b = torch.randn((K, N_per_dev)).float()
            _scale_tiles_random_formats(b, formats_per_device[dev_idx], fmt_distribution, fmt_ratios)
            per_dev.append(b)
        torch_b_all[eidx] = per_dev
        logger.info(f"  torch_b expert {eidx}/{num_experts} created")
    return torch_b_all


def _build_weight_tensors_replicated(
    num_experts,
    K,
    N_dram_per_device,
    formats,
    num_devices,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """Build per-expert weight tensors, identical across devices (for tp_expert=False)."""
    torch_b_all = {}
    for eidx in range(num_experts):
        torch.manual_seed(eidx * 1000 + 42)
        b = torch.randn((K, N_dram_per_device)).float()
        _scale_tiles_random_formats(b, formats, fmt_distribution, fmt_ratios)
        # All devices share the same tensor.
        torch_b_all[eidx] = [b] * num_devices
        logger.info(f"  torch_b expert {eidx}/{num_experts} created (replicated)")
    return torch_b_all


def _build_dram_experts_replicated(
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
    subblock_k,
    subblock_n,
    Kt,
    tile_w,
):
    """Build DRAM CompressedTensors replicated across all devices (for tp_expert=False)."""
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
        b_shuffled = shuffle_tensor_tiles(torch_b_all[eidx][0], tile_w, num_banks, subblock_n=subblock_n)
        b_4d = b_shuffled.reshape(1, 1, K, N_dram_per_device)
        ct = CompressedTensor.from_torch(
            b_4d,
            assigner,
            device=mesh_device,
            memory_config=dram_b_mem,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        )
        logger.info(f"DRAM expert {eidx} tile counts: {ct.tile_counts}")
        dram_cts.append(ct)
        logger.info(f"  DRAM expert {eidx} uploaded replicated (packed idx {len(dram_cts) - 1})")

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
        subblock_n=subblock_n,
    )
    return dram_cts, dram_meta_tensors


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


def _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags):
    """Create HEIGHT_SHARDED uint16 index tensor, replicated across devices.

    SRAM experts (is_dram_flags[eid]==0) get bit 15 set in the index value.
    """
    encoded_ids = encode_expert_indices(active_expert_ids, is_dram_flags)
    index_torch = torch.zeros(num_cores, 16, dtype=torch.int32)
    for i, eid in enumerate(encoded_ids):
        index_torch[:, i] = eid
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


def _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, sram_k_per_core, sram_per_core_n, Kt):
    """Build SRAM format tensors and K-offset core values."""
    from models.demos.deepseek_v3_b1.micro_ops.matmul_expert.op import create_expert_fmt_tensors

    sram_fmt_tensors, sram_base_addr_tensors = create_expert_fmt_tensors(
        sram_cts, mesh_device, sram_core_grid, sram_k_per_core, sram_per_core_n
    )

    sram_k_offsets = None
    if sram_k_per_core < Kt:
        sram_cores = ttnn.corerange_to_cores(sram_core_grid)
        n_parallel = len(sram_cores) * sram_k_per_core // Kt
        sram_k_offsets = [(sram_cores[i], (i // n_parallel) * sram_k_per_core) for i in range(len(sram_cores))]

    return sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets


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
    subblock_n,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """Standard path: WIDTH_SHARDED SRAM, per-expert output slices on compute_core_grid."""
    tile_w = 32
    sram_id_set = set(sram_expert_ids)
    has_sram = bool(sram_expert_ids)
    num_sram_cores = sram_k_parallel * sram_n_parallel
    assert tp_expert or not has_sram, "Expert parallel (tp_expert=False) only supports DRAM matmul"

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, num_sram_cores, sram_cores_override, has_sram)
    sram_cores_list = grids["sram_cores_list"]
    sram_core_grid = grids["sram_core_grid"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(grids["dram_cores_list"])

    num_devices = mesh_device.get_num_devices()
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
        subblock_n,
    )

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)
    dram_meta_flags, is_dram_flags = _build_expert_flags(num_experts, sram_expert_ids, dram_expert_ids)

    if tp_expert:
        assert len(formats_per_device) == num_devices
        assigner = _build_assigner(formats_per_device)
        torch_b_all = _build_weight_tensors(
            num_experts,
            K,
            N,
            N_dram_per_device,
            sram_id_set,
            formats_per_device,
            num_devices,
            fmt_distribution,
            fmt_ratios,
        )
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
            subblock_n,
            Kt,
            tile_w,
        )
    else:
        assert len(formats_per_device) == 1, "tp_expert=False: all devices share same formats"
        assigner = _build_assigner(formats_per_device)
        torch_b_all = _build_weight_tensors_replicated(
            num_experts,
            K,
            N_dram_per_device,
            formats_per_device[0],
            num_devices,
            fmt_distribution,
            fmt_ratios,
        )
        dram_cts, dram_meta_tensors = _build_dram_experts_replicated(
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
            subblock_k,
            subblock_n,
            Kt,
            tile_w,
        )

    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)

    active_sram = [eid for eid in active_expert_ids if eid in sram_id_set]
    active_dram = [eid for eid in active_expert_ids if eid not in sram_id_set]
    num_active_experts = len(active_sram) + len(active_dram)
    num_sram_cores_active = len(sram_cores_list) if sram_expert_ids else 0
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
    # Expert parallel: each device processes 1 expert, output sized for 1.
    num_dram_for_output = 1 if not tp_expert else num_active_dram

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
        max(num_dram_for_output, 1),
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
    )

    sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets = (
        _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, Kt, sram_per_core_N, Kt)
        if has_sram
        else ({}, {}, None)
    )
    result = ExpertKernel.op(
        a_tensor,
        sram_cts,
        dram_cts,
        out_tensor,
        index_tensor,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        subblock_n=subblock_n,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_base_addr_tensors=sram_base_addr_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=Kt,
        sram_output_tensor=sram_out_tensor,
        dram_fuse_silu=dram_fuse_silu,
        tp_expert=tp_expert,
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
            tp_expert=tp_expert,
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
    subblock_n,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """Accumulation path: WIDTH_SHARDED SRAM, expert outputs summed in-place."""
    assert tp_expert, "Expert parallel (tp_expert=False) not supported in accum path"
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

    Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
        subblock_n,
    )

    assert len(formats_per_device) == num_devices
    torch.manual_seed(0)
    # Per-expert activations: each expert gets a distinct activation, laid out
    # in index-tensor order so the kernel can offset incrementally.
    num_active = len(active_expert_ids)
    torch_a_per_expert = {eid: torch.randn((M, K), dtype=torch.bfloat16) for eid in active_expert_ids}
    torch_a_all = torch.cat([torch_a_per_expert[eid] for eid in active_expert_ids], dim=-1)
    assigner = _build_assigner(formats_per_device)
    torch_b_all = _build_weight_tensors(
        num_experts,
        K,
        N,
        N_dram_per_device,
        sram_id_set,
        formats_per_device,
        num_devices,
        fmt_distribution,
        fmt_ratios,
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
        subblock_n,
        Kt,
        tile_w,
    )
    a_tensor = _build_activation_tensor(
        torch_a_all, mesh_device, compute_core_grid, num_cores, M, K * num_active, tile_w
    )
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)

    active_sram = [eid for eid in active_expert_ids if eid in sram_id_set]
    active_dram = [eid for eid in active_expert_ids if eid not in sram_id_set]
    num_active_experts = len(active_sram) + len(active_dram)
    num_sram_cores_active = len(sram_cores_list) if sram_expert_ids else 0
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

    sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets = (
        _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, Kt, sram_per_core_N, Kt)
        if has_sram
        else ({}, {}, None)
    )
    result = ExpertKernel.op(
        a_tensor,
        sram_cts,
        dram_cts,
        out_tensor,
        index_tensor,
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        subblock_n=subblock_n,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_base_addr_tensors=sram_base_addr_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        accum_experts=True,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=Kt,
        sram_output_tensor=sram_out_tensor,
        tp_expert=tp_expert,
    )
    if active_sram:
        _validate_sram_output_accum(
            sram_out_tensor,
            torch_a_per_expert,
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
            torch_a_per_expert,
            torch_b_all,
            active_dram,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            tile_w,
            tp_expert=tp_expert,
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
    subblock_n,
    cores_per_dram_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """K-sliced path: HEIGHT_SHARDED SRAM, separate output grids."""
    assert tp_expert, "Expert parallel (tp_expert=False) not supported in slice_k path"
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

    Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device = _compute_dram_matmul_params(
        K,
        N,
        tile_w,
        num_banks,
        num_dram_cores,
        num_dram_cores,
        cores_per_dram_bank,
        subblock_k,
        subblock_n,
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
        fmt_distribution,
        fmt_ratios,
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
        subblock_n,
        Kt,
        tile_w,
    )
    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)

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

    sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets = _build_sram_fmt_data(
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
        num_active_experts=num_active_experts,
        subblock_k=subblock_k,
        subblock_n=subblock_n,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        has_sram=has_sram,
        sram_core_grid=sram_core_grid,
        sram_fmt_tensors=sram_fmt_tensors,
        sram_base_addr_tensors=sram_base_addr_tensors,
        sram_k_offsets=sram_k_offsets,
        cores_per_dram_bank=cores_per_dram_bank,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=sram_k_per_core,
        sram_output_tensor=sram_out_tensor,
        dram_fuse_silu=dram_fuse_silu,
        tp_expert=tp_expert,
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
    subblock_n=None,
    cores_per_dram_bank=1,
    pcc_threshold=0.97,
    accum_experts=False,
    sram_cores_override=None,
    sram_k_parallel=1,
    sram_n_parallel=1,
    dram_fuse_silu=False,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
):
    """Dispatcher: delegate to the appropriate variant."""
    assert dram_expert_ids, "DRAM expert path is always required"
    assert len(dram_expert_ids) <= num_experts, "dram_expert_ids exceeds num_experts"
    if not tp_expert:
        assert not sram_expert_ids, "Expert parallel (tp_expert=False) only supports DRAM matmul"
        assert (
            not accum_experts
        ), "Expert parallel (tp_expert=False) processes 1 expert per device, accum not applicable"
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
            subblock_n,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
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
            subblock_n,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
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
            subblock_n,
            cores_per_dram_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
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


# ---------------------------------------------------------------------------
# MoE production shape tests — DRAM-only expert parallel (1 expert per device)
#   gate/up: [1, 7168] x [7168, 2048]
#   down:    [1, 2048] x [2048, 7168]
# ---------------------------------------------------------------------------


@pytest.mark.requires_grid_size((12, 10))
def test_moe_gate_proj_shape(device):
    """MoE gate-proj shape: 256 experts, 8 selected (1 per device), K=7168, N=2048, DRAM-only, fuse_silu."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=2048,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp2"]],
        dram_fuse_silu=True,
        tp_expert=False,
    )


@pytest.mark.requires_grid_size((12, 10))
def test_moe_up_proj_shape(device):
    """MoE up-proj shape: 8 experts, 8 selected (1 per device), K=7168, N=2048, DRAM-only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=2048,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp2"]],
        tp_expert=False,
    )


@pytest.mark.requires_grid_size((12, 10))
def test_moe_down_proj_shape(device):
    """MoE down-proj shape: 8 experts, 8 selected (1 per device), K=2048, N=7168, DRAM-only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=2048,
        N=7168,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp2"]],
        tp_expert=False,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_moe_gate_proj_shape_multi_device(bh_2d_mesh_device):
    """MoE gate-proj: 256 experts, 8 active (1 per device), K=7168, N=2048, DRAM-only, fuse_silu, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=2048,
        num_experts=256,
        sram_expert_ids=[],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[10, 42, 80, 120, 150, 190, 220, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        dram_fuse_silu=True,
        tp_expert=False,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_moe_up_proj_shape_multi_device(bh_2d_mesh_device):
    """MoE up-proj: 256 experts, 8 active (1 per device), K=7168, N=2048, DRAM-only, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=2048,
        num_experts=256,
        sram_expert_ids=[],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[10, 42, 80, 120, 150, 190, 220, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        tp_expert=False,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_moe_down_proj_shape_multi_device(bh_2d_mesh_device):
    """MoE down-proj: 256 experts, 8 active (1 per device), K=2048, N=7168, DRAM-only, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=2048,
        N=7168,
        num_experts=256,
        sram_expert_ids=[],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[10, 42, 80, 120, 150, 190, 220, 250],
        formats_per_device=[["bfp4", "bfp2"]],
        tp_expert=False,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_gate_grid_multi_device(bh_2d_mesh_device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=7168, N=256, gate-proj A-cores (64 irregular), 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    a_cores, _ = _build_ab_grids(mesh)
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=256,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
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
        sram_cores_override=a_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
        dram_fuse_silu=True,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_up_grid_multi_device(bh_2d_mesh_device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=7168, N=256, up-proj B-cores (64 irregular), 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _, b_cores = _build_ab_grids(mesh)
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=7168,
        N=256,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
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
        sram_cores_override=b_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_hybrid_expert_irregular_sram_down_grid_multi_device(bh_2d_mesh_device):
    """256 experts, 8 active (2 SRAM + 6 DRAM), K=256, N=7168, down-proj 112-core grid, 8 devices."""
    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")
    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    down_cores = _build_down_grid(mesh)
    _run_hybrid_expert_multi_device(
        mesh,
        M=1,
        K=256,
        N=7168,
        num_experts=256,
        sram_expert_ids=[112, 156],
        dram_expert_ids=list(range(256)),
        active_expert_ids=[96, 112, 156, 200, 212, 220, 240, 250],
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
        sram_cores_override=down_cores,
        accum_experts=True,
    )


# ---------------------------------------------------------------------------
# BSPM real-assignment tests — Step 2.3
# ---------------------------------------------------------------------------


def _run_dram_bspm(
    mesh_device,
    M,
    K,
    N,
    bspm_path,
    bspm_expert_configs,
    active_expert_ids,
    pcc_threshold=0.90,
    subblock_k=None,
    cores_per_dram_bank=1,
):
    """DRAM-only ExpertKernel with real BSPM assignments.

    bspm_expert_configs: list of (expert_idx, proj_idx) — one entry per registered expert slot.
    All slots use the same bspm_path; assignments are loaded per (expert_idx, proj_idx).
    Weights are random so no HF checkpoint is required; the purpose is to validate that the
    kernel correctly decompresses a real mixed-precision {bfp4, bfp2, zero} tile map.

    Requires BSPM_RESULTS_DIR env var and the corresponding .bspm file to be present.
    """
    import numpy as np

    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert
    from models.demos.deepseek_v3_b1.weights.transforms.moe import shuffle_dram_assignment

    tile_w = 32
    num_devices = mesh_device.get_num_devices()
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]
    num_experts = len(bspm_expert_configs)

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, 0, None, has_sram=False)
    dram_cores_list = grids["dram_cores_list"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(dram_cores_list)
    num_banks = num_dram_cores // cores_per_dram_bank
    num_cores = compute_core_grid.num_cores()

    Kt, dram_per_core_N, subblock_k_v, _subblock_n, N_dram_per_device = _compute_dram_matmul_params(
        K, N, tile_w, num_banks, num_dram_cores, num_dram_cores, cores_per_dram_bank, subblock_k
    )
    assert (
        N_dram_per_device == N
    ), f"N_dram_per_device ({N_dram_per_device}) != N ({N}): DRAM padding not supported in BSPM tests"

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
        ttnn.ShardSpec(
            dram_grid,
            [K, dram_per_core_N * tile_w * cores_per_dram_bank],
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)

    dram_cts = []
    torch_b_all = {}

    for slot_idx, (expert_idx, proj_idx) in enumerate(bspm_expert_configs):
        assignment = load_bspm_for_expert(
            str(bspm_path),
            expert_idx=expert_idx,
            proj_idx=proj_idx,
            tile_rows=K // tile_w,
            tile_cols=N // tile_w,
        )
        assignment_shuffled = shuffle_dram_assignment(assignment, num_banks)

        # Random per-device weights (tensor parallel: each device gets distinct N columns)
        per_dev_weights = []
        per_dev_shuffled = []
        for dev_idx in range(num_devices):
            torch.manual_seed(slot_idx * 100 + dev_idx + 7)
            b = torch.randn(K, N_dram_per_device).float()
            per_dev_weights.append(b)
            per_dev_shuffled.append(shuffle_tensor_tiles(b, tile_w, num_banks))
        torch_b_all[slot_idx] = per_dev_weights

        # 4D tensor (mesh_rows, mesh_cols, K, N_per_dev) + assignment stacked per device
        b_4d = torch.stack(per_dev_shuffled).reshape(mesh_rows, mesh_cols, K, N_dram_per_device)
        # tiles_h = num_devices * K // tile_w, tiles_w = N // tile_w
        assignment_4d = np.tile(assignment_shuffled, (num_devices, 1))

        ct = CompressedTensor(
            b_4d,
            assignment_4d,
            device=mesh_device,
            memory_config=dram_b_mem,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        logger.info(f"  BSPM slot {slot_idx} (expert={expert_idx}, proj={proj_idx}): {ct.tile_counts}")
        assert ct.tile_counts.get("bfp4", 0) > 0, f"Slot {slot_idx}: expected bfp4 tiles from 3.5 b/e assignment"
        low_p = ct.tile_counts.get("bfp2", 0) + ct.tile_counts.get("bfp0", 0)
        assert low_p > 0, f"Slot {slot_idx}: expected bfp2/bfp0 tiles from 3.5 b/e assignment"
        dram_cts.append(ct)

    dram_meta_flags = [1] * num_experts
    is_dram_flags = list(dram_meta_flags)
    num_subblocks_k = Kt // subblock_k_v

    dram_meta_tensors = create_dram_expert_tensors_multi_device(
        mesh_device,
        dram_cts,
        subblock_k_v,
        num_subblocks_k,
        dram_per_core_N,
        cores_per_dram_bank=cores_per_dram_bank,
        num_total_experts=num_experts,
        is_dram_flags=dram_meta_flags,
    )

    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)
    expert_selection_meta = _build_expert_selection_meta(mesh_device, a_tensor, is_dram_flags)

    num_active = len(active_expert_ids)
    out_tensor = _build_dram_output(
        mesh_device, M, dram_per_core_N, num_active, num_dram_cores, num_devices, dram_core_grid, tile_w
    )

    result = ExpertKernel.op(
        a_tensor,
        [],
        dram_cts,
        out_tensor,
        index_tensor,
        num_active_experts=num_active,
        subblock_k=subblock_k_v,
        dram_core_grid=dram_core_grid,
        dram_meta_tensors=dram_meta_tensors,
        dram_per_core_n=dram_per_core_N,
        expert_selection_meta=expert_selection_meta,
        has_sram=False,
        sram_core_grid=None,
        sram_fmt_tensors={},
        sram_k_offsets=None,
        cores_per_dram_bank=cores_per_dram_bank,
        sram_per_core_n=0,
        sram_k_per_core=Kt,
        sram_output_tensor=None,
        dram_fuse_silu=False,
        tp_expert=True,
    )

    _validate_dram_output(
        result,
        torch_a,
        torch_b_all,
        active_expert_ids,
        dram_per_core_N,
        num_dram_cores,
        pcc_threshold,
        False,
        tile_w,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_matmul_expert_bspm_single_device(device):
    """1 DRAM expert, real BSPM assignment at 3.5 b/e — gate_proj shape (K=7168, N=2048).

    Validates the DRAM streaming kernel correctly decompresses a real {bfp4, bfp2, zero}
    tile map from a binary .bspm file. Weights are random; only the assignment is real.
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    _run_dram_bspm(
        device,
        M=1,
        K=7168,
        N=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(0, 0)],  # expert 0, proj_idx=0 (gate_proj)
        active_expert_ids=[0],
        pcc_threshold=0.90,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_matmul_expert_bspm_multi_device(bh_2d_mesh_device):
    """2 DRAM experts, real BSPM assignments, 4×2 mesh — gate_proj shape (K=7168, N=2048).

    Slot 0 uses expert_idx=0 gate_proj assignment; slot 1 uses expert_idx=1 gate_proj.
    Both experts are active. Validates tensor-parallel DRAM streaming with real mixed-precision
    tile maps across all 8 devices simultaneously.
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")

    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_dram_bspm(
        mesh,
        M=1,
        K=7168,
        N=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(0, 0), (1, 0)],  # experts 0 and 1, both gate_proj
        active_expert_ids=[0, 1],
        pcc_threshold=0.90,
    )


@pytest.mark.skip_post_commit
@pytest.mark.requires_grid_size((12, 10))
def test_matmul_expert_bspm_sparse_activation(bh_2d_mesh_device):
    """4 DRAM experts registered, only 2 active — validates no output corruption in inactive slots.

    Expert slots 0–3 all use real BSPM gate_proj assignments (experts 0–3, proj_idx=0).
    The router selects only slots 0 and 2; slots 1 and 3 have CTs allocated but are never
    computed. Verifies that active-expert outputs pass PCC ≥ 0.90 and the kernel correctly
    skips inactive slots without corrupting the output tensor.
    Requires BSPM_RESULTS_DIR.
    """
    import os
    from pathlib import Path

    if bh_2d_mesh_device.shape[0] * bh_2d_mesh_device.shape[1] < 8:
        pytest.skip("Test requires at least 8 devices (4x2 mesh)")

    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if not bspm_dir:
        pytest.skip("BSPM_RESULTS_DIR not set")
    bspm_path = Path(bspm_dir) / "deepseek-r1-0528" / "layer_4" / "precision_eval" / "precision_map_B_3.5.bspm"
    if not bspm_path.exists():
        pytest.skip(f"BSPM file not found: {bspm_path}")

    mesh = bh_2d_mesh_device.create_submesh(ttnn.MeshShape(4, 2))
    _run_dram_bspm(
        mesh,
        M=1,
        K=7168,
        N=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(0, 0), (1, 0), (2, 0), (3, 0)],  # 4 registered experts
        active_expert_ids=[0, 2],  # sparse: only experts 0 and 2 selected this step
        pcc_threshold=0.90,
    )


@pytest.mark.skip_post_commit
def test_benchmark_down_proj(device):
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=256,
        N=7168,
        num_experts=1,
        sram_expert_ids=[],
        dram_expert_ids=list(range(1)),
        active_expert_ids=[0],
        formats_per_device=[
            ["bfp4", "bfp0"],
        ],
        subblock_n=4,
        accum_experts=True,
    )


@pytest.mark.skip_post_commit
def test_benchmark_up_proj(device):
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=1,
        sram_expert_ids=[],
        dram_expert_ids=list(range(1)),
        active_expert_ids=[0],
        formats_per_device=[
            ["bfp4", "bfp0"],
        ],
        subblock_n=1,
        accum_experts=False,
    )


# ---------------------------------------------------------------------------
# Benchmark tests — DRAM-only, 1 expert per device, production MoE shapes
# ---------------------------------------------------------------------------


def test_benchmark_gate_proj(device):
    """MoE gate-proj shape: 256 experts, 8 selected (1 per device), K=7168, N=256, DRAM-only, fuse_silu."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4"]],
        dram_fuse_silu=False,  # disable to reduce runtime for benchmarking
        subblock_n=1,
        fmt_distribution="uniform",
        # fmt_ratios={"bfp4": 3, "bfp0": 1},
    )


def test_benchmark_up_proj(device):
    """MoE up-proj shape: 8 experts, 8 selected (1 per device), K=7168, N=256, DRAM-only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp0"]],
        subblock_n=1,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 3, "bfp0": 1},
    )


def test_benchmark_down_proj(device):
    """MoE down-proj shape: 8 experts, 8 selected (1 per device), K=256, N=7168, DRAM-only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=256,
        N=7168,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp0"]],
        accum_experts=True,
        subblock_n=4,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 3, "bfp0": 1},
    )
