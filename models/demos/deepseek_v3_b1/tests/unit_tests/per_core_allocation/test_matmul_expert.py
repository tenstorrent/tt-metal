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
    _pad_to_face_r_dim,
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


def _software_quantize_per_tile(b_np, assignment, tile_w=32):
    """Apply per-tile BFP quantize-dequantize matching CompressedTensor's hardware behavior.

    Replicates what the hardware sees after DRAM decompression: each 32x32 tile is
    quantize-dequantized according to its precision code (0=bfp8, 1=bfp4, 2=bfp2, 3=bfp0).
    Result is the software reference for "this is what the hardware should compute against".
    """
    import numpy as np

    from models.demos.deepseek_v3_b1.compressed_tensor.tile_utils import quantize_dequantize_bfp

    code_to_mant_bits = {0: 7, 1: 3, 2: 1}  # bfp8, bfp4, bfp2; bfp0 (code 3) zeros the tile
    K, N = b_np.shape
    tiles_h, tiles_w = K // tile_w, N // tile_w
    assert assignment.shape == (tiles_h, tiles_w), f"assignment shape {assignment.shape} != ({tiles_h}, {tiles_w})"
    out = np.asarray(b_np, dtype=np.float32).copy()
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            code = int(assignment[tr, tc])
            r0, c0 = tr * tile_w, tc * tile_w
            tile = out[r0 : r0 + tile_w, c0 : c0 + tile_w]
            if code == 3:
                tile[:] = 0.0
            elif code in code_to_mant_bits:
                tile[:] = quantize_dequantize_bfp(tile, code_to_mant_bits[code])
            else:
                raise ValueError(f"unknown precision code {code} at tile ({tr}, {tc})")
    return out


def _compute_pcc(golden, calculated):
    """Direct Pearson correlation between two tensors (any shape, broadcast to flat).

    Returns a float in [-1, 1]. Used when we need the numerical PCC value rather than
    a pass/fail boolean from comp_pcc.
    """
    a = golden.float().flatten()
    b = calculated.float().flatten()
    a_c = a - a.mean()
    b_c = b - b.mean()
    denom = a_c.norm() * b_c.norm()
    if denom < 1e-12:
        return 1.0 if (a_c.norm() < 1e-12 and b_c.norm() < 1e-12) else 0.0
    return float((a_c * b_c).sum() / denom)


def _shuffle_assignment_blocked(assignment, num_banks, subblock_k=None, subblock_n=1):
    """Apply the same per-shard blocked permutation as shuffle_tensor_tiles to a 2D
    assignment-code array. For (subblock_k=K_tiles, subblock_n=1) reduces to the simple
    column-major-within-shard shuffle that moe.shuffle_dram_assignment applies. For
    smaller subblock_k (K-parallel mode) groups tiles into subblock_k blocks contiguously
    so each K-slice's tiles land at the start of the shard.
    """
    import numpy as np

    tiles_h, tiles_w = assignment.shape
    K_tiles = tiles_h
    if tiles_w % num_banks != 0:
        raise ValueError(f"tiles_w ({tiles_w}) must be divisible by num_banks ({num_banks})")
    per_N_tiles = tiles_w // num_banks
    num_tiles_per_shard = K_tiles * per_N_tiles

    if subblock_k is None:
        subblock_k = K_tiles
    assert K_tiles % subblock_k == 0, f"K_tiles ({K_tiles}) must be divisible by subblock_k ({subblock_k})"
    assert per_N_tiles % subblock_n == 0, f"per_N_tiles ({per_N_tiles}) must be divisible by subblock_n ({subblock_n})"

    num_n_groups = per_N_tiles // subblock_n
    block_size = subblock_k * subblock_n

    i = np.arange(num_tiles_per_shard)
    block_idx = i // block_size
    pos_in_block = i % block_size
    local_k = pos_in_block // subblock_n
    local_n = pos_in_block % subblock_n
    n_group = block_idx % num_n_groups
    k_sub = block_idx // num_n_groups
    global_k = k_sub * subblock_k + local_k
    global_n = n_group * subblock_n + local_n
    source_idx = global_k * per_N_tiles + global_n

    result = np.empty_like(assignment)
    for b in range(num_banks):
        shard = assignment[:, b * per_N_tiles : (b + 1) * per_N_tiles].reshape(-1)
        result[:, b * per_N_tiles : (b + 1) * per_N_tiles] = shard[source_idx].reshape(K_tiles, per_N_tiles)
    return result


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
            raw_weights = [float(weights.get(f, 1)) for f in formats]
        else:
            raw_weights = [1.0] * len(formats)
        total_w = sum(raw_weights)
        # Largest-remainder apportionment → exact share counts that sum to tiles_h.
        quotas = [w * tiles_h / total_w for w in raw_weights]
        floor_shares = [int(q) for q in quotas]
        remainders = [q - fs for q, fs in zip(quotas, floor_shares)]
        leftover = tiles_h - sum(floor_shares)
        order = sorted(range(len(raw_weights)), key=lambda i: -remainders[i])
        shares = list(floor_shares)
        for i in order[:leftover]:
            shares[i] += 1
        # Interleaved placement (greedy largest-deficit): at each row, pick the
        # format whose expected running count (share × row / tiles_h) is most
        # behind its actual placed count. Spreads each format evenly across K
        # rather than concatenating blocks — avoids a long contiguous bfp0/bfp2
        # tail that exposes numerical edge cases in the reducer's silu path.
        col = torch.empty(tiles_h, dtype=torch.long)
        placed = [0.0] * len(shares)
        for row in range(tiles_h):
            best_f = 0
            best_deficit = -float("inf")
            for f, s in enumerate(shares):
                if placed[f] >= s:
                    continue
                deficit = s * (row + 1) / tiles_h - placed[f]
                if deficit > best_deficit:
                    best_deficit = deficit
                    best_f = f
            col[row] = best_f
            placed[best_f] += 1
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
    num_active_experts,
    num_sram_cores,
    num_devices,
    sram_core_grid,
    tile_w,
):
    """SRAM output tensor on sram_core_grid.

    Sized for num_active_experts (worst-case static layout) — the kernel decodes
    the SRAM/DRAM split from the index tensor at runtime, so all active experts
    could hypothetically route to SRAM. Pass 1 for accum-mode callers.
    """
    sram_out_per_core = sram_per_core_N * tile_w * num_active_experts
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
    num_active_experts,
    num_dram_cores_active,
    num_devices,
    dram_core_grid,
    tile_w,
    dram_fuse_silu=False,
):
    """DRAM output tensor on dram_core_grid.

    Sized for num_active_experts, not num_active_dram — the kernel decodes the
    SRAM/DRAM split from the index tensor at runtime, so the static CB layout
    must cover the case where every active expert could route to DRAM.

    When dram_fuse_silu=True, cb_out_silu aliases out_tensor with a fat tile of
    height silu_tile_h = pad_to_face_r_dim(num_active_experts * dram_per_core_N * M).
    Per-core width must be a multiple of silu_tile_h * tile_w, so we pad the
    per-core tile count up to silu_tile_h / M.
    """
    per_core_tiles = num_active_experts * dram_per_core_N
    if dram_fuse_silu:
        silu_tile_h = _pad_to_face_r_dim(per_core_tiles * M)
        assert silu_tile_h % M == 0, f"silu_tile_h ({silu_tile_h}) must be divisible by M ({M})"
        per_core_tiles = silu_tile_h // M
    dram_out_per_core = per_core_tiles * tile_w
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
    num_active_experts,
    pcc_threshold,
    tile_w,
):
    """Validate SRAM expert output from separate sram_out_tensor (no K-reduction).

    Per-core shard is sized for num_active_experts (worst-case static layout).
    The kernel packs SRAM outputs consecutively into [0, num_active_sram); trailing
    slots are garbage skipped by the stride-by-num_active_experts walk below.
    """
    sram_core_width = sram_per_core_N * tile_w
    sram_out_shard = sram_core_width * num_active_experts

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
    num_active_experts,
    dram_per_core_N,
    num_dram_cores_active,
    pcc_threshold,
    dram_fuse_silu,
    tile_w,
    M=1,
    tp_expert=True,
    cores_per_dram_bank=1,
    k_parallel_per_bank=1,
    pccs_out=None,
):
    """Validate per-expert DRAM output from dram_core_grid output tensor.

    When pccs_out is a list, append (dev_idx, expert_id, label, pcc_value) tuples to it
    and skip the per-expert assertion. Caller can then aggregate / compare PCCs across
    configurations. Default (None) preserves existing assertion-mode behavior.

    For k_parallel_per_bank > 1 (cross-core K-reduction):
      - Reducer cores (k_slice_idx == k_parallel-1) hold the final full-K result
        (silu-fused when enabled).
      - Sender cores (k_slice_idx < k_parallel-1) hold their K-slice partial
        (pre-reduction, no silu). We PCC-check these separately to isolate
        per-core matmul correctness from the K-reduction path.
    """
    assert cores_per_dram_bank % k_parallel_per_bank == 0
    n_parallel_per_bank = cores_per_dram_bank // k_parallel_per_bank
    num_banks = num_dram_cores_active // cores_per_dram_bank
    dram_core_width = dram_per_core_N * tile_w

    # Mirror _build_dram_output's silu padding so the stride matches the actual shard.
    per_core_slots = num_active_experts
    if dram_fuse_silu:
        silu_tile_h = _pad_to_face_r_dim(num_active_experts * dram_per_core_N * M)
        per_core_slots = silu_tile_h // M // dram_per_core_N

    # K per slice — each K-slice core sees only a contiguous chunk of K.
    K_total = torch_a.shape[-1]
    assert K_total % k_parallel_per_bank == 0, f"K ({K_total}) must be divisible by k_parallel ({k_parallel_per_bank})"
    K_per_slice = K_total // k_parallel_per_bank

    for dev_idx, out_dev in enumerate(ttnn.get_device_tensors(result)):
        output_dev = ttnn.to_torch(out_dev)
        dev_active_dram = active_dram if tp_expert else [active_dram[dev_idx]]

        # Walk every k_slice_idx (senders + reducer). Per-core shard is sized
        # for per_core_slots; the kernel packs DRAM outputs consecutively into
        # [0, num_active_dram); trailing slots are garbage we skip.
        for k_slice_idx in range(k_parallel_per_bank):
            is_reducer = k_slice_idx == k_parallel_per_bank - 1
            k_start = k_slice_idx * K_per_slice
            k_end = k_start + K_per_slice
            label = "reducer" if is_reducer else f"sender k_slice={k_slice_idx}"

            for exp_offset, eidx in enumerate(dev_active_dram):
                slices = []
                for bank_idx in range(num_banks):
                    for n_slice_in_bank in range(n_parallel_per_bank):
                        offset = n_slice_in_bank * k_parallel_per_bank + k_slice_idx
                        core_flat_idx = bank_idx * cores_per_dram_bank + offset
                        start = core_flat_idx * dram_core_width * per_core_slots + exp_offset * dram_core_width
                        slices.append(output_dev[..., start : start + dram_core_width])
                expert_output = torch.cat(slices, dim=-1)
                if is_reducer:
                    # Reducer holds the full A @ B (silu if fused).
                    mm_result = torch_a.float() @ torch_b_all[eidx][dev_idx].float()
                    if dram_fuse_silu:
                        mm_result = torch.nn.functional.silu(mm_result)
                else:
                    # Sender holds its K-slice partial — raw, no silu.
                    mm_result = torch_a[:, k_start:k_end].float() @ torch_b_all[eidx][dev_idx][k_start:k_end, :].float()
                torch_expected = mm_result.bfloat16()
                if pccs_out is not None:
                    pcc_val = _compute_pcc(torch_expected, expert_output)
                    pccs_out.append((dev_idx, eidx, label, pcc_val))
                    logger.info(f"Device {dev_idx} expert {eidx} (DRAM {label}) PCC: {pcc_val:.6f} (collected)")
                else:
                    passing, msg = comp_pcc(torch_expected, expert_output, pcc_threshold)
                    logger.info(f"Device {dev_idx} expert {eidx} (DRAM {label}) PCC: {msg}")
                    assert passing, f"Device {dev_idx} expert {eidx} (DRAM {label}) failed: {msg}"


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
    num_active_experts,
    sram_cores_list,
    sram_k_parallel,
    sram_n_parallel,
    M,
    N_sram_per_device,
    pcc_threshold,
    tile_w,
):
    """K-sliced SRAM verification — reduce K-partials from separate sram_out_tensor.

    Per-core shard is sized for num_active_experts (worst-case static layout),
    same rationale as _validate_sram_output.
    """
    for dev_idx, sram_dev in enumerate(ttnn.get_device_tensors(sram_out_tensor)):
        sram_output_dev = ttnn.to_torch(sram_dev)
        sram_out_shard = sram_per_core_N * tile_w * num_active_experts
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
    k_parallel_per_bank=1,
):
    """Build DRAM CompressedTensors and dram_meta_tensors for the kernel.

    For K-split: shuffle the tensor with subblock_k=Kt/k_parallel so each K-slice's
    tiles are contiguous at the start of the shard. n_parallel cores share the N-slice,
    each K-parallel core reads a contiguous K-slice region.
    """
    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        ]
    )
    assert cores_per_dram_bank % k_parallel_per_bank == 0
    n_parallel_per_bank = cores_per_dram_bank // k_parallel_per_bank
    # Shard width = per_core_N × tile_w × n_parallel (cores sharing a bank's N-slice).
    # For K-split (n_parallel=1), shard width = per_core_N × tile_w (one N-slice per bank).
    shard_width = dram_per_core_N * tile_w * n_parallel_per_bank
    dram_b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [K, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Shuffle with subblock_k=Kt/k_parallel lays out each K-slice's tiles contiguously.
    # For k_parallel=1 (no K-split), subblock_k defaults to Kt (one big block per N col).
    shuffle_subblock_k = Kt // k_parallel_per_bank if k_parallel_per_bank > 1 else None
    dram_cts = []
    for eidx in dram_expert_ids:
        slices_shuffled = [
            shuffle_tensor_tiles(b, tile_w, num_banks, subblock_k=shuffle_subblock_k, subblock_n=subblock_n)
            for b in torch_b_all[eidx]
        ]
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
        n_parallel_per_bank=n_parallel_per_bank,
        num_total_experts=num_experts,
        is_dram_flags=dram_meta_flags,
        subblock_n=subblock_n,
        k_parallel_per_bank=k_parallel_per_bank,
    )
    return dram_cts, dram_meta_tensors


def _compute_dram_matmul_params(
    K,
    N,
    tile_w,
    num_banks,
    num_dram_cores,
    num_dram_cores_active,
    cores_per_dram_bank,
    num_subblocks_k,
    num_subblocks_n=None,
    k_parallel_per_bank=1,
):
    """Compute DRAM per-core tiling.

    Test callers pass num_subblocks_k / num_subblocks_n (dimensionless split counts)
    and this helper derives subblock_k = Kt // num_subblocks_k, subblock_n =
    dram_per_core_N // num_subblocks_n. Defaults: num_subblocks_k = 4 when Kt>8
    else 1; num_subblocks_n = dram_per_core_N (→ subblock_n=1).

    For K-split (k_parallel_per_bank > 1): cores within a bank share the same
    N-slice and split K. N padding/sizing uses n_parallel_per_bank, not cores_per_dram_bank.
    """
    Kt = K // tile_w
    assert cores_per_dram_bank % k_parallel_per_bank == 0
    n_parallel_per_bank = cores_per_dram_bank // k_parallel_per_bank
    n_dram_padded = _pad_to_dram_banks(N, tile_w, tile_w * num_banks * n_parallel_per_bank)
    # Per-core N tiles = N / (num_banks × n_parallel). For K-split, n_parallel=1, so
    # all cores in a bank share the bank's full N-slice.
    dram_per_core_N = n_dram_padded // (num_banks * n_parallel_per_bank) // tile_w

    if num_subblocks_k is None:
        num_subblocks_k = 4 if Kt > 8 else 1
    assert Kt % num_subblocks_k == 0, f"Kt ({Kt}) must be divisible by num_subblocks_k ({num_subblocks_k})"
    subblock_k = Kt // num_subblocks_k
    # Kernel subblock_k must be even when > 1.
    if subblock_k % 2 != 0 and subblock_k > 1:
        subblock_k = max(2, subblock_k - 1)
        assert Kt % subblock_k == 0
        num_subblocks_k = Kt // subblock_k

    if num_subblocks_n is None:
        num_subblocks_n = dram_per_core_N  # subblock_n=1
    assert (
        dram_per_core_N % num_subblocks_n == 0
    ), f"dram_per_core_N ({dram_per_core_N}) must be divisible by num_subblocks_n ({num_subblocks_n})"
    subblock_n = dram_per_core_N // num_subblocks_n

    assert (
        num_subblocks_k % k_parallel_per_bank == 0
    ), f"num_subblocks_k ({num_subblocks_k}) must be divisible by k_parallel_per_bank ({k_parallel_per_bank})"
    N_dram_per_device = dram_per_core_N * tile_w * num_banks * n_parallel_per_bank
    logger.info(
        f"DRAM matmul params: Kt={Kt}, dram_per_core_N={dram_per_core_N}, "
        f"num_subblocks_k={num_subblocks_k}, num_subblocks_n={num_subblocks_n}, "
        f"subblock_k={subblock_k}, subblock_n={subblock_n}, N_dram_per_device={N_dram_per_device}, "
        f"k_parallel_per_bank={k_parallel_per_bank}, n_parallel_per_bank={n_parallel_per_bank}"
    )
    return Kt, dram_per_core_N, subblock_k, subblock_n, N_dram_per_device


def _build_assigner(formats_per_device):
    """Create CompressedTensorAssigner from per-device format lists."""
    all_formats = list({fmt for fmts in formats_per_device for fmt in fmts})
    # op.py sizes DRAM CB slots for bfp4 — bfp8 tiles would overflow the slot.
    assert "bfp8" not in all_formats, "DRAM expert CBs are sized for bfp4 only; remove bfp8 from formats_per_device"
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
        n_parallel_per_bank=cores_per_dram_bank,
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
    num_subblocks_k,
    num_subblocks_n,
    n_parallel_per_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
    k_parallel_per_bank=1,
    num_loop_iters=1,
    n_program_invocations=1,
):
    """Standard path: WIDTH_SHARDED SRAM, per-expert output slices on compute_core_grid.

    n_program_invocations > 1 calls ExpertKernel.op repeatedly with identical inputs and
    asserts every invocation's output is bitwise identical to the first. Probes for races
    and inter-call state leakage (e.g. the documented K-reduction race at
    matmul_expert_compressed_dram.hpp:470-474). Output tensors are rebuilt per invocation;
    DRAM weight tensors, fmt tables, and the activation/index tensors are reused.
    """
    cores_per_dram_bank = n_parallel_per_bank * k_parallel_per_bank
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
        num_subblocks_k,
        num_subblocks_n,
        k_parallel_per_bank=k_parallel_per_bank,
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
            k_parallel_per_bank=k_parallel_per_bank,
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
    # Expert parallel: each device processes 1 expert. Otherwise size the DRAM
    # output for ALL active experts — the runtime split (sram vs dram) comes
    # from the index tensor, so the static layout must cover the worst case.
    num_dram_for_output = 1 if not tp_expert else num_active_experts

    sram_out_tensor = (
        _build_sram_output(
            mesh_device,
            M,
            sram_per_core_N,
            num_active_experts,
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
        num_dram_for_output,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
        dram_fuse_silu=dram_fuse_silu,
    )

    sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets = (
        _build_sram_fmt_data(sram_cts, mesh_device, sram_core_grid, Kt, sram_per_core_N, Kt)
        if has_sram
        else ({}, {}, None)
    )

    reference_outputs = None
    mismatches = []
    extra_mismatch_count = 0
    MAX_STORED_MISMATCHES = 1000
    progress_every = max(1, n_program_invocations // 10)

    for invocation in range(n_program_invocations):
        if invocation > 0:
            out_tensor = _build_dram_output(
                mesh_device,
                M,
                dram_per_core_N,
                num_dram_for_output,
                num_dram_cores,
                num_devices,
                dram_core_grid,
                tile_w,
                dram_fuse_silu=dram_fuse_silu,
            )
            if has_sram:
                sram_out_tensor = _build_sram_output(
                    mesh_device,
                    M,
                    sram_per_core_N,
                    num_active_experts,
                    num_sram_cores_active,
                    num_devices,
                    sram_core_grid,
                    tile_w,
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
            n_parallel_per_bank=n_parallel_per_bank,
            k_parallel_per_bank=k_parallel_per_bank,
            sram_per_core_n=sram_per_core_N,
            sram_k_per_core=Kt,
            sram_output_tensor=sram_out_tensor,
            dram_fuse_silu=dram_fuse_silu,
            tp_expert=tp_expert,
            num_loop_iters=num_loop_iters,
        )

        if n_program_invocations > 1:
            current = [ttnn.to_torch(d).clone() for d in ttnn.get_device_tensors(result)]
            if reference_outputs is None:
                reference_outputs = current
            else:
                for dev_idx, (ref_dev, cur_dev) in enumerate(zip(reference_outputs, current)):
                    if not torch.equal(ref_dev, cur_dev):
                        if len(mismatches) < MAX_STORED_MISMATCHES:
                            n_diff = int((ref_dev != cur_dev).sum().item())
                            delta = (ref_dev.float() - cur_dev.float()).abs()
                            mismatches.append(
                                (
                                    invocation,
                                    dev_idx,
                                    n_diff,
                                    float(delta.max().item()),
                                    float(delta.mean().item()),
                                )
                            )
                        else:
                            extra_mismatch_count += 1

            if (invocation + 1) % progress_every == 0:
                total = len(mismatches) + extra_mismatch_count
                logger.info(
                    f"  multi-invocation: {invocation + 1}/{n_program_invocations} done, {total} divergent so far"
                )

    if n_program_invocations > 1:
        total_mismatches = len(mismatches) + extra_mismatch_count
        if total_mismatches:
            logger.error(
                f"NON-DETERMINISTIC: {total_mismatches} divergent (invocation, device) pairs "
                f"across {n_program_invocations} invocations"
            )
            for inv, dev, n_diff, max_d, mean_d in mismatches[:10]:
                logger.error(
                    f"  invocation {inv} device {dev}: {n_diff} differing elements, "
                    f"max|delta|={max_d:.4g}, mean|delta|={mean_d:.4g}"
                )
        assert not total_mismatches, (
            f"ExpertKernel non-deterministic across {n_program_invocations} program invocations: "
            f"{total_mismatches} divergent (invocation, device) pairs "
            f"(first at invocation {mismatches[0][0]} device {mismatches[0][1]})"
        )

    if active_sram:
        _validate_sram_output(
            sram_out_tensor,
            torch_a,
            torch_b_all,
            active_sram,
            sram_per_core_N,
            num_sram_cores_active,
            num_active_experts,
            pcc_threshold,
            tile_w,
        )
    if active_dram:
        _validate_dram_output(
            result,
            torch_a,
            torch_b_all,
            active_dram,
            num_active_experts,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            dram_fuse_silu,
            tile_w,
            M=M,
            tp_expert=tp_expert,
            cores_per_dram_bank=cores_per_dram_bank,
            k_parallel_per_bank=k_parallel_per_bank,
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
    num_subblocks_k,
    num_subblocks_n,
    n_parallel_per_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
    num_loop_iters=1,
):
    """Accumulation path: WIDTH_SHARDED SRAM, expert outputs summed in-place."""
    cores_per_dram_bank = n_parallel_per_bank
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
        num_subblocks_k,
        num_subblocks_n,
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
        n_parallel_per_bank=n_parallel_per_bank,
        accum_experts=True,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=Kt,
        sram_output_tensor=sram_out_tensor,
        tp_expert=tp_expert,
        num_loop_iters=num_loop_iters,
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
    num_subblocks_k,
    num_subblocks_n,
    n_parallel_per_bank,
    sram_cores_override,
    sram_k_parallel,
    sram_n_parallel,
    pcc_threshold,
    dram_fuse_silu,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
    k_parallel_per_bank=1,
    num_loop_iters=1,
):
    """K-sliced path: HEIGHT_SHARDED SRAM, separate output grids."""
    cores_per_dram_bank = n_parallel_per_bank * k_parallel_per_bank
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
        num_subblocks_k,
        num_subblocks_n,
        k_parallel_per_bank=k_parallel_per_bank,
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
        k_parallel_per_bank=k_parallel_per_bank,
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

    sram_cts = (
        _build_sram_cts_slice_k(
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
        if has_sram
        else []
    )

    sram_out_tensor = (
        _build_sram_output(
            mesh_device,
            M,
            sram_per_core_N,
            num_active_experts,
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
        num_active_experts,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
        dram_fuse_silu=dram_fuse_silu,
    )

    sram_fmt_tensors, sram_base_addr_tensors, sram_k_offsets = (
        _build_sram_fmt_data(
            sram_cts,
            mesh_device,
            sram_core_grid,
            sram_k_per_core,
            sram_per_core_N,
            Kt,
        )
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
        n_parallel_per_bank=n_parallel_per_bank,
        k_parallel_per_bank=k_parallel_per_bank,
        sram_per_core_n=sram_per_core_N,
        sram_k_per_core=sram_k_per_core,
        sram_output_tensor=sram_out_tensor,
        dram_fuse_silu=dram_fuse_silu,
        tp_expert=tp_expert,
        num_loop_iters=num_loop_iters,
    )
    if active_sram:
        _validate_sram_output_slice_k(
            sram_out_tensor,
            torch_a,
            torch_b_all,
            active_sram,
            sram_per_core_N,
            num_active_experts,
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
            num_active_experts,
            dram_per_core_N,
            num_dram_cores,
            pcc_threshold,
            dram_fuse_silu,
            tile_w,
            M=M,
            cores_per_dram_bank=cores_per_dram_bank,
            k_parallel_per_bank=k_parallel_per_bank,
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
    num_subblocks_k=None,
    num_subblocks_n=None,
    n_parallel_per_bank=1,
    pcc_threshold=0.97,
    accum_experts=False,
    sram_cores_override=None,
    sram_k_parallel=1,
    sram_n_parallel=1,
    dram_fuse_silu=False,
    tp_expert=True,
    fmt_distribution="random",
    fmt_ratios=None,
    k_parallel_per_bank=1,
    num_loop_iters=1,
    n_program_invocations=1,
):
    """Dispatcher: delegate to the appropriate variant.

    num_subblocks_k / num_subblocks_n are DRAM-only knobs (dimensionless split
    counts along K / per-core-N). subblock_k = Kt // num_subblocks_k and
    subblock_n = dram_per_core_N // num_subblocks_n are derived inside
    _compute_dram_matmul_params.
    """
    assert dram_expert_ids, "DRAM expert path is always required"
    assert len(dram_expert_ids) <= num_experts, "dram_expert_ids exceeds num_experts"
    if not tp_expert:
        assert not sram_expert_ids, "Expert parallel (tp_expert=False) only supports DRAM matmul"
        assert (
            not accum_experts
        ), "Expert parallel (tp_expert=False) processes 1 expert per device, accum not applicable"
    slice_k = sram_k_parallel > 1
    if n_program_invocations > 1:
        assert not slice_k and not accum_experts, (
            "n_program_invocations > 1 is only supported on the standard path "
            "(no SRAM K-parallel, no expert accumulation)"
        )
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
            num_subblocks_k,
            num_subblocks_n,
            n_parallel_per_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
            k_parallel_per_bank=k_parallel_per_bank,
            num_loop_iters=num_loop_iters,
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
            num_subblocks_k,
            num_subblocks_n,
            n_parallel_per_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
            num_loop_iters=num_loop_iters,
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
            num_subblocks_k,
            num_subblocks_n,
            n_parallel_per_bank,
            sram_cores_override,
            sram_k_parallel,
            sram_n_parallel,
            pcc_threshold,
            dram_fuse_silu,
            tp_expert,
            fmt_distribution,
            fmt_ratios,
            k_parallel_per_bank=k_parallel_per_bank,
            num_loop_iters=num_loop_iters,
            n_program_invocations=n_program_invocations,
        )


# ---------------------------------------------------------------------------
# Hybrid expert test variants — each exercises different SRAM/DRAM
# configurations, formats, and router selections.
# ---------------------------------------------------------------------------


def test_hybrid_expert_1sram_1dram(device):
    """1 SRAM expert + 1 DRAM expert, bfp4 only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=128,
        N=256,
        num_experts=2,
        sram_expert_ids=[0],
        dram_expert_ids=[1],
        active_expert_ids=[0, 1],
        formats_per_device=[["bfp4"]],
        sram_n_parallel=4,
        num_subblocks_k=2,
    )


def test_hybrid_expert_0sram_2dram(device):
    """0 SRAM experts + 2 DRAM experts, bfp4 only."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=128,
        N=256,
        num_experts=2,
        sram_expert_ids=[],
        dram_expert_ids=[0, 1],
        active_expert_ids=[0, 1],
        formats_per_device=[["bfp4"]],
        sram_n_parallel=4,
        num_subblocks_k=2,
    )


def test_hybrid_expert_2sram_2dram_mixed(device):
    """4 experts, 2 SRAM + 2 DRAM, bfp4+bfp2, router selects all 4."""
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=4,
        sram_expert_ids=[0, 2],
        dram_expert_ids=[1, 3],
        active_expert_ids=[0, 1, 2, 3],
        formats_per_device=[["bfp4", "bfp2"]],
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
        formats_per_device=[["bfp4", "bfp2"]],
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
        formats_per_device=[["bfp4", "bfp2"], ["bfp4", "bfp2"]] * 4,
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
        n_parallel_per_bank=2,
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
        num_subblocks_k=1,
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
        num_subblocks_k=1,
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
        formats_per_device=[["bfp4", "bfp0"]],
        sram_cores_override=a_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
        dram_fuse_silu=True,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 3, "bfp0": 1},
        num_loop_iters=100,
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
        formats_per_device=[["bfp4", "bfp0"]],
        sram_cores_override=b_cores,
        sram_k_parallel=8,
        sram_n_parallel=8,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 3, "bfp0": 1},
        num_loop_iters=100,
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
        formats_per_device=[["bfp4", "bfp0"]],
        sram_cores_override=down_cores,
        accum_experts=True,
        num_subblocks_n=2,
        n_parallel_per_bank=2,
        k_parallel_per_bank=1,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 3, "bfp0": 1},
        num_loop_iters=100,
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
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        num_loop_iters=100,
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
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        num_loop_iters=100,
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
        num_subblocks_n=2,
        n_parallel_per_bank=2,
        k_parallel_per_bank=1,
        fmt_distribution="uniform",
        num_loop_iters=100,
    )


# ---------------------------------------------------------------------------
# Benchmark tests — DRAM-only, 8 experts, production MoE shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formats_per_device, fmt_ratios",
    [
        (["bfp4", "bfp0"], {"bfp4": 77.8, "bfp0": 22.2}),
        (["bfp4", "bfp2", "bfp0"], {"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3}),
    ],
    ids=["bfp4_bfp0", "bfp4_bfp2_bfp0"],
)
def test_benchmark_gate_proj(device, formats_per_device, fmt_ratios):
    """MoE gate-proj shape: 256 experts, 8 selected (1 per device), K=7168, N=256, DRAM-only, fuse_silu.

    cores_per_dram_bank=2, k_parallel_per_bank=2 → 2 cores per bank split K (n_parallel=1).
    Each K-slice core produces a partial matmul; partial-K PCC validator checks each core.
    """
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[formats_per_device],
        dram_fuse_silu=True,
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios=fmt_ratios,
        num_loop_iters=100,
    )


@pytest.mark.parametrize(
    "formats_per_device, fmt_ratios",
    [
        (["bfp4", "bfp0"], {"bfp4": 77.8, "bfp0": 22.2}),
        (["bfp4", "bfp2", "bfp0"], {"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3}),
    ],
    ids=["bfp4_bfp0", "bfp4_bfp2_bfp0"],
)
def test_benchmark_up_proj(device, formats_per_device, fmt_ratios):
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
        formats_per_device=[formats_per_device],
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios=fmt_ratios,
        num_loop_iters=100,
    )


@pytest.mark.parametrize(
    "formats_per_device, fmt_ratios",
    [
        (["bfp4", "bfp0"], {"bfp4": 77.8, "bfp0": 22.2}),
        (["bfp4", "bfp2", "bfp0"], {"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3}),
    ],
    ids=["bfp4_bfp0", "bfp4_bfp2_bfp0"],
)
def test_benchmark_down_proj(device, formats_per_device, fmt_ratios):
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
        formats_per_device=[formats_per_device],
        accum_experts=True,
        num_subblocks_k=1,
        num_subblocks_n=2,
        n_parallel_per_bank=2,
        k_parallel_per_bank=1,
        fmt_distribution="uniform",
        fmt_ratios=fmt_ratios,
        num_loop_iters=100,
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


# ---------------------------------------------------------------------------
# Race / determinism probes — multi-invocation under K-parallel mixed precision.
#
# These tests target the inter-call race documented at
# matmul_expert_compressed_dram.hpp:470-474 (sender can issue next iteration's
# writes before reducer finishes, overwriting L1). The race lives in the
# k_parallel_per_bank > 1 path and is masked by uniform precision because both
# fmt double-buffer slots carry identical bytes — only mixed precision makes
# slot-vs-slot content differ, so race surfaces here are mixed-precision-only.
#
# Shape and params mirror test_benchmark_gate_proj (production R1 gate-proj).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "formats_per_device, fmt_ratios",
    [
        (["bfp4", "bfp0"], {"bfp4": 77.8, "bfp0": 22.2}),
        (["bfp4", "bfp2", "bfp0"], {"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3}),
    ],
    ids=["bfp4_bfp0", "bfp4_bfp2_bfp0"],
)
def test_matmul_expert_race_probe_back_to_back(device, formats_per_device, fmt_ratios):
    """Probe documented inter-call race: call ExpertKernel.op twice with identical inputs.

    Production gate-proj config (K=7168, N=256, k_parallel_per_bank=2, dram_fuse_silu).
    On a clean kernel both invocations must produce bitwise-identical output. If the
    second invocation diverges, the K-reduction back-to-back race is reproduced.
    """
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[formats_per_device],
        dram_fuse_silu=True,
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios=fmt_ratios,
        num_loop_iters=1,
        n_program_invocations=2,
    )


def test_matmul_expert_determinism(device):
    """100-invocation determinism probe at production gate-proj config (3-way mixed).

    Catches any intra-call or inter-call non-determinism that 2-invocation back-to-back
    might miss. If the back-to-back test passes but this fails, the race is rare and
    requires more invocations to surface. If both pass, the kernel is reproducible at
    this shape/grid — bug is elsewhere (real-BSPM specifics, inter-op state, etc).
    """
    _run_hybrid_expert_multi_device(
        device,
        M=1,
        K=7168,
        N=256,
        num_experts=8,
        sram_expert_ids=[],
        dram_expert_ids=list(range(8)),
        active_expert_ids=[2, 3, 4, 5, 6, 7, 1, 0],
        formats_per_device=[["bfp4", "bfp2", "bfp0"]],
        dram_fuse_silu=True,
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        fmt_distribution="uniform",
        fmt_ratios={"bfp4": 74.4, "bfp2": 7.3, "bfp0": 18.3},
        num_loop_iters=1,
        n_program_invocations=100,
    )


# ---------------------------------------------------------------------------
# Real-BSPM race probes — production tile patterns from BitSculpt layer file.
#
# Synthetic mixed-precision tests use uniform per-column precision (every N-column
# carries the same K-row pattern, every expert carries the same fmt_ratios). Real
# BSPMs vary per-tile in both K and N, and per-expert distributions differ wildly.
# This stresses block_sizes / fmt_per_expert_bytes indexing and the fmt
# double-buffer's slot-to-slot byte delta under irregular patterns the synthetic
# path cannot generate.
#
# Default path expects: $BSPM_RESULTS_DIR/deepseek-r1-0528/layer_16/precision_eval/
#   precision_map_B_3.5.bspm (10.5 MB, 256 experts × 3 projs at 3.5 b/e).
# ---------------------------------------------------------------------------


def _run_dram_bspm_kparallel(
    mesh_device,
    M,
    K,
    N,
    bspm_path,
    bspm_expert_configs,
    active_expert_ids,
    pcc_threshold=0.85,
    num_subblocks_k=None,
    num_subblocks_n=None,
    n_parallel_per_bank=1,
    k_parallel_per_bank=1,
    dram_fuse_silu=False,
    n_program_invocations=1,
    num_loop_iters=1,
    precision_override=None,
    return_pccs=False,
    bspm_full_n=None,
    software_quantized_golden=False,
):
    """DRAM-only ExpertKernel with real BSPM assignments + K-parallel + multi-invocation.

    Mirrors _run_standard's structure (which is known to work with K-parallel mixed
    precision) but replaces the synthetic CompressedTensorAssigner with real BSPM-loaded
    per-tile precision codes. Weights remain random per slot; only the assignment
    pattern is from BitSculpt.

    bspm_expert_configs: list of (expert_idx, proj_idx) tuples — one per slot.
    active_expert_ids: slot indices to mark active in the index tensor.
    """
    import numpy as np

    from models.demos.deepseek_v3_b1.compressed_tensor.bspm_loader import load_bspm_for_expert

    tile_w = 32
    cores_per_dram_bank = n_parallel_per_bank * k_parallel_per_bank
    num_experts = len(bspm_expert_configs)
    num_devices = mesh_device.get_num_devices()
    mesh_rows, mesh_cols = mesh_device.shape[0], mesh_device.shape[1]

    grids = _setup_core_grids(mesh_device, cores_per_dram_bank, 0, None, has_sram=False)
    dram_cores_list = grids["dram_cores_list"]
    dram_core_grid = grids["dram_core_grid"]
    compute_core_grid = grids["compute_core_grid"]
    num_dram_cores = len(dram_cores_list)
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
        num_subblocks_k,
        num_subblocks_n,
        k_parallel_per_bank=k_parallel_per_bank,
    )
    assert N_dram_per_device == N, f"N_dram_per_device ({N_dram_per_device}) != N ({N}): DRAM padding not supported"

    num_subblocks_k_v = Kt // subblock_k
    shuffle_subblock_k = Kt // k_parallel_per_bank if k_parallel_per_bank > 1 else None

    dram_grid_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(
                ttnn.CoreCoord(0, 0),
                ttnn.CoreCoord(dram_grid_size.x - 1, dram_grid_size.y - 1),
            )
        ]
    )
    shard_width = dram_per_core_N * tile_w * n_parallel_per_bank
    dram_b_mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(dram_grid, [K, shard_width], ttnn.ShardOrientation.ROW_MAJOR),
    )

    torch.manual_seed(0)
    torch_a = torch.randn((M, K), dtype=torch.bfloat16)

    # tt-metal COMPRESSED_FORMATS indices: bfp8=0, bfp4=1, bfp2=2, bfp0=3.
    _UNIFORM_CODE = {"bfp8": 0, "bfp4": 1, "bfp2": 2, "bfp0": 3}
    if isinstance(precision_override, str):
        assert (
            precision_override in _UNIFORM_CODE
        ), f"precision_override must be one of {list(_UNIFORM_CODE)} or a dict of ratios, got {precision_override!r}"
    elif isinstance(precision_override, dict):
        assert precision_override, "precision_override dict must be non-empty"
        for fmt in precision_override:
            assert fmt in _UNIFORM_CODE, f"unknown format {fmt!r} in precision_override (valid: {list(_UNIFORM_CODE)})"

    # BSPM file is built at the full N width (e.g. 2048 for R1 gate_proj/up_proj). When the
    # test runs at a smaller N (e.g. the per-device tensor-parallel slice 256), set bspm_full_n
    # to the on-disk width and we'll slice the loaded assignment to the first N columns.
    bspm_load_n = bspm_full_n if bspm_full_n is not None else N
    assert bspm_load_n >= N, f"bspm_full_n ({bspm_load_n}) must be >= N ({N})"
    bspm_tile_cols = bspm_load_n // tile_w
    n_tile_cols = N // tile_w

    dram_cts = []
    torch_b_all = {}
    for slot_idx, (expert_idx, proj_idx) in enumerate(bspm_expert_configs):
        if precision_override is None:
            assignment_full = load_bspm_for_expert(
                str(bspm_path),
                expert_idx=expert_idx,
                proj_idx=proj_idx,
                tile_rows=K // tile_w,
                tile_cols=bspm_tile_cols,
            )
            assignment = assignment_full[:, :n_tile_cols] if bspm_load_n != N else assignment_full
        elif isinstance(precision_override, dict):
            # Synthetic mix: per-tile random sampling from the supplied ratios.
            rng = np.random.RandomState(slot_idx + 42)
            formats_list = list(precision_override.keys())
            codes_arr = np.array([_UNIFORM_CODE[f] for f in formats_list], dtype=np.int8)
            probs = np.array([float(precision_override[f]) for f in formats_list], dtype=float)
            probs = probs / probs.sum()
            flat_codes = rng.choice(codes_arr, size=(K // tile_w) * n_tile_cols, p=probs)
            assignment = flat_codes.astype(np.int8).reshape(K // tile_w, n_tile_cols)
        else:
            assignment = np.full((K // tile_w, n_tile_cols), _UNIFORM_CODE[precision_override], dtype=np.int8)
        assignment_shuffled = _shuffle_assignment_blocked(
            assignment, num_banks, subblock_k=shuffle_subblock_k, subblock_n=subblock_n
        )

        per_dev_weights = []
        per_dev_shuffled = []
        for dev_idx in range(num_devices):
            torch.manual_seed(slot_idx * 1000 + dev_idx + 7)
            b = torch.randn(K, N_dram_per_device).float()
            per_dev_weights.append(b)
            per_dev_shuffled.append(
                shuffle_tensor_tiles(b, tile_w, num_banks, subblock_k=shuffle_subblock_k, subblock_n=subblock_n)
            )
        if software_quantized_golden:
            # Replace each slot's golden weight with its software-quantized version so
            # _validate_dram_output's golden computes A @ B_software_quant — PCC then
            # reflects ONLY the kernel-vs-software discrepancy, not the quantization cost.
            torch_b_all[slot_idx] = [
                torch.from_numpy(_software_quantize_per_tile(b.numpy(), assignment, tile_w)) for b in per_dev_weights
            ]
        else:
            torch_b_all[slot_idx] = per_dev_weights

        b_4d = torch.stack(per_dev_shuffled).reshape(mesh_rows, mesh_cols, K, N_dram_per_device)
        assignment_4d = np.tile(assignment_shuffled, (num_devices, 1))

        ct = CompressedTensor(
            b_4d,
            assignment_4d,
            device=mesh_device,
            memory_config=dram_b_mem,
            per_core_allocation=False,
            mesh_mapper_config=ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)]),
        )
        label = f"override={precision_override}" if precision_override else f"expert={expert_idx}, proj={proj_idx}"
        logger.info(f"  CT slot {slot_idx} ({label}): {ct.tile_counts}")
        if isinstance(precision_override, str):
            assert (
                ct.tile_counts.get(precision_override, 0) > 0
            ), f"Slot {slot_idx}: expected {precision_override} tiles from uniform override"
        elif isinstance(precision_override, dict):
            for fmt, weight in precision_override.items():
                if weight > 0:
                    assert (
                        ct.tile_counts.get(fmt, 0) > 0
                    ), f"Slot {slot_idx}: expected {fmt} tiles from mix override {precision_override}"
        dram_cts.append(ct)

    if precision_override is None:
        total_counts = {"bfp8": 0, "bfp4": 0, "bfp2": 0, "bfp0": 0}
        for ct in dram_cts:
            for fmt, count in ct.tile_counts.items():
                total_counts[fmt] = total_counts.get(fmt, 0) + count
        total_tiles = sum(total_counts.values()) or 1
        mixed_tiles = total_counts.get("bfp2", 0) + total_counts.get("bfp0", 0)
        mixed_pct = 100.0 * mixed_tiles / total_tiles
        logger.info(
            f"  BSPM slice global distribution across {len(dram_cts)} slots: {total_counts} "
            f"({mixed_pct:.2f}% bfp2+bfp0)"
        )
        if mixed_tiles == 0:
            logger.warning(
                f"All {len(dram_cts)} slots loaded uniform precision (no bfp2/bfp0 tiles). "
                f"This slice does NOT exercise mixed-precision behavior — pick different "
                f"(expert, proj) tuples, a different layer, or a different bspm_n_offset."
            )

    dram_meta_flags = [1] * num_experts
    is_dram_flags = list(dram_meta_flags)

    dram_meta_tensors = create_dram_expert_tensors_multi_device(
        mesh_device,
        dram_cts,
        subblock_k,
        num_subblocks_k_v,
        dram_per_core_N,
        n_parallel_per_bank=n_parallel_per_bank,
        num_total_experts=num_experts,
        is_dram_flags=dram_meta_flags,
        subblock_n=subblock_n,
        k_parallel_per_bank=k_parallel_per_bank,
    )

    a_tensor = _build_activation_tensor(torch_a, mesh_device, compute_core_grid, num_cores, M, K, tile_w)
    index_tensor = _build_index_tensor(active_expert_ids, mesh_device, compute_core_grid, num_cores, is_dram_flags)

    num_active_experts = len(active_expert_ids)
    out_tensor = _build_dram_output(
        mesh_device,
        M,
        dram_per_core_N,
        num_active_experts,
        num_dram_cores,
        num_devices,
        dram_core_grid,
        tile_w,
        dram_fuse_silu=dram_fuse_silu,
    )

    reference_outputs = None
    mismatches = []
    extra_mismatch_count = 0
    MAX_STORED_MISMATCHES = 1000
    progress_every = max(1, n_program_invocations // 10)

    for invocation in range(n_program_invocations):
        if invocation > 0:
            out_tensor = _build_dram_output(
                mesh_device,
                M,
                dram_per_core_N,
                num_active_experts,
                num_dram_cores,
                num_devices,
                dram_core_grid,
                tile_w,
                dram_fuse_silu=dram_fuse_silu,
            )

        result = ExpertKernel.op(
            a_tensor,
            [],
            dram_cts,
            out_tensor,
            index_tensor,
            num_active_experts=num_active_experts,
            subblock_k=subblock_k,
            subblock_n=subblock_n,
            dram_core_grid=dram_core_grid,
            dram_meta_tensors=dram_meta_tensors,
            dram_per_core_n=dram_per_core_N,
            has_sram=False,
            sram_core_grid=None,
            sram_fmt_tensors={},
            sram_k_offsets=None,
            n_parallel_per_bank=n_parallel_per_bank,
            k_parallel_per_bank=k_parallel_per_bank,
            sram_per_core_n=0,
            sram_k_per_core=Kt,
            sram_output_tensor=None,
            dram_fuse_silu=dram_fuse_silu,
            tp_expert=True,
            num_loop_iters=num_loop_iters,
        )

        if n_program_invocations > 1:
            current = [ttnn.to_torch(d).clone() for d in ttnn.get_device_tensors(result)]
            if reference_outputs is None:
                reference_outputs = current
            else:
                for dev_idx, (ref_dev, cur_dev) in enumerate(zip(reference_outputs, current)):
                    if not torch.equal(ref_dev, cur_dev):
                        if len(mismatches) < MAX_STORED_MISMATCHES:
                            n_diff = int((ref_dev != cur_dev).sum().item())
                            delta = (ref_dev.float() - cur_dev.float()).abs()
                            mismatches.append(
                                (
                                    invocation,
                                    dev_idx,
                                    n_diff,
                                    float(delta.max().item()),
                                    float(delta.mean().item()),
                                )
                            )
                        else:
                            extra_mismatch_count += 1

            if (invocation + 1) % progress_every == 0:
                total = len(mismatches) + extra_mismatch_count
                logger.info(
                    f"  multi-invocation: {invocation + 1}/{n_program_invocations} done, {total} divergent so far"
                )

    if n_program_invocations > 1:
        total_mismatches = len(mismatches) + extra_mismatch_count
        if total_mismatches:
            logger.error(
                f"NON-DETERMINISTIC (real BSPM): {total_mismatches} divergent (invocation, device) "
                f"pairs across {n_program_invocations} invocations"
            )
            for inv, dev, n_diff, max_d, mean_d in mismatches[:10]:
                logger.error(
                    f"  invocation {inv} device {dev}: {n_diff} differing elements, "
                    f"max|delta|={max_d:.4g}, mean|delta|={mean_d:.4g}"
                )
        assert not total_mismatches, (
            f"ExpertKernel non-deterministic with real BSPMs across {n_program_invocations} invocations: "
            f"{total_mismatches} divergent (invocation, device) pairs "
            f"(first at invocation {mismatches[0][0]} device {mismatches[0][1]})"
        )

    collected_pccs = [] if return_pccs else None
    _validate_dram_output(
        result,
        torch_a,
        torch_b_all,
        active_expert_ids,
        num_active_experts,
        dram_per_core_N,
        num_dram_cores,
        pcc_threshold,
        dram_fuse_silu,
        tile_w,
        M=M,
        tp_expert=True,
        cores_per_dram_bank=cores_per_dram_bank,
        k_parallel_per_bank=k_parallel_per_bank,
        pccs_out=collected_pccs,
    )
    if return_pccs:
        return {
            "pccs": collected_pccs,
            "tile_counts_per_slot": [dict(ct.tile_counts) for ct in dram_cts],
            "slot_configs": list(bspm_expert_configs),
        }


def _resolve_layer16_bspm():
    """Locate layer-16 BSPM. Prefer $BSPM_RESULTS_DIR, else the on-disk default path."""
    import os
    from pathlib import Path

    bspm_dir = os.environ.get("BSPM_RESULTS_DIR")
    if bspm_dir:
        candidate = Path(bspm_dir) / "deepseek-r1-0528" / "layer_16" / "precision_eval" / "precision_map_B_3.5.bspm"
    else:
        candidate = Path(
            "/home/mtairum/bit_sculpt/results/deepseek-r1-0528/layer_16/precision_eval/precision_map_B_3.5.bspm"
        )
    if not candidate.exists():
        pytest.skip(f"Layer 16 BSPM not found at {candidate}")
    return candidate


@pytest.mark.skip_post_commit
def test_matmul_expert_real_bspm_pcc(device):
    """Three-way PCC comparison: all-bfp4, all-bfp2, and real layer-16 BSPMs.

    Runs the same kernel/activation/weight setup three times with different per-tile
    precision assignments, computing per-(device, expert, K-slice) PCC against the FP
    torch golden for each. Logs all three distributions for visual comparison.

    Expectation: all_bfp4_pcc > all_bfp2_pcc (sanity — kernel is internally consistent
    with the precision tier), and real BSPM PCC lands somewhere in between or near
    all_bfp2 (real BSPMs at 3.5 b/e zero out ~10–20% of tiles, which is a strict
    information loss vs uniform bfp2). If real BSPM PCC is dramatically below
    all_bfp2_pcc (e.g. < 0.5), the kernel likely has a real-BSPM-specific correctness
    bug — not just quantization noise.
    """
    import statistics

    bspm_path = _resolve_layer16_bspm()

    common_kwargs = dict(
        M=1,
        K=7168,
        N=256,  # per-device tensor-parallel slice of gate_proj's N=2048 (TP=8 in production)
        bspm_full_n=2048,  # BSPM file is built at the full gate_proj N width
        bspm_path=bspm_path,
        bspm_expert_configs=[(i, 0) for i in range(8)],
        active_expert_ids=list(range(8)),
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        dram_fuse_silu=True,
        n_program_invocations=1,
        return_pccs=True,
    )

    results = {}
    for label, override in [("all_bfp4", "bfp4"), ("all_bfp2", "bfp2"), ("real_bspm", None)]:
        logger.info(f"=== {label} run ===")
        ret = _run_dram_bspm_kparallel(device, precision_override=override, **common_kwargs)
        # ret is dict: {"pccs": [(dev_idx, eidx, label, pcc), ...],
        #               "tile_counts_per_slot": [...],
        #               "slot_configs": [...]}
        pccs_list = ret["pccs"]
        values = [v for _, _, _, v in pccs_list]
        reducer_values = [v for _, _, lbl, v in pccs_list if lbl == "reducer"]
        # Build per-(dev, slot) reducer PCC lookup for cross-config correlation below.
        reducer_by_slot = {(dev, eid): v for dev, eid, lbl, v in pccs_list if lbl == "reducer"}
        results[label] = {
            "all": values,
            "reducer": reducer_values,
            "mean": statistics.mean(values),
            "min": min(values),
            "max": max(values),
            "reducer_mean": statistics.mean(reducer_values) if reducer_values else float("nan"),
            "reducer_min": min(reducer_values) if reducer_values else float("nan"),
            "reducer_by_slot": reducer_by_slot,
            "tile_counts_per_slot": ret["tile_counts_per_slot"],
            "slot_configs": ret["slot_configs"],
        }
        logger.info(
            f"  {label}: n={len(values)} samples, mean={results[label]['mean']:.6f}, "
            f"min={results[label]['min']:.6f}, max={results[label]['max']:.6f}, "
            f"reducer_mean={results[label]['reducer_mean']:.6f}, reducer_min={results[label]['reducer_min']:.6f}"
        )

    logger.info("=== three-way summary (reducer only) ===")
    logger.info(
        f"  all_bfp4  mean={results['all_bfp4']['reducer_mean']:.6f}  min={results['all_bfp4']['reducer_min']:.6f}"
    )
    logger.info(
        f"  real_bspm mean={results['real_bspm']['reducer_mean']:.6f}  min={results['real_bspm']['reducer_min']:.6f}"
    )
    logger.info(
        f"  all_bfp2  mean={results['all_bfp2']['reducer_mean']:.6f}  min={results['all_bfp2']['reducer_min']:.6f}"
    )

    # Per-slot diagnostic: which (device, slot) shows the largest real-BSPM regression vs
    # uniform bounds? Sort ascending by real_bspm reducer PCC and dump the bottom slots
    # alongside their bfp4/bfp2 PCCs and the slot's tile distribution. This identifies
    # exactly which expert + tile pattern triggers low PCC.
    real_keys = sorted(results["real_bspm"]["reducer_by_slot"].items(), key=lambda kv: kv[1])
    logger.info("=== per-slot reducer PCC (real_bspm, sorted ascending) ===")
    logger.info(
        f"  {'dev':>3} {'slot':>4}  {'real_bspm':>10}  {'all_bfp4':>10}  {'all_bfp2':>10}   tile_counts (real_bspm)"
    )
    for (dev, eid), real_pcc in real_keys:
        bfp4_pcc = results["all_bfp4"]["reducer_by_slot"].get((dev, eid), float("nan"))
        bfp2_pcc = results["all_bfp2"]["reducer_by_slot"].get((dev, eid), float("nan"))
        # The slot's tile_counts (only differs in real_bspm config; uniform configs are uniform).
        tc = (
            results["real_bspm"]["tile_counts_per_slot"][eid]
            if eid < len(results["real_bspm"]["tile_counts_per_slot"])
            else {}
        )
        logger.info(f"  {dev:>3} {eid:>4}  {real_pcc:>10.6f}  {bfp4_pcc:>10.6f}  {bfp2_pcc:>10.6f}   {tc}")

    # Sanity: bfp4 should beat bfp2 (the kernel must be at least internally consistent
    # with the precision tier under uniform precision — this is the bound we already trust).
    assert results["all_bfp4"]["reducer_mean"] > results["all_bfp2"]["reducer_mean"], (
        f"bfp4 mean PCC ({results['all_bfp4']['reducer_mean']:.6f}) did not beat bfp2 mean "
        f"({results['all_bfp2']['reducer_mean']:.6f}) — kernel inconsistent across uniform precision tiers"
    )

    # Catch dramatic real-BSPM-specific failure: mixed precision shouldn't collapse to garbage.
    # Loose bound — bfp0 tiles can legitimately drop real-BSPM PCC below uniform bfp2.
    assert results["real_bspm"]["reducer_min"] > 0.3, (
        f"real BSPM min reducer PCC ({results['real_bspm']['reducer_min']:.6f}) is below 0.3 — "
        f"likely a real-BSPM-specific kernel correctness bug (not just quantization noise)"
    )


@pytest.mark.skip_post_commit
def test_matmul_expert_real_bspm_back_to_back(device):
    """Real layer-16 BSPMs, 2 back-to-back invocations at production K-parallel + fuse_silu.

    Combines the real-BSPM tile-pattern stress with the documented inter-call race
    precondition. If this fails but test_matmul_expert_real_bspm_pcc passes, the bug
    requires both real tile patterns AND back-to-back invocation.
    """
    bspm_path = _resolve_layer16_bspm()
    _run_dram_bspm_kparallel(
        device,
        M=1,
        K=7168,
        N=256,  # per-device tensor-parallel slice of gate_proj
        bspm_full_n=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(i, 0) for i in range(8)],
        active_expert_ids=list(range(8)),
        pcc_threshold=0.85,
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        dram_fuse_silu=True,
        n_program_invocations=2,
    )


@pytest.mark.skip_post_commit
def test_matmul_expert_synthetic_mix_pcc(device):
    """Four-way PCC comparison with synthetic bfp4/bfp2 mixes (no zeros).

    Runs the same kernel/activation/weight setup four times at production gate-proj shape
    (K=7168, N=256 per-device, k_parallel=2, fuse_silu) with per-tile precision assignments
    randomly sampled from controlled ratios. No bfp0, so isolates whether the kernel handles
    bfp4+bfp2 mixes correctly independent of any zero-tile-specific behavior.

    Expectation: PCC ordering should be monotonic in bfp4 fraction:
        all_bfp4 > mix_75_25 > mix_50_50 > all_bfp2.
    If the two mixes land in between, the kernel handles bfp4/bfp2 mixes correctly and the
    real-BSPM slot-3 anomaly is specifically a bfp0-related bug.
    """
    import statistics

    # BSPM path required by the helper signature but unused when precision_override is set.
    # Pass a real one anyway so we don't have to special-case its absence inside the helper.
    bspm_path = _resolve_layer16_bspm()

    common_kwargs = dict(
        M=1,
        K=7168,
        N=256,
        bspm_full_n=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(i, 0) for i in range(8)],
        active_expert_ids=list(range(8)),
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        dram_fuse_silu=True,
        n_program_invocations=1,
        return_pccs=True,
    )

    configs = [
        ("all_bfp4", "bfp4"),
        ("mix_75_25", {"bfp4": 75, "bfp2": 25}),
        ("mix_50_50", {"bfp4": 50, "bfp2": 50}),
        ("all_bfp2", "bfp2"),
    ]

    results = {}
    for label, override in configs:
        logger.info(f"=== {label} run ===")
        ret = _run_dram_bspm_kparallel(device, precision_override=override, **common_kwargs)
        pccs_list = ret["pccs"]
        reducer_values = [v for _, _, lbl, v in pccs_list if lbl == "reducer"]
        reducer_by_slot = {(dev, eid): v for dev, eid, lbl, v in pccs_list if lbl == "reducer"}
        results[label] = {
            "reducer": reducer_values,
            "reducer_mean": statistics.mean(reducer_values) if reducer_values else float("nan"),
            "reducer_min": min(reducer_values) if reducer_values else float("nan"),
            "reducer_max": max(reducer_values) if reducer_values else float("nan"),
            "reducer_by_slot": reducer_by_slot,
            "tile_counts_per_slot": ret["tile_counts_per_slot"],
        }
        logger.info(
            f"  {label}: reducer_mean={results[label]['reducer_mean']:.6f}, "
            f"reducer_min={results[label]['reducer_min']:.6f}, "
            f"reducer_max={results[label]['reducer_max']:.6f}"
        )

    logger.info("=== four-way summary (reducer only) ===")
    for label, _ in configs:
        logger.info(
            f"  {label:>10}  mean={results[label]['reducer_mean']:.6f}  "
            f"min={results[label]['reducer_min']:.6f}  max={results[label]['reducer_max']:.6f}"
        )

    bfp4_mean = results["all_bfp4"]["reducer_mean"]
    bfp2_mean = results["all_bfp2"]["reducer_mean"]
    mix_75_mean = results["mix_75_25"]["reducer_mean"]
    mix_50_mean = results["mix_50_50"]["reducer_mean"]

    # Sanity: uniform bounds must be ordered correctly.
    assert bfp4_mean > bfp2_mean, (
        f"bfp4 mean PCC ({bfp4_mean:.6f}) did not beat bfp2 mean ({bfp2_mean:.6f}); "
        f"kernel inconsistent across uniform precision tiers"
    )

    # Both mixes should land between the bounds. Allow a small tolerance (0.01) for noise.
    tol = 0.01
    assert bfp2_mean - tol <= mix_50_mean <= bfp4_mean + tol, (
        f"mix_50_50 mean PCC ({mix_50_mean:.6f}) outside [{bfp2_mean:.6f}, {bfp4_mean:.6f}] bounds — "
        f"kernel mishandles 50/50 bfp4/bfp2 mix"
    )
    assert bfp2_mean - tol <= mix_75_mean <= bfp4_mean + tol, (
        f"mix_75_25 mean PCC ({mix_75_mean:.6f}) outside [{bfp2_mean:.6f}, {bfp4_mean:.6f}] bounds — "
        f"kernel mishandles 75/25 bfp4/bfp2 mix"
    )

    # Monotonicity: more bfp4 should mean higher PCC. Allow small tolerance for noise.
    assert mix_75_mean + tol >= mix_50_mean, (
        f"mix_75_25 mean PCC ({mix_75_mean:.6f}) did not exceed mix_50_50 ({mix_50_mean:.6f}) — "
        f"non-monotonic precision response, possible kernel mix-handling bug"
    )


@pytest.mark.skip_post_commit
def test_matmul_expert_zero_fraction_sweep(device):
    """Sweep bfp0 fraction to find where kernel correctness collapses (if at all).

    Per-tile precision codes are randomly sampled from each config's ratios. Random
    distribution decouples ratio from spatial pattern — if a config collapses here, the
    bug is purely about per-tile bfp0 fraction; if it doesn't, the real-BSPM slot-3
    anomaly needs that specific *clustered* tile pattern, not just the ratio.

    `slot3_ratios` exactly matches real-BSPM slot 3 proportions
    (3% bfp4 / 9.5% bfp2 / 87.5% bfp0 → PCC 0.327 in test_matmul_expert_real_bspm_pcc).

    No tight assertions — this is a diagnostic sweep. Just confirms the kernel finishes
    and prints the per-config table for inspection.
    """
    import statistics

    bspm_path = _resolve_layer16_bspm()

    common_kwargs = dict(
        M=1,
        K=7168,
        N=256,
        bspm_full_n=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(i, 0) for i in range(8)],
        active_expert_ids=list(range(8)),
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        dram_fuse_silu=True,
        n_program_invocations=1,
        return_pccs=True,
    )

    configs = [
        ("all_bfp4", "bfp4"),
        ("mix_zero_25", {"bfp4": 75, "bfp0": 25}),
        ("mix_zero_50", {"bfp4": 50, "bfp0": 50}),
        ("mix_zero_75", {"bfp4": 25, "bfp0": 75}),
        ("mix_zero_90", {"bfp4": 10, "bfp0": 90}),
        ("slot3_ratios", {"bfp4": 3, "bfp2": 9.5, "bfp0": 87.5}),
    ]

    results = {}
    for label, override in configs:
        logger.info(f"=== {label} run ===")
        ret = _run_dram_bspm_kparallel(device, precision_override=override, **common_kwargs)
        pccs_list = ret["pccs"]
        reducer_values = [v for _, _, lbl, v in pccs_list if lbl == "reducer"]
        # Aggregate tile counts across slots so we can verify the realized distribution.
        agg_tile_counts = {"bfp8": 0, "bfp4": 0, "bfp2": 0, "bfp0": 0}
        for tc in ret["tile_counts_per_slot"]:
            for fmt, count in tc.items():
                agg_tile_counts[fmt] = agg_tile_counts.get(fmt, 0) + count
        total = sum(agg_tile_counts.values()) or 1
        bfp0_pct = 100.0 * agg_tile_counts.get("bfp0", 0) / total
        results[label] = {
            "reducer_mean": statistics.mean(reducer_values) if reducer_values else float("nan"),
            "reducer_min": min(reducer_values) if reducer_values else float("nan"),
            "reducer_max": max(reducer_values) if reducer_values else float("nan"),
            "realized_bfp0_pct": bfp0_pct,
            "tile_counts": agg_tile_counts,
        }
        logger.info(
            f"  {label}: reducer_mean={results[label]['reducer_mean']:.6f}, "
            f"reducer_min={results[label]['reducer_min']:.6f}, "
            f"reducer_max={results[label]['reducer_max']:.6f}, "
            f"realized_bfp0_pct={bfp0_pct:.2f}%, counts={agg_tile_counts}"
        )

    logger.info("=== bfp0-fraction sweep summary (reducer only) ===")
    logger.info(f"  {'config':>13}  {'bfp0%':>6}  {'mean':>10}  {'min':>10}  {'max':>10}")
    for label, _ in configs:
        r = results[label]
        logger.info(
            f"  {label:>13}  {r['realized_bfp0_pct']:>5.2f}%  "
            f"{r['reducer_mean']:>10.6f}  {r['reducer_min']:>10.6f}  {r['reducer_max']:>10.6f}"
        )

    # Single sanity check: uniform bfp4 must still work. Everything else is diagnostic —
    # the cliff (if any) is the signal we're after, not a pass/fail.
    assert results["all_bfp4"]["reducer_mean"] > 0.95, (
        f"all_bfp4 mean PCC ({results['all_bfp4']['reducer_mean']:.6f}) below 0.95 — "
        f"kernel regression independent of zero-fraction sweep"
    )


@pytest.mark.skip_post_commit
def test_matmul_expert_kernel_vs_software_quantized(device):
    """Direct hardware-vs-software comparison: PCC against a *software-quantized* golden.

    Standard tests compare kernel output to torch_a @ B_unquantized — PCC then mixes
    quantization cost and kernel correctness. This test compares kernel output to
    torch_a @ B_software_quant, where B_software_quant applies the same per-tile BFP
    quantize-dequantize that the hardware should be doing. If the kernel is doing the
    math correctly, PCC must be near 1.0 regardless of precision distribution.

    Configs span uniform bfp4 → 90% bfp0 → real BSPM slot 3, covering the same range
    where the FP-golden PCC collapsed from 0.99 → 0.29. If kernel is correct, all
    configs should produce kernel-vs-software PCC > 0.999. Any config below ~0.99 is
    a real hardware-vs-software discrepancy.
    """
    import statistics

    bspm_path = _resolve_layer16_bspm()

    common_kwargs = dict(
        M=1,
        K=7168,
        N=256,
        bspm_full_n=2048,
        bspm_path=bspm_path,
        bspm_expert_configs=[(i, 0) for i in range(8)],
        active_expert_ids=list(range(8)),
        num_subblocks_k=4,
        num_subblocks_n=1,
        n_parallel_per_bank=1,
        k_parallel_per_bank=2,
        dram_fuse_silu=True,
        n_program_invocations=100,
        return_pccs=True,
        software_quantized_golden=True,
    )

    configs = [
        ("all_bfp4", "bfp4"),
        ("all_bfp2", "bfp2"),
        ("mix_50_50_bfp4_bfp2", {"bfp4": 50, "bfp2": 50}),
        ("mix_zero_50", {"bfp4": 50, "bfp0": 50}),
        ("mix_zero_90", {"bfp4": 10, "bfp0": 90}),
        ("slot3_ratios", {"bfp4": 3, "bfp2": 9.5, "bfp0": 87.5}),
        ("real_bspm_layer16", None),
    ]

    results = {}
    for label, override in configs:
        logger.info(f"=== {label} run (vs software-quantized golden) ===")
        ret = _run_dram_bspm_kparallel(device, precision_override=override, **common_kwargs)
        pccs_list = ret["pccs"]
        reducer_values = [v for _, _, lbl, v in pccs_list if lbl == "reducer"]
        results[label] = {
            "reducer_mean": statistics.mean(reducer_values) if reducer_values else float("nan"),
            "reducer_min": min(reducer_values) if reducer_values else float("nan"),
            "reducer_max": max(reducer_values) if reducer_values else float("nan"),
            "reducer_values": reducer_values,
        }
        logger.info(
            f"  {label}: kernel-vs-software reducer_mean={results[label]['reducer_mean']:.6f}, "
            f"min={results[label]['reducer_min']:.6f}, max={results[label]['reducer_max']:.6f}"
        )

    logger.info("=== kernel-vs-software-quantized summary (reducer only) ===")
    logger.info(f"  {'config':>22}  {'mean':>10}  {'min':>10}  {'max':>10}")
    for label, _ in configs:
        r = results[label]
        logger.info(f"  {label:>22}  {r['reducer_mean']:>10.6f}  {r['reducer_min']:>10.6f}  {r['reducer_max']:>10.6f}")

    # If the kernel is correct, kernel output should match A @ B_software_quant to high
    # precision (bfloat16 accumulator + per-tile BFP roundoff matches the software emulator
    # bit-for-bit, modulo a tiny K-reduction summation order effect). Threshold 0.999 is
    # tight enough to catch any real discrepancy while tolerating accumulator order noise.
    SOFTWARE_MATCH_THRESHOLD = 0.999
    failures = []
    for label, _ in configs:
        if results[label]["reducer_min"] < SOFTWARE_MATCH_THRESHOLD:
            failures.append((label, results[label]["reducer_min"], results[label]["reducer_mean"]))

    if failures:
        logger.error("Hardware-vs-software discrepancy detected:")
        for label, lo, mean in failures:
            logger.error(f"  {label}: min PCC {lo:.6f}, mean {mean:.6f} (threshold {SOFTWARE_MATCH_THRESHOLD})")
    assert not failures, (
        f"Hardware-vs-software match failed for {len(failures)} config(s): "
        f"{[(lbl, f'{lo:.4f}') for lbl, lo, _ in failures]}. "
        f"Threshold {SOFTWARE_MATCH_THRESHOLD}."
    )
