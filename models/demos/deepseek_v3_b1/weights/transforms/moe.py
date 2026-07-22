# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MoE / shared-expert weight preprocessing transforms and fusion helpers."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_b1.weights.overlap.packing import OverlapEntry, OverlappedTensor, overlap_tensors
from models.demos.deepseek_v3_b1.weights.specs.overlap_configs import (
    DOWN_PROJ_SINGLE_DEVICE_SPEC,
    GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC,
)


def _tp_factors(device) -> tuple[int, int]:
    """Returns (mla_tp, moe_tp) derived from device topology."""
    if device.get_num_devices() == 1:
        return 1, 1
    return 2, 8


def shuffle_dram_tiles(
    tensor: torch.Tensor,
    tile_size: int,
    num_banks: int,
    subblock_k: int | None = None,
    subblock_n: int = 1,
) -> torch.Tensor:
    """Reorder tiles within each DRAM bank shard for the streaming matmul kernel.

    Within each bank shard, tiles are grouped into ``[subblock_k × subblock_n]``
    blocks laid out K-outer / N-inner inside the block (matches LLK
    ``compressed_custom_mm_block`` ct_dim walk). Block ordering inside the shard
    is ``k_sub`` outer / ``n_group`` inner.

    Defaults (``subblock_k=K_tiles``, ``subblock_n=1``) reproduce the original
    column-major-per-N-column layout (block per N-col holds full K).

    Note: when both ``num_subblocks_k > 1`` and ``num_n_groups > 1``, this
    block-ordering is ``k_sub``-outer while the kernel's ``_compute_expert_subblock_metadata``
    walks ``ng``-outer — only matches when ``num_subblocks_k == 1`` or
    ``num_n_groups == 1``. All current production projections satisfy this.
    """
    orig_shape = tensor.shape
    K, N = orig_shape[-2], orig_shape[-1]

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

    if K_tiles % subblock_k != 0:
        raise ValueError(f"K_tiles ({K_tiles}) must be divisible by subblock_k ({subblock_k})")
    if per_N_tiles % subblock_n != 0:
        raise ValueError(f"per_N_tiles ({per_N_tiles}) must be divisible by subblock_n ({subblock_n})")
    # Block ordering inside the shard is k_sub-outer / n_group-inner; the kernel
    # consumption order in `_compute_expert_subblock_metadata` is ng-outer /
    # sb_k-inner. Both agree when at least one factor is 1; the both-split case
    # would silently mis-order blocks and is intentionally disallowed.
    assert (K_tiles // subblock_k) == 1 or (per_N_tiles // subblock_n) == 1, (
        f"shuffle_dram_tiles: configs with both num_subblocks_k>1 ({K_tiles // subblock_k}) "
        f"and num_n_groups>1 ({per_N_tiles // subblock_n}) are not supported — block ordering "
        f"would not match the kernel's metadata walk. Pick subblock_k=K_tiles or subblock_n=per_N_tiles."
    )

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

    return shuffled.reshape(*orig_shape)


def shuffle_dram_assignment(
    assignment: np.ndarray,
    num_banks: int,
    subblock_k: int | None = None,
    subblock_n: int = 1,
) -> np.ndarray:
    """Apply the same tile permutation as :func:`shuffle_dram_tiles` to a BSPM assignment array.

    Defaults (``subblock_k=K_tiles``, ``subblock_n=1``) reproduce the original
    permutation. ``subblock_n > 1`` mirrors the parametric block-grouped layout.

    Args:
        assignment: ``(tiles_h, tiles_w)`` int8 tile format codes in logical order.
        num_banks: Number of DRAM banks (``device.dram_grid_size().x``).
        subblock_k: K-tile dim of one matmul block (defaults to full K).
        subblock_n: N-tile dim of one matmul block (default 1).

    Returns:
        ``(tiles_h, tiles_w)`` int8 array in DRAM-shuffled order — same counts, reordered.
    """
    tiles_h, tiles_w = assignment.shape
    if tiles_w % num_banks != 0:
        raise ValueError(
            f"shuffle_dram_assignment requires tiles_w ({tiles_w}) to be evenly divisible "
            f"by num_banks ({num_banks})"
        )
    per_N_tiles = tiles_w // num_banks
    K_tiles = tiles_h
    num_tiles_per_shard = K_tiles * per_N_tiles

    if subblock_k is None:
        subblock_k = K_tiles

    if K_tiles % subblock_k != 0:
        raise ValueError(f"K_tiles ({K_tiles}) must be divisible by subblock_k ({subblock_k})")
    if per_N_tiles % subblock_n != 0:
        raise ValueError(f"per_N_tiles ({per_N_tiles}) must be divisible by subblock_n ({subblock_n})")
    # See note in shuffle_dram_tiles: block ordering inside the shard is
    # k_sub-outer / n_group-inner, which only matches the kernel's metadata walk
    # when one factor is 1. Disallow the both-split case here too.
    assert (K_tiles // subblock_k) == 1 or (per_N_tiles // subblock_n) == 1, (
        f"shuffle_dram_assignment: configs with both num_subblocks_k>1 ({K_tiles // subblock_k}) "
        f"and num_n_groups>1 ({per_N_tiles // subblock_n}) are not supported — block ordering "
        f"would not match the kernel's metadata walk. Pick subblock_k=K_tiles or subblock_n=per_N_tiles."
    )

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
        shard = assignment[:, b * per_N_tiles : (b + 1) * per_N_tiles].ravel()
        result[:, b * per_N_tiles : (b + 1) * per_N_tiles] = shard[source_idx].reshape(K_tiles, per_N_tiles)
    return result


def shared_down_torch_for_cache(
    down_proj_weights: torch.Tensor, moe_tp: int, mesh_shape: tuple[int, int]
) -> torch.Tensor:
    """Produce the torch tensor layout for shared expert down_proj cache storage."""
    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    N_down = N_per_core * dp_spec.NUM_MATMUL_CORES
    expected_down_shape = (K_down_per_device * moe_tp, N_down)
    assert (
        tuple(down_proj_weights.shape) == expected_down_shape
    ), f"down_proj_weights must be {expected_down_shape}, got {tuple(down_proj_weights.shape)}"
    if moe_tp == 1:
        return down_proj_weights.contiguous()
    mesh_rows, mesh_cols = mesh_shape
    return (
        down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
        .permute(0, 2, 1, 3)
        .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
    ).contiguous()


def moe_routed_expert_torch_for_cache(w: torch.Tensor, num_banks: int) -> torch.Tensor:
    """Match the tensor passed to ``from_torch`` for one MoE routed expert (one projection).

    Returns shape ``(1, 1, K, N_padded)``.
    """
    tile_w = 32
    K, N = w.shape
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    if N_padded != N:
        w = torch.nn.functional.pad(w, (0, N_padded - N))
    w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
    return w_shuffled.reshape(1, 1, K, N_padded).contiguous()


def moe_routed_expert_bspm_tp8_torch_for_cache(
    w: torch.Tensor,
    assignment: np.ndarray | None,
    num_banks: int,
    mesh_shape: tuple[int, int],
    shard_dim: int,
    subblock_k: int | None = None,
    subblock_n: int = 1,
) -> tuple[torch.Tensor, np.ndarray]:
    """TP-shard a routed expert and matching BSPM assignment across a 2D mesh.

    When ``assignment`` is None, a uniform BFP4 assignment is generated. Returned
    tensors are already DRAM-bank shuffled per TP slice, and the assignment is
    laid out to match ``CompressedTensor``'s folded view of the stacked tensor.
    """
    tile_w = 32
    mesh_rows, mesh_cols = mesh_shape
    tp = mesh_rows * mesh_cols
    K, N = w.shape
    if K % tile_w != 0 or N % tile_w != 0:
        raise ValueError(f"Routed expert weight shape {(K, N)} must be tile-aligned")

    if assignment is None:
        assignment = np.ones((K // tile_w, N // tile_w), dtype=np.int8)
    else:
        assignment = np.asarray(assignment, dtype=np.int8)
        if assignment.shape[0] != K // tile_w:
            raise ValueError(f"BSPM assignment rows {assignment.shape[0]} do not match K tiles {K // tile_w}")
        if assignment.shape[1] < N // tile_w:
            raise ValueError(f"BSPM assignment cols {assignment.shape[1]} do not cover N tiles {N // tile_w}")

    def _pad_assignment_cols(a: np.ndarray, target_cols: int) -> np.ndarray:
        if a.shape[1] == target_cols:
            return np.ascontiguousarray(a)
        if a.shape[1] > target_cols:
            return np.ascontiguousarray(a[:, :target_cols])
        return np.pad(a, ((0, 0), (0, target_cols - a.shape[1])), constant_values=3).astype(np.int8)

    shuffled_slices = []
    shuffled_assignments = []
    if shard_dim == 1:
        assert N % tp == 0, f"N={N} must be divisible by tp={tp} for column-parallel shard"
        per_device_N = N // tp
        # ``N % tp == 0`` alone is not enough: we slice the assignment at tile
        # granularity (``col_start // tile_w``), so per_device_N must itself be
        # tile-aligned or the slice boundaries collapse onto the wrong tiles.
        assert per_device_N % tile_w == 0, (
            f"per_device_N={per_device_N} must be a multiple of tile_w={tile_w} "
            f"(N={N}, tp={tp}); N % tp == 0 alone is insufficient"
        )
        K_per_device = K
        N_padded = ((per_device_N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
        assignment_cols = N_padded // tile_w
        for tp_idx in range(tp):
            col_start = tp_idx * per_device_N
            col_end = col_start + per_device_N
            slc = w[:, col_start:col_end].contiguous()
            if N_padded != per_device_N:
                slc = torch.nn.functional.pad(slc, (0, N_padded - per_device_N))
            shuffled_slices.append(
                shuffle_dram_tiles(
                    slc.unsqueeze(0), tile_w, num_banks, subblock_k=subblock_k, subblock_n=subblock_n
                ).reshape(K_per_device, N_padded)
            )

            tile_col_start = col_start // tile_w
            tile_col_end = col_end // tile_w
            assignment_slice = assignment[:, tile_col_start:tile_col_end]
            assignment_slice = _pad_assignment_cols(assignment_slice, assignment_cols)
            shuffled_assignments.append(
                shuffle_dram_assignment(assignment_slice, num_banks, subblock_k=subblock_k, subblock_n=subblock_n)
            )
    elif shard_dim == 0:
        assert K % tp == 0, f"K={K} must be divisible by tp={tp} for row-parallel shard"
        per_device_K = K // tp
        # Row-parallel slices the assignment at tile granularity along K
        # (``row_start // tile_w``); per_device_K must be tile-aligned for the
        # slice boundaries to land on tile edges.
        assert per_device_K % tile_w == 0, (
            f"per_device_K={per_device_K} must be a multiple of tile_w={tile_w} "
            f"(K={K}, tp={tp}); K % tp == 0 alone is insufficient"
        )
        K_per_device = per_device_K
        N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
        assignment_cols = N_padded // tile_w
        assignment_padded = _pad_assignment_cols(assignment, assignment_cols)
        for tp_idx in range(tp):
            row_start = tp_idx * per_device_K
            row_end = row_start + per_device_K
            slc = w[row_start:row_end, :].contiguous()
            if N_padded != N:
                slc = torch.nn.functional.pad(slc, (0, N_padded - N))
            shuffled_slices.append(
                shuffle_dram_tiles(
                    slc.unsqueeze(0), tile_w, num_banks, subblock_k=subblock_k, subblock_n=subblock_n
                ).reshape(K_per_device, N_padded)
            )

            tile_row_start = row_start // tile_w
            tile_row_end = row_end // tile_w
            assignment_slice = assignment_padded[tile_row_start:tile_row_end, :]
            shuffled_assignments.append(
                shuffle_dram_assignment(assignment_slice, num_banks, subblock_k=subblock_k, subblock_n=subblock_n)
            )
    else:
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}")

    stacked = torch.stack(shuffled_slices).reshape(mesh_rows, mesh_cols, K_per_device, N_padded)
    stacked_assignment = (
        np.stack(shuffled_assignments)
        .reshape(mesh_rows, mesh_cols, K_per_device // tile_w, N_padded // tile_w)
        .reshape(mesh_rows * mesh_cols * (K_per_device // tile_w), N_padded // tile_w)
    )

    # B/E drift check (defensive): per-rank shuffles are permutations and the only
    # cells the helper introduces are pad cells (always code 3).  So the sum of
    # the output assignment must equal the sum of the *used* input portion plus
    # 3*pad_cells_added.  A mismatch means the slicing math went wrong.
    input_used = np.asarray(assignment)[: K // tile_w, : N // tile_w]
    pad_cells_added = stacked_assignment.size - input_used.size
    expected_sum = int(input_used.sum()) + 3 * pad_cells_added
    actual_sum = int(stacked_assignment.sum())
    if actual_sum != expected_sum:
        logger.warning(
            "BSPM TP8 assignment drift detected: input_sum={}, pad_cells_added={}, "
            "expected_output_sum={}, actual_output_sum={} (shard_dim={}, mesh_shape={})",
            int(input_used.sum()),
            pad_cells_added,
            expected_sum,
            actual_sum,
            shard_dim,
            mesh_shape,
        )

    return stacked.contiguous(), np.ascontiguousarray(stacked_assignment)


def preprocess_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    moe_tp: int,
    mesh_rows: int,
    mesh_cols: int,
) -> dict[str, torch.Tensor]:
    """Reshuffle and TP-stack gate/up into fusion-ready tensors."""
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    per_device_n = cfg.gate_proj_shape[1]
    stacked_h, stacked_w = cfg.stacked_shape
    gate_list, up_list = [], []
    for tp_idx in range(moe_tp):
        gate_list.append(
            cfg.reshuffle_block_to_height_sharded(
                gate[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n], cfg.gate_core_range_set
            )
        )
        up_list.append(
            cfg.reshuffle_block_to_height_sharded(
                up[:, tp_idx * per_device_n : (tp_idx + 1) * per_device_n], cfg.up_core_range_set
            )
        )

    def _stack(lst):
        if moe_tp == 1:
            return lst[0]
        return (
            torch.stack(lst)
            .reshape(mesh_rows, mesh_cols, stacked_h, stacked_w)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * stacked_h, mesh_cols * stacked_w)
            .contiguous()
        )

    return {"shared_gate_proj": _stack(gate_list), "shared_up_proj": _stack(up_list)}


def fuse_gate_up(
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[OverlappedTensor, OverlappedTensor, ttnn.Tensor]:
    """Fuse shared-expert gate/up projections and prepare down projection."""
    _, moe_tp = _tp_factors(device)
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC

    mesh_rows = device.shape[0] if device.get_num_devices() > 1 else 1
    mesh_cols = device.shape[1] if device.get_num_devices() > 1 else 1

    preprocessed = preprocess_gate_up(gate_proj_weights, up_proj_weights, moe_tp, mesh_rows, mesh_cols)
    gate_preprocessed = preprocessed["shared_gate_proj"]
    up_preprocessed = preprocessed["shared_up_proj"]

    gate_up_dict = overlap_tensors(
        [
            OverlapEntry(
                "gate_proj",
                gate_preprocessed,
                replace(
                    cfg.gate_shard_spec,
                    raw_tensor_shape=tuple(gate_preprocessed.shape),
                    dtype=dtype,
                    logical_tensor_shape=cfg.gate_proj_shape,
                ),
            ),
            OverlapEntry(
                "up_proj",
                up_preprocessed,
                replace(
                    cfg.up_shard_spec,
                    raw_tensor_shape=tuple(up_preprocessed.shape),
                    dtype=dtype,
                    logical_tensor_shape=cfg.up_proj_shape,
                ),
            ),
        ],
        device=device,
        move_to_device=move_to_device,
    )
    gate_ov = gate_up_dict["gate_proj"]
    up_ov = gate_up_dict["up_proj"]

    dp_spec = DOWN_PROJ_SINGLE_DEVICE_SPEC
    K_down_per_device = 256
    N_per_core = 64
    N_down = N_per_core * dp_spec.NUM_MATMUL_CORES
    matmul_core_grid = dp_spec.build_matmul_core_grid()

    if moe_tp == 1:
        dp_combined = down_proj_weights
        dp_mapper = ttnn.ReplicateTensorToMesh(device)
    else:
        dp_combined = (
            down_proj_weights.reshape(mesh_rows, mesh_cols, K_down_per_device, N_down)
            .permute(0, 2, 1, 3)
            .reshape(mesh_rows * K_down_per_device, mesh_cols * N_down)
        )
        dp_mapper = ttnn.ShardTensor2dMesh(device, mesh_shape=(mesh_rows, mesh_cols), dims=(0, 1))
    device_dp = device if move_to_device else None

    dp_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down_per_device, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
    dp_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, dp_shard_spec)

    down_tensor = ttnn.from_torch(
        dp_combined,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device_dp,
        memory_config=dp_mem,
        tile=ttnn.Tile([32, 32]),
        mesh_mapper=dp_mapper,
    )

    return gate_ov, up_ov, down_tensor


def create_moe_routed_expert_tensors(
    gate_proj_weights: torch.Tensor,
    up_proj_weights: torch.Tensor,
    down_proj_weights: torch.Tensor,
    device,
    *,
    dtype: ttnn.DataType = ttnn.bfloat4_b,
    move_to_device: bool = True,
) -> tuple[list[ttnn.Tensor], list[ttnn.Tensor], list[ttnn.Tensor]]:
    """Upload routed MoE expert gate/up/down weights as per-expert DRAM-sharded tensors."""
    tile_w = 32
    num_banks = device.dram_grid_size().x
    mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    device_for_torch = device if move_to_device else None

    def upload(expert_weights: torch.Tensor) -> list[ttnn.Tensor]:
        num_experts, K, N = expert_weights.shape
        N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
        per_core_N = N_padded // num_banks

        dram_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(device.dram_grid_size().x - 1, device.dram_grid_size().y - 1),
                )
            }
        )
        shard_spec = ttnn.ShardSpec(dram_grid, [K, per_core_N], ttnn.ShardOrientation.ROW_MAJOR)
        mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

        tensors = []
        for i in range(num_experts):
            w = expert_weights[i]
            if N_padded != N:
                w = torch.nn.functional.pad(w, (0, N_padded - N))

            w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
            w_shuffled = w_shuffled.reshape(1, 1, K, N_padded)

            tensors.append(
                ttnn.from_torch(
                    w_shuffled.contiguous(),
                    dtype=dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=device_for_torch,
                    memory_config=mem_config,
                    mesh_mapper=mesh_mapper,
                )
            )
            if (i + 1) % 32 == 0:
                logger.info(f"  Uploaded {i + 1}/{num_experts} experts")
        return tensors

    return upload(gate_proj_weights), upload(up_proj_weights), upload(down_proj_weights)
