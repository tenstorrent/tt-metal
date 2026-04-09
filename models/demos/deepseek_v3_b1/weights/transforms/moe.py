# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MoE / shared-expert weight preprocessing transforms and fusion helpers."""

from __future__ import annotations

from dataclasses import replace

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


def shuffle_dram_tiles(tensor: torch.Tensor, tile_size: int, num_banks: int) -> torch.Tensor:
    """Reorder tiles within each DRAM bank shard from row-major to column-major.

    WIDTH_SHARDED DRAM layout stores tiles row-major, but the streaming
    matmul kernel expects K tiles contiguous for each N column.  This
    function transposes the tile order within each shard so that the
    kernel can linearly read K tiles at a time.
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

    tensor = tensor.reshape(batch_size, K, num_banks, per_N)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    shards = tensor.reshape(-1, K, per_N)

    tiles = shards.reshape(-1, K_tiles, tile_size, per_N_tiles, tile_size)
    tiles = tiles.permute(0, 1, 3, 2, 4).contiguous()
    tiles = tiles.reshape(-1, num_tiles_per_shard, tile_size, tile_size)

    i = torch.arange(num_tiles_per_shard, device=tensor.device)
    source_idx = (i % K_tiles) * per_N_tiles + (i // K_tiles)
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


def mlp_routed_dense_stacked_torch_for_cache(
    experts: torch.Tensor, num_banks: int, mesh_shape: tuple[int, int]
) -> torch.Tensor:
    """Stacked torch before ``from_torch`` in dense MLP routed ``upload`` (all experts on mesh)."""
    tile_w = 32
    mesh_rows, mesh_cols = mesh_shape
    n_exp, K, N = experts.shape
    N_padded = ((N + num_banks * tile_w - 1) // (num_banks * tile_w)) * (num_banks * tile_w)
    processed = []
    for i in range(n_exp):
        w = experts[i]
        if N_padded != N:
            w = torch.nn.functional.pad(w, (0, N_padded - N))
        w_shuffled = shuffle_dram_tiles(w.unsqueeze(0), tile_w, num_banks)
        processed.append(w_shuffled.reshape(K, N_padded))
    stacked = torch.stack(processed).reshape(mesh_rows, mesh_cols, K, N_padded)
    return stacked.contiguous()


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
            ],
            [
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
