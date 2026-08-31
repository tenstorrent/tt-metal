# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed full-history staging cache for Gemma 4 sliding-attention KV."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

TILE = 32
HEAD_DIM = 256
NUM_KV_HEADS = 16
TOKENS_PER_CHUNK = TILE
HEADS_PER_TP_DEVICE = 4


def sliding_layer_indices(layer_types) -> tuple[int, ...]:
    return tuple(i for i, layer_type in enumerate(layer_types) if layer_type == "sliding_attention")


def sliding_k_perm(head_dim: int = HEAD_DIM) -> torch.Tensor:
    """HF NeoX K order -> tt-blaze adjacent-pair full-RoPE order."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


def pack_sliding_k_reference(k: torch.Tensor) -> torch.Tensor:
    if k.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected K width {HEAD_DIM}, got {k.shape[-1]}")
    return k[..., sliding_k_perm()]


@dataclass
class Gemma4SlidingMigrationCache:
    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    heads_per_device: int = HEADS_PER_TP_DEVICE


def allocate_sliding_migration_cache(
    mesh_device,
    *,
    num_users: int,
    num_layers: int,
    max_seq_len: int,
    sp_axis: int = 0,
    tp_axis: int = 1,
    dtype=ttnn.bfloat8_b,
) -> Gemma4SlidingMigrationCache:
    """Allocate separate full-history K/V caches in migration-friendly NdShard DRAM."""
    if dtype != ttnn.bfloat8_b:
        raise ValueError(f"Gemma 4 sliding migration requires bfloat8_b, got {dtype}")
    sp = int(mesh_device.shape[sp_axis])
    tp = int(mesh_device.shape[tp_axis])
    if tp != 4:
        raise ValueError(f"Gemma 4 sliding migration requires TP=4, got {tp}")
    if NUM_KV_HEADS % tp:
        raise ValueError(f"{NUM_KV_HEADS} sliding KV heads cannot shard over TP={tp}")
    if max_seq_len % (sp * TOKENS_PER_CHUNK):
        raise ValueError(
            f"sliding cache length {max_seq_len} must be divisible by "
            f"SP*{TOKENS_PER_CHUNK} ({sp * TOKENS_PER_CHUNK})"
        )
    if num_users <= 0 or num_layers <= 0:
        raise ValueError(f"num_users and num_layers must be positive, got {num_users}, {num_layers}")

    heads_per_device = NUM_KV_HEADS // tp
    local_shape = ttnn.Shape([num_users * num_layers, heads_per_device, max_seq_len // sp, HEAD_DIM])
    bank_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0))
            for bank in range(get_num_dram_banks(mesh_device))
        ]
    )
    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, TOKENS_PER_CHUNK, HEAD_DIM],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )

    dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    placements = [None, None]
    placements[sp_axis] = ttnn.PlacementShard(2)
    placements[tp_axis] = ttnn.PlacementShard(1)
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())]) for coord in ttnn.MeshCoordinateRange(dist_shape)
    ]

    def _allocate():
        tensor = ttnn.allocate_tensor_on_device(local_shape, dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config)
        DRAMZeroFill.op(tensor)
        tensor.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
        return tensor

    return Gemma4SlidingMigrationCache(
        k=_allocate(),
        v=_allocate(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        heads_per_device=heads_per_device,
    )


_K_INDEX_CACHE = {}
_PAD_MASK_CACHE = {}


def _k_indices(mesh_device, seq_len: int, heads_per_device: int):
    key = (id(mesh_device), seq_len, heads_per_device)
    cached = _K_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    host = (
        sliding_k_perm()
        .to(torch.uint32)
        .reshape(1, 1, 1, HEAD_DIM)
        .expand(1, heads_per_device, seq_len, HEAD_DIM)
        .contiguous()
    )
    cached = ttnn.from_torch(
        host,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    _K_INDEX_CACHE[key] = cached
    return cached


def pack_sliding_k_device(k: ttnn.Tensor) -> ttnn.Tensor:
    if int(k.shape[-1]) != HEAD_DIM or int(k.shape[0]) != 1:
        raise ValueError(f"expected K shape [1, local_heads, seq, {HEAD_DIM}], got {k.shape}")
    heads = int(k.shape[1])
    k_input = k if k.dtype == ttnn.bfloat16 else ttnn.typecast(k, ttnn.bfloat16)
    packed = ttnn.gather(
        k_input,
        -1,
        _k_indices(k.device(), int(k.shape[-2]), heads),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if k_input is not k:
        k_input.deallocate(True)
    return packed


def _pad_mask(mesh_device, *, seq_len: int, heads: int, valid_in_chunk: int):
    key = (id(mesh_device), seq_len, heads, valid_in_chunk)
    cached = _PAD_MASK_CACHE.get(key)
    if cached is not None:
        return cached
    rows, cols = tuple(mesh_device.shape)
    shards = []
    for row in range(rows):
        local_valid = max(0, min(seq_len, valid_in_chunk - row * seq_len))
        host = torch.zeros(1, heads, seq_len, HEAD_DIM, dtype=torch.bfloat16)
        host[:, :, :local_valid, :] = 1
        for _col in range(cols):
            shards.append(ttnn.from_torch(host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT))
    host_mesh = ttnn.from_host_shards(shards, mesh_device.shape)
    cached = ttnn.to_device(host_mesh, mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _PAD_MASK_CACHE[key] = cached
    return cached


def _write_one(cache, values, *, slot_idx: int, layer_idx: int, num_layers: int, kv_actual: int, sp_axis: int):
    src = values if values.dtype == cache.dtype else ttnn.typecast(values, cache.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        src,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if src is not values:
        src.deallocate(True)


def write_sliding_kv_chunk(
    cache: Gemma4SlidingMigrationCache,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    slot_idx: int,
    layer_idx: int,
    kv_actual: int,
    valid_global: int,
    sp_axis: int = 0,
) -> None:
    """Append one CP-sharded sliding K/V chunk and clear its padded tail."""
    if not 0 <= slot_idx < cache.num_users:
        raise ValueError(f"slot_idx {slot_idx} outside [0, {cache.num_users})")
    if not 0 <= layer_idx < cache.num_layers:
        raise ValueError(f"layer_idx {layer_idx} outside [0, {cache.num_layers})")
    if tuple(k.shape) != tuple(v.shape) or int(k.shape[1]) != cache.heads_per_device:
        raise ValueError(f"unexpected sliding K/V shapes: {k.shape}, {v.shape}")
    if kv_actual % TOKENS_PER_CHUNK:
        raise ValueError(f"kv_actual must be {TOKENS_PER_CHUNK}-token aligned, got {kv_actual}")

    chunk_size_global = int(k.shape[-2]) * cache.sp
    if cache.max_seq_len % chunk_size_global:
        raise ValueError(
            f"sliding cache length {cache.max_seq_len} must be divisible by chunk size {chunk_size_global}"
        )
    if kv_actual + chunk_size_global > cache.max_seq_len:
        raise ValueError(
            f"chunk ending at {kv_actual + chunk_size_global} exceeds sliding cache length {cache.max_seq_len}"
        )
    if not kv_actual < valid_global <= kv_actual + chunk_size_global:
        raise ValueError(
            f"valid_global={valid_global} must be within chunk " f"({kv_actual}, {kv_actual + chunk_size_global}]"
        )

    packed_k = pack_sliding_k_device(k)
    write_k = packed_k
    write_v = v
    if valid_global < kv_actual + chunk_size_global:
        mask = _pad_mask(
            k.device(),
            seq_len=int(k.shape[-2]),
            heads=cache.heads_per_device,
            valid_in_chunk=valid_global - kv_actual,
        )
        write_k = ttnn.mul(packed_k, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_input = v if v.dtype == ttnn.bfloat16 else ttnn.typecast(v, ttnn.bfloat16)
        write_v = ttnn.mul(v_input, mask, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if v_input is not v:
            v_input.deallocate(True)
    _write_one(
        cache.k,
        write_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
    _write_one(
        cache.v,
        write_v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
    packed_k.deallocate(True)
    if write_k is not packed_k:
        write_k.deallocate(True)
        write_v.deallocate(True)
