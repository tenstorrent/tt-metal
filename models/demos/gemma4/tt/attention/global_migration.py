# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decode-compatible staging cache for Gemma 4 global-attention KV migration.

The ordinary prefill cache stores full, separate K and V rows in Hugging Face
channel order.  tt-blaze stores one compact row instead::

    [K_roped_rotary (128 channels) | V (512 channels)]

The helpers in this module build that representation while the post-norm,
post-RoPE K/V tensors are still live, then write it to a user-major,
global-layer-compact cache.  The staging cache deliberately uses the same
32-token DRAM NdShard layout as the common prefill migration tables: one
``[32, 640]`` chunk is contiguous in one DRAM bank.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

TILE = 32
HEAD_DIM = 512
ROTARY_DIM = 128
ROW_DIM = ROTARY_DIM + HEAD_DIM
TOKENS_PER_CHUNK = TILE


def global_layer_indices(layer_types) -> tuple[int, ...]:
    """Return semantic indices of Gemma full-attention layers."""
    return tuple(i for i, layer_type in enumerate(layer_types) if layer_type == "full_attention")


def interleave_perm(head_dim: int = HEAD_DIM) -> torch.Tensor:
    """HF NeoX channel order -> tt-blaze adjacent-pair RoPE order."""
    if head_dim % 2:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    half = head_dim // 2
    perm = torch.empty(head_dim, dtype=torch.long)
    perm[0::2] = torch.arange(half)
    perm[1::2] = torch.arange(half, head_dim)
    return perm


def merged_kv_perms(head_dim: int = HEAD_DIM, rotary_dim: int = ROTARY_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the K-rotary and V source-column orders used by tt-blaze."""
    if not 0 < rotary_dim < head_dim:
        raise ValueError(f"rotary_dim must be in (0, {head_dim}), got {rotary_dim}")
    if rotary_dim % TILE:
        raise ValueError(f"rotary_dim must be tile aligned, got {rotary_dim}")
    # The HF NeoX halves span the full 512-wide head. Consequently the active
    # 128 device lanes come from HF ids [0..63, 256..319], interleaved as
    # 0,256,1,257,... — exactly tt-blaze's merged_kv_perms implementation.
    perm = interleave_perm(head_dim)
    k_rotary = perm[:rotary_dim]
    nonrot_hf = torch.tensor(sorted(perm[rotary_dim:].tolist()), dtype=torch.long)
    rot_hf = torch.tensor(sorted(k_rotary.tolist()), dtype=torch.long)
    return k_rotary, torch.cat((nonrot_hf, rot_hf))


def pack_global_kv_reference(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Host reference for the exact 640-channel migration row."""
    if k.shape != v.shape or k.shape[-1] != HEAD_DIM:
        raise ValueError(f"expected equal K/V shapes ending in {HEAD_DIM}, got {tuple(k.shape)} and {tuple(v.shape)}")
    k_cols, v_cols = merged_kv_perms()
    return torch.cat((k[..., k_cols], v[..., v_cols]), dim=-1)


@dataclass
class Gemma4GlobalMigrationCache:
    """Packed global-layer cache owned by the common prefill runner."""

    kv: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def allocate_global_migration_cache(
    mesh_device,
    *,
    num_users: int,
    num_layers: int,
    max_seq_len: int,
    sp_axis: int = 0,
    tp_axis: int = 1,
    dtype=ttnn.bfloat8_b,
) -> Gemma4GlobalMigrationCache:
    """Allocate a zeroed TP4/CP-sharded decode-row staging cache."""
    if dtype != ttnn.bfloat8_b:
        raise ValueError(f"initial Gemma 4 migration support requires bfloat8_b, got {dtype}")
    sp = int(mesh_device.shape[sp_axis])
    tp = int(mesh_device.shape[tp_axis])
    if tp != 4:
        raise ValueError(f"Gemma 4 global migration requires TP=4, got {tp}")
    if max_seq_len % (sp * TOKENS_PER_CHUNK):
        raise ValueError(
            f"max_seq_len={max_seq_len} must be divisible by SP*{TOKENS_PER_CHUNK} ({sp * TOKENS_PER_CHUNK})"
        )
    if num_users <= 0 or num_layers <= 0:
        raise ValueError(f"num_users and num_layers must be positive, got {num_users}, {num_layers}")

    local_shape = ttnn.Shape([num_users * num_layers, 1, max_seq_len // sp, ROW_DIM])
    bank_grid = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(bank, 0), ttnn.CoreCoord(bank, 0))
            for bank in range(get_num_dram_banks(mesh_device))
        ]
    )
    memory_config = ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=ttnn.NdShardSpec(
            shard_shape=[1, 1, TOKENS_PER_CHUNK, ROW_DIM],
            grid=bank_grid,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        ),
    )
    cache = ttnn.allocate_tensor_on_device(local_shape, dtype, ttnn.TILE_LAYOUT, mesh_device, memory_config)
    DRAMZeroFill.op(cache)

    placements = [None, None]
    placements[sp_axis] = ttnn.PlacementShard(2)
    placements[tp_axis] = ttnn.PlacementShard(1)
    dist_shape = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [
        ttnn.MeshCoordinate([coord[i] for i in range(coord.dims())]) for coord in ttnn.MeshCoordinateRange(dist_shape)
    ]
    cache.update_tensor_topology(ttnn.TensorTopology(dist_shape, placements, coords))
    return Gemma4GlobalMigrationCache(cache, num_users, num_layers, max_seq_len, sp)


_PACK_INDEX_CACHE = {}


def _pack_indices(mesh_device, seq_len: int):
    key = (id(mesh_device), seq_len)
    cached = _PACK_INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    k_cols, v_cols = merged_kv_perms()

    def _index(cols):
        host = cols.to(torch.uint32).reshape(1, 1, 1, -1).expand(1, 1, seq_len, -1).contiguous()
        return ttnn.from_torch(
            host,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    cached = (_index(k_cols), _index(v_cols))
    _PACK_INDEX_CACHE[key] = cached
    return cached


def pack_global_kv_device(k: ttnn.Tensor, v: ttnn.Tensor) -> ttnn.Tensor:
    """Build the tt-blaze 640-channel row without a host round trip.

    ``ttnn.gather`` applies the exact same fixed index vectors as the host
    reference.  Index tensors are cached per local chunk length and reused.
    """
    if tuple(k.shape) != tuple(v.shape) or int(k.shape[-1]) != HEAD_DIM:
        raise ValueError(f"expected equal K/V shapes ending in {HEAD_DIM}, got {k.shape} and {v.shape}")
    if int(k.shape[0]) != 1 or int(k.shape[1]) != 1:
        raise ValueError(f"migration pack expects one local KV head, got shape {k.shape}")

    k_input = k if k.dtype == ttnn.bfloat16 else ttnn.typecast(k, ttnn.bfloat16)
    v_input = v if v.dtype == ttnn.bfloat16 else ttnn.typecast(v, ttnn.bfloat16)
    k_index, v_index = _pack_indices(k.device(), int(k.shape[-2]))
    k_rotary = ttnn.gather(k_input, -1, k_index, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v_ordered = ttnn.gather(v_input, -1, v_index, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    merged = ttnn.concat([k_rotary, v_ordered], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    k_rotary.deallocate(True)
    v_ordered.deallocate(True)
    if k_input is not k:
        k_input.deallocate(True)
    if v_input is not v:
        v_input.deallocate(True)
    return merged


def write_global_kv_chunk(
    cache: Gemma4GlobalMigrationCache,
    k: ttnn.Tensor,
    v: ttnn.Tensor,
    *,
    slot_idx: int,
    layer_idx: int,
    kv_actual: int,
    valid_global: int,
    sp_axis: int = 0,
) -> None:
    """Pack and append one CP-sharded global-layer chunk."""
    if not 0 <= slot_idx < cache.num_users:
        raise ValueError(f"slot_idx {slot_idx} outside [0, {cache.num_users})")
    if not 0 <= layer_idx < cache.num_layers:
        raise ValueError(f"layer_idx {layer_idx} outside [0, {cache.num_layers})")
    if kv_actual % TOKENS_PER_CHUNK:
        raise ValueError(f"kv_actual must be {TOKENS_PER_CHUNK}-token aligned, got {kv_actual}")
    chunk_size_global = int(k.shape[-2]) * cache.sp
    if not kv_actual < valid_global <= kv_actual + chunk_size_global:
        raise ValueError(
            f"valid_global={valid_global} must be within chunk " f"({kv_actual}, {kv_actual + chunk_size_global}]"
        )

    merged = pack_global_kv_device(k, v)
    src = merged if merged.dtype == cache.kv.dtype else ttnn.typecast(merged, cache.kv.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache.kv,
        src,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=cache.num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if valid_global < kv_actual + chunk_size_global:
        # Migration moves a whole final 32-token chunk. Do not publish K/V
        # computed from pad-token embeddings in its unused tail.
        ttnn.experimental.deepseek_prefill.zero_padded_kv_cache(
            cache.kv,
            slot_idx,
            layer_idx,
            cache.num_layers,
            valid_global,
            chunk_size_global,
            sp_axis,
            TOKENS_PER_CHUNK,
        )
    if src is not merged:
        src.deallocate(True)
    merged.deallocate(True)
