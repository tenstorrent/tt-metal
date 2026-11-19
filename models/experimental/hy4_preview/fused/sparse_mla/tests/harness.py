# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test fixtures of the sparse-MLA piece: a packed KV storage built with the prefill stack's own pack ops and laid
out in DRAM the way the prefill stack allocates it, selection lists, and the host read-back."""

from __future__ import annotations

import os

import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheGeometry, reconstruct_scaled_fp8_kv_cache
from models.experimental.hy4_preview.fused.sparse_mla.layout import SparseMlaConfig, selection_table

MOCK = bool(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"))


def to_dev(t, device, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def kv_memory_config(device, cfg: SparseMlaConfig):
    """The prefill stack's KV memory config (utils/kv_cache_utils.py init_kvpe_cache): 32-row shards round-robin
    over the chip's DRAM banks."""
    banks = device.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, cfg.shard_rows, cfg.row_bytes],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def pack_rows(device, latent: torch.Tensor, rope: torch.Tensor):
    """The prefill stack's pack: latent [B, 1, T, 512], rope [B, 1, T, 64] (bf16 values) -> packed device tensor
    [B, 1, T, 656] fp8_e4m3 ROW_MAJOR (MlaKvCache._pack_scaled_fp8: per_token_cast_to_fp8 with power-of-two scales,
    then pack_scaled_fp8_kv_cache)."""
    tt_latent = to_dev(latent.to(torch.bfloat16), device, ttnn.bfloat16)
    tt_rope = to_dev(rope.to(torch.bfloat16), device, ttnn.bfloat16)
    fp8, scales = ttnn.experimental.deepseek_prefill.per_token_cast_to_fp8(tt_latent, round_scale_to_power_of_two=True)
    packed = ttnn.experimental.deepseek_prefill.pack_scaled_fp8_kv_cache(fp8, scales, tt_rope)
    for t in (tt_latent, tt_rope, fp8, scales):
        ttnn.deallocate(t)
    return packed


def make_storage(device, cfg: SparseMlaConfig, slots: int, latent: torch.Tensor, rope: torch.Tensor):
    """Storage [slots, 1, R, 656] holding latent [slots, R, 512] | rope [slots, R, 64], ND-sharded like the
    prefill cache. Returns (device tensor, logical rows [slots, R, 576] bf16 as the prefill stack's host reader
    decodes them)."""
    r = cfg.rows_per_slot
    assert latent.shape == (slots, r, cfg.latent) and rope.shape == (slots, r, cfg.rope)
    packed = pack_rows(device, latent[:, None], rope[:, None])
    storage = ttnn.to_memory_config(packed, kv_memory_config(device, cfg))
    ttnn.deallocate(packed)
    return storage, read_storage(storage, device, cfg)


def read_storage(storage, device, cfg: SparseMlaConfig) -> torch.Tensor:
    """Host read-back through the prefill stack's unpack: [slots, R, 576] bf16."""
    physical = ttnn.to_torch(storage, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    geometry = MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope)
    return reconstruct_scaled_fp8_kv_cache(physical, geometry)[:, 0]


def fp8_subnormal(storage, device, cfg: SparseMlaConfig) -> torch.Tensor:
    """bool [slots, R, 576]: latent values whose stored FP8 e4m3 code is subnormal (0 < |code value| < 2**-6).
    The host unpack decodes them; the device's unpacker flushes them to zero (a Tensix register holds no
    subnormals), so these are the only values a row read on the device may differ in."""
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import unpack_scaled_fp8_kv_cache

    physical = ttnn.to_torch(storage, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))
    fp8, _, rope = unpack_scaled_fp8_kv_cache(physical, MlaKvCacheGeometry(latent_dim=cfg.latent, rope_dim=cfg.rope))
    sub = (fp8 != 0) & (fp8.abs() < 2.0**-6)
    return torch.cat([sub, torch.zeros_like(rope, dtype=torch.bool)], dim=-1)[:, 0]


def random_latents(slots: int, rows: int, cfg: SparseMlaConfig, seed: int):
    """Latents with block magnitudes spread over three decades (so every scale block gets its own scale) and a
    unit-scale RoPE part."""
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(slots, rows, cfg.latent, generator=gen)
    latent *= torch.logspace(-2, 0.5, cfg.scale_blocks).repeat_interleave(cfg.scale_block)
    rope = torch.randn(slots, rows, cfg.rope, generator=gen)
    return latent, rope


def make_selection(device, cfg: SparseMlaConfig, lists):
    table = selection_table(cfg, lists)
    return to_dev(table[None, None], device, ttnn.uint32)


def spread_ids(cfg: SparseMlaConfig, n: int, seed: int) -> torch.Tensor:
    """n distinct rows drawn over the whole slab, unsorted: rows of many DRAM pages and banks."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(cfg.rows_per_slot, generator=gen)[:n]
