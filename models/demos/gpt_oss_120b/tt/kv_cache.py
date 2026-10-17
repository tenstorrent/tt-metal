# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV cache (serving_contract.md "KV cache"), after gpt_oss_d_p/tt/attention/kv_cache.py for 2 KV heads
per TP column.

Per chip (row r, column c) and per cache tensor (K, V): [n_batch, 2, max_seq / 8, 64] bfloat8_b TILE, DRAM ND-sharded
[1, 1, 32, 64] ROUND_ROBIN_1D over every bank, so one 32-token record of one head sits in one bank. Column c holds KV
heads 2c, 2c+1; row r holds the positions p with (p // 640) % 8 == r, at local row (p // 5120) * 640 + p % 640 (the
block-cyclic order update_padded_kv_cache writes). Batch slot = slot * num_layers + layer. K is post-RoPE in meta
(interleaved-pair) column order, V in checkpoint order. Full length on every layer, sliding ones too.
"""

import torch

import ttnn

RECORD_TOKENS = 32


def block_cyclic_reorder(t: torch.Tensor, chunk_local: int, sp: int, seq_dim: int = 2) -> torch.Tensor:
    """[.., seq, ..] natural order -> block-cyclic order: an SP-contiguous split of ``seq_dim`` then gives chip c the
    640-row blocks c, c + sp, c + 2 sp, ... (deepseek_v3_d_p mla/utils.block_cyclic_reorder)."""
    n = t.shape[seq_dim]
    assert n % (chunk_local * sp) == 0, f"seq {n} must be a multiple of chunk_local * sp = {chunk_local * sp}"
    blocks = list(torch.split(t, chunk_local, dim=seq_dim))
    order = [b for c in range(sp) for b in range(c, len(blocks), sp)]
    return torch.cat([blocks[b] for b in order], dim=seq_dim)


def cache_memory_config(mesh, head_dim: int):
    banks = mesh.dram_grid_size().x
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, RECORD_TOKENS, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)


def cache_from_host(mesh, kv: torch.Tensor, chunk: int) -> ttnn.Tensor:
    """Host [n_batch, n_kv, max_seq, 64] in natural position order -> the device cache (heads over TP, positions
    block-cyclic over SP). Zeros allocate an empty cache; the component harness loads a golden prefix with it."""
    rows, cols = tuple(mesh.shape)
    return ttnn.from_torch(
        block_cyclic_reorder(kv.float(), chunk // rows, rows),
        device=mesh,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        memory_config=cache_memory_config(mesh, kv.shape[-1]),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(2, 1)),
    )


def bfp8_round(t: torch.Tensor) -> torch.Tensor:
    """Host only (harness boundaries, load time): ``t`` rounded to bfloat8_b exactly as cache_from_host stores it
    (ttnn's host conversion, TILE layout: the shared exponents run along the last dim, which the reorder keeps)."""
    return ttnn.to_torch(ttnn.from_torch(t.float(), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
