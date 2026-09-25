# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block-cyclic SP KV cache for MiMo-V2 (GQA, separate K / V head dims).

DeepSeek chunked-KV DRAM layout (as gpt_oss_d_p / gemma4_d_p): per chip ``[users*layers, n_kv_local,
seq_local, D]``, 32-token ND shards round-robin over DRAM banks, written by ``update_padded_kv_cache``.
MiMo has K head_dim 192 and V head_dim 128 on both layer types. One cache per layer type (GA 4 KV heads,
SWA 8).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class MiMoKVCache(KvCaches):
    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    n_kv_local: int
    k_dim: int
    v_dim: int


def _cache_mem(mesh_device, head_dim):
    core_ranges = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(get_num_dram_banks(mesh_device))]
    nd = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd)


def allocate_kv_cache(
    mesh_device, *, num_layers, max_seq_len, n_kv_local, k_dim, v_dim, sp_axis=0, num_users=1, cache_dtype=ttnn.bfloat8_b
) -> MiMoKVCache:
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, f"max_seq_len {max_seq_len} must be a multiple of 32*sp ({32 * sp})"
    seq_local = max_seq_len // sp

    def _alloc(d):
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, n_kv_local, seq_local, d),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=_cache_mem(mesh_device, d),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return MiMoKVCache(_alloc(k_dim), _alloc(v_dim), num_users, num_layers, max_seq_len, sp, n_kv_local, k_dim, v_dim)
