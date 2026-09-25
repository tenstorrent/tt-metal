# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Block-cyclic SP KV cache for Gemma-4 (GQA, multiple KV heads per chip).

Same DeepSeek chunked-KV DRAM layout as gpt_oss_d_p (per-chip ``[users*layers, n_kv_local, seq_local, D]``,
32-token ND shards round-robin over DRAM banks, written by ``update_padded_kv_cache``), generalised to
``n_kv_local`` heads per chip: at TP < n_kv a chip holds several KV heads (sliding layers: 8/TP).
Gemma-4 has two cache geometries (sliding 8x256, full 2x512), so the model allocates one cache per
layer type.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches
from models.demos.common.prefill.runners.migration import get_num_dram_banks

NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32


@dataclass
class Gemma4KVCache(KvCaches):
    k: ttnn.Tensor
    v: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    n_kv_local: int
    head_dim: int


def allocate_kv_cache(
    mesh_device, *, num_layers, max_seq_len, sp_axis=0, num_users=1, n_kv_local=1, head_dim=256, cache_dtype=ttnn.bfloat8_b
) -> Gemma4KVCache:
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % (ttnn.TILE_SIZE * sp) == 0, f"max_seq_len {max_seq_len} must be a multiple of 32*sp ({32 * sp})"
    seq_local = max_seq_len // sp
    core_ranges = [ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(get_num_dram_banks(mesh_device))]
    nd = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd)

    def _alloc():
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, n_kv_local, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return Gemma4KVCache(_alloc(), _alloc(), num_users, num_layers, max_seq_len, sp, n_kv_local, head_dim)
