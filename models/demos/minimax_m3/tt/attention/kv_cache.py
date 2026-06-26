# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import torch

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK


@dataclass
class MiniMaxKVCache:
    """Externally-owned, user-major packed prefill KV caches for the SP chunked-KV path.

    Three persistent device caches, each per-chip shape ``[num_users*num_layers, 1, seq_local, head_dim]``
    (DeepSeek chunked-KV NdShard DRAM layout). They are written by
    ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``:

      * ``k``, ``v``   — main GQA K/V. Under TP=cols each chip holds 1 head (heads sharded on the TP cols
                         at write time); the sequence is SP-sharded block-cyclic on the ``sp`` rows.
      * ``index_k``    — MSA lightning-indexer key (a single shared head). Same per-chip shape; at write
                         time it is REPLICATED across the TP cols (one shared head, NOT head-sharded), and
                         only the MSA layers (3-59) populate it — dense layers leave their slots zeroed.

    The batch dim is user-major: ``slot = user_id * num_layers + layer_idx`` so each user's layers stay
    contiguous, matching ``update_padded_kv_cache``'s ``slot_idx`` / ``layer_idx`` indexing.

    STEP 1: allocation + plumbing only. The write (step 2) and the cache-read attention path (step 4)
    are not wired yet — the single-shot prefill still runs the no-cache forward for its logits.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    index_k: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
) -> MiniMaxKVCache:
    """Allocate the three external prefill KV caches (K, V, index_k). See :class:`MiniMaxKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (that is MLA-specific and allocates a single cache): this owns
    the M3 GQA triple and the user-major packing. It reuses the same DRAM NdShard spec (same bank grid +
    32-token contiguous shard) so ``update_padded_kv_cache`` can write into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 60). All three caches allocate all layers; only the MSA
            layers will write ``index_k`` (dense slots stay zeroed — capacity is cheap, packing stays simple).
        max_seq_len: per-user cache capacity in tokens, a multiple of ``sp``. ``seq_local = max_seq_len // sp``.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for M3 main K/V and the index head alike).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate + the device golden check).
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc():
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds (or whether index_k is
        # replicated across cols) is decided at write time by how the input chunk is mesh-mapped, not
        # here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer; content
        # diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=cache_dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    return MiniMaxKVCache(
        k=_alloc(),
        v=_alloc(),
        index_k=_alloc(),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
    )
