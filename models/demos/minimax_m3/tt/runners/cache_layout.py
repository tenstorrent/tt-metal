# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The MiniMax-M3 prefill caches declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Each cache of each pipeline stage is a ``CacheConfig``: the tensor's ``KvLayoutSpec`` (its logical shape,
dtype and DRAM ND-shard spec, and how its sequence is retained), the mesh distribution its chunks really
have, the writer's ``GenerationPolicy``, and the stage's mesh, layer range and base address.
``ttnn.experimental.disaggregation.to_chunk_map`` derives the KV chunk address table from the model.

The allocated tensors carry a 1-D Replicate topology (which head or sequence block a chip holds is decided
at write time), so the distribution is declared here instead:

    configs 00..N-1   k head h:   the sequence sharded over the SP rows of TP column h
    configs N..2N-1   v head h:   likewise
    config  2N        index_k:    the sequence sharded over the SP rows, replicated over the TP columns

A stage's caches are ``[num_users * stage_layers, 1, seq_len, head_dim]``, user-major, so ``to_chunk_map``
folds the stage's layers into the batch axis. The writer's layout is the Natural scheme with the Identity
bank order: 32-token blocks round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp,
+chunk/sp)`` of each prefill chunk n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``).
"""

import ttnn
from models.demos.common.prefill.runners.cache_layout import CacheLayoutNotDerivable, row_major_mesh_id
from models.demos.minimax_m3.tt.attention.kv_cache import (
    BH_NUM_DRAM_BANKS,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    cache_memory_config,
)

disaggregation = ttnn.experimental.disaggregation

# The cache's ND-shard grid is the BH_NUM_DRAM_BANKS Blackhole DRAM banks.
CACHE_ARCH = ttnn.device.Arch.BLACKHOLE


def _spec(dtype, *, num_users, stage_layers, seq_len, head_dim):
    shape = ttnn.Shape([num_users * stage_layers, 1, seq_len, head_dim])
    tensor_spec = ttnn.TensorSpec(shape, dtype, ttnn.TILE_LAYOUT, cache_memory_config(head_dim))
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), CACHE_ARCH)


def _head_topology(sp, head):
    """K/V head ``head``: the sequence sharded over the SP rows of TP column ``head``."""
    coords = [ttnn.MeshCoordinate(row, head) for row in range(sp)]
    return ttnn.TensorTopology(ttnn.MeshShape([sp]), [ttnn.PlacementShard(2)], coords)


def _replicated_topology(sp, cols):
    """index_k: the sequence sharded over the SP rows, replicated over the TP columns."""
    shape = ttnn.MeshShape(sp, cols)
    coords = [ttnn.MeshCoordinate(row, col) for row in range(sp) for col in range(cols)]
    return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementReplicate()], coords)


def m3_cache_layout_model(
    *,
    kv_cache,
    stage_layouts,
    sp,
    cols,
    num_users,
    seq_len,
    chunk_size,
    num_kv_heads,
    head_dim,
    index_k_layers=None,
):
    """The ``CacheLayoutModel`` of the M3 prefill caches over ``stage_layouts`` (one layout per cache — k,
    v, index_k — each a list of pipeline stages with ``first_layer``, ``count``, ``base_addr``,
    ``num_banks`` and ``fnids[row][col]``; a single rank is one stage). ``index_k_layers`` limits index_k's
    rows to those model layers (the MSA layers), as the decode peer holds index_k only there. Config names
    are the zero-padded positions the decode table uses.

    Raises CacheLayoutNotDerivable for a TP-deduped index_k (its sequence is sharded over both mesh axes),
    a stage whose chips are not numbered row-major, or a bank count other than the cache's."""
    from models.demos.minimax_m3.tt.runners.kv_chunk_table import config_name

    if getattr(kv_cache, "index_k_tp_axis", None) is not None:
        raise CacheLayoutNotDerivable("a TP-deduped index_k shards its sequence over both mesh axes")
    assert num_kv_heads == cols, f"num_kv_heads ({num_kv_heads}) must equal the TP column count ({cols})"
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one DRAM block.
        k_chunk_size=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        device_chunk_size=chunk_size // sp,
    )
    num_configs = 2 * num_kv_heads + 1
    # (cache index into stage_layouts, dtype, topology) per config, in config-id order.
    configs = [(0, kv_cache.k.dtype, _head_topology(sp, h)) for h in range(num_kv_heads)]
    configs += [(1, kv_cache.v.dtype, _head_topology(sp, h)) for h in range(num_kv_heads)]
    configs.append((2, kv_cache.index_k.dtype, _replicated_topology(sp, cols)))
    index_k_published = sorted(index_k_layers) if index_k_layers is not None else []

    caches = []
    for config_id, (cache_idx, dtype, topology) in enumerate(configs):
        is_index_k = cache_idx == 2
        for stage in stage_layouts[cache_idx]:
            if stage["num_banks"] != BH_NUM_DRAM_BANKS:
                raise CacheLayoutNotDerivable(
                    f"stage at layer {stage['first_layer']} has {stage['num_banks']} DRAM banks, "
                    f"not the cache's {BH_NUM_DRAM_BANKS}"
                )
            caches.append(
                disaggregation.CacheConfig(
                    _spec(dtype, num_users=num_users, stage_layers=stage["count"], seq_len=seq_len, head_dim=head_dim),
                    topology,
                    policy,
                    stage["base_addr"],
                    name=config_name(config_id, num_configs),
                    layers=index_k_published if is_index_k else [],
                    first_layer=stage["first_layer"],
                    num_layers=stage["count"],
                    mesh_id=row_major_mesh_id(stage["fnids"], sp, cols, f"stage at layer {stage['first_layer']}"),
                )
            )
    num_layers = sum(stage["count"] for stage in stage_layouts[0])
    geometry = disaggregation.MapGeometry(
        num_layers, num_users, seq_len, position_step=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    return disaggregation.CacheLayoutModel(caches, caches[0].mesh_id, ttnn.MeshShape(sp, cols), geometry)
