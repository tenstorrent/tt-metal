# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The DeepSeek-V3 MLA prefill KVPE cache declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

The single-config table's one cache, config "0", is a ``CacheConfig`` per pipeline stage: the tensor's
``KvLayoutSpec`` (its logical shape, dtype and DRAM ND-shard spec, and how its sequence is retained), the
mesh distribution its chunks really have, the writer's ``GenerationPolicy``, and the stage's mesh, layer
range and base address. ``ttnn.experimental.disaggregation.to_chunk_map`` derives the KV chunk address
table from the model.

The allocated tensor carries a 1-D Replicate topology (which sequence block a chip holds is decided at
write time), so the distribution is declared here instead: the sequence sharded over the SP rows,
replicated over the TP columns (a whole TP row is one replica group).

A stage's cache is ``[num_users * stage_layers, 1, seq_len, kvpe_dim]``, user-major, so ``to_chunk_map``
folds the stage's layers into the batch axis. The writer's layout is the Natural scheme with the Identity
bank order: 32-token blocks round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp,
+chunk/sp)`` of each prefill chunk n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``).
"""

import ttnn
from models.demos.common.prefill.runners.cache_layout import CacheLayoutNotDerivable, row_major_mesh_id
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    BH_NUM_DRAM_BANKS,
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    kvpe_cache_memory_config,
)

disaggregation = ttnn.experimental.disaggregation

# The cache's ND-shard grid is the BH_NUM_DRAM_BANKS Blackhole DRAM banks.
CACHE_ARCH = ttnn.device.Arch.BLACKHOLE

# The single-config table's one config.
KVPE_CONFIG_NAME = "0"


def _spec(kvpe_cache, *, num_users, stage_layers, seq_len):
    kvpe_dim = kvpe_cache.shape[-1]
    shape = ttnn.Shape([num_users * stage_layers, 1, seq_len, kvpe_dim])
    memory_config = kvpe_cache_memory_config(kvpe_dim, BH_NUM_DRAM_BANKS)
    tensor_spec = ttnn.TensorSpec(shape, kvpe_cache.dtype, kvpe_cache.layout, memory_config)
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), CACHE_ARCH)


def _replicated_topology(sp, tp):
    """The sequence sharded over the SP rows, replicated over the TP columns."""
    shape = ttnn.MeshShape(sp, tp)
    coords = [ttnn.MeshCoordinate(row, col) for row in range(sp) for col in range(tp)]
    return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementReplicate()], coords)


def deepseek_cache_layout_model(
    *,
    kvpe_cache,
    stage_layout,
    mesh_shape,
    sp_axis,
    num_users,
    seq_len,
    chunk_size,
    kv_dedup,
    layer_rows=None,
):
    """The ``CacheLayoutModel`` of the KVPE cache over ``stage_layout`` (the gathered pipeline stages, each
    with ``first_layer``, ``count``, ``base_addr``, ``num_banks`` and ``fnids[row][col]``). ``kvpe_cache``
    is this rank's cache tensor (its format and row width hold on every stage).

    Raises CacheLayoutNotDerivable for stages that were not gathered, a TP-deduped cache (its sequence is
    sharded over both mesh axes), layers published at remapped rows, an SP axis other than the mesh rows, a
    cache format other than bfp8 TILE, a stage whose chips are not numbered row-major, or a bank count other
    than the cache's."""
    if stage_layout is None:
        raise CacheLayoutNotDerivable("the stages were not gathered (the walk gathers them inline)")
    if kv_dedup:
        raise CacheLayoutNotDerivable("a TP-deduped cache shards its sequence over both mesh axes")
    if layer_rows is not None:
        raise CacheLayoutNotDerivable("the cache's layers are published at remapped table rows")
    if sp_axis != 0:
        raise CacheLayoutNotDerivable(f"the SP axis is mesh axis {sp_axis}, not the rows")
    if (kvpe_cache.dtype, kvpe_cache.layout) != (ttnn.bfloat8_b, ttnn.TILE_LAYOUT):
        raise CacheLayoutNotDerivable(f"a {kvpe_cache.dtype} {kvpe_cache.layout} cache is not the bfp8 TILE format")
    sp, tp = mesh_shape
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one DRAM block.
        k_chunk_size=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        device_chunk_size=chunk_size // sp,
    )
    topology = _replicated_topology(sp, tp)

    caches = []
    for stage in stage_layout:
        if stage["num_banks"] != BH_NUM_DRAM_BANKS:
            raise CacheLayoutNotDerivable(
                f"stage at layer {stage['first_layer']} has {stage['num_banks']} DRAM banks, "
                f"not the cache's {BH_NUM_DRAM_BANKS}"
            )
        caches.append(
            disaggregation.CacheConfig(
                _spec(kvpe_cache, num_users=num_users, stage_layers=stage["count"], seq_len=seq_len),
                topology,
                policy,
                stage["base_addr"],
                name=KVPE_CONFIG_NAME,
                layers=[],
                first_layer=stage["first_layer"],
                num_layers=stage["count"],
                mesh_id=row_major_mesh_id(stage["fnids"], sp, tp, f"stage at layer {stage['first_layer']}"),
            )
        )
    num_layers = sum(stage["count"] for stage in stage_layout)
    geometry = disaggregation.MapGeometry(
        num_layers, num_users, seq_len, position_step=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    return disaggregation.CacheLayoutModel(caches, caches[0].mesh_id, ttnn.MeshShape(sp, tp), geometry)
