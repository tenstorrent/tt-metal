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
replicated over the TP columns (a whole TP row is one replica group). A TP-deduped cache shards its
sequence over both mesh axes instead (row-major, SP outer), each chip its own group.

A stage's cache is ``[num_users * stage_layers, 1, seq_len, kvpe_dim]``, user-major, so ``to_chunk_map``
folds the stage's layers into the batch axis. The writer's layout is the Natural scheme with the Identity
bank order: 32-token blocks round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp,
+chunk/sp)`` of each prefill chunk n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``); deduped,
chip (r, c) owns ``[n*chunk + r*chunk/sp + c*chunk/(sp*tp), +chunk/(sp*tp))`` (``device_chunk_size =
chunk/(sp*tp)``).

A compacted cache (``layer_rows``: a hybrid stack whose cache holds only its MLA layers, its stages numbered
in compacted slab space) holds just the model layers it publishes, slot x those layers on its batch axis.
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


def _sequence_topology(sp, tp, *, kv_dedup):
    """The sequence sharded over the SP rows, and over the TP columns too when ``kv_dedup`` (replicated
    over them otherwise)."""
    shape = ttnn.MeshShape(sp, tp)
    coords = [ttnn.MeshCoordinate(row, col) for row in range(sp) for col in range(tp)]
    tp_placement = ttnn.PlacementShard(2) if kv_dedup else ttnn.PlacementReplicate()
    return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), tp_placement], coords)


def _stage_model_layers(stage, layer_rows):
    """The model layers ``stage`` holds: its layer range, or with ``layer_rows`` the rows its compacted slabs
    publish at."""
    first, count = stage["first_layer"], stage["count"]
    if layer_rows is None:
        return list(range(first, first + count))
    return list(layer_rows[first : first + count])


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
    config_name=KVPE_CONFIG_NAME,
):
    """The ``CacheLayoutModel`` of the KVPE cache over ``stage_layout`` (the gathered pipeline stages, each
    with ``first_layer``, ``count``, ``base_addr``, ``num_banks`` and ``fnids[row][col]``). ``kvpe_cache``
    is this rank's cache tensor (its format and row width hold on every stage). ``kv_dedup`` declares the
    sequence sharded over the TP columns as well. ``layer_rows`` maps the stages' compacted slabs to the
    model layers they publish at; ``config_name`` names the table config.

    A stage holding no layers (a rank with no MLA layer) publishes nothing and gets no cache.

    Raises CacheLayoutNotDerivable for stages that were not gathered or hold no layer, layer rows not
    strictly ascending (a cache holds its published layers in layer order), an SP axis other than the mesh
    rows, a cache format other than bfp8 TILE, a stage whose chips are not numbered row-major, or a bank
    count other than the cache's."""
    if stage_layout is None:
        raise CacheLayoutNotDerivable("the stages were not gathered (the walk gathers them inline)")
    stages = [stage for stage in stage_layout if stage["count"] > 0]
    if not stages:
        raise CacheLayoutNotDerivable("no stage holds a layer")
    rows_ascend = layer_rows is None or all(a < b for a, b in zip(layer_rows, layer_rows[1:]))
    if not rows_ascend:
        raise CacheLayoutNotDerivable(f"the layer rows {list(layer_rows)} are not strictly ascending")
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
        device_chunk_size=chunk_size // (sp * tp if kv_dedup else sp),
    )
    topology = _sequence_topology(sp, tp, kv_dedup=kv_dedup)
    compacted = layer_rows is not None

    caches = []
    for stage in stages:
        if stage["num_banks"] != BH_NUM_DRAM_BANKS:
            raise CacheLayoutNotDerivable(
                f"stage at layer {stage['first_layer']} has {stage['num_banks']} DRAM banks, "
                f"not the cache's {BH_NUM_DRAM_BANKS}"
            )
        model_layers = _stage_model_layers(stage, layer_rows)
        caches.append(
            disaggregation.CacheConfig(
                _spec(kvpe_cache, num_users=num_users, stage_layers=stage["count"], seq_len=seq_len),
                topology,
                policy,
                stage["base_addr"],
                name=config_name,
                layers=model_layers if compacted else [],
                first_layer=model_layers[0],
                num_layers=model_layers[-1] + 1 - model_layers[0],
                mesh_id=row_major_mesh_id(stage["fnids"], sp, tp, f"stage at layer {stage['first_layer']}"),
                holds_only_published_layers=compacted,
            )
        )
    num_layers = max(layer_rows) + 1 if compacted else sum(stage["count"] for stage in stages)
    geometry = disaggregation.MapGeometry(
        num_layers, num_users, seq_len, position_step=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    return disaggregation.CacheLayoutModel(caches, caches[0].mesh_id, ttnn.MeshShape(sp, tp), geometry)
