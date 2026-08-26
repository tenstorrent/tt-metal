# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV chunk address table builder for the MLA prefill cache.

This is the MODEL-specific half of migration bring-up: build the KvChunkAddressTable
from the device KV layout and serialize it to a protobuf file. ``TtPrefillRuntime``
calls this via ``runtime.build_kv_chunk_table(path)``; the runner then publishes the
serialized table to the migration worker over the generic handshake in
``models.demos.common.prefill.runners.migration`` (the runner owns the comms).

Serialization uses the ttnn binding
``ttnn.experimental.disaggregation.export_to_protobuf_file`` (no separate _migration
extension needed).

NOTE: per-layer LayerAck channel + scheduler-driven migration are NOT here
(owned by the runner / scheduler / worker side).
"""

import ttnn
from models.demos.common.prefill.runners.migration import (
    allgather_kv_stage_layout,
    serialize_kv_chunk_table,
    serialize_prebuilt_kv_chunk_table,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_kimi,
    merged_num_layers,
    populate_kv_chunk_address_table_dflash,
    populate_kv_chunk_address_table_kimi,
)

# A KV chunk is one DRAM bank's worth of tokens (NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK=32) x head_dim.
_TILE_DIM = 32  # bfp8 is tiled 32x32
_BFP8_TILE_BYTES = 1088  # one 32x32 bfp8 tile: 1024 data + 64 exponent bytes


def dflash_config_name(kind: str, head_idx: int) -> str:
    """Table config name for one global kv-head of the DFlash drafter's K or V context cache.

    ``kind`` is "k" or "v". K names sort ahead of V names, so config-id order is
    k_h00..k_hN-1, v_h00..v_hN-1 — the same K-then-V order the M3 builder documents as the
    src<->dst migration contract."""
    assert kind in ("k", "v"), f"dflash config kind must be 'k' or 'v', got {kind!r}"
    return f"dflash_{kind}_h{head_idx:02d}"


def _dram_chunk_size_bytes(cache) -> int:
    """Bytes of one 32-token DRAM-bank chunk ([.., 32, head_dim]) of `cache`, from its dtype:
      * bfp8_b  (block-float, TILE):  (head_dim / 32) tiles x 1088 B/tile (1024 data + 64 exponent).
      * bfloat16/fp8_e4m3 (ROW_MAJOR): 32 native row pages, including any DRAM page alignment.
    Derived from the tensor so dense tiled-bfp8 KVPE, sparse BF16 or packed scaled-FP8 KVPE, and the
    tiled-bfp8 index cache each size themselves from their physical representation."""
    head_dim = cache.shape[-1]
    if cache.dtype == ttnn.bfloat8_b:
        # bfp8 is tiled 32x32, so head_dim must be a whole number of tiles — otherwise integer division
        # would silently undersize the chunk and corrupt the address table.
        if head_dim % _TILE_DIM != 0:
            raise ValueError(f"bfloat8_b KV cache head_dim {head_dim} must be a multiple of {_TILE_DIM} (tiled)")
        return (head_dim // _TILE_DIM) * _BFP8_TILE_BYTES
    if cache.dtype in (ttnn.bfloat16, ttnn.fp8_e4m3):
        # Each token is one native row-major buffer page. Use its physical aligned size rather than
        # head_dim * element_size: the migration worker copies raw DRAM bytes and must include padding.
        if cache.layout != ttnn.ROW_MAJOR_LAYOUT:
            raise ValueError(
                f"{cache.dtype} KV cache must be ROW_MAJOR for contiguous chunk sizing, got {cache.layout}"
            )
        return NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK * cache.buffer_aligned_page_size()
    raise ValueError(f"unsupported KV cache dtype for chunk sizing: {cache.dtype}")


def _num_layers_from_cache(cache, num_users: int) -> int:
    """Layer count a KV cache holds, recovered from its folded batch dim. init_kvpe_cache lays caches
    out user-major with shape[0] = num_users * num_layers, so dividing the batch dim by num_users gives
    this cache's layer count — all layers for the KVPE cache, full-layers-only for the GLM-5.2 index
    cache (which allocate_kv_cache sizes to num_full)."""
    return cache.shape[0] // num_users


def build_and_serialize_kv_chunk_table(
    *,
    mesh_device,
    kvpe_cache,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    chunk_size_global,
    path,
    index_kv_cache=None,
    dflash_caches=None,
    tp_axis=1,
    first_layer_idx=0,
    num_my_layers=None,
    stage_layouts=None,
    index_layer_ids=None,
) -> str:
    """Build the MLA block-cyclic KV chunk address table and serialize it to ``path`` for the
    inference server's SET_TABLE. Returns the path on success.

    Chunked prefill stores KV positions block-cyclic across the SP shards, so the table maps each
    natural position to its true storage chip + offset. The migration worker copies the chunks the
    table lists for the migrated range. A contiguous (wrong) table still works for a blanket copy of
    the WHOLE cache, but any sub-cache migration (a [0, N) prefix, or a prompt shorter than
    max_seq_len) lists the wrong, block-cyclically-scattered chunks and fails its PCC check.

    ``chunk_size_global`` is the block-cyclic period; the kimi builder hardcodes it as
    PREFILL_CHUNK_OUTPUT_TOKENS, so a non-default period is rejected here rather than mismapped.

    ``index_kv_cache`` (sparse/DSA models only): when given, a single MERGED table describes BOTH
    caches — config 0 = the KVPE cache, config 1 = the index-key cache — sharing one device-group
    side table. None (dense models) → the usual single-config table over the KVPE cache alone.

    ``dflash_caches`` (DFlash drafter only): ``(k_cache, v_cache)`` for the drafter's context-KV, which
    also joins the merged table — ``2 * num_kv_heads`` further configs, named by
    :func:`dflash_config_name`, because the table key is (layer, position, slot) with no head axis. The
    drafter's shapes carry everything else: layer count from ``shape[0] // num_users`` (user-major fold),
    ``num_kv_heads`` from ``shape[1] * tp`` (dim 1 is this chip's TP head slice), head_dim from
    ``shape[-1]``. Passing these turns even a DENSE model's table into a merged one.

    ``first_layer_idx`` / ``num_my_layers`` / ``stage_layouts`` (pipeline-parallel only): this rank owns
    layers [first_layer_idx, first_layer_idx + num_my_layers); ``stage_layouts`` holds ONE all-gathered
    per-stage layout per block-cyclic cache (config order), so rank 0 builds one table spanning every
    stage while the collectives ran on all ranks. Leave it None to gather inline (single-rank / tests)."""
    assert chunk_size_global == PREFILL_CHUNK_OUTPUT_TOKENS, (
        f"create_kv_chunk_address_table_kimi assumes a block-cyclic period of "
        f"PREFILL_CHUNK_OUTPUT_TOKENS={PREFILL_CHUNK_OUTPUT_TOKENS}, but chunk_size_global={chunk_size_global}. "
        f"A different period would mismap every position; re-introduce a parametrized builder if needed."
    )

    primary_cache = kvpe_cache.storage
    # One tagged list of every cache this rank owns, merged into ONE table downstream. The tag routes
    # each cache to its populate path: "kvpe" / "index" are block-cyclic MLA caches named positionally
    # (KVPE at config 0, index at 1); "dflash" is the drafter's (k, v) pair that fans out per-head.
    all_caches = [("kvpe", primary_cache)]
    if index_kv_cache is not None:
        all_caches.append(("index", index_kv_cache))
    if dflash_caches is not None:
        all_caches.append(("dflash", dflash_caches))
    if index_kv_cache is not None or dflash_caches is not None:
        return _build_and_serialize_merged_kv_chunk_table(
            mesh_device=mesh_device,
            caches=all_caches,
            seq_len=seq_len,
            num_layers=num_layers,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            num_users=num_users,
            chunk_size_global=chunk_size_global,
            path=path,
            stage_layouts=stage_layouts,
            index_layer_ids=index_layer_ids,
        )

    # Single config: the KVPE cache is the only one described, so its layout is the only one gathered.
    stage_layout = stage_layouts[0] if stage_layouts else None

    def _builder(*, config, chunk_size_bytes, num_users):
        return create_kv_chunk_address_table_kimi(
            config=config,
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            seq_len=seq_len,
            sp_axis=sp_axis,
            kvpe_cache=primary_cache,
            chunk_size_bytes=chunk_size_bytes,
            num_users=num_users,
            first_layer_idx=first_layer_idx,
            num_my_layers=num_my_layers,
            stage_layout=stage_layout,
        )

    return serialize_kv_chunk_table(
        table_builder=_builder,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_users=num_users,
        chunk_n_tokens=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        chunk_size_bytes=_dram_chunk_size_bytes(primary_cache),
        path=path,
    )


def _build_and_serialize_merged_kv_chunk_table(
    *,
    mesh_device,
    caches,
    seq_len,
    num_layers,
    mesh_shape,
    sp_axis,
    num_users,
    path,
    tp_axis=1,
    chunk_size_global=None,
    stage_layouts=None,
    index_layer_ids=None,
) -> str:
    """Build ONE KvChunkAddressTable over every cache this rank owns and serialize it to ``path``.
    ``caches`` is a tagged list of ``(kind, payload)``: ``("kvpe", tensor)`` / ``("index", tensor)`` for
    the block-cyclic MLA caches, named "0" (KVPE), "1" (GLM-5.2 index); ``("dflash", (k_cache, v_cache))``
    for the DFlash drafter (Kimi-only), which adds one config per (K|V, kv-head) via
    :func:`dflash_config_name`. Names must stay in sorted order (asserted) so the protobuf round-trip
    keeps KVPE at config id 0 and the index at 1 — see the naming note at the top of this module.

    ``index_layer_ids``: dense row -> global layer map; publishes config 1 on the LAYER axis so one
    layer number selects the same layer in every config. None keeps the compacted axis.

    ``stage_layouts`` is one all-gathered layout per block-cyclic cache, in the same order — each config
    needs its own, since a layout carries one cache's DRAM base and one layer-index space. Only rank 0
    reaches this function, so the gather cannot happen here; None gathers inline, correct only
    single-rank."""
    disagg = ttnn.experimental.disaggregation

    # (name, cache, head_idx); head_idx is None for the block-cyclic MLA caches ("kvpe" / "index"), the
    # global kv-head otherwise ("dflash" drafter). Block-cyclic entries come first so config names sort
    # globally (asserted below) with KVPE at id 0.
    entries = []
    dflash_kv_heads = 0
    n_block_cyclic = 0
    index_config_name = None
    for kind, payload in caches:
        if kind in ("kvpe", "index"):  # block-cyclic MLA caches -> populate_kv_chunk_address_table_kimi
            if kind == "index":
                index_config_name = str(n_block_cyclic)
            entries.append((str(n_block_cyclic), payload, None))
            n_block_cyclic += 1
        elif kind == "dflash":
            k_cache, v_cache = payload
            # Distinct allocations, else every V config aliases K's addresses (a same-address table
            # still looks well-formed). dim 1 is this chip's TP head slice, so the GLOBAL head count is
            # shape[1]*tp.
            assert (
                k_cache.buffer_address() != v_cache.buffer_address()
            ), "drafter K and V must be distinct allocations (same buffer => V configs alias K)"
            assert v_cache.shape == k_cache.shape, f"drafter V shape {v_cache.shape} != K shape {k_cache.shape}"
            dflash_kv_heads = k_cache.shape[1] * mesh_shape[tp_axis]
            entries += [
                (dflash_config_name(k_or_v, h), cache, h)
                for k_or_v, cache in (("k", k_cache), ("v", v_cache))
                for h in range(dflash_kv_heads)
            ]
        else:
            raise ValueError(f"unknown KV table cache kind: {kind!r} (expected 'kvpe', 'index', or 'dflash')")

    block_cyclic = [(name, cache) for name, cache, head_idx in entries if head_idx is None]
    if stage_layouts is None:
        stage_layouts = [
            allgather_kv_stage_layout(
                mesh_device,
                int(cache.buffer_address()),
                mesh_shape,
                first_layer_idx=0,
                num_my_layers=_num_layers_from_cache(cache, num_users),
            )
            for _, cache in block_cyclic
        ]
    if len(stage_layouts) != len(block_cyclic):
        raise RuntimeError(
            f"merged table has {len(block_cyclic)} block-cyclic caches but got {len(stage_layouts)} gathered "
            "stage layouts; the runtime must describe every config it merges (one KvCacheStage per cache, "
            "in config order)."
        )
    layout_of = {name: layout for (name, _), layout in zip(block_cyclic, stage_layouts)}

    # int(): the binding hands back a strong Rank type, which never compares equal to the plain int
    # the gathered stages are keyed by.
    my_rank = int(ttnn.distributed_context_get_rank())

    def _table_config(cache, stage_layout):
        cfg = disagg.KvChunkAddressTableConfig()
        if stage_layout is None:
            # Drafter cache: single-stage, so its layer count comes off the cache itself (the 6 draft
            # layers, user-major shape[0] // num_users).
            cfg.num_layers = _num_layers_from_cache(cache, num_users)
        else:
            # Match a layout to its cache by DRAM base, so a runtime returning its stages out of config
            # order is caught here instead of silently addressing one cache with the other's layout.
            base_addr = int(cache.buffer_address())
            if not any(s["rank"] == my_rank and s["base_addr"] == base_addr for s in stage_layout):
                raise RuntimeError(
                    f"gathered stage layout does not describe this cache: rank {my_rank} has no stage at "
                    f"{hex(base_addr)} in {[(s['rank'], hex(s['base_addr'])) for s in stage_layout]} — "
                    "stages are out of config order."
                )
            # Size the config to the GLOBAL layer total, summed over the gathered stages: the KVPE cache's
            # every layer, and the index cache's full-indexer layers only (GLM-5.2 cross-layer reuse — the
            # shared layers own no indexer slot; GLM-5.1 / dense have one per layer, so it equals num_layers).
            cfg.num_layers = merged_num_layers(stage_layout)
        cfg.max_sequence_length = seq_len
        cfg.num_slots = num_users
        cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        cfg.chunk_size_bytes = _dram_chunk_size_bytes(cache)
        return cfg

    names = [name for name, _, _ in entries]
    assert names == sorted(names), f"config names must already be sorted (protobuf renumbers by name): {names}"
    configs = {name: _table_config(cache, layout_of.get(name)) for name, cache, _ in entries}

    # Widen the index config to the layer axis before the table is built (extents are fixed at construction).
    # The compacted extent stays the DRAM row count: only the published axis grows, so the stage layouts
    # gathered above (dense, one per cache) keep addressing the same rows.
    if index_config_name is not None and index_layer_ids is not None:
        index_dense_layers = configs[index_config_name].num_layers
        # Global layer total: under PP the `num_layers` arg is this rank's slice, config 0 spans every stage.
        global_layers = configs["0"].num_layers
        assert len(index_layer_ids) == index_dense_layers, (
            f"index_layer_ids has {len(index_layer_ids)} entries but the index config spans "
            f"{index_dense_layers} compacted layers; every dense row needs a global layer id"
        )
        assert (
            max(index_layer_ids) < global_layers
        ), f"index_layer_ids reaches layer {max(index_layer_ids)} but the table spans {global_layers} layers"
        configs[index_config_name].num_layers = global_layers
    table = disagg.KvChunkAddressTable(configs)

    for name, cache, head_idx in entries:
        cfg, config_id = configs[name], table.config_id_of(name)
        if head_idx is None:  # MLA/kimi block-cyclic model cache
            populate_kv_chunk_address_table_kimi(
                lookup_table=table,
                config=cfg,
                mesh_device=mesh_device,
                mesh_shape=mesh_shape,
                seq_len=seq_len,
                sp_axis=sp_axis,
                tt_kvpe_cache=cache,
                chunk_size_bytes=cfg.chunk_size_bytes,
                num_users=num_users,
                config_id=config_id,
                stage_layout=layout_of[name],
                layer_rows=index_layer_ids if name == index_config_name else None,
            )
        else:  # one global kv-head of the drafter's K or V cache
            populate_kv_chunk_address_table_dflash(
                lookup_table=table,
                config=cfg,
                mesh_device=mesh_device,
                mesh_shape=mesh_shape,
                seq_len=seq_len,
                sp_axis=sp_axis,
                tp_axis=tp_axis,
                kv_cache=cache,
                chunk_size_bytes=cfg.chunk_size_bytes,
                num_kv_heads=dflash_kv_heads,
                head_idx=head_idx,
                num_users=num_users,
                config_id=config_id,
                chunk_size_global=chunk_size_global,
            )

    return serialize_prebuilt_kv_chunk_table(table=table, path=path)
