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
from models.demos.common.prefill.runners.migration import serialize_kv_chunk_table, serialize_prebuilt_kv_chunk_table
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_kimi,
    populate_kv_chunk_address_table_kimi,
)

# A KV chunk is one DRAM bank's worth of tokens (NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK=32) x head_dim.
_TILE_DIM = 32  # bfp8 is tiled 32x32
_BFP8_TILE_BYTES = 1088  # one 32x32 bfp8 tile: 1024 data + 64 exponent bytes


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
    side table. None (dense models) → the usual single-config table over the KVPE cache alone."""
    assert chunk_size_global == PREFILL_CHUNK_OUTPUT_TOKENS, (
        f"create_kv_chunk_address_table_kimi assumes a block-cyclic period of "
        f"PREFILL_CHUNK_OUTPUT_TOKENS={PREFILL_CHUNK_OUTPUT_TOKENS}, but chunk_size_global={chunk_size_global}. "
        f"A different period would mismap every position; re-introduce a parametrized builder if needed."
    )

    primary_cache = kvpe_cache.storage
    all_caches = (primary_cache,) + ((index_kv_cache,) if index_kv_cache is not None else ())
    if len(all_caches) > 1:
        return _build_and_serialize_merged_kv_chunk_table(
            mesh_device=mesh_device,
            caches=all_caches,
            seq_len=seq_len,
            num_layers=num_layers,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_users=num_users,
            path=path,
        )

    def _builder(*, config, chunk_size_bytes, num_users):
        return create_kv_chunk_address_table_kimi(
            config=config,
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tt_kvpe_cache=primary_cache,
            chunk_size_bytes=chunk_size_bytes,
            num_users=num_users,
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
    *, mesh_device, caches, seq_len, num_layers, mesh_shape, sp_axis, num_users, path
) -> str:
    """Sparse (DSA) path: build ONE KvChunkAddressTable holding BOTH caches instead of two tables —
    config 0 = the KVPE cache, config 1 = the index-key cache. Each config carries its own
    chunk_size_bytes (derived from the cache's dtype + head_dim); the device-group / fabric-host side
    table is shared across both. Mirrors test_glm_kv_cache_table's merged readback."""
    disagg = ttnn.experimental.disaggregation

    def _table_config(cache):
        cfg = disagg.KvChunkAddressTableConfig()
        # KVPE = all layers; the GLM-5.2 index cache = full-layers-only, so config 1 holds only num_full
        # entries and populate_kv_chunk_address_table_kimi (which iterates config.num_layers) skips the
        # shared-layer slots. GLM-5.1 / dense: index cache is all-layers, so this equals num_layers.
        cfg.num_layers = _num_layers_from_cache(cache, num_users)
        cfg.max_sequence_length = seq_len
        cfg.num_slots = num_users
        cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
        cfg.chunk_size_bytes = _dram_chunk_size_bytes(cache)
        return cfg

    # BF16 and scaled FP8 both own one primary KVPE tensor; scaled FP8 stores its mixed fields in one
    # packed row. The optional index cache is therefore always the next stable config.
    configs = [_table_config(c) for c in caches]
    table = disagg.KvChunkAddressTable(configs)

    for config_id, (cache, cfg) in enumerate(zip(caches, configs)):
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
        )

    return serialize_prebuilt_kv_chunk_table(table=table, path=path)
