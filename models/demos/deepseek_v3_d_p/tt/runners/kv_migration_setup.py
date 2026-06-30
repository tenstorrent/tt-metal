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

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_kimi,
)

# bfp8 [1, 1, 32, 576] KV chunk: 18 tiles * 1088 B = 19584 B.
_CHUNK_SIZE_BYTES = 19584


def _disaggregation():
    """KvChunkAddressTable + serializer come from ttnn.experimental.disaggregation
    (no _migration extension needed)."""
    return ttnn.experimental.disaggregation


def _serialize_table_to_path(table, path: str) -> None:
    """Serialize a KvChunkAddressTable to a protobuf file for the worker's SET_TABLE."""
    _disaggregation().export_to_protobuf_file(table, path)


def build_and_serialize_kv_chunk_table(
    *, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, sp_axis, num_users, chunk_size_global, path
) -> str:
    """Build the KV chunk address table from the device KV layout and serialize it to ``path``
    for the inference server to forward via SET_TABLE. Returns the path on success.

    Uses create_kv_chunk_address_table_kimi: chunked prefill stores KV positions block-cyclic across
    the SP shards (every model variant), so the table must map each natural position to its true
    block-cyclic storage chip + offset. The migration worker copies the chunks the table lists for
    the migrated position range. A contiguous (wrong) table still works for a blanket copy of the
    WHOLE cache — every chunk is copied regardless of its label — but any sub-cache migration (a
    prefix copy of [0, N), or a prompt shorter than max_seq_len) lists the wrong, block-cyclically-
    scattered chunks and copies mostly un-prefilled storage, so the migrated KV fails its PCC check.

    ``chunk_size_global`` is the prefill chunk size (the block-cyclic period; the same value passed
    to blockcyclic_positions). The kimi builder hardcodes this period as PREFILL_CHUNK_OUTPUT_TOKENS,
    so a non-default PREFILL_CHUNK_SIZE is rejected here rather than silently mismapped."""
    assert chunk_size_global == PREFILL_CHUNK_OUTPUT_TOKENS, (
        f"create_kv_chunk_address_table_kimi assumes a block-cyclic period of "
        f"PREFILL_CHUNK_OUTPUT_TOKENS={PREFILL_CHUNK_OUTPUT_TOKENS}, but chunk_size_global={chunk_size_global}. "
        f"A different period would mismap every position; re-introduce a parametrized builder if needed."
    )
    cfg = _disaggregation().KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = seq_len
    cfg.num_slots = num_users
    cfg.chunk_n_tokens = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    cfg.chunk_size_bytes = _CHUNK_SIZE_BYTES
    table = create_kv_chunk_address_table_kimi(
        config=cfg,
        mesh_device=mesh_device,
        mesh_shape=mesh_shape,
        seq_len=seq_len,
        sp_axis=sp_axis,
        tt_kvpe_cache=kvpe_cache,
        chunk_size_bytes=_CHUNK_SIZE_BYTES,
        num_users=num_users,
    )
    _serialize_table_to_path(table, path)
    logger.info(f"[migration] KV chunk address table serialized to {path} (entries={table.total_entries()})")
    return path
