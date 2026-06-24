# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""KV chunk address table publish for the prefill runner.

STARTUP step: build the KvChunkAddressTable from the device KV layout and
serialize it to a protobuf file. The runner owns the device, so it is the ONLY
component that knows the KV cache's NoC addresses (kvpe_cache.buffer_address()).
It writes the table to a path; the inference server / orchestrator feeds that
path to the migration_worker via SET_TABLE
(`MigrationLayerClient.send_kv_chunk_table(table_path)`). The runner never talks
to the worker directly.

Serialization uses the ttnn binding
``ttnn.experimental.disaggregation.export_to_protobuf_file`` (no separate
_migration extension needed).

This is the MODEL-SPECIFIC half of the disaggregation integration: building the table from the
deepseek KV layout (block-cyclic, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK). The generic half — forwarding
the serialized .pb to the migration worker (`send_kv_chunk_table`) — is model-agnostic and lives in
`models.common.prefill_runner.migration`. The deepseek adapter exposes this builder over the runner
seam via `build_and_serialize_kv_chunk_table`.

NOTE: only the table-publish half of the disaggregation integration lives here.
The per-layer LayerAck channel + scheduler-driven migration are intentionally
left out (owned by the scheduler/worker side).
"""

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
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
    *, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, sp_axis, num_users, path
) -> str:
    """Build the KV chunk address table from the device KV layout and serialize it to ``path`` for the
    inference server to forward via SET_TABLE. Returns the path on success.

    Chunked prefill always uses the block-cyclic (non-balanced) cache, so the table is built with the
    block-cyclic / multi-user builder for every variant — it folds slots as user*num_layers + layer and
    supports num_users >= 1. (The balanced create_kv_chunk_address_table_ds is single-user only and is
    not used here, despite the _kimi name the block-cyclic builder still carries.)"""
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
