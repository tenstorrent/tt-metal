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

from models.demos.common.prefill.runners.migration import serialize_kv_chunk_table
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    PREFILL_CHUNK_OUTPUT_TOKENS,
    create_kv_chunk_address_table_kimi,
)

# bfp8 [1, 1, 32, 576] KV chunk: 18 tiles * 1088 B = 19584 B.
_CHUNK_SIZE_BYTES = 19584


def build_and_serialize_kv_chunk_table(
    *, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, sp_axis, num_users, chunk_size_global, path
) -> str:
    """Build the MLA block-cyclic KV chunk address table and serialize it to ``path`` for the
    inference server's SET_TABLE. Returns the path on success.

    Chunked prefill stores KV positions block-cyclic across the SP shards, so the table maps each
    natural position to its true storage chip + offset. The migration worker copies the chunks the
    table lists for the migrated range. A contiguous (wrong) table still works for a blanket copy of
    the WHOLE cache, but any sub-cache migration (a [0, N) prefix, or a prompt shorter than
    max_seq_len) lists the wrong, block-cyclically-scattered chunks and fails its PCC check.

    ``chunk_size_global`` is the block-cyclic period; the kimi builder hardcodes it as
    PREFILL_CHUNK_OUTPUT_TOKENS, so a non-default period is rejected here rather than mismapped."""
    assert chunk_size_global == PREFILL_CHUNK_OUTPUT_TOKENS, (
        f"create_kv_chunk_address_table_kimi assumes a block-cyclic period of "
        f"PREFILL_CHUNK_OUTPUT_TOKENS={PREFILL_CHUNK_OUTPUT_TOKENS}, but chunk_size_global={chunk_size_global}. "
        f"A different period would mismap every position; re-introduce a parametrized builder if needed."
    )

    def _builder(*, config, chunk_size_bytes, num_users):
        return create_kv_chunk_address_table_kimi(
            config=config,
            mesh_device=mesh_device,
            mesh_shape=mesh_shape,
            seq_len=seq_len,
            sp_axis=sp_axis,
            tt_kvpe_cache=kvpe_cache,
            chunk_size_bytes=chunk_size_bytes,
            num_users=num_users,
        )

    return serialize_kv_chunk_table(
        table_builder=_builder,
        num_layers=num_layers,
        max_seq_len=seq_len,
        num_users=num_users,
        chunk_n_tokens=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        chunk_size_bytes=_CHUNK_SIZE_BYTES,
        path=path,
    )
