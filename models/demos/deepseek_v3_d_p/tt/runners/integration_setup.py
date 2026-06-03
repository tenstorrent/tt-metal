# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Prefill-node disaggregation integration for the prefill runner.

In the target architecture (tt-llm-engine docs/scheduler/prefill.md) the prefill
runner has NO IPC with the migration_worker and does not drive migration. Its
entire integration footprint is two things:

  (1) STARTUP — build the KvChunkAddressTable and serialize it to a file. The
      runner owns the device, so it is the ONLY component that knows the KV
      cache's NoC addresses (kvpe_cache.buffer_address()). It writes the table
      to a path; the inference server (which owns prefill->decode handoff) feeds
      that path to the worker via SET_TABLE. The runner never talks to the worker.

  (2) PER LAYER — increment a single 64-bit counter in a named POSIX shm segment
      (the LayerAckChannel, prefill.md §3.11). The scheduler reads the delta and
      decides what to migrate; slot/range/layer correlation is the scheduler's
      job (its InFlightChunkFIFO), so the ack carries no payload.

Everything else — CONNECT, MIGRATE, completion tracking — belongs to the
scheduler + worker, not the runner.

Serialization uses the ttnn binding
``ttnn.experimental.disaggregation.export_to_protobuf_file`` (added so the
runner needs no separate _migration extension). The table is built from the
ttnn disaggregation types, whose API differs from the legacy _migration one:
  * add_device_group takes a Sequence[FabricNodeId] (not (mesh,chip) tuples)
  * set_fabric_node_host takes (FabricNodeId, host_name) (2 args)
  * KvCacheLocation is default-constructed; fields set by assignment
"""

from loguru import logger

import ttnn

# Endpoint ids — disaggregated prefill/decode convention (lower id = decode).
DECODE_EP_ID = 0
PREFILL_EP_ID = 1

# Sentinel dst_slot sent by the inference server when a request must NOT trigger
# migration (e.g. warmup probes, scheduler-skipped requests).
INVALID_SLOT_ID = 0xFFFFFFFF

# DRAM banks per Blackhole device — used when packing NOC addresses for the table.
BH_NUM_DRAM_BANKS = 8


def _disaggregation():
    """KvChunkAddressTable + serializer come from ttnn.experimental.disaggregation
    (no _migration extension needed)."""
    return ttnn.experimental.disaggregation


def _build_prefill_table(mesh, tt_kvpe_cache, seq_len, num_layers, mesh_shape):
    """KvChunkAddressTable for a kvpe_cache laid out by init_kvpe_cache().

    NOC address encoding: (bank_id << 32) | (base_addr + bank_offset), where
    bank_offset accumulates by chunk_size_bytes each time the bank counter wraps.
    """
    d = _disaggregation()

    # Match kvpe geometry — keep in sync with kv_cache_utils.NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    # and the bfp8 [1, 1, 32, 576] chunk size used by DeepSeek V3.
    chunk_n_tokens = 32
    chunk_size_bytes = 19584

    cfg = d.KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = seq_len
    cfg.num_slots = 1
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = d.KvChunkAddressTable(cfg)

    base_addr = tt_kvpe_cache.buffer_address()
    logger.info(f"[migration][prefill] base_addr = {base_addr}")

    rows, cols = mesh_shape
    dg_per_row = []
    for row in range(rows):
        # ttnn add_device_group wants a Sequence[FabricNodeId]; pass the node ids
        # straight from the mesh (NOT (mesh_id, chip_id) tuples).
        fnids = [mesh.get_fabric_node_id(ttnn.MeshCoordinate(row, col)) for col in range(cols)]
        dg_idx = table.add_device_group(fnids)
        for fnid in fnids:
            table.set_fabric_node_host(fnid, f"mesh-{int(fnid.mesh_id)}")
        dg_per_row.append(dg_idx)

    # Two-strip layout: each row owns a low strip (front) and a high strip (back).
    num_tokens_in_strip = seq_len // (rows * 2)
    num_chunks_in_strip = num_tokens_in_strip // chunk_n_tokens
    chunks_per_dg = num_chunks_in_strip * 2

    low_start = 0
    high_end = seq_len - 1
    low_strips, high_strips = [], []
    for _ in range(rows):
        low_end = low_start + num_tokens_in_strip - 1
        high_start = high_end - num_tokens_in_strip + 1
        low_strips.append((low_start, low_end))
        high_strips.append((high_start, high_end))
        low_start = low_end + 1
        high_end = high_start - 1

    for row in range(rows):
        curr_bank_id = 0
        curr_bank_offset = 0
        for layer in range(num_layers):
            pos = low_strips[row][0]
            for c in range(chunks_per_dg):
                noc_addr = (curr_bank_id << 32) | (base_addr + curr_bank_offset)
                loc = d.KvCacheLocation()
                loc.noc_addr = noc_addr
                loc.size_bytes = chunk_size_bytes
                loc.device_group_index = dg_per_row[row]
                table.set(layer, pos, 0, loc)
                curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                if curr_bank_id == 0:
                    curr_bank_offset += chunk_size_bytes
                pos += chunk_n_tokens
                if c == num_chunks_in_strip - 1:
                    pos = high_strips[row][0]
    return table


def _serialize_table_to_path(table, path: str) -> None:
    """Serialize a KvChunkAddressTable to a protobuf file for the worker's SET_TABLE."""
    _disaggregation().export_to_protobuf_file(table, path)


def build_and_serialize_kv_chunk_table(*, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, path) -> str:
    """STARTUP step (1): build the KV chunk address table from the device layout
    and serialize it to ``path`` for the inference server to forward via SET_TABLE.

    Returns the path on success."""
    table = _build_prefill_table(mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape)
    _serialize_table_to_path(table, path)
    logger.info(f"[migration] KV chunk address table serialized to {path} (entries={table.total_entries()})")
    return path
