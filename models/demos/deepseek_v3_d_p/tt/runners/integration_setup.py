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

==============================================================================
STATUS: written to the FINAL shape; the blocked pieces are marked.

  BLOCKED(serialize):  KvChunkAddressTable has no serialize-to-file binding yet
                       (only set/add_device_group/total_entries are exposed). Add
                       it to ttnn/cpp/ttnn-nanobind/disaggregation.cpp.

  NOTE(layer-ack):     ShmLayerAckChannel below is a minimal stand-in matching
                       the documented datum (one uint64 at offset 0 of a
                       /dev/shm segment). Swap for the engine's binding when it
                       lands; the on-wire layout is intended to match.
==============================================================================
"""

import mmap
import os
import struct

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


def _table_types():
    """KvChunkAddressTable types come from ttnn (not _migration), so building the
    table needs no migration extension."""
    d = ttnn.experimental.disaggregation
    return {
        "KvCacheLocation": d.KvCacheLocation,
        "KvChunkAddressTable": d.KvChunkAddressTable,
        "KvChunkAddressTableConfig": d.KvChunkAddressTableConfig,
    }


def _build_prefill_table(mesh, tt_kvpe_cache, seq_len, num_layers, mesh_shape, m):
    """KvChunkAddressTable for a kvpe_cache laid out by init_kvpe_cache().

    NOC address encoding: (bank_id << 32) | (base_addr + bank_offset), where
    bank_offset accumulates by chunk_size_bytes each time the bank counter wraps.
    """
    KvCacheLocation = m["KvCacheLocation"]
    KvChunkAddressTable = m["KvChunkAddressTable"]
    KvChunkAddressTableConfig = m["KvChunkAddressTableConfig"]

    # Match kvpe geometry — keep in sync with kv_cache_utils.NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    # and the bfp8 [1, 1, 32, 576] chunk size used by DeepSeek V3.
    chunk_n_tokens = 32
    chunk_size_bytes = 19584

    cfg = KvChunkAddressTableConfig()
    cfg.num_layers = num_layers
    cfg.max_sequence_length = seq_len
    cfg.num_slots = 1
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = chunk_size_bytes
    table = KvChunkAddressTable(cfg)

    base_addr = tt_kvpe_cache.buffer_address()
    logger.info(f"[migration][prefill] base_addr = {base_addr}")

    rows, cols = mesh_shape
    dg_per_row = []
    for row in range(rows):
        devices = []
        for col in range(cols):
            fnid = mesh.get_fabric_node_id(ttnn.MeshCoordinate(row, col))
            devices.append((int(fnid.mesh_id), int(fnid.chip_id)))
        dg_idx = table.add_device_group(devices)
        for mid, cid in devices:
            table.set_fabric_node_host(mid, cid, f"mesh-{mid}")
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
                table.set(layer, pos, 0, KvCacheLocation(noc_addr, chunk_size_bytes, dg_per_row[row]))
                curr_bank_id = (curr_bank_id + 1) % BH_NUM_DRAM_BANKS
                if curr_bank_id == 0:
                    curr_bank_offset += chunk_size_bytes
                pos += chunk_n_tokens
                if c == num_chunks_in_strip - 1:
                    pos = high_strips[row][0]
    return table


def _serialize_table_to_path(table, path: str) -> None:
    """Serialize a KvChunkAddressTable to a protobuf file for the worker's SET_TABLE.

    BLOCKED(serialize): no serialize-to-file binding exists on KvChunkAddressTable
    yet. When it lands this becomes e.g. ``table.serialize(path)``.
    """
    raise NotImplementedError(
        f"KvChunkAddressTable has no serialize-to-path binding yet "
        f"(BLOCKED(serialize)); cannot write {path}. Add it in "
        f"ttnn/cpp/ttnn-nanobind/disaggregation.cpp."
    )


def build_and_serialize_kv_chunk_table(*, mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, path) -> str:
    """STARTUP step (1): build the KV chunk address table from the device layout
    and serialize it to ``path`` for the inference server to forward via SET_TABLE.

    Returns the path. On BLOCKED(serialize) it logs and returns None so the rest
    of runner setup (LayerAck wiring) can still proceed during bring-up.
    """
    table = _build_prefill_table(mesh_device, kvpe_cache, seq_len, num_layers, mesh_shape, _table_types())
    try:
        _serialize_table_to_path(table, path)
    except NotImplementedError as exc:
        logger.warning(f"[migration] KV chunk table built but NOT serialized — {exc}")
        return None
    logger.info(f"[migration] KV chunk address table serialized to {path}")
    return path


class ShmLayerAckChannel:
    """Producer side of the LayerAck channel (prefill.md §3.11).

    A named POSIX shm segment holding a single 64-bit counter the runner bumps
    once per layer; the scheduler reads the delta. Lock-free SPSC: an aligned
    8-byte store on x86-64 is atomic, and a monotonic counter needs nothing more.

    NOTE(layer-ack): minimal stand-in for the engine's ShmLayerAckChannel; the
    layout (one little-endian uint64 at offset 0) is intended to match.
    """

    _SIZE = 8  # one uint64 counter

    def __init__(self, name: str, create: bool = True):
        # POSIX shm name "/foo" maps to /dev/shm/foo on Linux.
        self._name = name
        shm_path = "/dev/shm/" + name.lstrip("/")
        flags = os.O_RDWR | (os.O_CREAT if create else 0)
        fd = os.open(shm_path, flags, 0o600)
        try:
            if create:
                os.ftruncate(fd, self._SIZE)
            self._buf = mmap.mmap(fd, self._SIZE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        finally:
            os.close(fd)
        if create:
            struct.pack_into("<Q", self._buf, 0, 0)
        self._count = 0
        logger.info(f"[migration] LayerAck channel ready at {shm_path}")

    def increment(self) -> None:
        """Bump the counter by one (called once per layer from the prefill loop)."""
        self._count += 1
        struct.pack_into("<Q", self._buf, 0, self._count)

    def close(self) -> None:
        self._buf.close()
