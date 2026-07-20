#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parametrized H2D producer for the prefill runner.

It behaves like a small inference scheduler: it drives N concurrent user "slots", each running a
request made of token chunks, and pushes those chunks to the runner over the H2D socket. The order
and timing of the pushes is described entirely by a flat `ProducerConfig` — the producer has no notion
of named "scenarios" or "modes". Scenarios (single-user, round-robin, random stress, ...) live in the
test suite as ProducerConfig data (see tests/test_producer_runner_e2e.py); this engine just runs one.

After pushing, it drains the runner's per-layer LayerAcks and, if asked, reads the resulting KV cache
back and PCC-checks it against the golden trace. The KV read (read_dram_umd over a bare UMD cluster)
and the bfp8 decode are BOTH device-less on purpose: touching a real ttnn device here would take the
CHIP_IN_USE lock the runner already holds, and deadlock.

`run_schedule()` takes injectable seams (push_fn / now_fn / sleep_fn / rng) so the scheduling logic
can be unit-tested with no device and reproduced deterministically.

Env — schedule knobs (flat; the defaults describe a 1-user, 11-chunk, in-order run):
  PREFILL_NUM_USERS              concurrent cache slots (default 1)
  PREFILL_PRODUCER_CHUNKS        chunks per request: "N" fixed, or "min,max" random (default "11")
  PREFILL_PRODUCER_MAX_REQUESTS  total requests across all slots before stopping (default 1)
  PREFILL_PRODUCER_DURATION_S    wall-clock bound (default "inf" = stop on request count)
  PREFILL_PRODUCER_P_GAP         per-step probability of an idle gap (default 0.0)
  PREFILL_PRODUCER_P_BURST       per-step probability of a 2-3 chunk burst (default 0.0)
  PREFILL_PRODUCER_GAP_MS        "min,max" idle-gap milliseconds (default "200,2000")
  PREFILL_PRODUCER_MID_END_PROB  probability a request ends mid-chunk (default 0.0)
  PREFILL_PRODUCER_INTERLEAVE    slot order: "random" (default) | "round_robin"
  PREFILL_PRODUCER_SEED          RNG seed (default 1234)
  PREFILL_PRODUCER_CHECK_PCC     "1" to read KV back and PCC vs golden per slot (default 0)
  PREFILL_SEND_SHUTDOWN          "1" to close the stream with an all -1 sentinel so the runner exits
                                 gracefully after the run (sent after the KV read; default 0). PR #48718.
Env — transport (must match the runner): PREFILL_SP / PREFILL_TP / PREFILL_CHUNK_SIZE /
  PREFILL_MAX_SEQ_LEN / PREFILL_NUM_LAYERS / PREFILL_H2D_SERVICE_ID / PREFILL_H2D_CONNECT_TIMEOUT.

Usage:
    # 1 user, full depth, with PCC:
    PREFILL_PRODUCER_CHUNKS=11 PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_producer
    # 8-user random stress + PCC:
    PREFILL_NUM_USERS=8 PREFILL_PRODUCER_CHUNKS=1,4 PREFILL_PRODUCER_MAX_REQUESTS=200 \
      PREFILL_PRODUCER_P_GAP=0.2 PREFILL_PRODUCER_P_BURST=0.3 PREFILL_PRODUCER_MID_END_PROB=0.33 \
      PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_producer
"""

import os
import random
import struct
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir

# PrefillMetadata on the wire: 3 x uint32 = [slot_id, actual_start, actual_end].
METADATA_SIZE_BYTES = 12

SP_AXIS = int(os.environ.get("PREFILL_SP", 8))
TP_AXIS = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (SP_AXIS, TP_AXIS)
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 60 * 1024))
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    """Pack one chunk's PrefillMetadata (3 little-endian uint32s)."""
    return struct.pack("<III", slot_id, actual_start, actual_end)


def _chunk_to_host_array(chunk_token_ids):
    """One chunk's tokens as the un-sharded [SP, 1, chunk_local] uint32 buffer the H2D service expects.
    Block-cyclic / chip-major layout, matching the runner's prepare_prefill_input_tensor; the connected
    service resplits it across SP coordinates, so this process needs no MeshDevice."""
    sp = GLOBAL_MESH_SHAPE[0]
    chunk_local = CHUNK_SIZE // sp
    return (
        torch.tensor(chunk_token_ids, dtype=torch.int64)
        .reshape(sp, 1, chunk_local)
        .to(torch.uint32)
        .contiguous()
        .numpy()
    )


# ---------------------------------------------------------------------------
# Device-less helpers: read the KV table / device map, attach the LayerAck channel, decode bfp8.
# All device-less on purpose — none may touch a real ttnn device (that would take the CHIP_IN_USE
# lock the runner holds and deadlock).
# ---------------------------------------------------------------------------


def _read_kv_chunk_table(timeout_s: int):
    """Poll for and deserialize the KV chunk address table the runner published to
    PREFILL_MIGRATION_TABLE_PATH. Fully device-less (import_from_protobuf_file rebuilds it from the
    protobuf alone). Returns the table, or None if it never appears (the producer can still push)."""
    table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
    deadline = time.perf_counter() + timeout_s
    while not os.path.exists(table_path):
        if time.perf_counter() > deadline:
            logger.warning(f"[producer] KV chunk table {table_path} not found after {timeout_s}s; skipping table read.")
            return None
        time.sleep(0.1)

    import_from_protobuf_file = getattr(ttnn.experimental.disaggregation, "import_from_protobuf_file", None)
    if import_from_protobuf_file is None:
        logger.error("[producer] ttnn import_from_protobuf_file missing — rebuild ttnn; skipping table read.")
        return None

    table = import_from_protobuf_file(table_path)
    table_cfg = table.config()
    logger.info(
        f"[producer] read KV chunk table {table_path}: entries={table.total_entries()} "
        f"num_layers={table_cfg.num_layers} num_slots={table_cfg.num_slots} "
        f"max_seq_len={table_cfg.max_sequence_length} chunk_n_tokens={table_cfg.chunk_n_tokens}"
    )
    return table


def _read_device_map(timeout_s: int) -> dict:
    """Poll for and read the runner's fabric_node -> ASIC-unique_id sidecar (JSON), so read_dram_umd can
    pick chips by unique_id without touching the ControlPlane. Returns {(mesh_id, chip_id): unique_id}."""
    import json

    path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
    deadline = time.perf_counter() + timeout_s
    while not os.path.exists(path):
        if time.perf_counter() > deadline:
            logger.warning(f"[producer] device map {path} not found after {timeout_s}s; skipping KV read.")
            return {}
        time.sleep(0.1)

    with open(path) as f:
        raw_map = json.load(f)
    device_map = {tuple(int(x) for x in key.split(":")): int(unique_id) for key, unique_id in raw_map.items()}
    logger.info(f"[producer] read device map {path}: {len(device_map)} chips")
    return device_map


def _connect_layer_ack_channel(timeout_s: int):
    """Attach (consumer side) to the runner's per-layer LayerAck channel
    (/tt_prefill_layer_acks_<service_id>). Returns the channel, or None if it isn't available (only the
    single-rank runner creates it)."""
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm_name = f"/tt_prefill_layer_acks_{service_id}"
    try:
        channel = ttnn.InterProcessCounterChannel.connect(shm_name, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(f"[producer] could not connect LayerAck channel {shm_name}: {e}; skipping ack wait.")
        return None
    logger.info(f"[producer] connected LayerAck channel {shm_name}")
    return channel


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
    """Block until `expected` per-layer acks (NUM_LAYERS per chunk) have been drained, or timeout.
    Returns the count actually drained."""
    if ack_channel is None:
        return 0
    drained = 0
    last_logged = -1
    start = time.perf_counter()
    while drained < expected:
        drained += ack_channel.try_consume_all()
        if drained != last_logged:
            logger.info(f"[producer] layer acks {drained}/{expected}")
            last_logged = drained
        if drained >= expected:
            break
        if time.perf_counter() - start > timeout_s:
            logger.warning(f"[producer] timed out at {drained}/{expected} acks after {timeout_s}s")
            break
        time.sleep(0.01)
    logger.info(f"[producer] drained {drained}/{expected} layer acks in {(time.perf_counter() - start):.2f}s")
    return drained


def _decode_bfp8_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    """Decode a [32, head_dim] bfp8_b tile chunk (raw device bytes) to a float32 [32, head_dim] tensor,
    in pure numpy (no ttnn tensor ops, so it never inits the device context / takes the CHIP_IN_USE
    lock). Validated bit-exact against ttnn._ttnn.bfp_utils.unpack_bfp8.

    Layout per 1088-byte tile: 64 exponent bytes (one per (face, row)) then 1024 mantissa bytes
    ((face, row, col)); value = (-1)^sign * (mantissa & 0x7F) * 2^(exponent - 133). Face f = fr*2 + fc
    maps to tile rows fr*16 + r and cols fc*16 + c; tiles lie along the head_dim (column) axis.
    """
    TILE = 32
    n_tiles = head_dim // TILE
    raw_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(n_tiles, 1088)

    exponents = raw_u8[:, :64].astype(np.int32).reshape(n_tiles, 4, 16)  # (tile, face, row)
    mantissas = raw_u8[:, 64:].reshape(n_tiles, 4, 16, 16)  # (tile, face, row, col)
    signs = (mantissas >> 7).astype(np.int32)
    magnitude = (mantissas & 0x7F).astype(np.float32)
    scale = np.exp2((exponents - 133).astype(np.float32))[..., None]
    values = np.where(signs > 0, -(magnitude * scale), magnitude * scale)  # (tile, face, row, col)

    # face (fr, fc) -> tile rows fr*16+r, cols fc*16+c
    by_face = values.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)  # tile t -> columns [t*32 : (t+1)*32]
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _decode_bf16_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    """Decode a ``[32, head_dim]`` bf16 TILE chunk (raw device bytes) to float32, device-less. Same face/
    tile de-swizzle as ``_decode_bfp8_chunk`` but bf16 has no exponent block: each 2048-byte tile is 1024
    bf16 values (uint16 -> float32 via a 16-bit left shift, which is exact). NOTE: not validated against a
    ttnn unpack the way the bf8 path is."""
    TILE = 32
    n_tiles = head_dim // TILE
    u16 = np.frombuffer(raw, dtype="<u2").reshape(n_tiles, 4, 16, 16)  # (tile, face, row, col)
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    by_face = f32.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, TILE, TILE)
    decoded = by_face.transpose(1, 0, 2).reshape(TILE, n_tiles * TILE)
    return torch.from_numpy(np.ascontiguousarray(decoded))


def _resolve_unique_id(fabric_node_ids, device_map: dict) -> int:
    """ASIC unique_id for any replica fabric node present in the device map (replicas hold identical KV,
    and add_device_group sorts the ids so index 0 is not a fixed chip). Raises if none are mapped."""
    for node in fabric_node_ids:
        key = (int(node.mesh_id), int(node.chip_id))
        if key in device_map:
            return device_map[key]
    tried = [(int(n.mesh_id), int(n.chip_id)) for n in fabric_node_ids]
    raise KeyError(f"no fabric node {tried} in device map ({len(device_map)} chips; single-rank/one-galaxy only)")


# ---------------------------------------------------------------------------
# Config + scheduler engine (mode-unaware; run_schedule touches no device/ttnn, so it is unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class ProducerConfig:
    """A flat description of a push schedule. A "scenario" is just a set of these values."""

    num_users: int  # number of concurrent cache slots (users)
    chunks_min: int  # per-request chunk count is rng.randint(chunks_min, chunks_max)
    chunks_max: int
    max_requests: int  # total requests across all slots before stopping
    duration_s: float  # wall-clock bound (inf => stop on request count only)
    p_gap: float  # per-step probability of an idle gap
    p_burst: float  # per-step probability of a 2-3 chunk burst to one slot
    gap_ms: tuple  # (min, max) idle-gap milliseconds
    mid_chunk_end_prob: float  # probability a request ends mid-chunk (exercises the actual_end clamp)
    seed: int
    verify: bool  # read KV back and PCC each resident slot vs golden
    pcc_threshold: float
    interleave: str = "random"  # slot order: "random" | "round_robin" (fair alternation)


def _config_from_env() -> ProducerConfig:
    """Build a ProducerConfig from the flat PREFILL_PRODUCER_* env vars (every knob independent)."""
    max_chunks = MAX_SEQ_LEN // CHUNK_SIZE  # a request can't exceed the per-user cache
    chunk_bounds = [int(x) for x in os.environ.get("PREFILL_PRODUCER_CHUNKS", "11").split(",")]
    chunks_max = min(chunk_bounds[-1], max_chunks)
    chunks_min = min(chunk_bounds[0], chunks_max)
    gap_lo, gap_hi = (float(x) for x in os.environ.get("PREFILL_PRODUCER_GAP_MS", "200,2000").split(","))

    interleave = os.environ.get("PREFILL_PRODUCER_INTERLEAVE", "random")
    if interleave not in ("random", "round_robin"):
        raise ValueError(f"PREFILL_PRODUCER_INTERLEAVE must be 'random' or 'round_robin', got {interleave!r}")

    return ProducerConfig(
        num_users=int(os.environ.get("PREFILL_NUM_USERS", "1")),
        chunks_min=chunks_min,
        chunks_max=chunks_max,
        max_requests=int(os.environ.get("PREFILL_PRODUCER_MAX_REQUESTS", "1")),
        duration_s=float(os.environ.get("PREFILL_PRODUCER_DURATION_S", "inf")),
        p_gap=float(os.environ.get("PREFILL_PRODUCER_P_GAP", "0.0")),
        p_burst=float(os.environ.get("PREFILL_PRODUCER_P_BURST", "0.0")),
        gap_ms=(gap_lo, gap_hi),
        mid_chunk_end_prob=float(os.environ.get("PREFILL_PRODUCER_MID_END_PROB", "0.0")),
        seed=int(os.environ.get("PREFILL_PRODUCER_SEED", "1234")),
        verify=os.environ.get("PREFILL_PRODUCER_CHECK_PCC", "0") == "1",
        pcc_threshold=float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.93")),
        interleave=interleave,
    )


class _Slot:
    """One cache-user slot holding one in-flight request. `next_chunk` advances per push; `actual_isl`
    is the request's real (non-pad) token count, so the last chunk reports a clamped actual_end."""

    def __init__(self, slot_id: int):
        self.slot_id = slot_id
        self.req_id = -1
        self.target_chunks = 0
        self.next_chunk = 0
        self.actual_isl = 0

    @property
    def done(self) -> bool:
        return self.next_chunk >= self.target_chunks


def _new_request(slot: _Slot, req_id: int, cfg: ProducerConfig, rng: random.Random) -> None:
    """(Re)assign a fresh request to `slot`: a random chunk count, starting at chunk 0, optionally
    ending mid-chunk."""
    slot.req_id = req_id
    slot.target_chunks = rng.randint(cfg.chunks_min, cfg.chunks_max)
    slot.next_chunk = 0
    full_tokens = slot.target_chunks * CHUNK_SIZE
    if cfg.mid_chunk_end_prob > 0 and rng.random() < cfg.mid_chunk_end_prob and slot.target_chunks >= 1:
        slot.actual_isl = full_tokens - rng.randint(1, CHUNK_SIZE - 1)
    else:
        slot.actual_isl = full_tokens


@dataclass
class RunStats:
    resident: dict  # slot_id -> (chunks_pushed_for_resident_request, actual_isl)
    total_pushes: int
    push_ms: list
    completed: int
    wall_s: float


def run_schedule(cfg: ProducerConfig, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    """Execute the push schedule described by `cfg`.

    Device-free: `push_fn(slot_id, chunk_idx, actual_start, actual_end) -> elapsed_ms` performs and
    times the actual push, and now_fn/sleep_fn/rng are injectable so tests run instantly and
    deterministically. Records, per slot, the resident request as (chunks_pushed, actual_isl) at push
    time — i.e. what is physically in that slot's KV cache (a recycled slot overwrites from chunk 0),
    which the caller PCC-checks. Returns RunStats.
    """
    rng = rng if rng is not None else random.Random(cfg.seed)
    slots = [_Slot(i) for i in range(cfg.num_users)]
    resident: dict = {}

    # Give every slot an initial request; `next_req_id` counts total requests (initial + recycled).
    next_req_id = 0
    for slot in slots:
        _new_request(slot, next_req_id, cfg, rng)
        next_req_id += 1

    push_ms: list = []
    total_pushes = 0
    completed = 0
    round_robin_cursor = -1  # only used when cfg.interleave == "round_robin"
    start = now_fn()

    def send_chunk(slot: _Slot) -> None:
        nonlocal total_pushes, completed, next_req_id
        chunk_idx = slot.next_chunk
        actual_start = chunk_idx * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        push_ms.append(push_fn(slot.slot_id, chunk_idx, actual_start, actual_end))
        total_pushes += 1
        slot.next_chunk += 1
        resident[slot.slot_id] = (chunk_idx + 1, slot.actual_isl)  # what's now resident in this slot
        if slot.done:
            completed += 1
            if next_req_id < cfg.max_requests:  # recycle the slot as a fresh request
                _new_request(slot, next_req_id, cfg, rng)
                next_req_id += 1

    while (now_fn() - start) < cfg.duration_s and completed < cfg.max_requests:
        active_slots = [s for s in slots if not s.done]
        if not active_slots:
            break

        # One random draw classifies the step: [0, p_gap) gap, [p_gap, p_gap+p_burst) burst, else single.
        roll = rng.random()
        if roll < cfg.p_gap:
            sleep_fn(rng.uniform(*cfg.gap_ms) / 1000.0)
            continue

        if cfg.interleave == "round_robin":
            # Advance the cursor to the next non-done slot (active_slots is non-empty, so this finds one).
            for _ in range(len(slots)):
                round_robin_cursor = (round_robin_cursor + 1) % len(slots)
                if not slots[round_robin_cursor].done:
                    break
            slot = slots[round_robin_cursor]
        else:
            slot = rng.choice(active_slots)

        if roll < cfg.p_gap + cfg.p_burst:  # burst: 2-3 chunks to this slot
            for _ in range(rng.randint(2, 3)):
                if slot.done:
                    break
                send_chunk(slot)
        else:
            send_chunk(slot)

    return RunStats(
        resident=resident, total_pushes=total_pushes, push_ms=push_ms, completed=completed, wall_s=now_fn() - start
    )


# ---------------------------------------------------------------------------
# Per-slot KV read-back + PCC (device-less: read_dram_umd over UMD + pure-numpy bfp8 decode)
# ---------------------------------------------------------------------------


def _read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """Read slot `slot_id`'s KV over [0, real_len) via the published table and PCC-check it against the
    golden trace. Dispatches on the model: MLA (single merged kvpe config) vs M3 (multi-config triple
    cache). Returns the min PCC across layers."""
    if ADAPTER.name == "minimax_m3":
        return _read_slot_kv_and_check_pcc_m3(table, device_map, slot_id, real_len, trace_dir)
    return _read_slot_kv_and_check_pcc_mla(table, device_map, slot_id, real_len, trace_dir)


def _read_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim, decode):
    """Read one config's KV chunks over [0, read_len) for (layer, slot) via the address table and return
    the decoded ``[read_len, head_dim]`` tensor in natural token order."""
    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK

    rows = []
    for pos in range(0, read_len, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
        loc = table.lookup(layer, pos, slot_id, config_id)
        unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
        raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
        rows.append(decode(raw, head_dim))
    return torch.cat(rows, dim=0)[:read_len]


def _read_slot_kv_and_check_pcc_m3(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """M3 multi-config read-back: reconstruct per-head K/V + index_k from the 9-config table and PCC vs the
    separate_k_v golden. Config layout matches the builder: k_h0..N-1 = 0..N-1, v_h0..N-1 = N..2N-1,
    index_k = 2N."""
    from pathlib import Path

    from safetensors import safe_open

    from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from models.demos.minimax_m3.tt.runners.prefill_kv_validation import _hf_to_meta_rotary_perm
    from tests.ttnn.utils_for_testing import comp_pcc

    mc = ADAPTER.model_config
    n_kv, head_dim, rotary_dim = mc.NUM_KEY_VALUE_HEADS, mc.HEAD_DIM, mc.ROTARY_DIM
    perm = _hf_to_meta_rotary_perm(head_dim, rotary_dim)  # golden HF -> device Meta rotary swizzle
    # index_k dtype from its config's chunk size (bf8 vs bf16) -> the right decoder.
    ik_cfg = 2 * n_kv
    ik_bf16 = table.config(ik_cfg).chunk_size_bytes == (head_dim // 32) * 2048
    ik_decode = _decode_bf16_chunk if ik_bf16 else _decode_bfp8_chunk

    read_len = ((real_len + NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK - 1) // NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK) * (
        NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    kv_dir = Path(trace_dir) / "kv_cache"
    mins = {"k": 1.0, "v": 1.0, "index_k": 1.0}
    for layer in range(NUM_LAYERS):
        dev_k = torch.stack(
            [
                _read_kv_slice(table, device_map, h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[
            :, :real_len
        ]  # [n_kv, real_len, head_dim]
        dev_v = torch.stack(
            [
                _read_kv_slice(table, device_map, n_kv + h, layer, slot_id, read_len, head_dim, _decode_bfp8_chunk)
                for h in range(n_kv)
            ],
            dim=0,
        )[:, :real_len]

        with safe_open(str(kv_dir / f"layer_{layer}.safetensors"), framework="pt") as h:
            keys = set(h.keys())
            g_k = h.get_tensor(f"key_cache_layer_{layer}").float()[0, :, :real_len, :][..., perm]  # HF -> Meta
            g_v = h.get_tensor(f"value_cache_layer_{layer}").float()[0, :, :real_len, :]
            has_ik = f"index_k_cache_layer_{layer}" in keys
            g_ik = (
                h.get_tensor(f"index_k_cache_layer_{layer}").float()[0, 0, :real_len, :][..., perm] if has_ik else None
            )

        pcc_k = float(comp_pcc(g_k, dev_k, 0.0)[1])
        pcc_v = float(comp_pcc(g_v, dev_v, 0.0)[1])
        mins["k"], mins["v"] = min(mins["k"], pcc_k), min(mins["v"], pcc_v)
        line = f"  layer {layer:>2}: K={pcc_k:.5f} V={pcc_v:.5f}"
        if has_ik:
            dev_ik = _read_kv_slice(table, device_map, ik_cfg, layer, slot_id, read_len, head_dim, ik_decode)[:real_len]
            pcc_ik = float(comp_pcc(g_ik, dev_ik, 0.0)[1])
            mins["index_k"] = min(mins["index_k"], pcc_ik)
            line += f" index_k={pcc_ik:.5f}"
        logger.info(line)

    min_pcc = min(mins.values())
    logger.info(
        f"[producer] slot {slot_id} M3 KV PCC over [0,{real_len}) across {NUM_LAYERS} layers -> "
        f"K={mins['k']:.5f} V={mins['v']:.5f} index_k={mins['index_k']:.5f} (min {min_pcc:.6f})"
    )
    return min_pcc


def _read_slot_kv_and_check_pcc_mla(table, device_map: dict, slot_id: int, real_len: int, trace_dir):
    """Read slot `slot_id`'s KV over [0, real_len) via the table and validate it. Config 0 (the KVPE
    cache) is PCC'd vs the golden trace. For a sparse/DSA model the merged table also carries config 1
    (the index-key cache), which has NO golden — it is sanity-checked (finite + non-zero over its written
    layers) so a broken config-1 migration / address table still fails. Returns the min KVPE PCC across
    layers; raises on an index-cache sanity failure."""
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    KV_LORA = ADAPTER.model_config.KV_LORA_RANK  # "nope" part: device_kv[:, :KV_LORA]
    HEAD_DIM = KV_LORA + ADAPTER.model_config.QK_ROPE_HEAD_DIM  # + rope "pe" part: device_kv[:, KV_LORA:]
    tokens_per_block = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    read_len = ((real_len + tokens_per_block - 1) // tokens_per_block) * tokens_per_block  # round up to a block

    min_pcc = 1.0
    for layer in range(NUM_LAYERS):
        # Read this layer's KV block by block over UMD, decode each bfp8 chunk, concat to [real_len, 576].
        decoded_rows = []
        for pos in range(0, read_len, tokens_per_block):
            loc = table.lookup(layer, pos, slot_id)
            unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
            decoded_rows.append(_decode_bfp8_chunk(raw, HEAD_DIM))
        device_kv = torch.cat(decoded_rows, dim=0)[:real_len]  # natural order (table un-rotates block-cyclic)

        golden = _load_golden_kv_post(trace_dir, layer, real_len)
        _, pcc_nope = comp_pcc(golden[:, :KV_LORA], device_kv[:, :KV_LORA])

        # The rope "pe" columns are stored interleaved in HF layout; re-interleave the golden to Meta layout.
        golden_pe = golden[:, KV_LORA:]
        pe_dim = golden_pe.shape[-1]
        golden_pe = torch.stack([golden_pe[:, : pe_dim // 2], golden_pe[:, pe_dim // 2 :]], dim=-1).reshape(-1, pe_dim)
        _, pcc_pe = comp_pcc(golden_pe, device_kv[:, KV_LORA:])

        min_pcc = min(min_pcc, pcc_nope, pcc_pe)

    logger.info(f"[producer] slot {slot_id} KV PCC over [0,{real_len}) across {NUM_LAYERS} layers -> {min_pcc:.6f}")

    # config 1: index cache (sparse/DSA only). No golden -> sanity-check finite + non-zero so a broken
    # config-1 migration / address table still fails. Config 1 holds all layers on GLM-5.1 and only the
    # full-indexer layers on GLM-5.2, so iterate its OWN layer count, not NUM_LAYERS.
    if table.num_configs() > 1:
        index_head_dim = ADAPTER.model_config.INDEX_HEAD_DIM
        n_index_layers = table.config(1).num_layers
        empty = []
        for layer in range(n_index_layers):
            decoded_rows = []
            for pos in range(0, read_len, tokens_per_block):
                loc = table.lookup(layer, pos, slot_id, 1)  # config 1 = index cache
                unique_id = _resolve_unique_id(
                    table.get_device_group(loc.device_group_index).fabric_node_ids, device_map
                )
                raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
                decoded_rows.append(_decode_bfp8_chunk(raw, index_head_dim))
            dev_ik = torch.cat(decoded_rows, dim=0)[:real_len]
            if not torch.isfinite(dev_ik).all():
                raise AssertionError(
                    f"[producer] index cache slot={slot_id} layer={layer} has non-finite values "
                    "(bad config-1 migration or address table)"
                )
            if dev_ik.abs().sum() == 0:
                empty.append(layer)
        if empty:
            raise AssertionError(
                f"[producer] index cache slot={slot_id} all-zero over [0,{real_len}) for layers {empty} "
                "(expected written index keys — bad config-1 migration or address table)"
            )
        logger.info(
            f"[producer] slot {slot_id} index cache sanity OK over [0,{real_len}) across {n_index_layers} "
            "layers (config 1; no golden -> finite + non-zero check only)"
        )

    return min_pcc


def _verify_resident_slots(kv_table, stats: RunStats, threshold: float) -> bool:
    """PCC-check every slot that holds resident trace-derived KV. Returns True only if at least one slot
    was checked and all of them met the threshold."""
    device_map = _read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error("[producer] no device map available; skipping KV read/PCC.")
        return False
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))

    min_pcc_overall = 1.0
    checked = 0
    failures = []
    for slot_id, (chunks_pushed, actual_isl) in sorted(stats.resident.items()):
        real_len = min(chunks_pushed * CHUNK_SIZE, actual_isl)
        if real_len <= 0:
            continue
        pcc = _read_slot_kv_and_check_pcc(kv_table, device_map, slot_id, real_len, trace_dir)
        min_pcc_overall = min(min_pcc_overall, pcc)
        checked += 1
        if pcc < threshold:
            failures.append((slot_id, real_len, pcc))

    print(f"[producer] kv_cache_pcc_complete slots_checked={checked} min_pcc={min_pcc_overall:.6f}")
    if failures:
        logger.error(f"[producer] KV cache PCC below {threshold} for (slot, real_len, pcc): {failures}")
        return False
    if not checked:
        logger.error("[producer] verify requested but no resident slots had data to check.")
        return False
    logger.success(f"[producer] KV cache PCC PASSED (min {min_pcc_overall:.6f} >= {threshold} across {checked} slots)")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list, p: float) -> float:
    """p-th percentile (p in 0..1) of an already-sorted list; 0.0 if empty."""
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(p * len(sorted_values)))]


def _load_token_pool(trace_dir, num_tokens: int) -> list:
    """The shared token pool every request replays from chunk 0, padded up to `num_tokens` if the
    trace is shorter."""
    pool = load_trace_token_ids(trace_dir, num_tokens)
    if len(pool) < num_tokens:
        pool = pool + [1] * (num_tokens - len(pool))
    return pool[:num_tokens]


def main() -> None:
    cfg = _config_from_env()
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(
        f"[producer] service_id={service_id!r} users={cfg.num_users} chunks=[{cfg.chunks_min},{cfg.chunks_max}] "
        f"max_requests={cfg.max_requests} duration={cfg.duration_s}s p_gap={cfg.p_gap} p_burst={cfg.p_burst} "
        f"mid_end={cfg.mid_chunk_end_prob} interleave={cfg.interleave} verify={cfg.verify} seed={cfg.seed}"
    )

    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    payload_bytes = service.payload_size_bytes()
    logger.info(f"[producer] attached; payload={payload_bytes}B")

    # Read the KV table + attach the LayerAck channel BEFORE pushing (the runner publishes them at setup).
    kv_table = _read_kv_chunk_table(timeout_s)
    ack_channel = _connect_layer_ack_channel(timeout_s)

    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))
    token_pool = _load_token_pool(trace_dir, cfg.chunks_max * CHUNK_SIZE)

    def push_chunk(slot_id: int, chunk_idx: int, actual_start: int, actual_end: int) -> float:
        chunk_bytes = _chunk_to_host_array(token_pool[actual_start : actual_start + CHUNK_SIZE])
        assert (
            chunk_bytes.nbytes == payload_bytes
        ), f"payload {chunk_bytes.nbytes}B != service-expected {payload_bytes}B"
        logger.info(f"[producer] push slot={slot_id} cidx={chunk_idx} start={actual_start} end={actual_end}")
        push_start = time.perf_counter()
        service.forward_to_tensor_bytes(chunk_bytes, metadata=_pack_metadata(slot_id, actual_start, actual_end))
        return (time.perf_counter() - push_start) * 1000.0

    stats = run_schedule(cfg, push_fn=push_chunk)
    service.barrier()

    sorted_ms = sorted(stats.push_ms)
    total_tokens = stats.total_pushes * CHUNK_SIZE
    logger.info(
        f"[producer] DONE wall={stats.wall_s:.1f}s pushes={stats.total_pushes} requests={stats.completed} "
        f"tokens={total_tokens} throughput={total_tokens / stats.wall_s if stats.wall_s else 0:.0f} tok/s "
        f"push_ms p50={_percentile(sorted_ms, 0.5):.1f} p90={_percentile(sorted_ms, 0.9):.1f} "
        f"p99={_percentile(sorted_ms, 0.99):.1f}"
    )

    # Wait for the runner's per-layer LayerAcks: NUM_LAYERS per chunk, for every chunk pushed.
    _drain_layer_acks(ack_channel, NUM_LAYERS * stats.total_pushes)

    # Opt-in: read the generated KV back per resident slot and PCC-check vs the golden trace.
    verify_ok = True
    if cfg.verify and kv_table is not None:
        try:
            verify_ok = _verify_resident_slots(kv_table, stats, cfg.pcc_threshold)
        except Exception as e:
            logger.error(f"[producer] KV read/PCC failed: {type(e).__name__}: {e}")
            verify_ok = False
    elif cfg.verify:
        logger.error("[producer] PREFILL_PRODUCER_CHECK_PCC=1 but no KV chunk table available; skipping PCC.")
        verify_ok = False

    # Optional graceful shutdown (PR #48718): close the stream with an all -1 PrefillMetadata sentinel so
    # the runner breaks its request loop and tears down cleanly instead of blocking to SIGKILL. Sent LAST,
    # after the KV read, because read_dram_umd needs the mesh/DRAM alive (the runner is idle until now).
    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        assert len(sentinel) == METADATA_SIZE_BYTES
        sentinel_payload = _chunk_to_host_array(token_pool[:CHUNK_SIZE])  # ignored by the runner; size must match
        assert sentinel_payload.nbytes == payload_bytes
        logger.info("[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(sentinel_payload, metadata=sentinel)
        service.barrier()  # drain the sentinel before releasing the descriptor
        logger.info("[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.")
    else:
        logger.info("[producer] exiting (the runner keeps its sync-op loop running).")

    # Non-zero exit on PCC failure so a CI / scripted run can gate on the exit code (after the sentinel,
    # so the runner is still told to drain even when verification failed).
    if cfg.verify and not verify_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
