#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Parametrized H2D producer: the multi-user stress driver (prefill_h2d_stress_producer.py) plus the
LayerAck + KV-cache-PCC verification from prefill_h2d_producer.py.

This producer is deliberately MODE-UNAWARE: it knows nothing about "single" vs "stress". It takes a
flat ProducerConfig (num_users, per-request chunk count, gap/burst probabilities, mid-chunk-end
probability, request/duration bounds, verify) and executes whatever push schedule those knobs
describe. Named scenarios (single-user in-order, multi-user stress, bursty, mid-chunk-ends, ...) live
in the test suite (tests/test_scenario_producer.py) as ProducerConfig data that parametrizes the
engine — the producer never enumerates them. Single-user in-order is simply num_users=1, one request,
no gaps/bursts, chunk-aligned; there is no code path named for it.

On top of the schedule it drains per-layer LayerAcks (NUM_LAYERS per chunk, regardless of slot — see
tt_prefill_runtime.py) and, opt-in, PCC-checks each resident slot's KV against the golden trace. Per
slot it records the resident (last) request's (chunks_pushed, actual_isl); since every request replays
the same trace pool from chunk 0, KV[0 : chunks_pushed*CHUNK] holds golden[0 : chunks_pushed*CHUNK],
so we PCC the real [0, real_len) region. Only the last resident request per slot is checkable
(recycling overwrites from position 0); pad positions beyond actual_isl are excluded.

run_schedule() takes injectable seams (push_fn / now_fn / sleep_fn / rng) so it can be unit-tested
deterministically with no device (see tests/test_scenario_producer.py).

Env (flat schedule knobs — no preset/mode names; defaults = a plain 1-user, 11-chunk, in-order run):
  PREFILL_NUM_USERS              concurrent cache slots (default 1)
  PREFILL_PRODUCER_CHUNKS        per-request chunks: "N" fixed, or "min,max" random (default "11")
  PREFILL_PRODUCER_MAX_REQUESTS  total requests across slots before draining (default 1)
  PREFILL_PRODUCER_DURATION_S    wall-clock bound (default "inf" = count-bound)
  PREFILL_PRODUCER_P_GAP         per-step probability of an idle gap (default 0.0)
  PREFILL_PRODUCER_P_BURST       per-step probability of a 2-3 chunk burst (default 0.0)
  PREFILL_PRODUCER_GAP_MS        "min,max" idle gap ms (default "200,2000")
  PREFILL_PRODUCER_MID_END_PROB  prob a request ends mid-chunk (default 0.0)
  PREFILL_PRODUCER_INTERLEAVE    slot-selection policy: "random" (default) | "round_robin"
  PREFILL_PRODUCER_SEED          RNG seed (default 1234)
  PREFILL_PRODUCER_CHECK_PCC     1 to read KV back + PCC vs golden per resident slot (default 0)
  PREFILL_SEND_SHUTDOWN          1 to close the stream with an all -1 sentinel so the runner exits
                                 gracefully after the run (sent AFTER the KV read; default 0). See PR #48718.
Env (transport — must match the runner): PREFILL_SP / PREFILL_TP / PREFILL_CHUNK_SIZE /
  PREFILL_MAX_SEQ_LEN / PREFILL_NUM_LAYERS / PREFILL_H2D_SERVICE_ID / PREFILL_H2D_CONNECT_TIMEOUT.

Usage:
    # 1-user in-order + PCC (equivalent to prefill_h2d_producer.py):
    PREFILL_PRODUCER_CHUNKS=11 PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_h2d_scenario_producer
    # 8-user stress + verify every resident slot:
    PREFILL_NUM_USERS=8 PREFILL_PRODUCER_CHUNKS=1,4 PREFILL_PRODUCER_MAX_REQUESTS=200 \
      PREFILL_PRODUCER_P_GAP=0.2 PREFILL_PRODUCER_P_BURST=0.3 PREFILL_PRODUCER_MID_END_PROB=0.33 \
      PREFILL_PRODUCER_CHECK_PCC=1 \
      python -m models.demos.common.prefill.runners.prefill_h2d_scenario_producer
"""

import os
import random
import struct
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

import ttnn
from models.demos.common.prefill.adapter import DEFAULT_MODEL, get_adapter
from models.demos.common.prefill.runners.runner_utils import load_trace_token_ids, resolve_trace_dir

_METADATA_SIZE_BYTES = 12

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 60 * 1024))
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
    """3 little-endian uint32s [slot_id, actual_start, actual_end] — the runner's PrefillMetadata."""
    return struct.pack("<III", slot_id, actual_start, actual_end)


def _chunk_to_host_array(chunk_token_ids):
    """Un-sharded per-chunk token buffer [sp_factor, 1, chunk_local] uint32, block-cyclic (chip-major)
    to match the runner's prepare_prefill_input_tensor (is_balanced=False); the connected service
    splits it across SP coords via its descriptor-rebuilt mapper."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    chunk_local = CHUNK_SIZE // sp_factor
    return (
        torch.tensor(chunk_token_ids, dtype=torch.int64)
        .reshape(sp_factor, 1, chunk_local)
        .to(torch.uint32)
        .contiguous()
        .numpy()
    )


# ---------------------------------------------------------------------------
# Device-less KV table / device map / LayerAck / bfp8 decode helpers.
# Copied from prefill_h2d_producer.py so this producer is self-contained (identical logic; kept
# separate for side-by-side comparison of the three producers).
# ---------------------------------------------------------------------------


def _read_kv_chunk_table(timeout_s: int):
    """Read (deserialize) the KV chunk address table the runner serialized for this galaxy.

    The runner publishes it (PREFILL_MOCK_MIGRATION=1 → runtime.build_kv_chunk_table, or the full
    migration path) to PREFILL_MIGRATION_TABLE_PATH. This is fully device-less:
    import_from_protobuf_file rebuilds the KvChunkAddressTable from the protobuf alone (no device /
    ControlPlane). One galaxy => one complete table spanning all layers/slots. Polls for the file
    since the runner writes it during setup, possibly just after exporting the H2D descriptor.

    Returns the table, or None (so the producer can still push input even if the table isn't there).
    """
    table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
    t0 = time.perf_counter()
    while not os.path.exists(table_path):
        if time.perf_counter() - t0 > timeout_s:
            logger.warning(
                f"[producer] KV chunk table {table_path} not found after {timeout_s}s; "
                f"run the runner with PREFILL_MOCK_MIGRATION=1 to publish it. Skipping table read."
            )
            return None
        time.sleep(0.1)
    import_fn = getattr(ttnn.experimental.disaggregation, "import_from_protobuf_file", None)
    if import_fn is None:
        logger.error(
            "[producer] ttnn.experimental.disaggregation.import_from_protobuf_file is missing — "
            "rebuild ttnn after adding the import binding (disaggregation.cpp). Skipping table read."
        )
        return None
    table = import_fn(table_path)
    cfg = table.config()
    logger.info(
        f"[producer] read KV chunk table {table_path}: entries={table.total_entries()} "
        f"num_layers={cfg.num_layers} num_slots={cfg.num_slots} max_seq_len={cfg.max_sequence_length} "
        f"chunk_n_tokens={cfg.chunk_n_tokens} chunk_size_bytes={cfg.chunk_size_bytes}"
    )
    return table


def _read_device_map(timeout_s: int) -> dict:
    """Read the runner's fabric_node -> ASIC unique_id device-map sidecar (JSON) so the device-less
    UMD read (read_dram_umd) can select chips by unique_id without touching the ControlPlane. Returns
    {(mesh_id, chip_id): unique_id}, or {} if absent. Polls like the table (runner writes it at setup)."""
    import json

    path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
    t0 = time.perf_counter()
    while not os.path.exists(path):
        if time.perf_counter() - t0 > timeout_s:
            logger.warning(f"[producer] device map {path} not found after {timeout_s}s; skipping KV read.")
            return {}
        time.sleep(0.1)
    with open(path) as mp:
        raw = json.load(mp)
    device_map = {tuple(int(x) for x in key.split(":")): int(uid) for key, uid in raw.items()}
    logger.info(f"[producer] read device map {path}: {len(device_map)} chips")
    return device_map


def _connect_layer_ack_channel(timeout_s: int):
    """Attach (consumer side) to the runner's per-layer LayerAck channel. The single-rank runner
    creates `/tt_prefill_layer_acks_<service_id>` and injects 1 per layer (NUM_LAYERS per chunk);
    this connects and drains the deltas. Returns the channel, or None if it isn't available."""
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    shm = f"/tt_prefill_layer_acks_{service_id}"
    try:
        ch = ttnn.InterProcessCounterChannel.connect(shm, connect_timeout_ms=timeout_s * 1000)
    except Exception as e:
        logger.warning(
            f"[producer] could not connect LayerAck channel {shm}: {e} "
            f"(only the single-rank runner creates it). Skipping ack wait."
        )
        return None
    logger.info(f"[producer] connected LayerAck channel {shm}")
    return ch


def _drain_layer_acks(ack_channel, expected: int, timeout_s: float = 600.0) -> int:
    """Block until `expected` (= NUM_LAYERS * n_chunks) per-layer acks have been drained, or timeout.
    Non-blocking poll of try_consume_all(); returns the count actually drained."""
    if ack_channel is None:
        return 0
    got = 0
    last_logged = -1
    t0 = time.perf_counter()
    while got < expected:
        got += ack_channel.try_consume_all()
        if got != last_logged:
            logger.info(f"[producer] layer acks {got}/{expected}")
            last_logged = got
        if got >= expected:
            break
        if time.perf_counter() - t0 > timeout_s:
            logger.warning(f"[producer] timed out at {got}/{expected} acks after {timeout_s}s")
            break
        time.sleep(0.01)
    logger.info(f"[producer] drained {got}/{expected} layer acks in {(time.perf_counter() - t0):.2f}s")
    return got


def _decode_bfp8_chunk(raw: bytes, head_dim: int) -> torch.Tensor:
    """Decode a [1, 1, 32, head_dim] bfp8_b TILE chunk (raw device bytes) to a torch float32
    [32, head_dim] in PURE numpy — no ttnn tensor ops, so it never initializes tt-metal's device
    context (which would start_device and block on the CHIP_IN_USE lock the runner holds). Validated
    bit-exact against ttnn._ttnn.bfp_utils.unpack_bfp8. Per 1088-byte tile: [64 exponent bytes, one
    per (face, row)] then [1024 mantissa bytes, (face, row, col)]; value = (-1)^sign * (mant & 0x7F) *
    2^(exp - 133); face f = fr*2 + fc maps to tile rows fr*16+r, cols fc*16+c; tiles lie along the
    head_dim (column) axis."""
    tile = 32
    n_tiles = head_dim // tile
    b = np.frombuffer(raw, dtype=np.uint8).reshape(n_tiles, 1088)
    exps = b[:, :64].astype(np.int32).reshape(n_tiles, 4, 16)
    mants = b[:, 64:].reshape(n_tiles, 4, 16, 16)
    signs = (mants >> 7).astype(np.int32)
    m7 = (mants & 0x7F).astype(np.float32)
    scale = np.exp2((exps - 133).astype(np.float32))[..., None]
    vals = np.where(signs > 0, -(m7 * scale), m7 * scale)  # (T, face, row, col)
    v = vals.reshape(n_tiles, 2, 2, 16, 16).transpose(0, 1, 3, 2, 4).reshape(n_tiles, tile, tile)
    out = v.transpose(1, 0, 2).reshape(tile, n_tiles * tile)  # tile t -> cols [t*32:(t+1)*32]
    return torch.from_numpy(np.ascontiguousarray(out))


def _resolve_unique_id(fabric_node_ids, device_map: dict) -> int:
    """Return the ASIC unique_id for any replica fabric node present in the device map. Replicas hold
    byte-identical KV, so any that is mapped works; add_device_group sorts the ids, so we can't assume
    index 0 is a specific chip. Raises a clear error if none are mapped (e.g. a multi-rank table whose
    remote device groups aren't in this single-rank/one-galaxy sidecar)."""
    for fnid in fabric_node_ids:
        key = (int(fnid.mesh_id), int(fnid.chip_id))
        if key in device_map:
            return device_map[key]
    keys = [(int(f.mesh_id), int(f.chip_id)) for f in fabric_node_ids]
    raise KeyError(f"no fabric node {keys} in device map ({len(device_map)} chips; single-rank/one-galaxy only)")


# ---------------------------------------------------------------------------
# Config + scheduler engine (mode-unaware; testable — no device/ttnn in run_schedule)
# ---------------------------------------------------------------------------


@dataclass
class ProducerConfig:
    """A flat push schedule. No named modes — a scenario is just a set of these values."""

    num_users: int
    chunks_min: int  # per-request chunk count is rng.randint(chunks_min, chunks_max)
    chunks_max: int
    max_requests: int  # total requests across all slots before draining
    duration_s: float  # wall-clock bound (inf for count-bound runs)
    p_gap: float  # per-step probability of an idle gap
    p_burst: float  # per-step probability of a 2-3 chunk burst
    gap_ms: tuple  # (min, max) idle gap in ms
    mid_chunk_end_prob: float  # probability a request ends mid-chunk (exercises the actual_end clamp)
    seed: int
    verify: bool  # read KV back + PCC each resident slot vs golden
    pcc_threshold: float
    interleave: str = "random"  # slot-selection policy: "random" | "round_robin" (fair alternation)


def _config_from_env() -> ProducerConfig:
    """Flat env -> ProducerConfig. No preset/scenario names; every knob is independent. The neutral
    defaults happen to describe a 1-user, 11-chunk, single-request in-order run."""
    max_chunks_cap = MAX_SEQ_LEN // CHUNK_SIZE
    parts = [int(x) for x in os.environ.get("PREFILL_PRODUCER_CHUNKS", "11").split(",")]
    chunks_min, chunks_max = parts[0], parts[-1]
    chunks_max = min(chunks_max, max_chunks_cap)
    chunks_min = min(chunks_min, chunks_max)
    gap_lo, gap_hi = (float(x) for x in os.environ.get("PREFILL_PRODUCER_GAP_MS", "200,2000").split(","))
    interleave = os.environ.get("PREFILL_PRODUCER_INTERLEAVE", "random")
    if interleave not in ("random", "round_robin"):
        raise ValueError(f"PREFILL_PRODUCER_INTERLEAVE must be 'random' or 'round_robin', got {interleave!r}")
    return ProducerConfig(
        interleave=interleave,
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
        pcc_threshold=float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
    )


class _Slot:
    """One cache user slot = one in-flight request. next_chunk advances per push; actual_isl is the
    request's real (non-pad) token count so the final chunk reports a clamped actual_end."""

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
    slot.req_id = req_id
    slot.target_chunks = rng.randint(cfg.chunks_min, cfg.chunks_max)
    slot.next_chunk = 0
    full = slot.target_chunks * CHUNK_SIZE
    if cfg.mid_chunk_end_prob > 0 and rng.random() < cfg.mid_chunk_end_prob and slot.target_chunks >= 1:
        slot.actual_isl = full - rng.randint(1, CHUNK_SIZE - 1)
    else:
        slot.actual_isl = full


@dataclass
class RunStats:
    resident: dict  # slot_id -> (chunks_pushed_for_resident_request, actual_isl)
    total_pushes: int
    push_ms: list
    completed: int
    wall_s: float


def run_schedule(cfg: ProducerConfig, *, push_fn, now_fn=time.perf_counter, sleep_fn=time.sleep, rng=None):
    """Execute the push schedule described by `cfg`. Pure of device/ttnn — `push_fn(slot_id, chunk_idx,
    actual_start, actual_end) -> dt_ms` performs (and times) the actual push; now_fn/sleep_fn/rng are
    injectable so tests run the full schedule instantly and deterministically.

    Records, per slot, the resident request as (chunks_pushed, actual_isl) at push time — this is what
    is physically in the KV cache (a recycled slot's chunk 0 overwrites from position 0), which the
    caller PCC-checks. Returns RunStats."""
    rng = rng if rng is not None else random.Random(cfg.seed)
    slots = [_Slot(i) for i in range(cfg.num_users)]
    resident: dict = {}
    req = 0
    for s in slots:
        _new_request(s, req, cfg, rng)
        req += 1

    push_ms: list = []
    completed = 0
    total = 0
    rr_cursor = -1  # round-robin slot cursor (used only when cfg.interleave == "round_robin")
    t0 = now_fn()

    def _do_push(slot: _Slot) -> None:
        nonlocal completed, req, total
        c = slot.next_chunk
        actual_start = c * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        dt = push_fn(slot.slot_id, c, actual_start, actual_end)
        push_ms.append(dt)
        total += 1
        slot.next_chunk += 1
        # Resident = what is now in this slot's KV cache: the request that pushed up to chunk c.
        resident[slot.slot_id] = (c + 1, slot.actual_isl)
        if slot.done:
            completed += 1
            if req < cfg.max_requests:
                _new_request(slot, req, cfg, rng)  # recycle the slot as a fresh request
                req += 1

    while (now_fn() - t0) < cfg.duration_s and completed < cfg.max_requests:
        active = [s for s in slots if not s.done]
        if not active:
            break
        r = rng.random()
        if r < cfg.p_gap:
            sleep_fn(rng.uniform(*cfg.gap_ms) / 1000.0)
            continue
        if cfg.interleave == "round_robin":
            # Advance a persistent cursor to the next non-done slot: fair alternation across users
            # (active is non-empty here, so this finds one within a single pass over the slots).
            for _ in range(len(slots)):
                rr_cursor = (rr_cursor + 1) % len(slots)
                if not slots[rr_cursor].done:
                    break
            slot = slots[rr_cursor]
        else:
            slot = rng.choice(active)
        if r < cfg.p_gap + cfg.p_burst:
            for _ in range(rng.randint(2, 3)):
                if slot.done:
                    break
                _do_push(slot)
        else:
            _do_push(slot)

    return RunStats(resident=resident, total_pushes=total, push_ms=push_ms, completed=completed, wall_s=now_fn() - t0)


# ---------------------------------------------------------------------------
# Per-slot KV read + PCC (the verification feature from prefill_h2d_producer, per resident slot)
# ---------------------------------------------------------------------------


def _read_slot_kv_and_check_pcc(table, device_map: dict, slot_id: int, real_len: int, threshold: float, trace_dir):
    """Read slot `slot_id`'s KV over [0, real_len) via the table and PCC-check vs the golden trace.
    Device-less: read_dram_umd over UMD + pure-numpy bfp8 decode, so it never inits the device context
    (no start_device / CHIP_IN_USE lock)."""
    from models.demos.deepseek_v3_d_p.tt.runners.prefill_kv_validation import _load_golden_kv_post
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    from tests.ttnn.utils_for_testing import comp_pcc

    KVPE_HEAD_DIM = 576  # qk_rope_head_dim(64) + kv_lora_rank(512)
    KV_LORA = 512
    chunk_tok = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    read_len = ((real_len + chunk_tok - 1) // chunk_tok) * chunk_tok  # round up to a whole DRAM block

    min_pcc = 1.0
    for layer in range(NUM_LAYERS):
        rows = []
        for pos in range(0, read_len, chunk_tok):
            loc = table.lookup(layer, pos, slot_id)
            unique_id = _resolve_unique_id(table.get_device_group(loc.device_group_index).fabric_node_ids, device_map)
            raw = ttnn.experimental.disaggregation.read_dram_umd(unique_id, loc.noc_addr, loc.size_bytes)
            rows.append(_decode_bfp8_chunk(raw, KVPE_HEAD_DIM))
        dev = torch.cat(rows, dim=0)[:real_len]  # natural order (table un-rotates block-cyclic)
        g = _load_golden_kv_post(trace_dir, layer, real_len)
        _, pcc_nope = comp_pcc(g[:, :KV_LORA], dev[:, :KV_LORA])
        ref_pe = g[:, KV_LORA:]
        d = ref_pe.shape[-1]
        ref_pe = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe, dev[:, KV_LORA:])
        min_pcc = min(min_pcc, pcc_nope, pcc_pe)
    logger.info(f"[producer] slot {slot_id} KV PCC over [0,{real_len}) across {NUM_LAYERS} layers -> {min_pcc:.6f}")
    return min_pcc


def _verify_resident_slots(kv_table, stats: RunStats, threshold: float) -> None:
    """PCC-check every slot that holds resident trace-derived KV. Reads the device map lazily."""
    device_map = _read_device_map(int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60")))
    if not device_map:
        logger.error("[producer] no device map available; skipping KV read/PCC.")
        return
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))
    overall = 1.0
    checked = 0
    failures = []
    for slot_id, (chunks_pushed, actual_isl) in sorted(stats.resident.items()):
        real_len = min(chunks_pushed * CHUNK_SIZE, actual_isl)
        if real_len <= 0:
            continue
        pcc = _read_slot_kv_and_check_pcc(kv_table, device_map, slot_id, real_len, threshold, trace_dir)
        overall = min(overall, pcc)
        checked += 1
        if pcc < threshold:
            failures.append((slot_id, real_len, pcc))
    print(f"[producer] kv_cache_pcc_complete slots_checked={checked} min_pcc={overall:.6f}")
    if failures:
        logger.error(f"[producer] KV cache PCC below {threshold} for (slot, real_len, pcc): {failures}")
    elif checked:
        logger.success(f"[producer] KV cache PCC PASSED (min {overall:.6f} >= {threshold} across {checked} slots)")


def main() -> None:
    cfg = _config_from_env()
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "60"))
    logger.info(
        f"[producer] service_id={service_id!r} users={cfg.num_users} chunks=[{cfg.chunks_min},{cfg.chunks_max}] "
        f"max_requests={cfg.max_requests} duration={cfg.duration_s}s p_gap={cfg.p_gap} p_burst={cfg.p_burst} "
        f"mid_end={cfg.mid_chunk_end_prob} verify={cfg.verify} seed={cfg.seed}"
    )

    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    expected = service.payload_size_bytes()
    logger.info(f"[producer] attached; payload={expected}B")

    # Read the KV table + attach the LayerAck channel BEFORE pushing (the runner publishes them at setup).
    kv_table = _read_kv_chunk_table(timeout_s)
    ack_channel = _connect_layer_ack_channel(timeout_s)

    # One shared token pool, reused across users/requests; each request starts at chunk 0.
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))
    pool_len = cfg.chunks_max * CHUNK_SIZE
    pool = load_trace_token_ids(trace_dir, pool_len)
    if len(pool) < pool_len:
        pool = pool + [1] * (pool_len - len(pool))
    pool = pool[:pool_len]

    def _push(slot_id: int, chunk_idx: int, actual_start: int, actual_end: int) -> float:
        host = _chunk_to_host_array(pool[actual_start : actual_start + CHUNK_SIZE])
        assert host.nbytes == expected, f"payload {host.nbytes}B != service-expected {expected}B"
        metadata = _pack_metadata(slot_id, actual_start, actual_end)
        logger.info(f"[producer] push slot={slot_id} cidx={chunk_idx} start={actual_start} end={actual_end}")
        t = time.perf_counter()
        service.forward_to_tensor_bytes(host, metadata=metadata)
        return (time.perf_counter() - t) * 1000.0

    stats = run_schedule(cfg, push_fn=_push)
    service.barrier()

    sp = sorted(stats.push_ms)
    tok = stats.total_pushes * CHUNK_SIZE

    def _pct(p):
        return sp[min(len(sp) - 1, int(p * len(sp)))] if sp else 0.0

    logger.info(
        f"[producer] DONE wall={stats.wall_s:.1f}s pushes={stats.total_pushes} requests={stats.completed} "
        f"tokens={tok} throughput={tok / stats.wall_s if stats.wall_s else 0:.0f} tok/s "
        f"push_ms p50={_pct(0.5):.1f} p90={_pct(0.9):.1f} p99={_pct(0.99):.1f}"
    )

    # Wait for the runner's per-layer LayerAcks: NUM_LAYERS per chunk, for every chunk pushed.
    _drain_layer_acks(ack_channel, NUM_LAYERS * stats.total_pushes)

    # Opt-in: read the generated KV back per resident slot and PCC-check vs the golden trace.
    if cfg.verify and kv_table is not None:
        try:
            _verify_resident_slots(kv_table, stats, cfg.pcc_threshold)
        except Exception as e:
            logger.error(f"[producer] KV read/PCC failed: {type(e).__name__}: {e}")
    elif cfg.verify:
        logger.error("[producer] PREFILL_PRODUCER_CHECK_PCC=1 but no KV chunk table available; skipping PCC.")

    # Optional graceful shutdown (mirrors prefill_h2d_producer.py / PR #48718): close the request stream
    # with an all -1 PrefillMetadata sentinel so the runner breaks its request loop and tears down
    # cleanly instead of blocking to SIGKILL. Sent LAST — AFTER the KV read/PCC above — because
    # read_dram_umd needs the mesh/DRAM still alive; until now the runner is idle (blocked on recv).
    if os.environ.get("PREFILL_SEND_SHUTDOWN", "0") == "1":
        sentinel = struct.pack("<iii", -1, -1, -1)
        assert len(sentinel) == _METADATA_SIZE_BYTES, f"sentinel {len(sentinel)}B != {_METADATA_SIZE_BYTES}B"
        dummy = _chunk_to_host_array(pool[:CHUNK_SIZE])  # payload is ignored for a sentinel; size must still match
        assert dummy.nbytes == expected, f"sentinel payload {dummy.nbytes}B != service-expected {expected}B"
        logger.info("[producer] sending SHUTDOWN sentinel (metadata=-1,-1,-1)")
        service.forward_to_tensor_bytes(dummy, metadata=sentinel)
        service.barrier()  # drain the sentinel before releasing the descriptor
        logger.info("[producer] exiting; SHUTDOWN sentinel sent — runner will drain and shut down.")
    else:
        logger.info("[producer] exiting (the runner keeps its sync-op loop running).")


if __name__ == "__main__":
    main()
