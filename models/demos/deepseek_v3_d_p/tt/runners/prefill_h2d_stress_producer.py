#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Multi-user stress producer for the H2D-streamed prefill_runner (request mode).

Where prefill_h2d_producer.py pushes one user's chunks back-to-back, this drives a realistic mixed
workload against the running request pipeline: many concurrent user slots, chunks arriving interleaved
across users, occasional bursts of consecutive chunks for one user, and idle gaps where nothing is
pushed (the runner blocks in inbound_socket_service_sync). Finished requests recycle their slot as a fresh request
to sustain load. PCC is not meaningful here (users replay the same trace tokens) — the goal is
throughput and stability under load, not correctness.

The runner is stateless per chunk: each push carries its own PrefillMetadata (slot_id, actual_start,
actual_end), so the only invariant this producer must honor is per-slot ordering — a slot's chunks go
out in increasing actual_start (a request prefills in order). Across slots the schedule is free.

Env (must match the runner; see prefill_runner.py / binding global_env):
  PREFILL_NUM_USERS              cache user slots = concurrent users (default 10)
  PREFILL_MAX_SEQ_LEN            per-user cache length; caps a request's chunks (default 20480)
  PREFILL_CHUNK_SIZE / PREFILL_SP / PREFILL_TP   token packing (must match runner)
  PREFILL_H2D_SERVICE_ID         service descriptor id (default "ds_prefill")
  PREFILL_H2D_CONNECT_TIMEOUT    seconds to wait for the descriptor (default 120)
Stress schedule:
  PREFILL_STRESS_DURATION_S      wall-clock run length (default 180)
  PREFILL_STRESS_MAX_REQUESTS    total requests before draining (default 100000 = duration-bound)
  PREFILL_STRESS_SEED            RNG seed (default 1234)
  PREFILL_STRESS_P_GAP           per-step probability of an idle gap (default 0.2)
  PREFILL_STRESS_P_BURST         per-step probability of a 2-3 chunk burst (default 0.3)
  PREFILL_STRESS_GAP_MS          "min,max" idle gap range in ms (default "200,2000")
"""

import os
import random
import struct
import time

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import get_variant, load_trace_token_ids, resolve_trace_dir

_METADATA_SIZE_BYTES = 12

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 20 * 1024))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 10))
MAX_CHUNKS_PER_REQ = MAX_SEQ_LEN // CHUNK_SIZE

VARIANT = get_variant(os.environ.get("PREFILL_MODEL_VARIANT", "deepseek_v3_d_p"))


def _pack_metadata(slot_id: int, actual_start: int, actual_end: int) -> bytes:
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


def _new_request(slot: _Slot, req_id: int, rng: random.Random) -> None:
    slot.req_id = req_id
    slot.target_chunks = rng.randint(1, MAX_CHUNKS_PER_REQ)
    slot.next_chunk = 0
    # 1-in-3 requests end mid-chunk to exercise the actual_end clamp; else chunk-aligned.
    full = slot.target_chunks * CHUNK_SIZE
    if rng.random() < 0.33 and slot.target_chunks >= 1:
        slot.actual_isl = full - rng.randint(1, CHUNK_SIZE - 1)
    else:
        slot.actual_isl = full


def main() -> None:
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    timeout_s = int(os.environ.get("PREFILL_H2D_CONNECT_TIMEOUT", "120"))
    duration_s = float(os.environ.get("PREFILL_STRESS_DURATION_S", "180"))
    max_requests = int(os.environ.get("PREFILL_STRESS_MAX_REQUESTS", "100000"))
    seed = int(os.environ.get("PREFILL_STRESS_SEED", "1234"))
    p_gap = float(os.environ.get("PREFILL_STRESS_P_GAP", "0.2"))
    p_burst = float(os.environ.get("PREFILL_STRESS_P_BURST", "0.3"))
    gap_lo, gap_hi = (float(x) for x in os.environ.get("PREFILL_STRESS_GAP_MS", "200,2000").split(","))
    rng = random.Random(seed)

    logger.info(
        f"[stress] service_id={service_id!r} users={NUM_USERS} max_seq_len={MAX_SEQ_LEN} "
        f"max_chunks/req={MAX_CHUNKS_PER_REQ} duration={duration_s}s seed={seed} "
        f"p_gap={p_gap} p_burst={p_burst} gap_ms=[{gap_lo},{gap_hi}]"
    )

    service = ttnn.H2DStreamService.connect(service_id, timeout_ms=timeout_s * 1000)
    expected = service.payload_size_bytes()
    logger.info(f"[stress] attached; payload={expected}B")

    # One shared token pool reused across users (stress, not correctness). Pad if the trace is short.
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
    pool = load_trace_token_ids(trace_dir, MAX_SEQ_LEN)
    if len(pool) < MAX_SEQ_LEN:
        pool = pool + [1] * (MAX_SEQ_LEN - len(pool))
    pool = pool[:MAX_SEQ_LEN]

    slots = [_Slot(i) for i in range(NUM_USERS)]
    req_counter = 0
    for s in slots:
        _new_request(s, req_counter, rng)
        req_counter += 1

    push_ms: list[float] = []
    completed = 0
    last_push = time.perf_counter()
    t_start = time.perf_counter()

    def _push(slot: _Slot) -> None:
        nonlocal last_push, completed, req_counter
        c = slot.next_chunk
        actual_start = c * CHUNK_SIZE
        actual_end = min(actual_start + CHUNK_SIZE, slot.actual_isl)
        host = _chunk_to_host_array(pool[actual_start : actual_start + CHUNK_SIZE])
        gap_ms = (time.perf_counter() - last_push) * 1000.0
        logger.info(
            f"[stress] t={time.time():.6f} push slot={slot.slot_id} req={slot.req_id} "
            f"cidx={c}/{slot.target_chunks} start={actual_start} end={actual_end} gap_ms={gap_ms:.1f}"
        )
        t = time.perf_counter()
        service.forward_to_tensor_bytes(host, metadata=_pack_metadata(slot.slot_id, actual_start, actual_end))
        dt = (time.perf_counter() - t) * 1000.0
        push_ms.append(dt)
        last_push = time.perf_counter()
        logger.info(f"[stress]   slot={slot.slot_id} cidx={c} ret_ms={dt:.2f}")
        slot.next_chunk += 1
        if slot.done:
            completed += 1
            if req_counter < max_requests:
                _new_request(slot, req_counter, rng)  # recycle the slot as a fresh request
                req_counter += 1

    while (time.perf_counter() - t_start) < duration_s and completed < max_requests:
        active = [s for s in slots if not s.done]
        if not active:
            break
        r = rng.random()
        if r < p_gap:
            gap = rng.uniform(gap_lo, gap_hi) / 1000.0
            logger.info(f"[stress] t={time.time():.6f} GAP {gap * 1000:.0f}ms")
            time.sleep(gap)
            continue
        slot = rng.choice(active)
        if r < p_gap + p_burst:
            for _ in range(rng.randint(2, 3)):
                if slot.done:
                    break
                _push(slot)
        else:
            _push(slot)

    service.barrier()
    wall = time.perf_counter() - t_start
    pushed = len(push_ms)
    tok = pushed * CHUNK_SIZE
    sp = sorted(push_ms)

    def _pct(p):
        return sp[min(len(sp) - 1, int(p * len(sp)))] if sp else 0.0

    logger.info(
        f"[stress] DONE wall={wall:.1f}s pushes={pushed} requests_completed={completed} "
        f"tokens={tok} throughput={tok / wall:.0f} tok/s "
        f"push_ms p50={_pct(0.5):.1f} p90={_pct(0.9):.1f} p99={_pct(0.99):.1f} max={max(sp) if sp else 0:.1f}"
    )


if __name__ == "__main__":
    main()
