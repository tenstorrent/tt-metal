#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DeepSeek-V3 prefill runner — single-rank production serving and N-rank pipeline.

One entry point, two run modes selected at runtime:

  * Request mode (default, single rank): production serving. Token chunks + per-iter
    PrefillMetadata arrive over the H2D socket service from an external producer
    (prefill_h2d_producer.py / the scheduler); the loop is unbounded (runs to SIGTERM).
    Optional KV-chunk-table migration publish and per-layer LayerAck channel.

  * Standalone mode (PREFILL_STANDALONE=1): the model is split across N ranks under tt-run
    — each rank owns a contiguous layer slice and builds the same TtPrefillRuntime,
    parameterized with first_layer_idx / is_first_rank / is_last_rank. With >1 rank the
    cross-rank hidden state moves device-to-device over fabric sockets (needs a connected
    MGD + FABRIC_2D); N=1 is the single-galaxy bring-up case (one rank, no transport). The
    first rank's input is the golden trace (or the H2D socket with PREFILL_PP_H2D=1) for a
    fixed PREFILL_STANDALONE_NCHUNKS chunks. Ranks run decoupled (no per-chunk barrier; one
    warm-up barrier after compile). With PREFILL_STANDALONE_PCC=1 each rank checks the KV
    slice it populated against the global golden trace.

The model class is the single source of truth — this driver wires rank topology, input,
transport, and the per-chunk schedule; it does not reimplement embed / layers / forward.
"""

import os
import signal
import time

from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.d2d_socket_push_op import d2d_socket_push
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    build_h2d_service,
    get_variant,
    load_hf_config,
    load_trace_token_ids,
    open_mesh_device,
    prepare_prefill_input_tensor,
    resolve_trace_dir,
    resolve_weight_cache_path,
)
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

# H2D socket service config (first rank, PREFILL_PP_H2D=1). Mirrors prefill_runner.py: one worker
# core copies the pushed chunk into a fresh tensor; the producer packs a 12-byte PrefillMetadata
# (slot_id, actual_start, actual_end) alongside each push.
H2D_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
H2D_METADATA_SIZE_BYTES = 12
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])


# D2D socket transport (>1 rank): one persistent sender/receiver pair per rank boundary carries the
# sharded hidden state over inter-galaxy fabric. The activation is sharded [seq across SP rows, emb
# across TP cols] — the same layout the embedding output uses — so the receiver backing feeds the
# downstream model with no reshard. The per-chunk metadata (slot/start/end/is_last) rides inline as
# four uint32 words.
def _d2d_worker_grid() -> ttnn.CoreRange:
    """D2D push/sync worker grid, env-configurable as PREFILL_PP_D2D_WORKER_GRID='WxH' (default 1x1).
    A grid sweep showed compute + handoff gap are flat from 1x1 to 4x4 (the per-chunk overhead is the
    persistent service's fabric/NoC presence, not the push workers), so 1x1 is the cheapest footprint
    with no penalty."""
    spec = os.environ.get("PREFILL_PP_D2D_WORKER_GRID", "1x1").lower().split("x")
    w, h = (int(spec[0]), int(spec[1])) if len(spec) == 2 else (int(spec[0]), int(spec[0]))
    return ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(w - 1, h - 1))


D2D_SYNC_WORKER_CORES = _d2d_worker_grid()
D2D_METADATA_SIZE_BYTES = 16
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 64 * 1024))

VARIANT = get_variant(os.environ.get("PREFILL_MODEL_VARIANT", "deepseek_v3_d_p"))
MODEL_CFG = VARIANT.model_config

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
# Chunks this run drives. The per-user KV cache is sized to exactly hold them
# (max_seq_len = chunk_size * num_chunks), so there is no separate cache-length knob to keep in sync.
# PREFILL_MAX_SEQ_LEN still overrides if a larger cache is wanted.
NUM_CHUNKS = int(os.environ.get("PREFILL_STANDALONE_NCHUNKS", 4))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * NUM_CHUNKS))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", VARIANT.default_gate_mode)

os.environ.setdefault("PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def compute_layer_split(num_layers: int, num_ranks: int) -> list[tuple[int, int]]:
    """Contiguous (first_layer_idx, count) per rank. PREFILL_PP_LAYER_COUNTS, a
    comma-separated count list summing to num_layers, overrides the default even
    split (remainder handed to the earlier ranks)."""
    override = os.environ.get("PREFILL_PP_LAYER_COUNTS")
    if override:
        counts = [int(x) for x in override.split(",")]
        if len(counts) != num_ranks or sum(counts) != num_layers:
            raise ValueError(
                f"PREFILL_PP_LAYER_COUNTS={override!r} must list {num_ranks} counts summing to "
                f"{num_layers} (got {len(counts)} counts summing to {sum(counts)})"
            )
    else:
        base, rem = divmod(num_layers, num_ranks)
        counts = [base + (1 if r < rem else 0) for r in range(num_ranks)]

    ranges = []
    start = 0
    for count in counts:
        ranges.append((start, count))
        start += count
    return ranges


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _load_token_ids() -> list[int]:
    """Load this run's token IDs (same source as the single-rank standalone loop).
    All ranks load identically so they agree on the chunk schedule."""
    import json

    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
    input_override = os.environ.get("PREFILL_STANDALONE_INPUT")
    if input_override:
        with open(input_override) as f:
            token_ids = list(json.load(f)["token_ids"])
        logger.info(f"[pp] input override: {len(token_ids)} token_ids from {input_override}")
    else:
        logger.info(f"[pp] reading input token_ids from {trace_dir}/metadata.json")
        token_ids = load_trace_token_ids(trace_dir)
    return token_ids


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _first_rank_chunk_tokens(runtime: TtPrefillRuntime, token_ids: list[int], kv_actual: int) -> ttnn.Tensor:
    cfg = runtime.config
    return prepare_prefill_input_tensor(
        token_ids[kv_actual : kv_actual + cfg.chunk_size],
        runtime.mesh_device,
        cfg.sp_factor,
        False,  # chunked prefill is block-cyclic (non-balanced)
        cfg.mesh_shape,
        cfg.sp_axis,
    )


def _h2d_request_mode() -> bool:
    """First rank reads tokens from the H2D socket (driven by prefill_h2d_producer.py) instead of the
    trace file. Off by default (file input)."""
    return os.environ.get("PREFILL_PP_H2D") == "1"


def _socket_next(h2d_service) -> tuple:
    """Block on the next producer push: returns (tt_tokens, {slot_id, actual_start, actual_end})
    decoded from the 12-byte PrefillMetadata. is_last is set by the caller — the socket carries no
    end-of-stream marker, so the first rank derives it from its chunk count vs PREFILL_STANDALONE_NCHUNKS."""
    import torch

    tt_tokens, tt_metadata = h2d_socket_sync(
        h2d_service, H2D_SYNC_WORKER_CORES, metadata_size_bytes=H2D_METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    return tt_tokens, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def _activation_global_spec(chunk_size: int, hidden_size: int) -> ttnn.TensorSpec:
    """Global spec of the inter-rank hidden state: [1, 1, chunk_size, hidden_size] bf16 TILE DRAM,
    sharded by D2D_MAPPER_CONFIG (seq across SP rows, emb across TP cols) to match the embedding
    output layout the downstream model consumes."""
    return ttnn.TensorSpec(
        shape=ttnn.Shape([1, 1, chunk_size, hidden_size]),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def build_d2d_pipeline_endpoints(mesh_device, rank: int, num_ranks: int, chunk_size: int, hidden_size: int):
    """Stand up this rank's persistent D2D endpoints for the pipeline: an inbound receiver from rank-1
    (every rank but the first) and an outbound sender to rank+1 (every rank but the last). Returns
    (inbound_receiver_or_None, outbound_sender_or_None).

    Setup order is inbound-then-outbound on every rank. create_sender/create_receiver rendezvous
    point-to-point between the two boundary ranks (no world barrier), and each MeshSocket ctor blocks
    until its peer's matching ctor. Doing inbound first chains the bring-up: rank 0's sender unblocks
    rank 1's receiver, which frees rank 1 to build its sender for rank 2's receiver, and so on — no
    deadlock. Both sides pass the identical worker-core grid and global spec."""
    global_spec = _activation_global_spec(chunk_size, hidden_size)

    def _common():
        # Fresh mapper per call: create_sender/create_receiver take the mapper by std::unique_ptr and
        # MOVE it, so a middle rank (builds BOTH a receiver and a sender) must not reuse one — the
        # second create would get a consumed/null mapper and fail overload resolution.
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=D2D_SYNC_WORKER_CORES,
            receiver_worker_cores=D2D_SYNC_WORKER_CORES,
            metadata_size_bytes=D2D_METADATA_SIZE_BYTES,
            share_fabric_links=True,
        )

    inbound = None
    if rank > 0:
        logger.info(f"[pp rank {rank}] [d2d] creating inbound receiver from rank {rank - 1}")
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common()
        )
    outbound = None
    if rank < num_ranks - 1:
        logger.info(f"[pp rank {rank}] [d2d] creating outbound sender to rank {rank + 1}")
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common()
        )
    logger.info(
        f"[pp rank {rank}] [d2d] endpoints up (inbound={'yes' if inbound else 'no'} "
        f"outbound={'yes' if outbound else 'no'}, workers={D2D_SYNC_WORKER_CORES}, fifo={D2D_FIFO_SIZE_BYTES}B)"
    )
    return inbound, outbound


def _d2d_recv(inbound) -> tuple:
    """Drain the next chunk that landed in the inbound receiver backing into a fresh device tensor and
    decode the inline metadata. The returned tensor already has the embedding-output sharding, so it
    feeds runtime.prefill with no reshard. Pairs with the upstream rank's _d2d_send."""
    import torch

    t0 = time.perf_counter()
    act, md = h2d_socket_sync(inbound, D2D_SYNC_WORKER_CORES, metadata_size_bytes=D2D_METADATA_SIZE_BYTES)
    m = ttnn.to_torch(ttnn.get_device_tensors(md)[0]).view(torch.int32).flatten()
    meta = {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2]), "is_last": bool(int(m[3]))}
    logger.info(
        f"[pp] RECV-d2d [{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']} "
        f"is_last={meta['is_last']} [xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )
    return act, meta


def _d2d_send(outbound, activation: ttnn.Tensor, rank: int, meta: dict) -> None:
    """Push this rank's output hidden state + metadata to the downstream rank's receiver. Coerce the
    activation to the sender backing's memory layout first so the op's page-for-page copy lines up,
    then free it."""
    t0 = time.perf_counter()
    backing = outbound.get_backing_tensor()
    if activation.memory_config() != backing.memory_config() or activation.layout != backing.layout:
        activation = ttnn.to_layout(activation, backing.layout)
        activation = ttnn.to_memory_config(activation, backing.memory_config())
    words = [meta["slot_id"], meta["actual_start"], meta["actual_end"], int(bool(meta["is_last"]))]
    d2d_socket_push(outbound, activation, D2D_SYNC_WORKER_CORES, metadata=words)
    ttnn.deallocate(activation)
    logger.info(
        f"[pp rank {rank}] SEND-d2d [{meta['actual_start']},{meta['actual_end']}) is_last={meta['is_last']} "
        f"[xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )


def run_request_loop(runtime: TtPrefillRuntime, h2d_service: ttnn.H2DStreamService) -> None:
    """Production single-rank serving: token IDs + per-iter PrefillMetadata arrive over the H2D socket
    service, pushed by a separate producer (prefill_h2d_producer.py today; the scheduler in production).

    `h2d_socket_sync` returns (tokens, metadata); we decode the 3xuint32 PrefillMetadata
    [slot_id, actual_start, actual_end] and pass them to runtime.prefill (actual_start is the chunk's
    cache write offset; actual_end - actual_start is its real-token count). Push-driven and unbounded —
    the producer/scheduler decides how many chunks a request spans; the loop runs until SIGTERM."""
    import time as _time

    import torch

    logger.info(
        "[request] entering request loop — blocks on h2d_socket_sync for each push, "
        "runs until SIGTERM/SIGINT. Drive it with prefill_h2d_producer.py / the scheduler."
    )

    i = 0
    while not _shutdown:
        # Blocks until the next producer push lands (workers wait on data_ready_sem), so the loop
        # idle-waits here between requests.
        tt_tokens, tt_metadata = h2d_socket_sync(
            h2d_service, H2D_SYNC_WORKER_CORES, metadata_size_bytes=H2D_METADATA_SIZE_BYTES
        )
        meta_host = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
        slot_id, actual_start, actual_end = int(meta_host[0]), int(meta_host[1]), int(meta_host[2])
        logger.info(
            f"[request] iter={i} metadata: slot_id={slot_id} actual_start={actual_start} "
            f"actual_end={actual_end} actual_isl={actual_end - actual_start}"
        )
        _t0 = _time.perf_counter()  # time only the compute, not the idle socket wait above
        runtime.prefill(tt_tokens, slot_id=slot_id, actual_start=actual_start, actual_end=actual_end)
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        logger.info(f"[request] iter={i} chunk slot={slot_id} [{actual_start},{actual_end}) prefill = {_dt_ms:.2f} ms")
        i += 1
    logger.info(f"[request] loop exited after {i} requests")


def run_standalone_loop(
    runtime: TtPrefillRuntime, rank: int, num_ranks: int, h2d_service=None, d2d_in=None, d2d_out=None
) -> None:
    """Decoupled per-rank loop: each rank runs its own loop, coordinated ONLY by the
    activation+metadata file handoff (atomic rename + polling recv) — no cross-rank barrier, so an
    upstream rank can race ahead of a downstream one (pipeline overlap). The first rank's input is
    either the trace file or the H2D socket (h2d_service set); the per-chunk metadata (slot /
    actual_start / actual_end / is_last) rides with each activation, so downstream ranks need no
    shared schedule and learn the end from is_last. Every rank checks the KV slice it populated."""
    cfg = runtime.config
    slot_id = 0  # standalone always fills slot 0 (request mode takes slot_id from the per-push metadata)

    token_ids = None
    n_chunks = NUM_CHUNKS if cfg.is_first_rank else None
    if cfg.is_first_rank:
        # The first rank drives n_chunks (NUM_CHUNKS); later ranks learn the end from is_last. The cache
        # is sized to chunk*num_chunks by default so this fits; the guard only catches a smaller override.
        if h2d_service is None:
            token_ids = _load_token_ids()
            token_ids = (token_ids + [1] * (n_chunks * cfg.chunk_size))[: n_chunks * cfg.chunk_size]
        if n_chunks * cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(
                f"{n_chunks} chunks x {cfg.chunk_size} exceeds per-user cache max_seq_len={cfg.max_seq_len}; "
                f"raise PREFILL_MAX_SEQ_LEN."
            )

    logger.info(
        f"[pp rank {rank}/{num_ranks}] decoupled loop start (is_first={cfg.is_first_rank} "
        f"is_last={cfg.is_last_rank} slot={slot_id} input={'h2d' if h2d_service is not None else 'trace'}"
        + (f" chunks={n_chunks}" if cfg.is_first_rank else "")
        + ")"
    )

    # Opt-in per-chunk timing breakdown. Adds a synchronize_device after compute so the reported
    # compute time is device-accurate (not just enqueue) — only meaningful for an isolated single-chunk
    # measurement, where there is no pipeline overlap to distort.
    time_chunks = os.environ.get("PREFILL_PP_TIME_CHUNKS", "0") == "1"

    t0 = time.perf_counter()
    c = 0
    n_done = 0
    loop_first_compute_start = None  # epoch of this rank's first prefill; rank0's = pipeline e2e start
    while not _shutdown:
        # (1) Reclaim my fabric links before this chunk's compute (their previous-iter D2D transfer has
        #     drained). No-op on iter 0 and in host transport. (2) Grant the inbound receiver BEFORE the
        #     recv so this chunk's incoming activation drains into its backing. The matching grant of the
        #     outbound sender happens AFTER the push, mirroring test_stream_pipeline.cpp's per-host cadence.
        t_lease = time.perf_counter()
        if d2d_in is not None:
            d2d_in.wait_for_fabric_links()
        if d2d_out is not None:
            d2d_out.wait_for_fabric_links()
        if d2d_in is not None:
            d2d_in.release_fabric_links()
        ms_lease = (time.perf_counter() - t_lease) * 1000.0

        if cfg.is_first_rank:
            if c >= n_chunks:
                break
            if h2d_service is not None:
                inp, meta = _socket_next(h2d_service)  # slot/start/end from the producer
                meta["is_last"] = c == n_chunks - 1
            else:
                kv_actual = c * cfg.chunk_size
                inp = _first_rank_chunk_tokens(runtime, token_ids, kv_actual)
                meta = {
                    "slot_id": slot_id,
                    "actual_start": kv_actual,
                    "actual_end": kv_actual + cfg.chunk_size,  # trace drives full chunks
                    "is_last": c == n_chunks - 1,
                }
        else:
            # Non-first rank: the upstream stage's hidden state + metadata arrive over the D2D socket
            # (the only inter-rank transport).
            inp, meta = _d2d_recv(d2d_in)
            slot_id = meta["slot_id"]

        # Per-chunk compute START (epoch, comparable across NTP-synced hosts). Always logged with NO
        # barrier — the real per-chunk wall is the delta between consecutive starts (CHUNK_START), which
        # equals device throughput wherever the host is throttled (the recv's metadata to_torch on every
        # non-first rank; rank0 can race ahead so its deltas are enqueue-bound until the CQ saturates).
        # time_chunks adds a synchronize so compute= reflects device completion (isolated-chunk use only,
        # at the cost of serializing the per-chunk pipeline and over-counting overlapped D2D work).
        t_start_epoch = time.time()
        if loop_first_compute_start is None:
            loop_first_compute_start = t_start_epoch
        t_c = time.perf_counter()
        out = runtime.prefill(
            inp, slot_id=meta["slot_id"], actual_start=meta["actual_start"], actual_end=meta["actual_end"]
        )
        if time_chunks:
            ttnn.synchronize_device(runtime.mesh_device)
        ms_compute = (time.perf_counter() - t_c) * 1000.0
        t_end_epoch = time.time()

        if not cfg.is_last_rank:
            _d2d_send(d2d_out, out, rank, meta)  # push + free; the grant below forwards it over fabric

        # (4) Grant the outbound sender AFTER the push so it forwards this chunk's output downstream.
        if d2d_out is not None:
            d2d_out.release_fabric_links()

        extra = (
            f" compute={ms_compute:.2f}ms end={t_end_epoch:.6f} lease_reclaim={ms_lease:.2f}ms" if time_chunks else ""
        )
        logger.info(f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start_epoch:.6f}{extra}")

        n_done += 1
        c += 1
        if meta["is_last"]:
            break

    # The last outbound forward is async after its grant; reclaim once more so it has fully drained into
    # the downstream receiver before this rank tears the mesh down (no teardown barrier across ranks).
    # For an isolated single-chunk run this drain time IS the D2D fabric-transfer latency on the producer.
    t_drain = time.perf_counter()
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    ms_drain = (time.perf_counter() - t_drain) * 1000.0
    if time_chunks and d2d_out is not None:
        logger.info(f"[pp rank {rank}] final D2D drain (fabric transfer) wait = {ms_drain:.2f}ms")
    ttnn.synchronize_device(runtime.mesh_device)
    # E2E clock: first prefill start (this rank) and last compute end (after the single post-loop
    # synchronize above, so it is device-accurate). No per-chunk synchronize is involved, so the true
    # wall-clock e2e = max over ranks(last_compute_end) - min over ranks(first_compute_start), with no
    # measurement inflation. Epochs are time.time() so they compare across hosts (NTP-synced).
    loop_last_compute_end = time.time()
    logger.info(
        f"[pp rank {rank}] E2E_CLOCK first_compute_start={loop_first_compute_start:.6f} "
        f"last_compute_end={loop_last_compute_end:.6f}"
    )
    logger.info(
        f"[pp rank {rank}] processed {n_done} chunks in "
        f"{(time.perf_counter() - t0) * 1000.0:.2f} ms (decoupled, no barrier)"
    )

    if os.environ.get("PREFILL_STANDALONE_PCC", "0") == "1":
        # Each rank PCC-checks the KV slice it populated against the golden trace (offset by
        # first_layer_idx); all ranks passing == the rank-sliced model + file handoff reproduce
        # single-rank KV.
        from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import kv_cache_pcc_check

        trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
        kv_cache_pcc_check(
            runtime, slot_id=slot_id, n_chunks=n_done, trace_dir=trace_dir, first_layer_idx=cfg.first_layer_idx
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_config() -> None:
    """Print every env var the pipeline runner reads at startup so each rank's config is visible in
    logs (mirrors prefill_runner._print_config). Values shown are the resolved effective values, not
    just what was set in the environment."""
    rows = [
        ("PREFILL_MODEL_VARIANT", VARIANT.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", VARIANT.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)),
        ("resolved weight_cache_path", str(resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_STANDALONE_NCHUNKS", str(NUM_CHUNKS)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_STANDALONE (pipeline/bring-up mode)", os.environ.get("PREFILL_STANDALONE", "0")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_PP_H2D", os.environ.get("PREFILL_PP_H2D", "<unset; 0>")),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_PCC", os.environ.get("PREFILL_STANDALONE_PCC", "0")),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
    ]
    sep = "=" * 70
    lines = [sep, "prefill_runner configuration", sep]
    lines += [f"  {label:<35} = {val}" for label, val in rows]
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    # tt-run launches the MPI ranks but does not stand up the distributed context;
    # do it here before reading rank/size (idempotent across re-entry).
    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks)
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG)

    hf_config = load_hf_config(VARIANT)
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE)
    runtime_config = TtPrefillRuntimeConfig(
        num_layers=num_my_layers,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        num_links=2,
        capacity_factor=CAPACITY_FACTOR,
        gate_fallback_mode=GateComputeMode[_gate_mode_name],
        weight_cache_path=cache_path,
        model_cfg=MODEL_CFG,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
    )

    runtime = TtPrefillRuntime(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={},
        config=runtime_config,
    )
    runtime.compile()

    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        _serve_standalone(runtime, mesh_device, hf_config, rank, num_ranks, is_first_rank)
    else:
        _serve_request(runtime, mesh_device, num_ranks)

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_standalone(runtime, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Bring-up / benchmark path: trace (or PREFILL_PP_H2D) input on rank 0, D2D-socket transport
    between ranks, per-rank KV PCC. Self-contained (no external producer); covers num_ranks 1..N."""
    # Warm-up sync — the ONLY barrier. Every rank finishes compile before any chunk enters the
    # pipeline, so a downstream rank isn't still warming up while an upstream one races ahead. The
    # per-chunk loop takes no barrier. Trade-off: a rank that dies during compile hangs the others here.
    ttnn.distributed_context_barrier()

    # First rank in H2D mode: stand up the socket service so a producer can push token chunks.
    h2d_service = None
    if is_first_rank and _h2d_request_mode():
        # compile() leaves a custom sub-device manager loaded; the service's init program validates its
        # worker cores against the default whole-chip sub-device, so revert first.
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            chunk_size=CHUNK_SIZE,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
        )
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        descriptor_path = h2d_service.export_descriptor(service_id)
        logger.info(f"[pp rank {rank}] [h2d] service up, descriptor {service_id!r} -> {descriptor_path}")

    # D2D transport: with >1 rank, every rank stands up its pipeline endpoints (revert the custom
    # sub-device as above). The post-compile barrier guarantees all ranks reach the chained create
    # rendezvous. A single rank owns the whole model — no transport.
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)

    logger.info(f"[pp rank {rank}] setup complete, entering decoupled standalone loop")
    run_standalone_loop(runtime, rank, num_ranks, h2d_service=h2d_service, d2d_in=d2d_in, d2d_out=d2d_out)

    if h2d_service is not None or d2d_in is not None or d2d_out is not None:
        # Free the services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        h2d_service = d2d_in = d2d_out = None
        gc.collect()


def _serve_request(runtime, mesh_device, num_ranks: int) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer; unbounded (runs to SIGTERM). Optional KV-chunk-table migration publish + per-layer
    LayerAck channel. Single-rank only for now — pipelined request serving (num_ranks>1) needs an
    end-of-request / clean-shutdown protocol over the socket; use standalone mode for multi-rank."""
    if num_ranks > 1:
        raise NotImplementedError(
            f"request mode is single-rank only (got num_ranks={num_ranks}); run multi-rank pipeline "
            f"with PREFILL_STANDALONE=1 (h2d-fed pipeline serving is not wired yet)"
        )

    # compile() leaves a custom sub-device manager loaded; H2DStreamService's init program validates
    # its cores against the default whole-chip sub-device, so revert first.
    mesh_device.clear_loaded_sub_device_manager()
    h2d_service = build_h2d_service(
        mesh_device,
        mesh_shape=GLOBAL_MESH_SHAPE,
        chunk_size=CHUNK_SIZE,
        mapper_config=H2D_MAPPER_CONFIG,
        worker_cores=H2D_SYNC_WORKER_CORES,
        metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
    )
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    descriptor_path = h2d_service.export_descriptor(service_id)
    logger.info(
        f"[h2d] exported descriptor service_id={service_id!r} -> {descriptor_path}; "
        f"run prefill_h2d_producer.py / the scheduler in another process to drive token pushes."
    )

    if os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1":
        # Standalone-worker model: build the KV chunk address table from the device KV layout and
        # serialize it; the orchestrator forwards the path to the migration_worker. Only the runner
        # knows the KV NoC addresses, so it must build the table (no IPC with the worker).
        from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import (
            build_and_serialize_kv_chunk_table,
            send_kv_chunk_table,
        )

        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        build_and_serialize_kv_chunk_table(
            mesh_device=mesh_device,
            kvpe_cache=runtime.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
            sp_axis=0,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=NUM_USERS,
            path=table_path,
        )
        send_kv_chunk_table(table_path)

    # Per-layer LayerAck: the runner bumps a counter once per layer; the scheduler reads the delta and
    # drives the migration worker. InterProcessCounterChannel owns a named POSIX shm the scheduler
    # connects to via shm_layer_ack_name(service_id).
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    _stale_ack_shm = f"/dev/shm/{ack_shm_name.lstrip('/')}"
    if os.path.exists(_stale_ack_shm):
        # A prior run that didn't tear down cleanly leaves the segment behind, so shm_open(O_CREAT|
        # O_EXCL) fails with EEXIST. Remove it first.
        logger.warning(f"[migration] removing stale LayerAck shm {_stale_ack_shm} from a prior run")
        os.remove(_stale_ack_shm)
    ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
    runtime.set_layer_ack_channel(ack_channel)
    logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")

    logger.info("Setup complete, entering request loop")
    run_request_loop(runtime, h2d_service)

    # Release the H2D service while the mesh + command queues + service core are still alive (its dtor
    # frees a command queue and service-core L1; running after close_mesh_device aborts).
    import gc

    del h2d_service
    gc.collect()
    ack_channel.shutdown()  # munmap + shm_unlink
    del ack_channel


if __name__ == "__main__":
    main()
