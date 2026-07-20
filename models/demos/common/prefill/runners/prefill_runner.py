#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Disaggregated prefill runner — one entry point driving an N-rank serving pipeline.

Model-agnostic: the model is selected by PREFILL_MODEL and driven through a PrefillModelAdapter
(see ../adapter.py and ADDING_A_PREFILL_MODEL.md). This driver wires rank topology, input,
transport, and the per-chunk schedule; the adapter supplies how to build the model, allocate the KV
cache, run a chunk, and validate/migrate it.

The model is split across N ranks under tt-run: each rank owns a contiguous layer slice and builds
the same TtPrefillRuntime (first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank
hidden state moves device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 is the
single-galaxy case (no transport). Ranks run decoupled (no per-chunk barrier; one warm-up barrier
after compile).

Serving is request-driven: rank 0's tokens + per-iter PrefillMetadata arrive over the H2D socket from
an external producer (prefill_producer.py / the scheduler); the loop is UNBOUNDED. KV-chunk-table
migration + per-layer LayerAck are wired for the single-rank case only (disabled for the pipeline for
now). Shutdown is graceful: the producer/scheduler closes the stream with an all -1 PrefillMetadata
sentinel that each rank forwards downstream and then exits on; a rank blocked in the recv can only be
released by a transfer (the recv device op has no timeout), so SIGTERM/SIGKILL remains the hard
fallback if no sentinel arrives.

The model class is the single source of truth — this driver wires rank topology, input, transport,
and the per-chunk schedule; it does not reimplement embed / layers / forward.
"""

import json
import os
import signal
import time

from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import DEFAULT_MODEL, PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.migration import publish_table_and_wait_ready, serialize_device_map
from models.demos.common.prefill.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    compute_layer_split,
    open_mesh_device,
)


def _apply_manifest_env():
    """If PREFILL_MANIFEST is set, load the shared run.json and populate the env vars
    the runner (and migration/validation helpers) read. setdefault => an explicitly
    exported env var still wins over the manifest. Must be invoked before the
    module-level env reads below (e.g. PREFILL_MAX_SEQ_LEN) so the values take effect."""
    manifest_path = os.environ.get("PREFILL_MANIFEST")
    if not manifest_path:
        return

    with open(manifest_path) as mp:
        manifest = json.load(mp)

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    # Generic model/run config: a flat PREFILL_* map applied verbatim (setdefault). This lets a
    # rank-binding stay topology-only (rank_bindings + mesh_graph_desc) and point at a per-model
    # manifest for all model config — PREFILL_MODEL, fabric mode, chunk count, etc.
    for key, val in manifest.get("env", {}).items():
        sd(key, val)

    # The migration/pairwise-validation runs additionally carry a users[] + migration{} block. A
    # plain model-config manifest omits it (env-only), so it's optional.
    users = manifest.get("users")
    if not users:
        return
    N = len(users)

    model = manifest.get("model", {})
    mig = manifest.get("migration", {})
    paths = manifest.get("paths", {})

    sd("PREFILL_MODEL", model.get("variant"))
    sd("DEEPSEEK_PREFILL_TRACE_DIR", paths.get("trace_dir"))
    sd("PREFILL_MIGRATION_CLIENT_DIR", paths.get("migration_client_dir"))
    sd("PREFILL_NUM_USERS", 2 * N)
    sd("PREFILL_MAX_SEQ_LEN", model.get("max_seq_len"))
    sd("PREFILL_STANDALONE_CHUNKED_NCHUNKS", sum(u["n_chunks"] for u in users))
    sd("PREFILL_MIGRATE_WAIT_S", mig.get("wait_s"))
    sd("PREFILL_MIGRATE_GOLDEN_PTS", ",".join(u.get("kv_cache", "") for u in users))

    # Mode: default to pairwise
    mode = mig.get("mode") or "pairwise"
    # Loud failure for incorrect mode
    if mode != "pairwise":
        raise ValueError(f"manifest migration.mode must be 'pairwise', got: {mode}")
    # Loud failure for empty users
    if N < 1:
        raise ValueError(f"manifest migration.mode 'pairwise' requires at least 1 user, got {N}")
    sd("PREFILL_MIGRATE", mode)

    # Each non-empty kv_cache must exist on disk.
    for i, u in enumerate(users):
        kv = u.get("kv_cache", "")
        if kv and not os.path.exists(kv):
            raise FileNotFoundError(f"PREFILL_MANIFEST user {i} kv_cache not found: {kv}")

    # PREFILL_NUM_USERS (derived or explicitly exported) must equal 2*N.
    num_users = int(os.environ["PREFILL_NUM_USERS"])
    if num_users != 2 * N:
        raise ValueError(
            f"PREFILL_NUM_USERS ({num_users}) inconsistent with manifest " f"({N} users => expected {2 * N})"
        )


# Populate env from the manifest BEFORE the module-level env reads below.
_apply_manifest_env()

# Both socket transports (H2D input on rank 0, D2D between ranks) share a 1x1 push/sync worker grid and
# the same 3-word PrefillMetadata (slot_id, actual_start, actual_end). The 1x1 grid is the cheapest
# footprint with no penalty: a grid sweep showed compute + handoff gap flat from 1x1 to 4x4 (the
# per-chunk overhead is the persistent service's fabric/NoC presence, not the push workers).
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12

# End-of-stream sentinel: the producer/scheduler closes the request stream with one final push whose
# PrefillMetadata words are all -1 (0xFFFFFFFF on the wire). -1 is out of range for slot_id and both KV
# positions, so it can't collide with a real chunk. On receipt a rank forwards it to the next rank
# (unblocking that rank's recv) and breaks its loop, so an N-rank pipeline drains and exits gracefully
# instead of every rank blocking in its recv until SIGKILL. Shared wire convention with the scheduler;
# see ADDING_A_PREFILL_MODEL.md.
SHUTDOWN_METADATA_WORD = -1

# H2D socket service (request mode, rank 0 input): one worker core copies each pushed chunk into a fresh
# tensor; the producer packs the PrefillMetadata alongside each push.
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

# D2D socket transport (>1 rank): one persistent sender/receiver pair per rank boundary carries the
# sharded hidden state over inter-galaxy fabric. The activation is sharded [seq across SP rows, emb
# across TP cols] — the same layout the embedding output uses — so the receiver backing feeds the
# downstream model with no reshard.
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])
D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 64 * 1024))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
MODEL_CFG = ADAPTER.model_config

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
# Per-user KV cache length. In request mode the external producer decides the chunk count, so this is
# the one cache-sizing knob; a chunk must not push a slot past it. Default holds 4 chunks.
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 4))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
# When on (default), the last transformer layer runs kv-only: it fills the KV cache for migration and
# skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head. In a pipeline only the last rank applies it.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
# Measurement-only: synchronize the device after each chunk's forward and log the isolated per-rank
# compute (CHUNK_COMPUTE). Off in production — the sync serializes dispatch and kills pipeline overlap.
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
# Some models (e.g. Kimi: single expert group, device gate) route the MoE routing all-gather's global
# semaphores to L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static
# CBs, which needs the mesh opened with an L1_SMALL region. The adapter owns both knobs.
_L1_SMALL_SIZE = ADAPTER.l1_small_size

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _is_shutdown_sentinel(meta: dict) -> bool:
    """True for the all -1 end-of-stream sentinel (see SHUTDOWN_METADATA_WORD); false for every real
    chunk, whose slot_id and KV positions are non-negative and in range."""
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    """Block on the next producer push: returns (tt_tokens, {slot_id, actual_start, actual_end})
    decoded from the 12-byte PrefillMetadata. Used only by the unbounded request loop (rank 0 input)."""
    import torch

    tt_tokens, tt_metadata = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
    return tt_tokens, {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def build_d2d_pipeline_endpoints(mesh_device, rank: int, num_ranks: int, chunk_size: int, hidden_size: int):
    """Stand up this rank's persistent D2D endpoints for the pipeline: an inbound receiver from rank-1
    (every rank but the first) and an outbound sender to rank+1 (every rank but the last). Returns
    (inbound_receiver_or_None, outbound_sender_or_None).

    Setup order is inbound-then-outbound on every rank. create_sender/create_receiver rendezvous
    point-to-point between the two boundary ranks (no world barrier), and each MeshSocket ctor blocks
    until its peer's matching ctor. Doing inbound first chains the bring-up: rank 0's sender unblocks
    rank 1's receiver, which frees rank 1 to build its sender for rank 2's receiver, and so on — no
    deadlock. Both sides pass the identical worker-core grid and global spec."""
    global_spec = activation_global_spec(chunk_size, hidden_size)

    def _common():
        # Fresh mapper per call: create_sender/create_receiver take the mapper by std::unique_ptr and
        # MOVE it, so a middle rank (builds BOTH a receiver and a sender) must not reuse one — the
        # second create would get a consumed/null mapper and fail overload resolution.
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
            # The service asserts L1-only (d2d_stream_service.cpp:260).
            socket_buffer_type=ttnn.BufferType.L1,
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
        f"outbound={'yes' if outbound else 'no'}, workers={SYNC_WORKER_CORES}, fifo={D2D_FIFO_SIZE_BYTES}B)"
    )
    return inbound, outbound


def _d2d_recv(inbound) -> tuple:
    """Drain the next chunk that landed in the inbound receiver backing into a fresh device tensor and
    decode the inline metadata. The returned tensor already has the embedding-output sharding, so it
    feeds runtime.prefill with no reshard. Pairs with the upstream rank's _d2d_send."""
    import torch

    t0 = time.perf_counter()
    act, md = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    m = ttnn.to_torch(ttnn.get_device_tensors(md)[0]).view(torch.int32).flatten()
    meta = {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}
    logger.info(
        f"[pp] RECV-d2d [{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']} "
        f"[xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )
    return act, meta


def _d2d_send(outbound, activation: ttnn.Tensor, rank: int, meta: dict) -> None:
    """Push this rank's output hidden state + metadata to the downstream rank's receiver, then free it.
    The model already emits the activation in the sender backing's spec, and outbound_socket_service_sync
    TT_FATALs on any spec mismatch, so no host-side relayout is needed."""
    t0 = time.perf_counter()
    backing = outbound.get_backing_tensor()
    import torch

    words = [meta["slot_id"], meta["actual_start"], meta["actual_end"]]
    # The outbound op ships metadata as a replicated device tensor (3 uint32 words), not a Python list.
    md_tensor = ttnn.from_torch(
        torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            backing.device(),
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        ),
    )
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, activation, metadata=md_tensor)
    ttnn.deallocate(activation)
    logger.info(
        f"[pp rank {rank}] SEND-d2d [{meta['actual_start']},{meta['actual_end']}) "
        f"[xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms"
    )


def _forward_shutdown(d2d_out, rank: int, hidden_size: int) -> None:
    """Forward the shutdown sentinel to the downstream rank so it unblocks in its own recv, then release
    the outbound link so the transfer ships (mirroring _compute_and_send's tail). The activation content
    is irrelevant — the downstream discards it once it sees the sentinel — but outbound_socket_service_sync
    requires the input's per-shard spec to equal the sender backing's, so build the dummy exactly like a
    real activation: the [1, 1, CHUNK_SIZE, hidden_size] bf16 TILE spec sharded by D2D_MAPPER_CONFIG."""
    import torch

    dev = d2d_out.get_backing_tensor().device()
    dummy = ttnn.from_torch(
        torch.zeros(1, 1, CHUNK_SIZE, hidden_size),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=dev,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(dev, D2D_MAPPER_CONFIG),
    )
    sentinel = {
        "slot_id": SHUTDOWN_METADATA_WORD,
        "actual_start": SHUTDOWN_METADATA_WORD,
        "actual_end": SHUTDOWN_METADATA_WORD,
    }
    _d2d_send(d2d_out, dummy, rank, sentinel)  # ships + frees the dummy
    d2d_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] forwarded SHUTDOWN sentinel to rank {rank + 1}")


def _lease_reclaim(d2d_in, d2d_out) -> None:
    """Before a chunk: reclaim this rank's fabric links (the previous-iter D2D transfer has drained),
    then grant the inbound receiver so this chunk's activation drains into its backing. No-op without
    D2D (single rank). The outbound grant happens AFTER the push, in _compute_and_send."""
    if d2d_in is not None:
        d2d_in.wait_for_fabric_links()
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    if d2d_in is not None:
        d2d_in.release_fabric_links()


def _compute_and_send(runtime, kv_caches, rank: int, c: int, inp, meta: dict, d2d_out) -> float:
    """Run one chunk: prefill into the engine-owned kv_caches, forward the output downstream (non-last
    rank) and grant the outbound sender so it ships over fabric. Returns the compute-start epoch
    (NTP-comparable). CHUNK_START is logged BEFORE the forward, with this chunk's metadata, so the
    slot/KV-range is visible per rank even if prefill_chunk hangs. The trailing metadata is kept after
    compute_start so the c=/compute_start= fields stay parseable (plot_pipeline_trace.py)."""
    t_start = time.time()
    logger.info(
        f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f} "
        f"slot={meta['slot_id']} [{meta['actual_start']},{meta['actual_end']})"
    )
    out = runtime.prefill_chunk(
        inp, kv_caches, slot_id=meta["slot_id"], actual_start=meta["actual_start"], actual_end=meta["actual_end"]
    )
    if SYNC_PER_CHUNK:
        # Block on device completion so the delta is this rank's forward alone, not the downstream-start
        # proxy. Serializes dispatch (no overlap) — measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={(time.time() - t_start) * 1000.0:.3f}")
    if not runtime.config.is_last_rank:
        _d2d_send(d2d_out, out, rank, meta)  # push + free; the grant below forwards it over fabric
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    return t_start


def _drain_and_log_e2e(runtime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float) -> None:
    """Per-rank teardown: drain the last outbound D2D forward, one synchronize so the e2e clock reflects
    device completion, then log E2E_CLOCK (first prefill start + last compute end, NTP-comparable epochs)
    and the chunk count. No teardown barrier across ranks."""
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    ttnn.synchronize_device(runtime.mesh_device)
    logger.info(
        f"[pp rank {rank}] E2E_CLOCK first_compute_start={first_compute_start:.6f} last_compute_end={time.time():.6f}"
    )
    logger.info(f"[pp rank {rank}] processed {n_done} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")


def run_request_loop(
    runtime, kv_caches, rank: int, num_ranks: int, *, hidden_size: int, h2d_service=None, d2d_in=None, d2d_out=None
):
    """Production serving loop — UNBOUNDED. rank 0 reads each chunk from the H2D socket (the external
    producer decides the count); downstream ranks read from D2D. Runs until the producer/scheduler
    closes the stream with the all -1 shutdown sentinel (each rank forwards it and exits gracefully) or,
    as a hard fallback, until SIGTERM/SIGKILL. No fixed chunk bound, no trace input, no PCC.

    Exception: in migration-validation mode (PREFILL_VALIDATE_MIGRATION=1) the scheduler driver never
    pushes the shutdown sentinel — it pushes PREFILL_STANDALONE_CHUNKED_NCHUNKS chunks, migrates, then
    writes the DONE sentinel for the runner to poll. So the loop exits after that many chunks and returns
    to validate_after_prefill. Returns (chunks_per_slot, real_end_per_slot, total_chunks)."""
    cfg = runtime.config
    if cfg.is_first_rank and h2d_service is None:
        raise ValueError("request mode requires the H2D service on the first rank for input")
    logger.info(
        f"[pp rank {rank}/{num_ranks}] request (unbounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input={'h2d' if cfg.is_first_rank else 'd2d'})"
    )
    t0 = time.perf_counter()
    c = 0
    first = None
    # Per-slot bookkeeping for the optional post-loop migration validation (validate_after_prefill):
    # how many chunks each slot received and its highest real (non-pad) end position.
    chunks_per_slot: dict = {}
    real_end_per_slot: dict = {}
    # If we run prefill validation, we need to know the expected number of chunks to exit the loop.
    # PREFILL_STANDALONE_CHUNKED_* is migration-validation config (not the removed standalone run mode).
    _expected_chunks = (
        int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "0"))
        if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1"
        else 0
    )
    while not _shutdown:
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            inp, meta = _socket_next(h2d_service)  # slot/start/end from the producer
        else:
            inp, meta = _d2d_recv(d2d_in)
        if _is_shutdown_sentinel(meta):
            # End of stream: drop the throwaway payload, hand the sentinel to the next rank so it too
            # unblocks and exits, then fall through to the graceful drain below.
            logger.info(f"[pp rank {rank}] SHUTDOWN sentinel received after {c} chunks; exiting request loop")
            ttnn.deallocate(inp)
            if d2d_out is not None:
                _forward_shutdown(d2d_out, rank, hidden_size)
            break
        slot = meta["slot_id"]
        chunks_per_slot[slot] = chunks_per_slot.get(slot, 0) + 1
        real_end_per_slot[slot] = max(real_end_per_slot.get(slot, 0), meta["actual_end"])
        t = _compute_and_send(runtime, kv_caches, rank, c, inp, meta, d2d_out)
        if first is None:
            first = t
        c += 1
        if _expected_chunks and c >= _expected_chunks:
            logger.info(
                f"[pp rank {rank}] processed {c}/{_expected_chunks} chunks "
                "(PREFILL_STANDALONE_CHUNKED_NCHUNKS reached); exiting request loop for migration validation"
            )
            if d2d_out is not None:
                _forward_shutdown(d2d_out, rank, hidden_size)
            break
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)
    return chunks_per_slot, real_end_per_slot, c


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_config() -> None:
    """Print every env var the runner (and its downstream model/runner_utils) reads at startup so each
    rank's config is visible in logs. Values shown are the resolved effective values, not just what was
    set in the environment."""
    rows = [
        ("PREFILL_MODEL", ADAPTER.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", ADAPTER.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)),
        ("resolved weight_cache_path", str(ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_PP_LAYER_COUNTS", os.environ.get("PREFILL_PP_LAYER_COUNTS", "<even split>")),
        ("PREFILL_KV_ONLY_LAST_LAYER", str(KV_ONLY_LAST_LAYER)),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        (
            "PREFILL_STANDALONE_CHUNKED_RECORD_ONLY",
            os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0"),
        ),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        ("PREFILL_MOCK_MIGRATION", os.environ.get("PREFILL_MOCK_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_WAIT_READY_MS", os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000")),
        ("MIGRATION_DONE_FILE", os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")),
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

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS))
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE)

    hf_config = ADAPTER.load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    params = PrefillRunParams(
        mesh_shape=GLOBAL_MESH_SHAPE,
        num_layers=num_my_layers,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        max_seq_len=MAX_SEQ_LEN,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        capacity_factor=CAPACITY_FACTOR,
        num_links=2 if is_blackhole() else 1,  # Blackhole trains 2 fabric routing planes, others 1
        gate_mode_name=_gate_mode_name,
        # Chunked prefill never samples (the populated KV cache is the output), so the final stage is
        # headless: its last layer runs KV-only and no norm/LM-head is built. Only the last rank does
        # this (single-rank inherits it); PREFILL_KV_ONLY_LAST_LAYER can force it off.
        kv_only_last_layer=is_last_rank and KV_ONLY_LAST_LAYER,
        weight_cache_path=ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE),
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    # The engine owns the KV cache(s): allocate them once (the adapter defines the layout) as an opaque
    # KvCaches, hand that container to every runtime call, and let it free with the mesh at shutdown. The
    # runner stays model-agnostic — it never unpacks the container; the (model-specific) runtime pulls out
    # the primary cache and any secondary cache (e.g. a sparse/DSA model's index cache) it needs, and folds
    # both into the merged migration table (see build_kv_chunk_table).
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    _serve_request(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer (prefill_producer.py / the scheduler); unbounded (runs to SIGTERM), num_ranks 1..N over D2D.

    Migration (KV-chunk-table publish) + per-layer LayerAck are wired for the single-rank case only;
    they are disabled for num_ranks>1 (pipelined migration is future work). Shutdown for num_ranks>1 is
    rough: downstream ranks block in D2D recv when rank 0 stops, so they exit on teardown / SIGKILL."""
    single_rank = num_ranks == 1

    # Migration is only wired for the single-rank case; on the pipeline it would silently no-op. Fail
    # loud so an enabled-migration pipeline run can't be mistaken for a working one.
    if not single_rank and os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1":
        raise ValueError(
            f"PREFILL_ENABLE_MIGRATION=1 is unsupported for num_ranks={num_ranks} (pipelined migration "
            "is not implemented); run single-rank or unset PREFILL_ENABLE_MIGRATION."
        )

    ttnn.distributed_context_barrier()  # warm-up: all ranks finish compile before chunks flow

    # H2D input service lives on the first rank only (downstream ranks read from D2D). compile() leaves
    # a custom sub-device manager loaded; the service's init program validates its cores against the
    # default whole-chip sub-device, so revert first.
    h2d_service = None
    if is_first_rank:
        mesh_device.clear_loaded_sub_device_manager()
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            chunk_size=CHUNK_SIZE,
            mapper_config=H2D_MAPPER_CONFIG,
            worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
        )
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        descriptor_path = h2d_service.export_descriptor(service_id)
        logger.info(
            f"[pp rank {rank}] [h2d] descriptor service_id={service_id!r} -> {descriptor_path}; "
            f"drive it with prefill_producer.py / the scheduler."
        )

    # D2D pipeline transport for num_ranks>1.
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)

    # Migration KV-chunk-table + LayerAck: single-rank only (disabled for the pipeline for now).
    ack_channel = None
    if single_rank:
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        if os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1":
            # Clear a stale DONE sentinel from a prior run so the validator can't read its pairs.
            _done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
            if os.path.exists(_done_file):
                logger.warning(f"[migration] removing stale DONE sentinel {_done_file} from a prior run")
                os.remove(_done_file)

            # Full migration bring-up: the runtime builds the model-specific KV chunk table from
            # its device cache layout; the runner publishes it (+ device map) to the worker and blocks
            # on WORKER_READY before the request loop opens (the worker gates on SetTable + AssignDevMap).
            # A sparse model's KvCaches carries its index cache too, so the table describes BOTH caches
            # in one (merged); a dense model's is a single-config table.
            table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
            wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))
            runtime.build_kv_chunk_table(kv_caches, path=table_path)
            publish_table_and_wait_ready(
                mesh_device=mesh_device,
                mesh_shape=GLOBAL_MESH_SHAPE,
                table_path=table_path,
                wait_ready_timeout_ms=wait_ready_ms,
            )
        elif os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1":
            # Mock integration (prefill_producer.py): serialize the KV chunk table so an external
            # producer can read it back via ttnn.experimental.disaggregation.import_from_protobuf_file
            # and locate each chunk — WITHOUT the migration_endpoint worker (no MigrationLayerClient,
            # no WORKER_READY). One galaxy => one complete table spanning all NUM_LAYERS / NUM_USERS
            # (both caches, merged, for a sparse model).
            table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
            runtime.build_kv_chunk_table(kv_caches, path=table_path)
            # Also publish the fabric_node -> ASIC unique_id device map so the producer can resolve chips
            # for its device-less UMD read (read_dram_umd) without touching the ControlPlane.
            device_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            serialize_device_map(mesh_device, device_map_path)
            logger.info(
                f"[mock-migration] KV chunk table -> {table_path}, device map -> {device_map_path} "
                f"(no migration worker); prefill_producer can import them"
            )

        # Per-layer LayerAck: the runner bumps a counter once per layer; the scheduler reads the delta.
        ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
        _stale_ack_shm = f"/dev/shm/{ack_shm_name.lstrip('/')}"
        if os.path.exists(_stale_ack_shm):
            # A prior run that didn't tear down cleanly leaves the segment behind (shm_open O_EXCL fails).
            logger.warning(f"[migration] removing stale LayerAck shm {_stale_ack_shm} from a prior run")
            os.remove(_stale_ack_shm)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")

    logger.info(f"[pp rank {rank}] setup complete, entering request loop")
    chunks_per_slot, real_end_per_slot, total_chunks = run_request_loop(
        runtime,
        kv_caches,
        rank,
        num_ranks,
        hidden_size=hf_config.hidden_size,
        h2d_service=h2d_service,
        d2d_in=d2d_in,
        d2d_out=d2d_out,
    )

    # Post-loop KV validation (bring-up / migration accuracy; never in production serving). Single-rank
    # only: only the last/single rank owns the whole cache. By now the scheduler has migrated the slots
    # out-of-band and written the DONE sentinel; the validator waits for it and PCCs the migrated pairs.
    if single_rank and os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1":
        from models.demos.common.prefill.runners.validation import validate_after_prefill

        validate_after_prefill(
            runtime,
            kv_caches,
            chunks_per_slot=chunks_per_slot,
            real_end_per_slot=real_end_per_slot,
            num_users=NUM_USERS,
            total_chunks=total_chunks,
        )

    # Release services while the mesh + command queues are still alive (their dtors free a command
    # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
    import gc

    h2d_service = d2d_in = d2d_out = None
    gc.collect()
    if ack_channel is not None:
        ack_channel.shutdown()  # munmap + shm_unlink
        ack_channel = None


if __name__ == "__main__":
    # Best-effort: some galaxies ship a small RLIMIT_NPROC soft limit that starves the runner's threads, so
    # raise it to the hard limit. Guarded — get/setrlimit can raise OSError/ValueError when the limit is
    # immutable or the process lacks permission, and that must not crash the runner before main().
    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (OSError, ValueError) as e:
        logger.warning(f"[prefill] could not raise RLIMIT_NPROC to the hard limit: {e}")

    main()
