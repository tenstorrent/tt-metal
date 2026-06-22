#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""DeepSeek-V3 prefill runner — one entry point, two run modes that share the same N-rank pipeline.

The model is split across N ranks under tt-run: each rank owns a contiguous layer slice and builds
the same TtPrefillRuntime (first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank
hidden state moves device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 is the
single-galaxy case (no transport). Ranks run decoupled (no per-chunk barrier; one warm-up barrier
after compile). The two modes run identical pipeline mechanics and differ only in the trigger:

  * Request mode (default): production serving. rank 0's tokens + per-iter PrefillMetadata arrive over
    the H2D socket from an external producer (prefill_h2d_producer.py / the scheduler); the loop is
    UNBOUNDED (runs to SIGTERM). KV-chunk-table migration + per-layer LayerAck are wired for the
    single-rank case only (disabled for the pipeline for now). Shutdown for >1 rank is rough: ranks
    block in the H2D/D2D recv device op and exit on teardown/SIGKILL (no end-of-request sentinel yet).

  * Standalone mode (PREFILL_STANDALONE=1): bring-up / benchmark. rank 0's input is the golden trace
    for a fixed PREFILL_STANDALONE_NCHUNKS chunks; the loop is BOUNDED and exits cleanly.
    PREFILL_STANDALONE_PCC=1 checks each rank's KV slice vs the golden.

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
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    get_variant,
    load_hf_config,
    load_trace_token_ids,
    open_mesh_device,
    resolve_trace_dir,
    resolve_weight_cache_path,
)
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig


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
    users = manifest["users"]
    N = len(users)

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    model = manifest.get("model", {})
    mig = manifest.get("migration", {})
    paths = manifest.get("paths", {})

    sd("PREFILL_MODEL_VARIANT", model.get("variant"))
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

# H2D socket service (request mode, rank 0 input): one worker core copies each pushed chunk into a fresh
# tensor; the producer packs the PrefillMetadata alongside each push.
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

# D2D socket transport (>1 rank): one persistent sender/receiver pair per rank boundary carries the
# sharded hidden state over inter-galaxy fabric. The activation is sharded [seq across SP rows, emb
# across TP cols] — the same layout the embedding output uses — so the receiver backing feeds the
# downstream model with no reshard.
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
# When on (default), the last transformer layer runs kv-only: it fills the KV cache for migration and
# skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head. In a pipeline only the last rank applies it.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
# Measurement-only: synchronize the device after each chunk's forward and log the isolated per-rank
# compute (CHUNK_COMPUTE). Off in production — the sync serializes dispatch and kills pipeline overlap.
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
# Kimi (single expert group, device gate) routes the MoE routing all-gather's global semaphores to
# L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static CBs. This
# requires opening the mesh with an L1_SMALL region. Off for DeepSeek (semaphores stay in main L1).
_ROUTING_USE_L1_SMALL_SEMAPHORES = VARIANT.name == "kimi_k2_6"
_L1_SMALL_SIZE = 512 if _ROUTING_USE_L1_SMALL_SEMAPHORES else 0

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
# Layer-completion routing (pipeline / num_ranks > 1)
# ---------------------------------------------------------------------------


def build_layer_completion_sink(producer, *, source_rank, num_layers, get_request_id):
    """Build the per-layer callback the runtime fires once per layer.

    Computes a globally-dense ordering key and pushes a full completion
    into `producer` (a ttnn.layer_completion.LayerCompletionQueue). The
    master router re-emits completions strictly in ascending `seq`.

    seq = request_id * num_layers + layer_idx — dense across all (request,
    layer) pairs. For pipelined prefill each rank owns a disjoint set of
    global layer indices per request, so the union of every rank's seqs
    tiles [0, num_requests*num_layers) with no gaps or collisions.

    Args:
        producer: connected LayerCompletionQueue (the host-local ring).
        source_rank: this rank's world rank (diagnostic in the payload).
        num_layers: total GLOBAL layers (the seq stride per request), NOT this rank's slice.
        get_request_id: zero-arg callable returning the current request/chunk id.
    """

    def on_layer_complete(layer_idx: int) -> None:
        request_id = get_request_id()
        seq = request_id * num_layers + layer_idx
        # Ring is sized well above in-flight depth; a full ring means the
        # router thread has stalled — surface it rather than drop silently.
        if not producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            raise RuntimeError(f"layer-completion ring full (seq={seq}); router not draining")

    return on_layer_complete


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _first_rank_chunk_tokens(runtime: TtPrefillRuntime, token_ids: list[int], kv_actual: int) -> ttnn.Tensor:
    """Slice this chunk's tokens and build the SP-sharded input tensor. Delegates to the runtime's own
    builder so the input format has one source of truth."""
    cfg = runtime.config
    return runtime._make_chunk_input(token_ids[kv_actual : kv_actual + cfg.chunk_size])


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


def _compute_and_send(runtime: TtPrefillRuntime, rank: int, c: int, inp, meta: dict, d2d_out) -> float:
    """Run one chunk: prefill, forward the output downstream (non-last rank) and grant the outbound
    sender so it ships over fabric, log CHUNK_START. Returns the compute-start epoch (NTP-comparable)."""
    t_start = time.time()
    # Record the chunk index so the pipelined layer-completion sink (if registered) can build its
    # globally-dense seq = current_chunk_idx * NUM_LAYERS + global_layer_idx. No-op single-rank.
    runtime.current_chunk_idx = c
    out = runtime.prefill(
        inp, slot_id=meta["slot_id"], actual_start=meta["actual_start"], actual_end=meta["actual_end"]
    )
    if SYNC_PER_CHUNK:
        # Block on device completion before the send so the delta is this rank's forward alone, not the
        # downstream-start proxy. Serializes dispatch (no overlap) — measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={(time.time() - t_start) * 1000.0:.3f}")
    if not runtime.config.is_last_rank:
        _d2d_send(d2d_out, out, rank, meta)  # push + free; the grant below forwards it over fabric
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f}")
    return t_start


def _drain_and_log_e2e(
    runtime: TtPrefillRuntime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float
) -> None:
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
    runtime: TtPrefillRuntime, rank: int, num_ranks: int, *, h2d_service=None, d2d_in=None, d2d_out=None
) -> None:
    """Production serving loop — UNBOUNDED. rank 0 reads each chunk from the H2D socket (the external
    producer decides the count); downstream ranks read from D2D. Runs until SIGTERM. There is no
    end-of-stream marker, so shutdown is rough: ranks block in the recv device op and exit on mesh
    teardown / SIGKILL. No fixed NUM_CHUNKS bound, no trace input, no PCC — see run_standalone_loop
    for those."""
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
    while not _shutdown:
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            inp, meta = _socket_next(h2d_service)  # slot/start/end from the producer
        else:
            inp, meta = _d2d_recv(d2d_in)
        t = _compute_and_send(runtime, rank, c, inp, meta, d2d_out)
        if first is None:
            first = t
        c += 1
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)


def run_standalone_loop(runtime: TtPrefillRuntime, rank: int, num_ranks: int, *, d2d_in=None, d2d_out=None) -> None:
    """Bring-up / benchmark loop — BOUNDED, golden-trace input. rank 0 drives NUM_CHUNKS chunks from the
    trace; downstream ranks receive the same count over D2D. Every rank knows NUM_CHUNKS (propagated via
    global_env), so each loops a fixed range independently — no end-of-stream marker needed. With
    PREFILL_STANDALONE_PCC=1 each rank checks the KV slice it populated vs the golden trace."""
    cfg = runtime.config
    slot_id = 0  # first rank fills slot 0; downstream ranks adopt the slot from the received metadata
    n_chunks = NUM_CHUNKS
    token_ids = None
    if cfg.is_first_rank:
        token_ids = _load_token_ids()
        token_ids = (token_ids + [1] * (n_chunks * cfg.chunk_size))[: n_chunks * cfg.chunk_size]
        if n_chunks * cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(
                f"{n_chunks} chunks x {cfg.chunk_size} exceeds per-user cache max_seq_len={cfg.max_seq_len}; "
                f"raise PREFILL_MAX_SEQ_LEN."
            )
    # Every rank loops a fixed range(n_chunks) independently — there is no end-of-stream marker, so all
    # ranks MUST resolve the same PREFILL_STANDALONE_NCHUNKS (set in the binding's global_env, not a
    # per-rank override). A mismatch strands the pipeline: a low downstream count exits early and leaves
    # rank 0's next send unconsumed. Log each rank's count so a mismatch is visible across the tag logs.
    logger.info(
        f"[pp rank {rank}/{num_ranks}] standalone (bounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input=trace chunks={n_chunks})"
    )
    t0 = time.perf_counter()
    first = None
    for c in range(n_chunks):
        _lease_reclaim(d2d_in, d2d_out)
        if cfg.is_first_rank:
            kv_actual = c * cfg.chunk_size
            inp = _first_rank_chunk_tokens(runtime, token_ids, kv_actual)
            meta = {"slot_id": slot_id, "actual_start": kv_actual, "actual_end": kv_actual + cfg.chunk_size}
        else:
            inp, meta = _d2d_recv(d2d_in)
            slot_id = meta["slot_id"]
        t = _compute_and_send(runtime, rank, c, inp, meta, d2d_out)
        if first is None:
            first = t
    # Every rank must finish receiving + forwarding the final chunk before any rank reclaims its
    # outbound fabric link in the drain. Without this, the producer reclaims the shared link
    # (share_fabric_links) right after its last send and strands the downstream's final recv —
    # the pipeline tail deadlocks (ranks 2/3 hang on the last chunk).
    if num_ranks > 1:
        ttnn.distributed_context_barrier()
    _drain_and_log_e2e(runtime, rank, d2d_out, first, n_chunks, t0)

    if os.environ.get("PREFILL_STANDALONE_PCC", "0") == "1":
        # Each rank PCC-checks the KV slice it populated against the golden trace (offset by
        # first_layer_idx); all ranks passing == the rank-sliced model reproduces single-rank KV.
        from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import kv_cache_pcc_check

        trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
        kv_cache_pcc_check(
            runtime, slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir, first_layer_idx=cfg.first_layer_idx
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _print_config() -> None:
    """Print every env var the runner (and its downstream model/runner_utils) reads at startup so each
    rank's config is visible in logs. Values shown are the resolved effective values, not just what was
    set in the environment."""
    rows = [
        ("PREFILL_MODEL_VARIANT", VARIANT.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", VARIANT.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)),
        ("resolved weight_cache_path", str(resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_PP_LAYER_COUNTS", os.environ.get("PREFILL_PP_LAYER_COUNTS", "<even split>")),
        ("PREFILL_KV_ONLY_LAST_LAYER", str(KV_ONLY_LAST_LAYER)),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_STANDALONE_NCHUNKS", str(NUM_CHUNKS)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_STANDALONE (pipeline/bring-up mode)", os.environ.get("PREFILL_STANDALONE", "0")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_PCC", os.environ.get("PREFILL_STANDALONE_PCC", "0")),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        (
            "PREFILL_STANDALONE_CHUNKED_RECORD_ONLY",
            os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0"),
        ),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
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

    layer_split = compute_layer_split(NUM_LAYERS, num_ranks)
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE)

    hf_config = load_hf_config(VARIANT)
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE)
    runtime_config = TtPrefillRuntimeConfig(
        num_layers=num_my_layers,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        num_links=2 if is_blackhole() else 1,  # Blackhole trains 2 fabric routing planes, others 1
        capacity_factor=CAPACITY_FACTOR,
        gate_fallback_mode=GateComputeMode[_gate_mode_name],
        weight_cache_path=cache_path,
        model_cfg=MODEL_CFG,
        first_layer_idx=first_layer_idx,
        is_first_rank=is_first_rank,
        is_last_rank=is_last_rank,
        # Chunked prefill never samples (the populated KV cache is the output), so the final stage is
        # headless: its last layer runs KV-only and no norm/LM-head is built. Only the last rank does
        # this (single-rank inherits it); PREFILL_KV_ONLY_LAST_LAYER can force it off.
        kv_only_last_layer=is_last_rank and KV_ONLY_LAST_LAYER,
        routing_use_l1_small_for_semaphores=_ROUTING_USE_L1_SMALL_SEMAPHORES,
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
        _serve_request(runtime, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_standalone(runtime, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Bring-up / benchmark path: golden-trace input on rank 0, D2D-socket transport between ranks,
    per-rank KV PCC. Self-contained (no external producer); covers num_ranks 1..N."""
    # Warm-up sync — the ONLY barrier. Every rank finishes compile before any chunk enters the
    # pipeline, so a downstream rank isn't still warming up while an upstream one races ahead. The
    # per-chunk loop takes no barrier. Trade-off: a rank that dies during compile hangs the others here.
    ttnn.distributed_context_barrier()

    # D2D transport: with >1 rank, every rank stands up its pipeline endpoints (revert the custom
    # sub-device as above). The post-compile barrier guarantees all ranks reach the chained create
    # rendezvous. A single rank owns the whole model — no transport.
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)
        # The chained D2D socket rendezvous finishes at staggered times per rank. Without this barrier
        # rank 0 enters its produce loop first, fills the socket, and stalls ~6s waiting for the
        # downstream ranks to enter their consume loops — moving that skew out of the timed chunk loop.
        ttnn.distributed_context_barrier()

    logger.info(f"[pp rank {rank}] setup complete, entering standalone loop")
    run_standalone_loop(runtime, rank, num_ranks, d2d_in=d2d_in, d2d_out=d2d_out)

    if d2d_in is not None or d2d_out is not None:
        # Free the services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        d2d_in = d2d_out = None
        gc.collect()


def _serve_request(runtime, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer (prefill_h2d_producer.py / the scheduler); unbounded (runs to SIGTERM). Same pipeline
    mechanics as standalone (num_ranks 1..N over D2D); the only difference is the trigger (H2D input)
    and that it runs forever.

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
            f"drive it with prefill_h2d_producer.py / the scheduler."
        )

    # D2D pipeline transport for num_ranks>1 (same as standalone).
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)

    # Per-layer LayerAck -> scheduler-driven migration. Two wirings by topology:
    #   * single-rank: the runtime owns the scheduler's counter channel and inject()s it
    #     directly (the original path).
    #   * pipeline (num_ranks > 1): each rank owns only a layer slice, so it cannot inject
    #     the scheduler channel directly. Every rank pushes full {seq, source_rank, layer_idx,
    #     request_id} completions into a host-local LayerCompletionQueue; a per-host
    #     LayerCompletionRouter forwards them to the master rank, which re-emits them in
    #     global seq order into the SAME counter channel the scheduler connects to. See
    #     build_layer_completion_sink() and ttnn.layer_completion.
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    ack_channel = None
    router = None
    producer = None

    def _unlink_stale_shm(name: str) -> None:
        # A prior run that didn't tear down cleanly leaves the segment behind (shm_open O_EXCL fails).
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    # KV-chunk-table publish stays single-rank for now: a pipeline rank only knows its own
    # slice's KV NoC addresses, so the per-rank/merged table is future work.
    if single_rank and os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1":
        # Clear a stale DONE sentinel from a prior run so the validator can't read its pairs.
        _done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
        if os.path.exists(_done_file):
            logger.warning(f"[migration] removing stale DONE sentinel {_done_file} from a prior run")
            os.remove(_done_file)

        # Full migration bring-up: publish the KV chunk table + device map, then block on
        # WORKER_READY before the request loop opens (the worker gates on SetTable + AssignDevMap).
        from models.demos.deepseek_v3_d_p.tt.runners.kv_migration_setup import publish_kv_chunk_table_and_wait_ready

        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))
        publish_kv_chunk_table_and_wait_ready(
            mesh_device=mesh_device,
            kvpe_cache=runtime.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
            sp_axis=0,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=NUM_USERS,
            chunk_size_global=CHUNK_SIZE,  # block-cyclic period (prefill chunk size)
            path=table_path,
            wait_ready_timeout_ms=wait_ready_ms,
        )

    if single_rank:
        # Direct path: the runtime owns + inject()s the scheduler counter channel.
        _unlink_stale_shm(ack_shm_name)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")
    else:
        # Pipeline path: route per-rank completions to the master, which re-emits in seq order.
        ring_shm_name = os.environ.get("PREFILL_LAYER_COMPLETION_RING", f"/tt_prefill_layer_completion_ring_{rank}")
        _unlink_stale_shm(ring_shm_name)
        if rank == master_rank:
            _unlink_stale_shm(ack_shm_name)
        # The router owns the host-local ring (and, on the master, the scheduler counter channel,
        # which it inject()s in order). Subordinate ranks MPI-forward completions to the master.
        router = ttnn.layer_completion.LayerCompletionRouter(
            rank=rank,
            world_size=num_ranks,
            master_rank=master_rank,
            ring_shm_name=ring_shm_name,
            scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
        )
        producer = ttnn.layer_completion.LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
        # seq stride is the GLOBAL layer total (NUM_LAYERS), NOT this rank's slice; the layer_idx
        # arriving at the sink is already global; the chunk index is set per chunk in _compute_and_send.
        runtime.set_layer_completion_sink(
            build_layer_completion_sink(
                producer,
                source_rank=rank,
                num_layers=NUM_LAYERS,
                get_request_id=lambda: runtime.current_chunk_idx,
            )
        )
        logger.info(
            f"[migration] pipelined layer-completion routing up: rank={rank}/{num_ranks} master={master_rank} "
            f"ring={ring_shm_name} "
            + (f"(owns scheduler channel {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
        )

    logger.info(f"[pp rank {rank}] setup complete, entering request loop")
    run_request_loop(runtime, rank, num_ranks, h2d_service=h2d_service, d2d_in=d2d_in, d2d_out=d2d_out)

    # Release services while the mesh + command queues are still alive (their dtors free a command
    # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
    import gc

    h2d_service = d2d_in = d2d_out = None
    gc.collect()
    # Teardown. NOTE: for num_ranks>1 the request loop only returns on a clean _shutdown; the known
    # rough-shutdown path (downstream ranks block in D2D recv, exit on SIGKILL) can bypass this. Clean
    # cross-rank router teardown (barrier + end-of-request sentinel) is future work.
    if producer is not None:
        producer.shutdown()
    if router is not None:
        router.stop()  # joins the listener; on the master, unlinks the scheduler counter channel
    if ack_channel is not None:
        ack_channel.shutdown()  # munmap + shm_unlink
        ack_channel = None


if __name__ == "__main__":
    main()
