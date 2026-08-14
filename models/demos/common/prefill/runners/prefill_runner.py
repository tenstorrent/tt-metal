#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Disaggregated prefill runner — one entry point, two run modes that share the same N-rank pipeline.

Model-agnostic: the model is selected by PREFILL_MODEL and driven through a PrefillModelAdapter
(see ../adapter.py and ADDING_A_PREFILL_MODEL.md). This driver wires rank topology, input,
transport, and the per-chunk schedule; the adapter supplies how to build the model, allocate the KV
cache, run a chunk, and validate/migrate it.

The model is split across N ranks under tt-run: each rank owns a contiguous layer slice and builds
the same TtPrefillRuntime (first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank
hidden state moves device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 is the
single-galaxy case (no transport). Ranks run decoupled (no per-chunk barrier; one warm-up barrier
after compile). The two modes run identical pipeline mechanics and differ only in the trigger:

  * Request mode (default): production serving. rank 0's tokens + per-iter PrefillMetadata arrive over
    the H2D socket from an external producer (prefill_producer.py / the scheduler); the loop is
    UNBOUNDED. KV-chunk-table migration + per-layer LayerAck are wired for the single-rank case only
    (disabled for the pipeline for now). Shutdown is graceful: the producer/scheduler closes the stream
    with an all -1 PrefillMetadata sentinel that each rank forwards downstream and then exits on; a rank
    blocked in the recv can only be released by a transfer (the recv device op has no timeout), so
    SIGTERM/SIGKILL remains the hard fallback if no sentinel arrives.

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
from models.demos.common.prefill.adapter import DEFAULT_MODEL, PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.migration import migration_file_export_enabled, serialize_device_map
from models.demos.common.prefill.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    load_trace_token_ids,
    open_mesh_device,
    resolve_trace_dir,
)

# NOTE: the layer_completion classes (the standalone `_layer_completion` extension)
# are imported lazily at point-of-use — its .so is built only under WITH_PYTHON_BINDINGS and may be
# absent in a packaged/wheel build, so a top-level import would hard-fail the runner for everyone
# (including single-rank runs that never touch layer completion).


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

D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 64 * 1024))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
MODEL_CFG = ADAPTER.model_config

# D2D socket transport (>1 rank): one sender/receiver pair per rank boundary carries the hidden state
# over inter-galaxy fabric, sharded seq across SP rows. The emb (TP) axis follows the adapter's residual
# layout (see pipeline_activation_emb_tp_sharded) so the receiver backing needs no reshard.
D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[
        ttnn.PlacementShard(2),
        ttnn.PlacementShard(3) if ADAPTER.pipeline_activation_emb_tp_sharded else ttnn.PlacementReplicate(),
    ]
)

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
# Chunks this run drives. The per-user KV cache is sized to exactly hold them
# (max_seq_len = chunk_size * num_chunks), so there is no separate cache-length knob to keep in sync.
# PREFILL_MAX_SEQ_LEN still overrides if a larger cache is wanted.
NUM_CHUNKS = int(os.environ.get("PREFILL_STANDALONE_NCHUNKS", 11))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * NUM_CHUNKS))
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
# Capture each rank's per-chunk forward as a (segmented) ttnn trace and replay it every chunk instead of
# re-dispatching op-by-op. Needs the mesh opened with a trace region; the segmented capture (sub-device
# swaps + per-layer acks) is handled by SubDeviceTraceController inside the runtime.
USE_TRACE = os.environ.get("PREFILL_USE_TRACE", "0") == "1"
_TRACE_REGION_SIZE = int(os.environ.get("PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024)) if USE_TRACE else 0

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


# ---------------------------------------------------------------------------
# Layer assignment
# ---------------------------------------------------------------------------


def _snap_counts_to_starts(counts, valid_starts, num_layers):
    """Nudge an even split's interior rank boundaries onto the nearest valid start (preserving
    sum == num_layers), for models that constrain where a rank may begin (layer_split_boundaries).
    Nearest by |distance| then lower index; each boundary is used at most once and stays increasing."""
    valid = sorted(valid_starts)
    boundaries, s = [], 0
    for c in counts[:-1]:
        s += c
        boundaries.append(s)
    snapped, prev = [], 0
    for b in boundaries:
        cand = min(
            (v for v in valid if prev < v < num_layers and v not in snapped),
            key=lambda v: (abs(v - b), v),
            default=None,
        )
        if cand is None:
            raise ValueError(f"cannot place {len(counts)} pipeline ranks on valid layer boundaries {valid}")
        snapped.append(cand)
        prev = cand
    out, prev = [], 0
    for b in [*snapped, num_layers]:
        out.append(b - prev)
        prev = b
    return out


def compute_layer_split(num_layers: int, num_ranks: int, valid_starts=None) -> list[tuple[int, int]]:
    """Contiguous (first_layer_idx, count) per rank. PREFILL_PP_LAYER_COUNTS, a
    comma-separated count list summing to num_layers, overrides the default even
    split (remainder handed to the earlier ranks).

    ``valid_starts`` (from the adapter's ``layer_split_boundaries``): layer indices at which a rank may
    begin. None => unconstrained. When set, the default even split is auto-snapped onto valid
    boundaries, and any split (explicit or snapped) whose rank starts fall off them is rejected early."""
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
        if valid_starts is not None:
            counts = _snap_counts_to_starts(counts, valid_starts, num_layers)

    ranges = []
    start = 0
    for count in counts:
        ranges.append((start, count))
        start += count

    if valid_starts is not None:
        for first_idx, _ in ranges:
            if first_idx not in valid_starts:
                near = sorted(b for b in valid_starts if abs(b - first_idx) <= 4)
                raise ValueError(
                    f"pipeline rank starts at layer {first_idx}, not a valid boundary for this model "
                    f"(nearest valid: {near}). Set PREFILL_PP_LAYER_COUNTS so every cumulative boundary "
                    f"is a valid start."
                )
    return ranges


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------


def _load_token_ids() -> list[int]:
    """Load this run's token IDs (same source as the single-rank standalone loop).
    All ranks load identically so they agree on the chunk schedule."""
    import json

    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default))
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

# When the completion ring is full, spin waiting for the router to drain rather than
# dropping/failing immediately. Bounded so a genuinely stalled router still surfaces.
LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001  # tiny yield so the spin doesn't peg a core


def build_layer_completion_sink(producer, *, source_rank, num_layers):
    """Build the per-layer completion sink the runtime fires once per layer.

    Computes a globally-dense ordering key and pushes a full completion
    into `producer` (a ttnn._experimental.layer_completion.LayerCompletionQueue). The
    master router re-emits completions strictly in ascending `seq`.

    seq = request_id * num_layers + layer_idx — dense across all (request,
    layer) pairs. For pipelined prefill each rank owns a disjoint set of
    global layer indices per request, so the union of every rank's seqs
    tiles [0, num_requests*num_layers) with no gaps or collisions.

    The runtime calls the returned sink as `sink(layer_idx, request_id)`,
    binding the current chunk's request_id per prefill() call — so this
    builder needs no request-id accessor and reads no shared mutable state.

    Args:
        producer: connected LayerCompletionQueue (the host-local ring).
        source_rank: this rank's world rank (diagnostic in the payload).
        num_layers: total GLOBAL layers (the seq stride per request), NOT this rank's slice.
    """

    def on_layer_complete(layer_idx: int, request_id: int) -> None:
        # Hot path: fired once per layer inside model.forward. Push directly (no per-call closure)
        # and return on the common success; only the rare full-ring case falls into the spin below.
        seq = request_id * num_layers + layer_idx
        if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            return

        # Ring is sized well above in-flight depth; a full ring means the router
        # thread is momentarily behind. Spin (don't drop) for up to PUSH_SPIN_TIMEOUT_S
        # waiting for it to drain; log on entry, every PUSH_SPIN_LOG_EVERY_S while waiting,
        # and on exit. Only surface an error if it never catches up.
        start = time.monotonic()
        next_log = start + LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
        logger.warning(
            f"[layer-completion] ring full (seq={seq}); spinning up to "
            f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s for router to drain"
        )
        while True:
            if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
                logger.info(f"[layer-completion] ring drained after {time.monotonic() - start:.1f}s; pushed seq={seq}")
                return
            if _shutdown:
                # Operator asked to stop (SIGTERM/SIGINT). Abort the spin immediately instead of
                # ignoring the signal for up to the full timeout; teardown runs via run_request_loop's
                # finally. Raising (vs. silently dropping) keeps the failure visible.
                raise RuntimeError(f"layer-completion ring full (seq={seq}); shutdown requested while spinning")
            now = time.monotonic()
            if now - start >= LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:
                logger.error(f"[layer-completion] gave up after {now - start:.1f}s spinning on full ring (seq={seq})")
                raise RuntimeError(
                    f"layer-completion ring full (seq={seq}); router not draining after "
                    f"{LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S:.0f}s"
                )
            if now >= next_log:
                logger.warning(f"[layer-completion] still spinning on full ring (seq={seq}) after {now - start:.0f}s")
                next_log += LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S
            time.sleep(LAYER_COMPLETION_PUSH_SPIN_SLEEP_S)

    return on_layer_complete


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------


def _first_rank_chunk_tokens(runtime, token_ids: list[int], kv_actual: int) -> ttnn.Tensor:
    """Slice this chunk's tokens and build the SP-sharded input tensor. Delegates to the runtime's own
    builder so the input format has one source of truth."""
    cfg = runtime.config
    return runtime.make_chunk_input(token_ids[kv_actual : kv_actual + cfg.chunk_size])


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


def _d2d_send(outbound, activation: ttnn.Tensor, rank: int, meta: dict, *, deallocate: bool = True) -> None:
    """Push this rank's output hidden state + metadata to the downstream rank's receiver, then free it.
    The model already emits the activation in the sender backing's spec, and outbound_socket_service_sync
    TT_FATALs on any spec mismatch, so no host-side relayout is needed.

    deallocate=False when the activation is the traced path's persistent _trace_output buffer: the socket
    sync copies it into the sender backing on the CQ (before the next replay, which reuses the same buffer,
    is enqueued), so it must NOT be freed — the next chunk's replay writes into it in place."""
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
    if deallocate:
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
        inp,
        kv_caches,
        slot_id=meta["slot_id"],
        actual_start=meta["actual_start"],
        actual_end=meta["actual_end"],
        request_id=c,
    )
    if SYNC_PER_CHUNK:
        # Block on device completion so the delta is this rank's forward alone, not the downstream-start
        # proxy. Serializes dispatch (no overlap) — measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={(time.time() - t_start) * 1000.0:.3f}")
    if not runtime.config.is_last_rank:
        # Traced: `out` is the runtime's persistent _trace_output (the next replay overwrites it in place),
        # so the send copies it into the socket backing but must not free it. Eager: `out` is fresh — free it.
        _d2d_send(d2d_out, out, rank, meta, deallocate=not runtime.config.use_trace)  # grant below ships it
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
    # first_compute_start is None if the loop exited before any chunk was computed (e.g. an immediate
    # shutdown sentinel, or SIGINT during the initial socket wait) — guard the float format.
    fcs = f"{first_compute_start:.6f}" if first_compute_start is not None else "n/a"
    logger.info(f"[pp rank {rank}] E2E_CLOCK first_compute_start={fcs} last_compute_end={time.time():.6f}")
    logger.info(f"[pp rank {rank}] processed {n_done} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")


def run_request_loop(
    runtime,
    kv_caches,
    rank: int,
    num_ranks: int,
    *,
    hidden_size: int,
    h2d_service=None,
    d2d_in=None,
    d2d_out=None,
    migration_driver=None,
) -> dict:
    """Production serving loop — UNBOUNDED. rank 0 reads each chunk from the H2D socket (the external
    producer decides the count); downstream ranks read from D2D. Runs until the producer/scheduler
    closes the stream with the all -1 shutdown sentinel (each rank forwards it and exits gracefully) or,
    as a hard fallback, until SIGTERM/SIGKILL. No fixed NUM_CHUNKS bound, no trace input — see
    run_standalone_loop for the bounded/trace variant.

    Exception: in migration-validation mode (PREFILL_VALIDATE_MIGRATION=1) the scheduler driver never
    pushes the shutdown sentinel — it pushes PREFILL_STANDALONE_CHUNKED_NCHUNKS chunks, migrates, then
    writes the DONE sentinel for the runner to poll. So the loop exits after that many chunks and returns
    to validate_after_prefill. Returns (chunks_per_slot, real_end_per_slot, total_chunks).

    PREFILL_REQUEST_LOOP_PCC=1 (single-rank, bring-up only) PCC-checks the populated KV against the golden
    trace once the stream closes — the production analogue of standalone's per-rank KV check, driven by the
    real H2D producer path (and, under use_trace, the replayed forward + post-compile LayerAck)."""
    cfg = runtime.config
    if cfg.is_first_rank and h2d_service is None:
        raise ValueError("request mode requires the H2D service on the first rank for input")
    logger.info(
        f"[pp rank {rank}/{num_ranks}] request (unbounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input={'h2d' if cfg.is_first_rank else 'd2d'})"
    )
    # Self-test bound: PREFILL_MIGRATION_SELFTEST=1 makes the loop run exactly NUM_CHUNKS chunks then
    # exit CLEANLY so the post-loop migrate + verify can run — without it the unbounded loop blocks in
    # recv and only SIGKILL exits, which kills before the verify. NUM_CHUNKS is the run's single chunk
    # count (the per-user KV cache is sized to exactly hold it, max_seq_len = chunk_size * NUM_CHUNKS),
    # and the producer pushes the same count, so they match by construction. 0 == unbounded serving.
    n_selftest = NUM_CHUNKS if os.environ.get("PREFILL_MIGRATION_SELFTEST", "0") == "1" else 0
    t0 = time.perf_counter()
    c = 0
    first = None
    # Per-slot bookkeeping for the optional post-loop migration validation (validate_after_prefill):
    # how many chunks each slot received and its highest real (non-pad) end position.
    chunks_per_slot: dict = {}
    real_end_per_slot: dict = {}
    # If we run prefill validation, we need to know the expected number of chunks to exit the loop.
    _expected_chunks = (
        int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "0"))
        if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1"
        else 0
    )
    slot_id = 0  # last chunk's slot — the PREFILL_REQUEST_LOOP_PCC check below reads the slice this rank populated
    while not _shutdown:
        if n_selftest and c >= n_selftest:
            break
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
        slot_id = slot  # for the PREFILL_REQUEST_LOOP_PCC check after the loop
        chunks_per_slot[slot] = chunks_per_slot.get(slot, 0) + 1
        real_end_per_slot[slot] = max(real_end_per_slot.get(slot, 0), meta["actual_end"])
        t = _compute_and_send(runtime, kv_caches, rank, c, inp, meta, d2d_out)
        # Interleaved migration: register this chunk's correlation, then migrate any chunk whose layers
        # have all acked (single-rank: chunk c's acks are visible the moment _compute_and_send returns).
        if migration_driver is not None:
            migration_driver.record_chunk(c, meta["slot_id"], meta["actual_start"], meta["actual_end"])
            logger.info(
                f"[interleave] prefilled chunk {c} (slot{meta['slot_id']} "
                f"pos[{meta['actual_start']},{meta['actual_end']})); pumping migration driver"
            )
            migration_driver.pump(current_prefill_chunk=c)
        # Track the real (non-pad) end position per slot: the producer clamps actual_end to the real
        # ISL, so the max over a slot's chunks is that slot's prompt length (== blaze's S).
        s = meta["slot_id"]
        real_end_per_slot[s] = max(real_end_per_slot.get(s, 0), meta["actual_end"])
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
    if num_ranks > 1 and n_selftest:
        ttnn.distributed_context_barrier()
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)

    # MUST stay above the return: this block sat BELOW it and was silently dead, so
    # PREFILL_REQUEST_LOOP_PCC=1 ran no check at all and reported nothing — a green run that had
    # verified nothing. Runs after _drain_and_log_e2e so the chunk's KV writes are flushed first.
    if os.environ.get("PREFILL_REQUEST_LOOP_PCC", "0") == "1" and c > 0:
        # Bring-up validation of the production path (golden-trace input): the same optional runtime hook
        # standalone uses. n_chunks = the count the producer actually pushed. Single-rank only (a pipeline
        # rank owns a layer slice; kv_cache_pcc_check offsets by first_layer_idx, but multi-rank KV PCC is
        # driven via the standalone loop).
        pcc_check = getattr(runtime, "kv_cache_pcc_check", None)
        if pcc_check is None:
            raise RuntimeError(
                f"PREFILL_REQUEST_LOOP_PCC=1 but {type(runtime).__name__} implements no kv_cache_pcc_check "
                "(optional bring-up hook; see ADDING_A_PREFILL_MODEL.md §2)."
            )
        pcc_check(
            kv_caches,
            slot_id=slot_id,
            n_chunks=c,
            trace_dir=os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default),
            first_layer_idx=cfg.first_layer_idx,
        )

    return chunks_per_slot, real_end_per_slot, c


def run_standalone_loop(runtime, kv_caches, rank: int, num_ranks: int, *, d2d_in=None, d2d_out=None) -> None:
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
        t = _compute_and_send(runtime, kv_caches, rank, c, inp, meta, d2d_out)
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
        # kv_cache_pcc_check is an OPTIONAL runtime hook (golden-trace bring-up only — never used in
        # production serving), so a model whose runtime doesn't implement it can't be checked this way.
        pcc_check = getattr(runtime, "kv_cache_pcc_check", None)
        if pcc_check is None:
            raise RuntimeError(
                f"PREFILL_STANDALONE_PCC=1 but {type(runtime).__name__} implements no kv_cache_pcc_check "
                "(optional bring-up hook; see ADDING_A_PREFILL_MODEL.md §2)."
            )
        # Pass the raw trace path; the validation helper resolves it (descends the vllm hash subdir).
        pcc_check(
            kv_caches,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default),
            first_layer_idx=cfg.first_layer_idx,
        )


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
        ("PREFILL_USE_TRACE", f"{USE_TRACE} (trace_region={_TRACE_REGION_SIZE >> 20} MB)"),
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
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_PCC", os.environ.get("PREFILL_STANDALONE_PCC", "0")),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        (
            "PREFILL_STANDALONE_CHUNKED_RECORD_ONLY",
            os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0"),
        ),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        ("PREFILL_MOCK_MIGRATION", os.environ.get("PREFILL_MOCK_MIGRATION", "0")),
        ("PREFILL_REQUEST_LOOP_PCC", os.environ.get("PREFILL_REQUEST_LOOP_PCC", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_WAIT_READY_MS", os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000")),
        ("PREFILL_MIGRATION_EXPORT_TO_FILE", os.environ.get("PREFILL_MIGRATION_EXPORT_TO_FILE", "0")),
        (
            "PREFILL_MIGRATION_DEVICE_MAP_PATH",
            os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "<transport-dependent default>"),
        ),
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

    mesh_device = open_mesh_device(
        GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE, trace_region_size=_TRACE_REGION_SIZE
    )

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
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=USE_TRACE,
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    # The engine owns the KV cache(s): allocate them once (the adapter defines the layout) as an opaque
    # KvCaches, hand that container to every runtime call, and let it free with the mesh at shutdown. The
    # runner stays model-agnostic — it never unpacks the container; the (model-specific) runtime pulls out
    # the primary cache and any secondary cache (e.g. a sparse/DSA model's index cache) it needs, and folds
    # both into the merged migration table (see build_kv_chunk_table).
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        _serve_standalone(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)
    else:
        _serve_request(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    # Release captured traces + the sub-device managers that own them BEFORE closing the mesh: the
    # trace buffers live inside the MoE-overlap SubDeviceManagers, so closing with both registered
    # frees them in the wrong order and segfaults in BankManager::deallocate_buffer (see
    # TtPrefillRuntime.release_trace). Optional hook — models without it are unaffected.
    _release_trace = getattr(runtime, "release_trace", None)
    if _release_trace is not None:
        _release_trace()

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_standalone(
    runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool
) -> None:
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

    # Capture the trace (use_trace) HERE — after D2D endpoints are built (their receiver-socket L1 must be
    # allocated before the trace records, or it corrupts replay on the last rank) and before the chunk loop,
    # so the one-time capture stays out of the timed loop.
    if getattr(runtime, "capture_trace", None) and runtime.config.use_trace:
        runtime.capture_trace(kv_caches)

    logger.info(f"[pp rank {rank}] setup complete, entering standalone loop")
    run_standalone_loop(runtime, kv_caches, rank, num_ranks, d2d_in=d2d_in, d2d_out=d2d_out)

    if d2d_in is not None or d2d_out is not None:
        # Free the services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        d2d_in = d2d_out = None
        gc.collect()


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer (prefill_producer.py / the scheduler); unbounded (runs to SIGTERM). Same pipeline
    mechanics as standalone (num_ranks 1..N over D2D); the only difference is the trigger (H2D input)
    and that it runs forever.

    Migration (KV-chunk-table publish) runs for any rank count: every rank all-gathers its stage into
    the merged table and rank 0 builds + publishes it. Per-layer completions feed the scheduler channel
    directly single-rank, or route through a per-host LayerCompletionRouter to the master for
    num_ranks>1. Shutdown for num_ranks>1 is rough: downstream ranks block in D2D recv when rank 0
    stops, so they exit on teardown / SIGKILL."""
    single_rank = num_ranks == 1

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

    # D2D pipeline transport for num_ranks>1 (same as standalone).
    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)
        # The chained D2D socket rendezvous finishes at staggered times per rank. Without this barrier
        # a rank can reach the loop's first fabric-link lease while an upstream/downstream rank is still
        # in rendezvous, deadlocking the lease handshake before any chunk flows.
        ttnn.distributed_context_barrier()

    # Per-layer LayerAck -> scheduler-driven migration. Two wirings by topology:
    #   * single-rank: the runtime owns the scheduler's counter channel and inject()s it
    #     directly (the original path).
    #   * pipeline (num_ranks > 1): each rank owns only a layer slice, so it cannot inject
    #     the scheduler channel directly. Every rank pushes full {seq, source_rank, layer_idx,
    #     request_id} completions into a host-local LayerCompletionQueue; a per-host
    #     LayerCompletionRouter forwards them to the master rank, which re-emits them in
    #     global seq order into the SAME counter channel the scheduler connects to. See
    #     build_layer_completion_sink() and ttnn._ttnn.layer_completion.
    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    ack_channel = None
    router = None
    producer = None
    # Completion checking (master rank only, test-only): a consumer that stands in for the scheduler
    # on the master's counter channel, to verify aggregated per-(chunk, layer) completions. See
    # scheduler_standins.CompletionCheckConsumer. Gated by PREFILL_CHECK_COMPLETIONS=1 so it never competes with a real
    # scheduler consuming the same channel in production.
    completion_check = None
    check_completions = os.environ.get("PREFILL_CHECK_COMPLETIONS", "0") == "1"
    # The single-rank LayerAck channel is the scheduler's per-layer signal (it drives migration).
    # Opt-in: creating it unconditionally makes two concurrent single-rank runs sharing a service_id
    # collide on the same /dev/shm segment. Defaults on when migration is enabled (its only consumer);
    # set PREFILL_ENABLE_LAYER_ACK=1 to force it on without full migration.
    enable_layer_ack = (
        os.environ.get("PREFILL_ENABLE_LAYER_ACK", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")) == "1"
    )

    def _unlink_stale_shm(name: str) -> None:
        # A prior run that didn't tear down cleanly leaves the segment behind (shm_open O_EXCL fails).
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    # Migration KV-chunk-table publish: runs for ANY rank count. The runner owns the control flow;
    # every rank joins the cross-host all-gather (barrier) that merges the table, then ONLY the first
    # rank asks its model runtime to build the merged table and sends it to the worker (mirroring
    # tt-blaze where all ranks all-gather but only mesh 0 builds + sends). Previously single-rank only.
    migration_endpoint = None
    # Single opt-in: PREFILL_MIGRATION_SELFTEST=1 runs the migrate + slot==slot verify AND implies the
    # table publish it depends on, so you don't also have to set PREFILL_ENABLE_MIGRATION. The latter
    # still works on its own for production publish-without-selftest.
    _selftest = os.environ.get("PREFILL_MIGRATION_SELFTEST", "0") == "1"
    # Mock integration: publish the KV chunk table + device map for an external reader
    # (prefill_producer.py's PREFILL_PRODUCER_CHECK_PCC) with NO migration worker. It must open this
    # block ON ITS OWN — gating it behind _migration_enabled made it unreachable, since
    # PREFILL_ENABLE_MIGRATION additionally drives the publish-and-block-on-WORKER_READY path below.
    _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
    _migration_enabled = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1" or _selftest
    _interleaved = os.environ.get("PREFILL_MIGRATION_INTERLEAVED", "0") == "1"

    _file_export = migration_file_export_enabled()
    # The selftest/interleaved paths drive migrate()/wait_complete() through the MigrationLayerClient,
    # which file-export mode never creates — reject the combination up front (rank-invariant:
    # env-only) rather than fail an assert after bring-up.
    if _file_export and (_selftest or _interleaved):
        raise ValueError(
            "PREFILL_MIGRATION_EXPORT_TO_FILE=1 is incompatible with PREFILL_MIGRATION_SELFTEST=1 / "
            "PREFILL_MIGRATION_INTERLEAVED=1: file-export mode has no MigrationLayerClient, "
            "so the runner cannot issue migrate() itself."
        )

    # Both flags put a scheduler stand-in on the master's ack channel, and try_consume_all() is a
    # destructive read against one shared cursor -- two consumers split the ack stream instead of each
    # seeing it whole. Rank-invariant (env + num_ranks only) and checked before the bring-up all-gather,
    # so every rank raises together rather than deadlocking the survivors. Single-rank never builds the
    # check consumer, so the combination is harmless there.
    if num_ranks > 1 and check_completions and _selftest and _interleaved:
        raise ValueError(
            "PREFILL_CHECK_COMPLETIONS=1 and PREFILL_MIGRATION_INTERLEAVED=1 cannot both be set in "
            f"pipeline mode (num_ranks={num_ranks}): both consume {ack_shm_name}, so each sees only part "
            "of the ack stream (the completion check can still report PASS while interleaved migration "
            "silently drops its tail chunks). Set exactly one."
        )

    # Mock integration (prefill_producer.py's PREFILL_PRODUCER_CHECK_PCC): publish the KV chunk table +
    # device map for an external device-less reader, with NO migration worker. Deliberately OUTSIDE the
    # _migration_enabled block below: that block's first step is
    # deliver_device_map_and_gather_stage_layout(), which imports the _migration_client .so and joins a
    # cross-rank all-gather. Mock has neither a client nor peers, so routing it through there raises
    # ImportError(_migration_client) — which is exactly what happens if you only make the old in-block
    # `elif PREFILL_MOCK_MIGRATION` reachable. Both writes here are local (build table + serialize map).
    if _mock_migration and not _migration_enabled:
        _mock_table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        _mock_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        runtime.build_kv_chunk_table(kv_caches, path=_mock_table_path)
        # fabric_node -> ASIC unique_id, so the producer can resolve chips for read_dram_umd without
        # touching the ControlPlane.
        serialize_device_map(mesh_device, _mock_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {_mock_table_path}, device map -> {_mock_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    if _migration_enabled:
        if is_first_rank:
            # Clear a stale DONE sentinel from a prior run so the validator can't read its pairs.
            # First rank only -- it owns the publish + validation handshake.
            _done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
            if os.path.exists(_done_file):
                logger.warning(f"[migration] removing stale DONE sentinel {_done_file} from a prior run")
                os.remove(_done_file)

        # Migration bring-up, split by ownership before the request loop opens (the worker gates on
        # SetTable + AssignDevMap, so this must finish first):
        #   * ALL RANKS deliver their local device map + join the all-gather barrier (COLLECTIVE —
        #     every rank must call it or the communicator deadlocks).
        #   * The model RUNTIME builds + serializes the model-specific KV chunk table and returns its
        #     path (runtime.build_kv_chunk_table — the model owns the cache layout / address math).
        #   * RANK 0 ONLY publishes that serialized table to the worker and blocks on WORKER_READY.
        # With PREFILL_MIGRATION_EXPORT_TO_FILE=1 the device map goes to a host-local text file
        # and the table stays on disk instead; no worker handshake.
        from models.demos.common.prefill.runners.migration import (
            allgather_kv_stage_layout,
            deliver_device_map_and_gather_stage_layout,
            export_device_map_file_and_gather_stage_layout,
            migration_device_map_file_path,
            publish_serialized_table_and_wait_ready,
        )

        # This rank's pipeline stage owns layers [first_layer_idx, first_layer_idx + num_my_layers).
        # The layer-aware merge gathers each rank's range so the table spans all stages; pass this
        # rank's range (same split the runtime/cache was built with).
        first_layer_idx, num_my_layers = compute_layer_split(NUM_LAYERS, num_ranks)[rank]
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))

        # Only rank 0 writes the merged table, but every rank (and every co-located producer) reads it
        # back. Multi-host therefore requires shared storage; the per-host default is invisible to the
        # other hosts. Reject it here — rank-invariant (same env + num_ranks), so all ranks raise
        # together before the stage-layout all-gather below rather than deadlocking the survivors.
        if num_ranks > 1:
            _abs_table = os.path.abspath(table_path)
            if any(_abs_table == p or _abs_table.startswith(p + "/") for p in ("/tmp", "/dev/shm", "/run", "/var/tmp")):
                raise ValueError(
                    f"PREFILL_MIGRATION_TABLE_PATH={_abs_table} is on per-host storage; with num_ranks="
                    f"{num_ranks} the table rank 0 writes is invisible to the other hosts' readers. Point "
                    "it at shared/NFS storage (e.g. /data/...)."
                )

        # Drop a stale table from a prior run before rank 0 rebuilds it, so a reader can never
        # deserialize last run's table (same rationale as the DONE sentinel above).
        if is_first_rank and os.path.exists(table_path):
            logger.warning(f"[migration] removing stale KV chunk table {table_path} from a prior run")
            os.remove(table_path)

        # ALL RANKS join the stage-layout all-gather (collective barrier; rank 0 needs the merged
        # layout to build the table). Real migration also delivers this rank's local FNID->UMD map to
        # its co-located worker first; mock has no worker (the producer reads a serialized JSON map),
        # so it joins the gather directly and never imports the worker client extension.
        #
        # Ask the runtime for this stage's KV base -- the engine must not introspect the opaque
        # KvCaches struct, whose shape is per-model.
        if not hasattr(runtime, "kv_migration_base_address"):
            raise RuntimeError(
                f"migration enabled but runtime {type(runtime).__name__} implements no "
                "kv_migration_base_address (see docs/ADDING_A_PREFILL_MODEL.md §2)."
            )
        kv_base_addr = runtime.kv_migration_base_address(kv_caches)
        _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
        if _mock_migration:
            stage_layout = allgather_kv_stage_layout(
                mesh_device, kv_base_addr, GLOBAL_MESH_SHAPE, first_layer_idx, num_my_layers
            )
        elif _file_export:
            stage_layout = export_device_map_file_and_gather_stage_layout(
                mesh_device,
                kv_base_addr,
                GLOBAL_MESH_SHAPE,
                first_layer_idx,
                num_my_layers,
                migration_device_map_file_path(),
            )
        else:
            stage_layout = deliver_device_map_and_gather_stage_layout(
                mesh_device, kv_base_addr, GLOBAL_MESH_SHAPE, first_layer_idx, num_my_layers, rank
            )

        if _mock_migration:
            # Mock integration (prefill_producer.py): the SAME merged table the real publish builds, but
            # serialized to disk for a device-less producer to read back via
            # ttnn.experimental.disaggregation.import_from_protobuf_file — WITHOUT the migration_endpoint
            # worker (no MigrationLayerClient, no WORKER_READY). Checked before is_first_rank so rank 0
            # also takes this path instead of blocking on a worker that isn't running.
            #
            # EVERY rank serializes its OWN local fabric_node -> ASIC unique_id device map so each
            # co-located producer can resolve only its host's chips for read_dram_umd (the multi-rank
            # merged table carries every host's fnids; a producer with just its local map naturally
            # filters to its own layers). The device-map path is host-local (each rank overwrites the
            # same name on its own host); the table path MUST be on shared storage (only rank 0 writes
            # it, but every host's reader resolves the same path) — enforced above for num_ranks > 1.
            device_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            serialize_device_map(mesh_device, device_map_path)
            if is_first_rank:
                # RANK 0 builds the merged table spanning every gathered stage — identical to the real
                # publish call below, minus the worker handshake.
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    stage_layout=stage_layout,
                )
                logger.info(f"[mock-migration] merged KV chunk table -> {table_path} (no migration worker)")
            logger.info(f"[mock-migration] rank {rank}: local device map -> {device_map_path}")
        elif _file_export:
            # The files on disk are the handoff: no SET_TABLE, no WORKER_READY.
            if is_first_rank:
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    stage_layout=stage_layout,
                )
                logger.info(f"[migration] merged KV chunk table -> {table_path} (file export; no worker handshake)")
            logger.info(f"[migration] rank {rank}: exported local device map -> {migration_device_map_file_path()}")
        elif is_first_rank:
            # RANK 0: model runtime builds + serializes the merged table (spanning all gathered
            # stages), then publish the serialized path + block on WORKER_READY.
            table_path = runtime.build_kv_chunk_table(
                kv_caches,
                table_path,
                first_layer_idx=first_layer_idx,
                num_my_layers=num_my_layers,
                stage_layout=stage_layout,
            )
            migration_endpoint = publish_serialized_table_and_wait_ready(
                table_path=table_path,
                wait_ready_timeout_ms=wait_ready_ms,
            )
        else:
            logger.info(
                f"[migration] rank {rank}: delivered local device map + contributed stage "
                f"(first_layer={first_layer_idx}, count={num_my_layers}); rank 0 sends the merged table."
            )

    elif os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1":
        # Mock integration (prefill_producer.py): serialize the KV chunk table so an external
        # producer can read it back via ttnn.experimental.disaggregation.import_from_protobuf_file
        # and locate each chunk — WITHOUT the migration_endpoint worker (no MigrationLayerClient,
        # no WORKER_READY). One galaxy => one complete table spanning all NUM_LAYERS / NUM_USERS
        # (both caches, merged, for a sparse model).
        # Sibling of `if _migration_enabled:` ON PURPOSE -- the mock path is what runs when migration
        # is OFF. One level deeper it was an elif of `if is_first_rank:`, so reaching it needed
        # _migration_enabled AND a non-first rank: unreachable in single-rank, and the runner then
        # entered the request loop having never published the table/device map its consumer waits on.
        # Single-rank only, and loudly so. Dedenting it out of `if _migration_enabled:` also took it
        # out of the pre-#48826 `if single_rank:` wrapper, so with num_ranks>1 every rank would build
        # a table covering only ITS layer slice and publish over the same paths -- and co-located
        # ranks would race serialize_device_map's `<path>.tmp` -> os.replace as well. Only the real
        # migration path merges stages (deliver_device_map_and_gather_stage_layout), and that needs
        # the worker. Same guard #48826 removed for PREFILL_ENABLE_MIGRATION, kept for the mock path.
        if not single_rank:
            raise ValueError(
                f"PREFILL_MOCK_MIGRATION=1 is unsupported for num_ranks={num_ranks} (each rank would "
                "publish a table covering only its own layer slice; a merged mock table is not "
                "implemented); run single-rank or unset PREFILL_MOCK_MIGRATION."
            )
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

    if single_rank and enable_layer_ack:
        # Direct path: the runtime owns + inject()s the scheduler counter channel.
        _unlink_stale_shm(ack_shm_name)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")
    elif single_rank:
        logger.info("[migration] LayerAck channel disabled (set PREFILL_ENABLE_LAYER_ACK=1 to enable)")
    else:
        # Pipeline path: route per-rank completions to the master, which re-emits in seq order.
        # Imported here (not at module top) so single-rank / no-extension builds never need the
        # standalone _layer_completion .so (built only with WITH_PYTHON_BINDINGS).
        from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

        # Each rank's router OWNS its own ring, so the name must be per-rank — append _{rank} even to
        # the env override (a single literal would make colocated ranks unlink each other's live ring
        # and collide on O_EXCL create).
        ring_base = os.environ.get("PREFILL_LAYER_COMPLETION_RING", "/tt_prefill_layer_completion_ring")
        ring_shm_name = f"{ring_base}_{rank}"
        _unlink_stale_shm(ring_shm_name)
        if rank == master_rank:
            _unlink_stale_shm(ack_shm_name)
        # The router owns the host-local ring (and, on the master, the scheduler counter channel,
        # which it inject()s in order). Subordinate ranks MPI-forward completions to the master.
        router = LayerCompletionRouter(
            rank=rank,
            world_size=num_ranks,
            master_rank=master_rank,
            ring_shm_name=ring_shm_name,
            scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
        )
        producer = LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
        # seq stride is the GLOBAL layer total (NUM_LAYERS), NOT this rank's slice; the layer_idx
        # arriving at the sink is already global; the chunk index is bound per prefill() call as
        # request_id (passed by _compute_and_send), so the sink reads no shared mutable state.
        runtime.set_layer_completion_sink(
            build_layer_completion_sink(
                producer,
                source_rank=rank,
                num_layers=NUM_LAYERS,
            )
        )
        logger.info(
            f"[migration] pipelined layer-completion routing up: rank={rank}/{num_ranks} master={master_rank} "
            f"ring={ring_shm_name} "
            + (f"(owns scheduler channel {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
        )

        if rank == master_rank and check_completions:
            from models.demos.common.prefill.runners.scheduler_standins import CompletionCheckConsumer

            completion_check = CompletionCheckConsumer(ack_shm_name, num_layers=NUM_LAYERS)

    # Interleaved migration self-test: rank 0 stands in for the scheduler — consume the per-layer ack
    # channel and migrate each chunk as its layers complete, overlapping later chunks' prefill (replaces
    # the post-loop bulk migrate). Rank 0 only (it holds the migration client); other ranks just verify.
    mig_driver = None
    if _selftest and is_first_rank and _interleaved:
        from models.demos.common.prefill.runners.scheduler_standins import InterleavedMigrationDriver

        assert migration_endpoint is not None, "rank 0 must hold the migration client for interleaved migrate"
        # Granularity flag: "layerwise" (default) migrates each chunk's layers as they ack — finer
        # overlap; "chunkwise" waits for a chunk's full 61-layer ack then migrates it in one shot.
        granularity = os.environ.get("PREFILL_MIGRATION_GRANULARITY", "layerwise").strip().lower()
        if granularity not in ("layerwise", "chunkwise"):
            raise ValueError(f"PREFILL_MIGRATION_GRANULARITY must be 'layerwise' or 'chunkwise', got {granularity!r}")
        mig_driver = InterleavedMigrationDriver(
            ack_shm_name,
            migration_endpoint,
            num_layers=NUM_LAYERS,
            src_slot=int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0")),
            dst_slot=int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1")),
            endpoint_id=int(os.environ.get("PREFILL_MIGRATION_ENDPOINT_ID", "1")),
            wait_complete_ms=int(os.environ.get("PREFILL_MIGRATE_WAIT_COMPLETE_MS", "120000")),
            # Diagnostics: the master router (pipeline only; None in single-rank) exposes .processed so the
            # driver can log injected-vs-consumed acks. router is None here in the single-rank path.
            router=router,
            granularity=granularity,
        )
        logger.info(
            f"[interleave] migration mode = INTERLEAVED, granularity={granularity} "
            f"(PREFILL_MIGRATION_INTERLEAVED=1, PREFILL_MIGRATION_GRANULARITY={granularity}): rank 0 migrates "
            "as layers ack, overlapping later chunks' prefill; one blocking wait at drain"
        )
    elif _selftest and is_first_rank:
        logger.info(
            "[interleave] migration mode = BULK (single post-loop migrate); set PREFILL_MIGRATION_INTERLEAVED=1 "
            "to interleave migrates with prefill"
        )

    # Capture the trace (use_trace) after the D2D endpoints AND the per-layer completion wiring
    # (LayerAck channel / layer-completion sink) are set up, but before the request loop: the capture
    # must split at each completion point, and doing it here keeps the one-time cost out of the loop.
    # No-op if already captured. See _serve_standalone / TtPrefillRuntime.capture_trace().
    if getattr(runtime, "capture_trace", None) and runtime.config.use_trace:
        runtime.capture_trace(kv_caches)

    logger.info(f"[pp rank {rank}] setup complete, entering request loop")

    try:
        chunks_per_slot, real_end_per_slot, total_chunks = run_request_loop(
            runtime,
            kv_caches,
            rank,
            num_ranks,
            hidden_size=hf_config.hidden_size,
            h2d_service=h2d_service,
            d2d_in=d2d_in,
            d2d_out=d2d_out,
            migration_driver=mig_driver,
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

        if _selftest:
            src_slot = int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0"))
            dst_slot = int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1"))

            # Interleaved-vs-bulk MUST be decided by an all-ranks-agree predicate (the env flag), NOT by
            # `mig_driver is None`: mig_driver is set ONLY on rank 0, so keying the pre-migrate barrier on it
            # made non-first ranks (mig_driver None) run an EXTRA barrier that rank 0 (draining) skipped ->
            # distributed_context_barrier() is an anonymous collective, so the mismatched count deadlocks
            # (rank 0's post-migrate barrier pairs with the others' pre-migrate barrier, then the others
            # block forever on their second barrier). Every rank evaluates `_interleaved` identically.
            if not _interleaved:
                # Bulk migrate path: ALL ranks sync + barrier so the KV cache is fully written before rank 0
                # issues the single post-loop migrate. (Interleaved mode instead drains on rank 0 below.)
                ttnn.synchronize_device(runtime.mesh_device)
                if num_ranks > 1:
                    ttnn.distributed_context_barrier()

            # RANK 0 ONLY issues the migrate (it holds the MigrationLayerClient).
            if is_first_rank and mig_driver is not None:
                # Interleaved: per-chunk migrates were already issued during the loop; drain the tail
                # (consume any remaining acks + wait_complete the deferred copies).
                mig_driver.drain(expected_chunks=NUM_CHUNKS)
            elif is_first_rank:
                assert migration_endpoint is not None, "rank 0 must hold the migration client for the self-test"
                # Loopback target is THIS endpoint's own id (A->B loopback; no peer, no connect_to).
                self_ep = int(os.environ.get("PREFILL_MIGRATION_ENDPOINT_ID", "1"))
                # Position range = the src slot's real prefilled length, aligned UP to the 32-token KV
                # migration chunk (blaze's _align_up(S)). Migrate the FULL global layer range [0, NUM_LAYERS)
                # the merged table was built for — the worker routes each layer to its owning stage.
                POS_CHUNK = 32
                real_end = real_end_per_slot.get(src_slot, 0)
                pos_end = ((real_end + POS_CHUNK - 1) // POS_CHUNK) * POS_CHUNK
                logger.info(
                    f"[migration-selftest] loopback migrate slot{src_slot}->slot{dst_slot} "
                    f"layers[0,{NUM_LAYERS}) pos[0,{pos_end}) (real_end={real_end}, self_ep={self_ep})"
                )
                # wait_complete's C++ default is only 30s; a full-prefill loopback copy (here ~2 GB:
                # 56320 pos x 61 layers) can exceed that, so make it configurable.
                wait_complete_ms = int(os.environ.get("PREFILL_MIGRATE_WAIT_COMPLETE_MS", "120000"))
                tok = migration_endpoint.migrate(1, self_ep, src_slot, dst_slot, 0, NUM_LAYERS, 0, pos_end)
                migration_endpoint.wait_complete(tok, wait_complete_ms)
                logger.success(f"[migration-selftest] migrate slot{src_slot}->slot{dst_slot} complete")

            # Barrier: every rank must wait for rank 0's migrate to finish before reading its local
            # dst slot (the migrate covers all stages; each rank then verifies its own layers).
            if num_ranks > 1:
                ttnn.distributed_context_barrier()
            ttnn.synchronize_device(runtime.mesh_device)

            from models.demos.common.prefill.runners.validation import validate_migrations_pairwise

            validate_migrations_pairwise(runtime, kv_caches, [(src_slot, dst_slot)])
    finally:
        # Always tear down — the request loop can raise (e.g. the layer-completion sink's ring-full
        # spin timing out on a stalled router); without this, producer/router/ack segments + the
        # router listener thread leak, and a downstream peer blocked in D2D recv deadlocks the pipeline.
        # Release services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).

        # Release services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        h2d_service = d2d_in = d2d_out = None
        gc.collect()
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()  # joins the listener; the master's final ring-drain + inject happens HERE
        if completion_check is not None:
            # Tally AFTER router.stop(): the master injects its own trailing completions during the
            # listener's final drain (inside stop()). The consumer's mapping survives the owner's
            # shm_unlink (POSIX), so it still reads those — tallying earlier would miss them and
            # falsely report "count short". router.stop() unlinks the channel on the master.
            completion_check.stop_and_report()
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
