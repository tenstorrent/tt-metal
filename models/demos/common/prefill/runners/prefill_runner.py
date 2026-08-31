#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Disaggregated prefill runner — one entry point driving an N-rank serving pipeline.

Model-agnostic: the model is selected by PREFILL_MODEL and driven through a PrefillModelAdapter
(see ../adapter.py and ADDING_A_PREFILL_MODEL.md). This driver wires rank topology, input,
transport, and the per-chunk schedule; the adapter supplies how to build the model, allocate the KV
cache, run a chunk, and describe the cache's layout as a KV-chunk address table.

The model is split across N ranks under tt-run: each rank owns a contiguous layer slice and builds
the same TtPrefillRuntime (first_layer_idx / is_first_rank / is_last_rank). With >1 rank the cross-rank
hidden state moves device-to-device over fabric sockets (connected MGD + FABRIC_2D); N=1 is the
single-galaxy case (no transport). Ranks run decoupled (no per-chunk barrier; one warm-up barrier
after compile).

Serving is request-driven: rank 0's tokens + per-iter PrefillMetadata arrive over the H2D socket from
an external producer (prefill_producer.py / the scheduler); the loop is UNBOUNDED. KV-chunk-table
migration and per-layer LayerAck run at any rank count: every rank joins the all-gather that merges
the chunk table and rank 0 publishes it, and pipeline layer completions are routed to the master
rank, which re-emits them into the same ack channel the scheduler connects to in the single-rank case
(only PREFILL_MOCK_MIGRATION stays single-rank). Shutdown is graceful: the producer/scheduler closes
the stream with an all -1 PrefillMetadata sentinel that each rank forwards downstream and then exits
on; a rank blocked in the recv can only be released by a transfer (the recv device op has no timeout),
so SIGTERM/SIGKILL remains the hard fallback if no sentinel arrives.

The model class is the single source of truth — this driver wires rank topology, input, transport,
and the per-chunk schedule; it does not reimplement embed / layers / forward.
"""

import json
import os
import signal
import time
from typing import Optional

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.common.prefill.adapter import DEFAULT_MODEL, PrefillRunParams, get_adapter
from models.demos.common.prefill.runners.migration import (
    migration_file_export_enabled,
    remove_stale_device_map_sidecars,
    serialize_device_map,
)
from models.demos.common.prefill.runners.runner_utils import (
    activation_global_spec,
    build_h2d_service,
    compute_layer_split,
    open_mesh_device,
)

# NOTE: the layer_completion classes (the standalone `_layer_completion` extension)
# are imported lazily at point-of-use — its .so is built only under WITH_PYTHON_BINDINGS and may be
# absent in a packaged/wheel build, so a top-level import would hard-fail the runner for everyone
# (including single-rank runs that never touch layer completion).


def _apply_manifest_env():
    """If PREFILL_MANIFEST is set, load the shared run.json and populate the env vars
    the runner reads. setdefault => an explicitly exported env var still wins over the
    manifest. Must be invoked before the module-level env reads below (e.g.
    PREFILL_MAX_SEQ_LEN) so the values take effect."""
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


# Populate env from the manifest BEFORE the module-level env reads below.
_apply_manifest_env()

# Both socket transports (H2D input on rank 0, D2D between ranks) share a 1x1 push/sync worker grid and
# the same 3-word PrefillMetadata (slot_id, actual_start, actual_end). The 1x1 grid is the cheapest
# footprint with no penalty: a grid sweep showed compute + handoff gap flat from 1x1 to 4x4 (the
# per-chunk overhead is the persistent service's fabric/NoC presence, not the push workers).
SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12

# The packed [1,1,1,3] uint32 message the sockets carry is `metadata_msg` everywhere; the model's
# `metadata` is the per-element triple of [1,1,1,1] operands the device ops take. There is no device-side
# conversion between them, so a name that blurs the two hides a required host round trip.

# LayerAck D2H FIFO. Records are METADATA_SIZE_BYTES (12 B) each; 4 KB is a PCIe-aligned
# one-page buffer with generous headroom for in-flight records.
LAYER_ACK_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_LAYER_ACK_FIFO_BYTES", 4 * 1024))

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

D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 256))

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
# Per-user KV cache length. In request mode the external producer decides the chunk count, so this is
# the one cache-sizing knob; a chunk must not push a slot past it. Default holds 11 chunks.
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 11))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
# When on (default), the last transformer layer runs kv-only: it fills the KV cache for migration and
# skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head. In a pipeline only the last rank applies it.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
# Build the DFlash drafter context-KV cache during this prefill. Three gates, ALL required: the selected
# model declares the capability (ADAPTER.supports_dflash — only Kimi K2.6/K2.7), the run explicitly opts in
# (PREFILL_DFLASH=1), and a drafter checkpoint is provided (DFLASH_HF_MODEL, resolved by the runtime). The
# capability gate keeps a non-dflash model from ever building a Kimi drafter; the explicit PREFILL_DFLASH
# switch keeps it off unless asked for, even when a checkpoint happens to be on disk.
DFLASH_ENABLED = (
    ADAPTER.supports_dflash and os.environ.get("PREFILL_DFLASH", "0") == "1" and bool(os.environ.get("DFLASH_HF_MODEL"))
)
# Measurement-only: synchronize the device after each chunk's forward and log the isolated per-rank
# compute (CHUNK_COMPUTE). Off in production — the sync serializes dispatch and kills pipeline overlap.
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
TIMING_DIR = os.environ.get("PREFILL_TIMING_DIR", "")
# Some models (e.g. Kimi: single expert group, device gate) route the MoE routing all-gather's global
# semaphores to L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static
# CBs, which needs the mesh opened with an L1_SMALL region. The adapter owns both knobs.
_L1_SMALL_SIZE = ADAPTER.l1_small_size
# Capture each rank's per-chunk forward as a (segmented) ttnn trace and replay it every chunk instead of
# re-dispatching op-by-op. Needs the mesh opened with a trace region; the segmented capture (sub-device
# swaps + per-layer acks) is handled by SubDeviceTraceController inside the runtime.
USE_TRACE = os.environ.get("PREFILL_USE_TRACE", "0") == "1"
_TRACE_REGION_SIZE = int(os.environ.get("PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024)) if USE_TRACE else 0
# DFlash is not trace-compatible: the drafter tap / pack-unpack / KV finalize run outside the runtime's
# captured per-chunk segment, so a replayed trace would silently skip them. Fail loudly rather than
# produce meaningless drafter KV. (Per offline discussion — keep DFlash on the eager per-op path.)
assert not (DFLASH_ENABLED and USE_TRACE), (
    "PREFILL_DFLASH=1 is incompatible with PREFILL_USE_TRACE=1: the DFlash drafter path is not "
    "trace-captured. Run DFlash with PREFILL_USE_TRACE=0."
)

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


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


def _decode_metadata(metadata_msg) -> dict:
    """Read the packed [1,1,1,3] metadata device tensor to host: {slot_id, actual_start, actual_end}."""
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_msg)[0]).view(torch.int32).flatten()
    return {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def _is_shutdown_sentinel(meta: dict) -> bool:
    """True for the all -1 end-of-stream sentinel (see SHUTDOWN_METADATA_WORD); false for every real
    chunk, whose slot_id and KV positions are non-negative and in range."""
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    """Block on the next producer push: returns (tt_tokens, meta, metadata_msg). The device metadata
    tensor is returned (not discarded) so it can be propagated into the model's per-layer ack send.
    Used only by the unbounded request loop (rank 0 input).

    The metadata is always read to host: the scalars are tiny and the read is what lets the loop see the
    in-band shutdown sentinel, which is the runner's graceful teardown path under trace and eager alike."""
    tt_tokens, metadata_msg = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    return tt_tokens, _decode_metadata(metadata_msg), metadata_msg


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
    """Drain the next chunk that landed in the inbound receiver backing into a fresh device tensor. The
    returned tensor already has the embedding-output sharding, so it feeds runtime.prefill with no
    reshard. Pairs with the upstream rank's _d2d_send."""
    t0 = time.perf_counter()
    act, metadata_msg = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    meta = _decode_metadata(metadata_msg)
    where = f"[{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']}"
    logger.info(f"[pp] RECV-d2d {where} [xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms")
    return act, meta, metadata_msg


def _d2d_send(
    outbound, activation: ttnn.Tensor, rank: int, meta: Optional[dict], *, deallocate: bool = True, metadata_msg=None
) -> None:
    """Push this rank's output hidden state + metadata to the downstream rank's receiver, then free it.
    The model already emits the activation in the sender backing's spec, and outbound_socket_service_sync
    TT_FATALs on any spec mismatch, so no host-side relayout is needed.

    metadata_msg (traced path): the packed device tensor received on this rank is forwarded downstream
    verbatim — it is already the [1,1,1,3] uint32 replicated form the op expects, so no host rebuild
    runs (meta may be None). When metadata_msg is None (eager path) the tensor is rebuilt from meta.

    deallocate=False when the activation is the traced path's persistent _trace_output buffer: the socket
    sync copies it into the sender backing on the CQ (before the next replay, which reuses the same buffer,
    is enqueued), so it must NOT be freed — the next chunk's replay writes into it in place."""
    t0 = time.perf_counter()

    if metadata_msg is not None:
        md_tensor = metadata_msg
    else:
        backing = outbound.get_backing_tensor()
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
    where = f"[{meta['actual_start']},{meta['actual_end']})" if meta is not None else "(on-device)"
    logger.info(f"[pp rank {rank}] SEND-d2d {where} [xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms")


def _forward_shutdown(d2d_out, rank: int, hidden_size: int) -> None:
    """Forward the shutdown sentinel to the downstream rank so it unblocks in its own recv, then release
    the outbound link so the transfer ships (mirroring _compute_and_send's tail). The activation content
    is irrelevant — the downstream discards it once it sees the sentinel — but outbound_socket_service_sync
    requires the input's per-shard spec to equal the sender backing's, so build the dummy exactly like a
    real activation: the [1, 1, CHUNK_SIZE, hidden_size] bf16 TILE spec sharded by D2D_MAPPER_CONFIG."""
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


def _record_chunk_timing(rank: int, c: int, compute_start: float, compute_ms: float) -> None:
    """Append one chunk's timing to this rank's CSV via a single O_APPEND write (per-rank file => lone
    writer, atomic even on NFS). Telemetry must never take down the run, so any write error is swallowed."""
    if not TIMING_DIR:
        return
    try:
        fd = os.open(os.path.join(TIMING_DIR, f"rank{rank}.csv"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, f"{rank},{c},{compute_start:.6f},{compute_ms:.3f}\n".encode())
        finally:
            os.close(fd)
    except OSError:
        pass


def _compute_and_send(
    runtime, kv_caches, rank: int, c: int, inp, meta: Optional[dict], d2d_out, d2h_service=None, metadata_msg=None
) -> float:
    """Run one chunk: prefill into the engine-owned kv_caches, forward the output downstream (non-last
    rank) and grant the outbound sender so it ships over fabric. Returns the compute-start epoch
    (NTP-comparable). CHUNK_START is logged BEFORE the forward, with this chunk's metadata, so the
    slot/KV-range is visible per rank even if prefill_chunk hangs. The trailing metadata is kept after
    compute_start so the c=/compute_start= fields stay parseable (plot_pipeline_trace.py)."""
    if SYNC_PER_CHUNK:
        ttnn.synchronize_device(runtime.mesh_device)
    t_start = time.time()
    t_perf = time.perf_counter()
    where = f"slot={meta['slot_id']} [{meta['actual_start']},{meta['actual_end']})"
    logger.info(f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f} {where}")
    out = runtime.prefill_chunk(
        inp,
        kv_caches,
        slot_id=meta["slot_id"],
        actual_start=meta["actual_start"],
        actual_end=meta["actual_end"],
        request_id=c,
        d2h_service=d2h_service,
        metadata_msg=metadata_msg,
    )
    if SYNC_PER_CHUNK:
        # Block on device completion so the delta is this rank's forward alone, not the downstream-start
        # proxy. Serializes dispatch (no overlap) — measurement runs only.
        ttnn.synchronize_device(runtime.mesh_device)
        compute_ms = (time.perf_counter() - t_perf) * 1000.0
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={compute_ms:.3f}")
        _record_chunk_timing(rank, c, t_start, compute_ms)
    if not runtime.config.is_last_rank:
        # Traced: `out` is the runtime's persistent _trace_output (the next replay overwrites it in place),
        # so the send copies it into the socket backing but must not free it. Forward the runtime's
        # persistent metadata, NOT the raw received metadata_msg: that raw tensor is a fresh socket output
        # the replay's writes can land on, so it can arrive corrupted downstream (per-shard-inconsistent),
        # tripping the D2H ack's cross-socket identity check. The persistent buffer survives replay. Eager:
        # `out` is fresh — free it, rebuild md from meta.
        forward_md = None
        if runtime.config.use_trace:
            persistent_md = getattr(runtime, "trace_metadata_msg", None)
            forward_md = persistent_md if persistent_md is not None else metadata_msg
        _d2d_send(
            d2d_out,
            out,
            rank,
            meta,
            deallocate=not runtime.config.use_trace,
            metadata_msg=forward_md,
        )  # grant below ships it
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
    d2h_service=None,
) -> None:
    """Production serving loop — UNBOUNDED. rank 0 reads each chunk from the H2D socket (the external
    producer decides the count); downstream ranks read from D2D. Runs until the producer/scheduler
    closes the stream with the all -1 shutdown sentinel (each rank forwards it and exits gracefully) or,
    as a hard fallback, until SIGTERM/SIGKILL. No fixed chunk bound, no trace input, no PCC, no
    migration — the runner serves; migration is issued from outside (migration_driver.py).

    The per-chunk metadata is read to host on every chunk (traced and eager alike): the scalars are tiny
    and the read is what lets the loop see the in-band shutdown sentinel and drain gracefully. Trace
    replay still consumes the metadata on-device from the persistent buffer; this host read is only the
    small sentinel/logging copy alongside it."""
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
            inp, meta, metadata_msg = _socket_next(h2d_service)  # slot/start/end from producer
        else:
            inp, meta, metadata_msg = _d2d_recv(d2d_in)
        if _is_shutdown_sentinel(meta):
            # End of stream: drop the throwaway payload + its metadata tensor, hand the sentinel to the
            # next rank so it too unblocks and exits, then fall through to the graceful drain below.
            logger.info(f"[pp rank {rank}] SHUTDOWN sentinel received after {c} chunks; exiting request loop")
            ttnn.deallocate(inp)
            ttnn.deallocate(metadata_msg)
            if d2d_out is not None:
                _forward_shutdown(d2d_out, rank, hidden_size)
            break
        t = _compute_and_send(
            runtime, kv_caches, rank, c, inp, meta, d2d_out, d2h_service=d2h_service, metadata_msg=metadata_msg
        )
        if first is None:
            first = t
        c += 1
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0)


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
        (
            "DFLASH_ENABLED",
            f"{DFLASH_ENABLED} (adapter.supports_dflash={ADAPTER.supports_dflash}, "
            f"DFLASH_HF_MODEL={os.environ.get('DFLASH_HF_MODEL') or '<unset>'})",
        ),
        ("PREFILL_USE_TRACE", f"{USE_TRACE} (trace_region={_TRACE_REGION_SIZE >> 20} MB)"),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "<auto: 1d if sp<=8 else 2d>")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        ("PREFILL_MOCK_MIGRATION", os.environ.get("PREFILL_MOCK_MIGRATION", "0")),
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
    ]
    sep = "=" * 70
    lines = [sep, "prefill_runner configuration", sep]
    lines += [f"  {label:<35} = {val}" for label, val in rows]
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))


def _assert_ranks_agree_on_config(rank: int, num_ranks: int) -> None:
    """Fail fast when the ranks of a pipeline did not resolve the SAME model/shape config.

    Every PREFILL_* knob is read from this process's environment, and tt-run only guarantees
    per-rank delivery for what a rank binding puts in `global_env` (it auto-propagates just the
    TT_/ARCH_/WH_/TTNN_/DEEPSEEK_/MESH_ prefixes, and an `-x FOO` in --mpi-args lands in mpirun's
    FIRST application context -> rank 0 only). So exporting PREFILL_MANIFEST in the launching shell
    sets rank 0 and silently leaves every other rank on adapter.py's DEFAULT_MODEL -- a DIFFERENT,
    perfectly valid model, whose weight cache is equally "complete". Nothing errors: the pipeline
    just runs one model's layers into another's, and only the downstream ranks' KV fails PCC.

    A one-int allgather of a config fingerprint turns that into an immediate, named failure.
    """
    if num_ranks <= 1:
        return
    import zlib

    fields = {
        "PREFILL_MODEL": os.environ.get("PREFILL_MODEL") or f"<unset -> default:{DEFAULT_MODEL}>",
        "adapter": ADAPTER.name,
        "num_layers": NUM_LAYERS,
        "chunk_size": CHUNK_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "num_users": NUM_USERS,
        "mesh_shape": GLOBAL_MESH_SHAPE,
        "PREFILL_MIGRATION_EXPORT_TO_FILE": migration_file_export_enabled(),
    }
    fingerprint = "|".join(f"{k}={v}" for k, v in fields.items())
    digest = zlib.crc32(fingerprint.encode()) & 0x7FFFFFFF
    all_digests = ttnn.distributed_context_allgather_int(digest)
    if len(set(all_digests)) == 1:
        logger.info(f"[pp rank {rank}/{num_ranks}] config fingerprint {digest} agrees across all ranks ({fingerprint})")
        return
    disagreeing = [i for i, d in enumerate(all_digests) if d != all_digests[0]]
    raise RuntimeError(
        f"Pipeline ranks resolved DIFFERENT prefill configs: fingerprints {all_digests} "
        f"(ranks {disagreeing} differ from rank 0). This rank ({rank}) has {fingerprint}. "
        f"Each rank prints its own values above — compare PREFILL_MODEL first. "
        f"Fix: pin PREFILL_MODEL (and any other PREFILL_* the run depends on) in the rank binding's "
        f"global_env, which tt-run applies to EVERY rank. Exporting PREFILL_MANIFEST in the shell only "
        f"reaches rank 0."
    )


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
    _assert_ranks_agree_on_config(rank, num_ranks)

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
        # NOT gated on is_last_rank: every rank builds its owned fc slices; the runtime derives the
        # last-rank KV tail from is_last_rank.
        dflash_enabled=DFLASH_ENABLED,
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


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    """Production serving: token chunks + PrefillMetadata arrive over the H2D socket from an external
    producer (prefill_producer.py / the scheduler); unbounded (runs to SIGTERM), num_ranks 1..N over D2D.

    Migration (KV-chunk-table publish) runs for any rank count: every rank all-gathers its stage into
    the merged table and rank 0 builds + publishes it. Per-layer completions feed the scheduler channel
    directly single-rank, or route through a per-host LayerCompletionRouter to the master for
    num_ranks>1. Shutdown for num_ranks>1 is rough: downstream ranks block in D2D recv when rank 0
    stops, so they exit on teardown / SIGKILL."""
    single_rank = num_ranks == 1
    # DFlash packs the drafter's FC partial alongside the hidden (concat on the feature dim), so the D2D
    # activation is 2H wide when enabled; the non-dflash path (every other model) stays H.
    d2d_activation_width = hf_config.hidden_size * (2 if DFLASH_ENABLED else 1)

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
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, d2d_activation_width)
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
    # Single-rank D2H layer-ack: the per-layer ack is emitted by a device op into this D2H FIFO and a
    # host reader thread (LayerAckService) bumps `ack_channel`, instead of the model calling back into
    # host code mid-forward. Set up further down, next to the ack_channel it feeds.
    d2h_service = None
    layer_ack_service = None
    producer = None
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
    # The runner never migrates; it only publishes. Rank 0 holds the client here for the process's
    # lifetime because dropping the reference destroys it and the worker loses the table it gated on.
    migration_endpoint = None
    # Mock integration: publish the KV chunk table + device map for an external reader
    # (prefill_producer.py's PREFILL_PRODUCER_CHECK_PCC) with NO migration worker. It must open this
    # block ON ITS OWN — gating it behind _migration_enabled made it unreachable, since
    # PREFILL_ENABLE_MIGRATION additionally drives the publish-and-block-on-WORKER_READY path below.
    _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
    _migration_enabled = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1"
    _file_export = migration_file_export_enabled()

    # Mock integration (prefill_producer.py's PREFILL_PRODUCER_CHECK_PCC): publish the KV chunk table +
    # device map for an external device-less reader, with NO migration worker. Deliberately OUTSIDE the
    # _migration_enabled block below: that block's first step is
    # deliver_device_map_and_gather_stage_layouts(), which imports the _migration_client .so and joins a
    # cross-rank all-gather. Mock has neither a client nor peers, so routing it through there raises
    # ImportError(_migration_client) — which is exactly what happens if you only make the old in-block
    # `elif PREFILL_MOCK_MIGRATION` reachable. Both writes here are local (build table + serialize map).
    if _mock_migration and not _migration_enabled:
        _mock_table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        _mock_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        runtime.build_kv_chunk_table(kv_caches, path=_mock_table_path)
        # fabric_node -> ASIC unique_id, so the producer can resolve chips for read_dram_umd without
        # touching the ControlPlane. Stale rank-scoped siblings from a prior multi-rank run would
        # merge into the reader's map, so drop them first.
        remove_stale_device_map_sidecars(_mock_map_path)
        serialize_device_map(mesh_device, _mock_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {_mock_table_path}, device map -> {_mock_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    if _migration_enabled:
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
            KvCacheStage,
            allgather_kv_stage_layouts,
            deliver_device_map_and_gather_stage_layouts,
            export_device_map_file_and_gather_stage_layouts,
            migration_device_map_file_path,
            publish_serialized_table_and_wait_ready,
            rank_scoped_device_map_path,
        )

        # This rank's pipeline stage owns layers [first_layer_idx, first_layer_idx + num_my_layers).
        # The layer-aware merge gathers each rank's range so the table spans all stages; pass this
        # rank's range -- via the adapter's boundaries, so it is the split the MODEL was built with in
        # main(). Without them a cross-layer-reuse model (GLM-5.2 snaps 39/39 to 38/40) describes a
        # partition its KV cache does not hold, mismapping every layer of the second stage.
        first_layer_idx, num_my_layers = compute_layer_split(
            NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS)
        )[rank]
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

        # Same rationale for the JSON device-map sidecars: a leftover rank-scoped file from a run with
        # a different rank count would silently merge into this run's map. Must stay BEFORE the
        # all-gather barrier below — every rank writes its fresh map only after it.
        if not _file_export:
            remove_stale_device_map_sidecars(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            )

        # ALL RANKS join the stage-layout all-gather (collective barrier; rank 0 needs the merged
        # layout to build the table). Real migration also delivers this rank's local FNID->UMD map to
        # its co-located worker first; mock has no worker (the producer reads a serialized JSON map),
        # so it joins the gather directly and never imports the worker client extension.
        #
        # Ask the runtime to describe its migratable caches -- the engine must not introspect the opaque
        # KvCaches struct, whose shape is per-model. One stage per config of the model's table, since a
        # layout carries one cache's DRAM base and one layer-index space. A runtime predating
        # `kv_migration_stages` exposes only the single-cache base address.
        _multi_cache_runtime = hasattr(runtime, "kv_migration_stages")
        if _multi_cache_runtime:
            kv_stages = runtime.kv_migration_stages(kv_caches, first_layer_idx, num_my_layers)
        elif hasattr(runtime, "kv_migration_base_address"):
            kv_stages = [KvCacheStage(runtime.kv_migration_base_address(kv_caches), first_layer_idx, num_my_layers)]
        else:
            raise RuntimeError(
                f"migration enabled but runtime {type(runtime).__name__} implements neither "
                "kv_migration_stages nor kv_migration_base_address "
                "(see docs/ADDING_A_PREFILL_MODEL.md §2)."
            )
        _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
        if _mock_migration:
            stage_layouts = allgather_kv_stage_layouts(mesh_device, kv_stages, GLOBAL_MESH_SHAPE)
        elif _file_export:
            stage_layouts = export_device_map_file_and_gather_stage_layouts(
                mesh_device, kv_stages, GLOBAL_MESH_SHAPE, migration_device_map_file_path()
            )
        else:
            stage_layouts = deliver_device_map_and_gather_stage_layouts(mesh_device, kv_stages, GLOBAL_MESH_SHAPE, rank)

        # A runtime predating `kv_migration_stages` describes ONE cache and takes the singular
        # `stage_layout=` -- its single gathered layout. Keep calling it that way: its single-rank guard
        # counts stages in that layout, and handing it the outer per-cache list would count caches (always
        # 1) and silently stop rejecting multi-rank migration.
        _layout_kwarg = {"stage_layouts": stage_layouts} if _multi_cache_runtime else {"stage_layout": stage_layouts[0]}

        if _mock_migration:
            # Mock integration (prefill_producer.py): the SAME merged table the real publish builds, but
            # serialized to disk for a device-less producer to read back via
            # ttnn.experimental.disaggregation.import_from_protobuf_file — WITHOUT the migration_endpoint
            # worker (no MigrationLayerClient, no WORKER_READY). Checked before is_first_rank so rank 0
            # also takes this path instead of blocking on a worker that isn't running.
            #
            # EVERY rank serializes its OWN local fabric_node -> ASIC unique_id device map so each
            # co-located producer can resolve only its host's chips for read_dram_umd (the multi-rank
            # merged table carries every host's fnids; a producer merges the local maps and skips
            # layers owned by another host). Rank-scoped filename for num_ranks > 1 — co-located ranks
            # would otherwise overwrite each other at the shared host-local path; the table path MUST
            # be on shared storage (only rank 0 writes it, but every host's reader resolves the same
            # path) — enforced above for num_ranks > 1.
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            if is_first_rank:
                # RANK 0 builds the merged table spanning every gathered stage — identical to the real
                # publish call below, minus the worker handshake.
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    **_layout_kwarg,
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
                    **_layout_kwarg,
                )
                logger.info(f"[migration] merged KV chunk table -> {table_path} (file export; no worker handshake)")
            logger.info(f"[migration] rank {rank}: exported local device map -> {migration_device_map_file_path()}")
        else:
            # EVERY rank serializes its own local fabric_node -> ASIC unique_id map, exactly as the
            # mock path above does. The real path used to skip this, which silently disabled every
            # device-less reader downstream: migration_driver's destination verification
            # (--verify-migration, both dst-bytes and dst-golden) and prefill_producer's source-KV
            # PCC each resolve chips through this file, log "device map ... not found", and FAIL —
            # so a real migration run could never verify what it copied. Host-local by design;
            # rank-scoped filename for num_ranks > 1 so co-located ranks don't overwrite each other.
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            logger.info(f"[migration] rank {rank}: local device map -> {device_map_path}")

            if is_first_rank:
                # RANK 0: model runtime builds + serializes the merged table (spanning all gathered
                # stages), then publish the serialized path + block on WORKER_READY.
                table_path = runtime.build_kv_chunk_table(
                    kv_caches,
                    table_path,
                    first_layer_idx=first_layer_idx,
                    num_my_layers=num_my_layers,
                    **_layout_kwarg,
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
        # migration path merges stages (deliver_device_map_and_gather_stage_layouts), and that needs
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

    # D2H layer-ack backend: the device sends one per-layer ack record over a metadata-only D2H socket
    # (outbound_socket_service_sync in each block) and a LayerAckService reader thread derives a
    # globally-dense seq per record and pushes it into the router-owned ring. This feeds the SAME
    # LayerCompletionRouter the multi-rank host-callback path uses, so it is multi-host compatible: it
    # works on any rank count (single-rank => world_size=1 router, local ring + counter channel, no MPI).
    use_d2h = os.environ.get("PREFILL_LAYER_ACK_D2H", "0") == "1"
    # The router path covers: (a) any multi-rank run (each rank owns only a layer slice, so it can't
    # inject the scheduler channel directly and must route to the master), and (b) the D2H backend on
    # any rank count (its LayerAckService is a pure producer into the ring). The single-rank non-D2H
    # path keeps the original direct-inject wiring.
    use_router = (not single_rank) or (use_d2h and enable_layer_ack)

    if use_router:
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
        # Constructed BEFORE the ring source below: the router creates the ring, the source connects.
        router = LayerCompletionRouter(
            rank=rank,
            world_size=num_ranks,
            master_rank=master_rank,
            ring_shm_name=ring_shm_name,
            scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
            teardown_timeout_ms=30000,
        )
        if use_d2h:
            # Device-record source. The reader thread reconstructs (chunk, global-layer) from a per-rank
            # record counter — valid because each rank emits exactly its layer-slice count (num_my_layers)
            # of records per chunk, in layer order. d2h_service is threaded into the request loop so
            # prefill_chunk drives the device-side send on every rank.
            #
            # MUST pass the adapter's split boundaries: seq = chunk * NUM_LAYERS + first_layer_idx + k, so
            # first_layer_idx/num_my_layers have to be the split the MODEL was actually built with (main(),
            # which snaps the even split onto ADAPTER.layer_split_boundaries). For a dense model boundaries
            # are None and this equals the even split; for a DSA/cross-layer-reuse model (GLM) the snapped
            # split differs, and an unsnapped one here would hand the router overlapping/gapped seqs, which
            # its reorder buffer cannot sequence — it would stall head-of-line instead of failing loudly.
            first_layer_idx, num_my_layers = compute_layer_split(
                NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS)
            )[rank]
            d2h_service = ttnn.D2HStreamService(
                mesh_device,
                global_spec=None,
                fifo_size_bytes=LAYER_ACK_FIFO_SIZE_BYTES,
                worker_cores=SYNC_WORKER_CORES,
                metadata_size_bytes=METADATA_SIZE_BYTES,
            )
            layer_ack_service = ttnn.LayerAckService(
                d2h_service,
                ring_shm_name,
                source_rank=rank,
                num_layers=NUM_LAYERS,
                first_layer_idx=first_layer_idx,
                local_layers=num_my_layers,
            )
            if runtime.config.use_trace:
                # Traced: the ack op is recorded inside the capture, so register the service up front and
                # defer the reader — it starts after the capture's warm records are drained below, or the
                # scheduler would see a phantom chunk's acks and migrate a slot that was never filled.
                runtime.set_d2h_ack_service(d2h_service)
            else:
                layer_ack_service.start()  # connects to the router-owned ring (created above)
            source_desc = "D2H device records"
        else:
            # Host-callback source: the runtime fires on_layer_complete(layer_idx, request_id) per layer
            # and the sink pushes into the ring (no device D2H). seq stride is the GLOBAL layer total
            # (NUM_LAYERS), NOT this rank's slice; layer_idx arriving at the sink is already global; the
            # chunk index is bound per prefill() call as request_id (passed by _compute_and_send), so the
            # sink reads no shared mutable state.
            producer = LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
            runtime.set_layer_completion_sink(
                build_layer_completion_sink(
                    producer,
                    source_rank=rank,
                    num_layers=NUM_LAYERS,
                )
            )
            source_desc = "host on_layer_complete callback"
        logger.info(
            f"[migration] layer-completion routing up: rank={rank}/{num_ranks} master={master_rank} "
            f"ring={ring_shm_name} source={source_desc} "
            + (f"(owns scheduler channel {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
        )
    elif single_rank and enable_layer_ack:
        # Single-rank non-D2H direct path: the runtime owns + inject()s the scheduler counter channel
        # directly (on_layer_complete fires per layer inside the model). Works traced or untraced.
        _unlink_stale_shm(ack_shm_name)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")
    elif single_rank:
        logger.info("[migration] LayerAck channel disabled (set PREFILL_ENABLE_LAYER_ACK=1 to enable)")

    # Capture the trace (use_trace) after the D2D endpoints AND the per-layer completion wiring
    # (LayerAck channel / layer-completion sink) are set up, but before the request loop: the capture
    # must split at each completion point, and doing it here keeps the one-time cost out of the loop.
    # No-op if already captured. See TtPrefillRuntime.capture_trace().
    if getattr(runtime, "capture_trace", None) and runtime.config.use_trace:
        runtime.capture_trace(kv_caches)
        # D2H ack under trace: the capture's warm forward compiles the ack programs by running them, so
        # num_layers real records already sit in the FIFO. Drain them before the reader starts, or the
        # scheduler sees a phantom chunk's worth of layer acks and migrates a slot that was never filled.
        # The host-callback source has no reader to defer and writes no D2H records, so it skips this.
        if use_d2h and layer_ack_service is not None:
            n_warm = getattr(runtime, "warmup_ack_count", lambda: 0)()
            for _ in range(n_warm):
                d2h_service.read_metadata()
            if n_warm:
                logger.info(f"[migration] drained {n_warm} D2H warm-up ack records from the trace capture")
            layer_ack_service.start()

    logger.info(f"[pp rank {rank}] setup complete, entering request loop")

    try:
        run_request_loop(
            runtime,
            kv_caches,
            rank,
            num_ranks,
            hidden_size=d2d_activation_width,
            h2d_service=h2d_service,
            d2d_in=d2d_in,
            d2d_out=d2d_out,
            d2h_service=d2h_service,
        )
    finally:
        # Always tear down — the request loop can raise (e.g. the layer-completion sink's ring-full
        # spin timing out on a stalled router); without this, producer/router/ack segments + the
        # router listener thread leak, and a downstream peer blocked in D2D recv deadlocks the pipeline.
        # Release services while the mesh + command queues are still alive (their dtors free a command
        # queue and service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        # Stop the D2H LayerAckService reader thread first (D2H backend only): it reads the D2H
        # service's sockets and pushes into the router-owned ring, so it must be joined before the
        # D2H service is dropped AND before router.stop() drains the ring (so the last records land).
        # No-op under the host-callback / direct-inject backends (layer_ack_service stays None).
        if layer_ack_service is not None:
            layer_ack_service.stop()
            layer_ack_service = None
        h2d_service = d2d_in = d2d_out = d2h_service = None
        gc.collect()
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()  # joins the listener; the master's final ring-drain + inject happens HERE
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
