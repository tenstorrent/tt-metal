#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

"""Pipeline-parallel ("pipeline prefill") runner.

Splits one prefill model across N ranks under tt-run: each rank owns a contiguous
layer slice and builds the same TtPrefillRuntime the single-rank runner uses
(prefill_runner.py), parameterized with first_layer_idx / is_first_rank /
is_last_rank. The model class is the single source of truth — this driver only
wires rank topology and the per-chunk schedule; it does not reimplement embed /
layers / forward.

Cross-rank activation transport is a host round-trip stand-in until the D2D-socket
sync ops land: each rank gathers its output activation to host, hands it to the next
rank through a file under PREFILL_PP_DIR, and reshards it on the far side (bit-exact,
so a multi-rank run PCC-matches single-rank). PREFILL_PP_DIR is /dev/shm for single-host;
for multi-host set it to a directory on a filesystem shared by all hosts (e.g. NFS) —
/dev/shm is per-host and won't cross hosts. When the sync ops arrive, only
send_activation / recv_activation change.

A chunk has a strict rank0->...->rankN-1 dependency, so ranks run as a barrier-ordered
wavefront (one rank at a time, fully serialized). Combined with the host round-trip
this is a correctness/PCC bring-up path, not a perf path.

Scope: standalone (file/trace) input, no H2D socket service, no KV migration. Chunked
prefill does not sample; the populated per-rank KV cache is the output. With
PREFILL_STANDALONE_PCC=1 each rank checks its KV slice against the global golden trace.
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


# D2D socket transport (PREFILL_PP_TRANSPORT=d2d): one persistent sender/receiver pair per rank
# boundary carries the sharded hidden state over inter-galaxy fabric, replacing the host/NFS file
# round-trip. The activation is sharded [seq across SP rows, emb across TP cols] — the same layout the
# embedding output uses — so the receiver backing feeds the downstream model with no reshard. The
# per-chunk metadata (slot/start/end/is_last) rides inline as four uint32 words.
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
# Cross-rank activation transport
# ---------------------------------------------------------------------------
#
# Gather the activation to host, hand it to the next rank through a file under PREFILL_PP_DIR,
# reshard on the far side. Bit-exact (no precision loss beyond the activation's own dtype) so a
# multi-rank run PCC-matches single-rank. Fully serialized + host round-trip => very slow; a
# correctness/PCC bring-up, not a perf path. A stand-in until the D2D-socket sync ops land — when
# they do, only send_activation / recv_activation change.
#
# PREFILL_PP_DIR is the handoff directory:
#   - single host: /dev/shm (default) — per-host tmpfs, fast.
#   - multi host:  a directory on a filesystem SHARED by all hosts (e.g. NFS). /dev/shm
#                  is per-host and will NOT work across hosts.
# The write is staged to a temp file and atomically renamed so a reader (esp. over NFS,
# whose write visibility is close-to-open) never observes a partial file; the wavefront
# barrier orders the rename before the consumer's read.

_TORCH_TO_TTNN_DTYPE = {}  # populated lazily (torch import is local to keep module import cheap)


def _act_dir() -> str:
    return os.environ.get("PREFILL_PP_DIR", "/dev/shm")


def _act_path(subctx: int, chunk_idx: int, producer_rank: int) -> str:
    return os.path.join(_act_dir(), f"pp_act_sub{subctx}_c{chunk_idx}_r{producer_rank}.pt")


def _gather_activation_to_host(mesh_device: ttnn.MeshDevice, activation: ttnn.Tensor):
    """SP+TP-sharded [1, 1, seq_per_chip, emb/tp] -> full host [1, 1, seq, emb].

    Mirrors TtPrefillTransformer._to_host's composer (mesh rows=SP concat seq dim -2,
    mesh cols=TP concat emb dim -1) but keeps 4D so the far side reshards symmetrically.
    """
    return ttnn.to_torch(
        activation,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
    )


def _reshard_activation_to_device(mesh_device: ttnn.MeshDevice, host_tensor, mesh_shape: tuple) -> ttnn.Tensor:
    """Inverse of _gather_activation_to_host: full host [1, 1, seq, emb] -> SP+TP-sharded
    device tensor matching the embedding output (seq across SP rows, emb across TP cols)."""
    import torch

    if not _TORCH_TO_TTNN_DTYPE:
        _TORCH_TO_TTNN_DTYPE[torch.bfloat16] = ttnn.bfloat16
        _TORCH_TO_TTNN_DTYPE[torch.float32] = ttnn.float32
    return ttnn.from_torch(
        host_tensor,
        device=mesh_device,
        dtype=_TORCH_TO_TTNN_DTYPE[host_tensor.dtype],
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=(-2, -1)),
    )


def send_activation(
    runtime: TtPrefillRuntime, activation: ttnn.Tensor, subctx: int, chunk_idx: int, rank: int, meta: dict
) -> None:
    """Publish this rank's output activation AND its per-chunk metadata to the downstream rank as one
    payload, then free the device tensor. The atomic rename makes the file appear whole, so the
    consumer's polling recv_activation never sees a partial write — that pair is the only cross-rank
    sync (no barrier). Bundling activation+metadata mirrors what the D2D-socket op will carry in one
    transfer, so the downstream rank needs no shared schedule."""
    import torch

    t_to = time.perf_counter()
    host = _gather_activation_to_host(runtime.mesh_device, activation)  # to_torch: device -> host
    ms_to_torch = (time.perf_counter() - t_to) * 1000.0
    t_w = time.perf_counter()
    final = _act_path(subctx, chunk_idx, rank)
    tmp = f"{final}.tmp.{rank}"  # same dir -> rename is atomic on the target FS
    torch.save({"act": host, "meta": meta}, tmp)
    os.rename(tmp, final)
    ms_write = (time.perf_counter() - t_w) * 1000.0
    ttnn.deallocate(activation)
    nbytes = host.element_size() * host.nelement()
    logger.info(
        f"[pp rank {rank}] SEND chunk={chunk_idx} -> {final} "
        f"shape={tuple(host.shape)} dtype={host.dtype} bytes={nbytes} meta={meta} "
        f"[xfer] to_torch={ms_to_torch:.2f}ms write={ms_write:.2f}ms"
    )


def recv_activation(runtime: TtPrefillRuntime, subctx: int, chunk_idx: int, producer_rank: int) -> tuple:
    """Receive the upstream rank's activation + metadata for this chunk and reshard the activation
    onto this rank's mesh. Returns (device_tensor, meta_dict); runtime.prefill() deallocates the tensor.

    Polls for the file — the producer's atomic rename is the only ordering, no barrier — so block here
    until it appears: the open fails while the producer is still computing/renaming (and, cross-host
    over NFS, while a stale negative dentry lingers). Retry up to PREFILL_PP_RECV_RETRIES * 0.1s; size
    that above the producer's worst-case per-chunk compute time."""
    import torch

    path = _act_path(subctx, chunk_idx, producer_rank)
    fname = os.path.basename(path)
    act_dir = _act_dir()
    attempts = int(os.environ.get("PREFILL_PP_RECV_RETRIES", "3000"))
    # Wait (poll) for the producer's atomic-renamed file; time it separately from the transfer cost.
    # Poll with os.listdir, NOT os.path.exists: a failed name lookup is cached by NFS as a negative
    # dentry for up to acdirmax (~60s here), so a consumer that polls before the producer writes would
    # not see the file for ~60s. os.listdir issues a READDIR that revalidates the directory each poll.
    t_wait = time.perf_counter()
    waits = 0
    for attempt in range(attempts):
        if fname in os.listdir(act_dir):
            break
        if attempt == attempts - 1:
            raise FileNotFoundError(path)
        waits += 1  # producer hasn't published this chunk yet (still computing)
        time.sleep(0.1)
    ms_wait = (time.perf_counter() - t_wait) * 1000.0
    t_r = time.perf_counter()
    payload = torch.load(path)
    ms_read = (time.perf_counter() - t_r) * 1000.0
    os.unlink(path)
    host, meta = payload["act"], payload["meta"]
    t_from = time.perf_counter()
    out = _reshard_activation_to_device(runtime.mesh_device, host, runtime.config.mesh_shape)  # from_torch
    ms_from_torch = (time.perf_counter() - t_from) * 1000.0
    nbytes = host.element_size() * host.nelement()
    logger.info(
        f"[pp] RECV chunk={chunk_idx} <- {path} (from rank {producer_rank}) "
        f"shape={tuple(host.shape)} bytes={nbytes} meta={meta} waits={waits} wait={ms_wait:.1f}ms "
        f"[xfer] read={ms_read:.2f}ms from_torch={ms_from_torch:.2f}ms"
    )
    return out, meta


def _subcontext_id() -> int:
    """0 when not launched under an MPI sub-context — subcontext_id() returns None there."""
    sub_id = ttnn.distributed_context_subcontext_id()
    return int(sub_id) if sub_id is not None else 0


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


def _transport_mode() -> str:
    """Cross-rank activation transport: 'host' (file/NFS round-trip, default) or 'd2d' (device-to-device
    socket over inter-galaxy fabric). d2d needs a connected MGD and FABRIC_2D."""
    return os.environ.get("PREFILL_PP_TRANSPORT", "host").strip().lower()


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


def run_pipeline_loop(
    runtime: TtPrefillRuntime, rank: int, num_ranks: int, h2d_service=None, d2d_in=None, d2d_out=None
) -> None:
    """Decoupled per-rank loop: each rank runs its own loop, coordinated ONLY by the
    activation+metadata file handoff (atomic rename + polling recv) — no cross-rank barrier, so an
    upstream rank can race ahead of a downstream one (pipeline overlap). The first rank's input is
    either the trace file or the H2D socket (h2d_service set); the per-chunk metadata (slot /
    actual_start / actual_end / is_last) rides with each activation, so downstream ranks need no
    shared schedule and learn the end from is_last. Every rank checks the KV slice it populated."""
    cfg = runtime.config
    subctx = _subcontext_id()
    slot_id = int(os.environ.get("PREFILL_STANDALONE_SLOT", "0")) % cfg.num_users

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
        elif d2d_in is not None:
            inp, meta = _d2d_recv(d2d_in)
            slot_id = meta["slot_id"]
        else:
            inp, meta = recv_activation(runtime, subctx, c, rank - 1)
            slot_id = meta["slot_id"]

        # Absolute (epoch) start/end of this rank's compute. time.time() so the timestamps are
        # comparable ACROSS hosts (NTP-synced) — perf_counter is process-local and would not be. The
        # synchronize before reading t_end makes end reflect actual device completion, not just enqueue,
        # so the cross-rank handoff gap (next rank's start - this rank's end) is the real transport cost.
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
            if d2d_out is not None:
                _d2d_send(d2d_out, out, rank, meta)  # push + free; the grant below forwards it
            else:
                send_activation(runtime, out, subctx, c, rank, meta)  # out is None on the last rank

        # (4) Grant the outbound sender AFTER the push so it forwards this chunk's output downstream.
        if d2d_out is not None:
            d2d_out.release_fabric_links()

        if time_chunks:
            logger.info(
                f"[pp rank {rank}] PREFILL c={c} compute={ms_compute:.2f}ms "
                f"start={t_start_epoch:.6f} end={t_end_epoch:.6f} lease_reclaim={ms_lease:.2f}ms"
            )

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
        ("PREFILL_PP_TRANSPORT", _transport_mode()),
        ("PREFILL_PP_DIR (host transport)", _act_dir()),
        ("PREFILL_PP_RECV_RETRIES", os.environ.get("PREFILL_PP_RECV_RETRIES", "3000")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_PP_H2D", os.environ.get("PREFILL_PP_H2D", "<unset; 0>")),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_PCC", os.environ.get("PREFILL_STANDALONE_PCC", "0")),
        ("PREFILL_STANDALONE_SLOT", os.environ.get("PREFILL_STANDALONE_SLOT", "0")),
    ]
    sep = "=" * 70
    lines = [sep, "pipeline_prefill_runner configuration", sep]
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

    # Warm-up sync — the ONLY barrier. Every rank finishes compile before any chunk enters the
    # pipeline, so a downstream rank isn't still warming up while an upstream one races ahead: the
    # ranks actually overlap and activations don't pile up. The per-chunk loop takes no barrier, so
    # ranks still run independently once started. Trade-off: a rank that dies during compile hangs the
    # others here — acceptable for this POC.
    ttnn.distributed_context_barrier()

    # The activation-handoff dir must exist before the first send (send_activation does not mkdir).
    os.makedirs(_act_dir(), exist_ok=True)

    # First rank in H2D mode: stand up the socket service so an external producer
    # (prefill_h2d_producer.py) can push token chunks. Other ranks / file mode: no service.
    h2d_service = None
    if is_first_rank and _h2d_request_mode():
        # compile() leaves a custom sub-device manager loaded; the service's init program validates
        # its worker cores against the default whole-chip sub-device, so revert first.
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
            f"[pp rank {rank}] [h2d] service up, descriptor service_id={service_id!r} -> {descriptor_path}; "
            f"drive it with prefill_h2d_producer.py on this host."
        )

    # D2D socket transport: every rank stands up its pipeline endpoints. Like the H2D service, the
    # create programs validate worker cores against the default sub-device, so revert compile()'s
    # custom sub-device manager first. The post-compile barrier above guarantees all ranks reach the
    # chained create_sender/create_receiver rendezvous together.
    d2d_in = d2d_out = None
    if _transport_mode() == "d2d":
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, hf_config.hidden_size)

    logger.info(f"[pp rank {rank}] setup complete, entering decoupled pipeline loop")
    run_pipeline_loop(runtime, rank, num_ranks, h2d_service=h2d_service, d2d_in=d2d_in, d2d_out=d2d_out)

    if h2d_service is not None or d2d_in is not None or d2d_out is not None:
        # Free the services while the mesh + command queues are still alive (their dtors free a command
        # queue and the service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        h2d_service = None
        d2d_in = None
        d2d_out = None
        gc.collect()

    # No teardown barrier: ranks finish asynchronously (an upstream rank exits once it has produced
    # its last chunk, before a downstream rank consumes it), so each tears down independently.
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


if __name__ == "__main__":
    main()
