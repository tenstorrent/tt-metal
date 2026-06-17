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

VARIANT = get_variant(os.environ.get("PREFILL_MODEL_VARIANT", "deepseek_v3_d_p"))
MODEL_CFG = VARIANT.model_config

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 4 * CHUNK_SIZE))
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


def run_pipeline_loop(runtime: TtPrefillRuntime, rank: int, num_ranks: int, h2d_service=None) -> None:
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
    n_chunks = None
    if cfg.is_first_rank:
        # Only the first rank needs the schedule length; later ranks learn the end from is_last.
        if h2d_service is not None:
            # Socket mode: tokens arrive per push; the producer streams PREFILL_STANDALONE_NCHUNKS of them.
            n_chunks = int(os.environ["PREFILL_STANDALONE_NCHUNKS"])
        else:
            token_ids = _load_token_ids()
            actual_isl = len(token_ids)
            nchunks_env = os.environ.get("PREFILL_STANDALONE_NCHUNKS")
            n_chunks = int(nchunks_env) if nchunks_env else ((actual_isl + cfg.chunk_size - 1) // cfg.chunk_size)
            token_ids = (token_ids + [1] * (n_chunks * cfg.chunk_size))[: n_chunks * cfg.chunk_size]
        if n_chunks * cfg.chunk_size > cfg.max_seq_len:
            raise ValueError(
                f"{n_chunks} chunks x {cfg.chunk_size} exceeds per-user cache max_seq_len={cfg.max_seq_len}. "
                f"Lower PREFILL_STANDALONE_NCHUNKS or bump PREFILL_MAX_SEQ_LEN."
            )

    logger.info(
        f"[pp rank {rank}/{num_ranks}] decoupled loop start (is_first={cfg.is_first_rank} "
        f"is_last={cfg.is_last_rank} slot={slot_id} input={'h2d' if h2d_service is not None else 'trace'}"
        + (f" chunks={n_chunks}" if cfg.is_first_rank else "")
        + ")"
    )

    t0 = time.perf_counter()
    c = 0
    n_done = 0
    while not _shutdown:
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
            inp, meta = recv_activation(runtime, subctx, c, rank - 1)
            slot_id = meta["slot_id"]

        out = runtime.prefill(
            inp, slot_id=meta["slot_id"], actual_start=meta["actual_start"], actual_end=meta["actual_end"]
        )
        if not cfg.is_last_rank:
            send_activation(runtime, out, subctx, c, rank, meta)  # out is None on the last rank

        n_done += 1
        c += 1
        if meta["is_last"]:
            break

    ttnn.synchronize_device(runtime.mesh_device)
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


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

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

    logger.info(f"[pp rank {rank}] setup complete, entering decoupled pipeline loop")
    run_pipeline_loop(runtime, rank, num_ranks, h2d_service=h2d_service)

    if h2d_service is not None:
        # Free the service while the mesh + command queues are still alive (its dtor frees a command
        # queue and the service-core L1; running after close_mesh_device aborts with cq_id-out-of-range).
        import gc

        del h2d_service
        gc.collect()

    # No teardown barrier: ranks finish asynchronously (an upstream rank exits once it has produced
    # its last chunk, before a downstream rank consumes it), so each tears down independently.
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


if __name__ == "__main__":
    main()
