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
sync ops land (PREFILL_PP_TRANSPORT, see the transport section): "host" gathers each
rank's output activation to host, hands it to the next rank through a file under
PREFILL_PP_DIR, and reshards it on the far side (bit-exact, so a 4-rank run
PCC-matches single-rank); "placeholder" feeds zeros (compile/timing smoke only).
PREFILL_PP_DIR is /dev/shm for single-host; for multi-host set it to a directory on a
filesystem shared by all hosts (e.g. NFS) — /dev/shm is per-host and won't cross hosts.
When the sync ops arrive, only send_activation / recv_activation change.

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
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    get_variant,
    load_hf_config,
    load_trace_token_ids,
    open_mesh_device,
    prepare_prefill_input_tensor,
    resolve_trace_dir,
    resolve_weight_cache_path,
)
from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig

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
# Two modes (PREFILL_PP_TRANSPORT):
#   "host"        — gather the activation to host, hand it to the next rank through a file
#                   under PREFILL_PP_DIR, reshard on the far side. Bit-exact (no precision
#                   loss beyond the activation's own dtype) so a 4-rank run PCC-matches
#                   single-rank. Fully serialized + host round-trip => very slow; a
#                   correctness/PCC bring-up, not a perf path.
#   "placeholder" — non-first ranks consume zeros; cross-rank output is meaningless.
#                   Compile/timing smoke test only.
#
# Both are stand-ins until the D2D-socket sync ops land; "host" is the one that lets us
# validate the rank-sliced model now. The transport touches only these helpers.
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


def send_activation(runtime: TtPrefillRuntime, activation: ttnn.Tensor, subctx: int, chunk_idx: int, rank: int) -> None:
    """Publish this rank's output activation to the downstream rank, then free the device tensor.
    Atomic-rename so the consumer (single shared FS / NFS) never reads a partial file; the wavefront
    barrier orders this rename before the consumer's read."""
    import torch

    if _transport_mode() == "placeholder":
        ttnn.deallocate(activation)
        return
    host = _gather_activation_to_host(runtime.mesh_device, activation)
    final = _act_path(subctx, chunk_idx, rank)
    tmp = f"{final}.tmp.{rank}"  # same dir -> rename is atomic on the target FS
    torch.save(host, tmp)
    os.rename(tmp, final)
    ttnn.deallocate(activation)


def recv_activation(runtime: TtPrefillRuntime, subctx: int, chunk_idx: int, producer_rank: int) -> ttnn.Tensor:
    """Receive the upstream rank's activation for this chunk and reshard it onto this rank's mesh.
    In placeholder mode, returns zeros instead. runtime.prefill() deallocates the returned tensor.

    The producer's rename is barrier-ordered before this read, but NFS can briefly serve a stale
    negative dentry, so retry the open a bounded number of times before giving up."""
    import time as _time

    import torch

    if _transport_mode() == "placeholder":
        return runtime.make_placeholder_activation()
    path = _act_path(subctx, chunk_idx, producer_rank)
    attempts = int(os.environ.get("PREFILL_PP_RECV_RETRIES", "50"))
    for attempt in range(attempts):
        try:
            host = torch.load(path)
            break
        except FileNotFoundError:
            if attempt == attempts - 1:
                raise
            _time.sleep(0.1)  # absorb NFS negative-dentry cache; barrier already ordered the write
    os.unlink(path)
    return _reshard_activation_to_device(runtime.mesh_device, host, runtime.config.mesh_shape)


def _transport_mode() -> str:
    mode = os.environ.get("PREFILL_PP_TRANSPORT", "host")
    if mode not in ("host", "placeholder"):
        raise ValueError(f"PREFILL_PP_TRANSPORT={mode!r} must be 'host' or 'placeholder'")
    return mode


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


def run_standalone_loop(runtime: TtPrefillRuntime, rank: int, num_ranks: int) -> None:
    cfg = runtime.config
    subctx = int(ttnn.distributed_context_subcontext_id())
    token_ids = _load_token_ids()
    actual_isl = len(token_ids)
    slot_id = int(os.environ.get("PREFILL_STANDALONE_SLOT", "0")) % cfg.num_users

    nchunks_env = os.environ.get("PREFILL_STANDALONE_NCHUNKS")
    n_chunks = int(nchunks_env) if nchunks_env else ((actual_isl + cfg.chunk_size - 1) // cfg.chunk_size)
    total_len = n_chunks * cfg.chunk_size
    if total_len > cfg.max_seq_len:
        raise ValueError(
            f"{n_chunks} chunks x {cfg.chunk_size} = {total_len} exceeds per-user cache "
            f"max_seq_len={cfg.max_seq_len}. Lower PREFILL_STANDALONE_NCHUNKS or bump PREFILL_MAX_SEQ_LEN."
        )
    token_ids = (token_ids + [1] * total_len)[:total_len]

    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))
    logger.info(
        f"[pp rank {rank}] transport={_transport_mode()} actual_isl={actual_isl} slot={slot_id} "
        f"chunks={n_chunks} iters={num_iterations}"
    )

    iter_times_ms = []
    for it in range(num_iterations):
        if _shutdown:
            logger.info(f"[pp rank {rank}] shutdown requested, breaking loop")
            break
        t0 = time.perf_counter()
        for c in range(n_chunks):
            kv_actual = c * cfg.chunk_size
            # Barrier-ordered wavefront: a chunk has a strict rank0->rank1->...->rankN-1 data
            # dependency, so ranks run one at a time. The barrier after each stage guarantees the
            # producer's activation-file write lands before the next rank reads it. Fully serialized
            # — a PCC/bring-up path, not a perf path.
            for stage in range(num_ranks):
                if rank == stage:
                    if cfg.is_first_rank:
                        inp = _first_rank_chunk_tokens(runtime, token_ids, kv_actual)
                    else:
                        inp = recv_activation(runtime, subctx, c, rank - 1)
                    out = runtime.prefill(
                        inp,
                        slot_id=slot_id,
                        actual_start=kv_actual,
                        actual_end=kv_actual + cfg.chunk_size,  # standalone drives full chunks
                    )
                    if not cfg.is_last_rank:
                        send_activation(runtime, out, subctx, c, rank)
                ttnn.distributed_context_barrier()
        ttnn.synchronize_device(runtime.mesh_device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        iter_times_ms.append(dt_ms)
        logger.info(
            f"[pp rank {rank}] [prefill timing] iter={it} num_tokens={total_len} "
            f"chunks={n_chunks} slot={slot_id} wavefront = {dt_ms:.2f} ms"
        )
    logger.info(f"[pp rank {rank}] [iter timing summary] per-iter ms = {[round(t, 2) for t in iter_times_ms]}")

    if os.environ.get("PREFILL_STANDALONE_PCC", "0") == "1":
        # Each rank PCC-checks its own KV slice against the global golden trace (offset by
        # first_layer_idx). All ranks passing == the rank-sliced model + host transport reproduce
        # single-rank KV. Placeholder transport will fail this for non-first ranks, as intended.
        from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import kv_cache_pcc_check

        trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
        kv_cache_pcc_check(
            runtime,
            slot_id=slot_id,
            n_chunks=n_chunks,
            trace_dir=trace_dir,
            first_layer_idx=cfg.first_layer_idx,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    # tt-run pre-initializes the distributed context; bind the Python handle.
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
    ttnn.distributed_context_barrier()

    # The activation-handoff dir must exist before the first send (send_activation does not mkdir).
    if _transport_mode() == "host":
        os.makedirs(_act_dir(), exist_ok=True)

    logger.info(f"[pp rank {rank}] setup complete, entering standalone loop")
    run_standalone_loop(runtime, rank, num_ranks)

    ttnn.distributed_context_barrier()
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


if __name__ == "__main__":
    main()
