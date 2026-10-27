#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


import contextlib
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
from models.demos.common.prefill.runners.layer_completion_layout import layer_completion_layout
from models.demos.common.prefill.runners.layer_completion_ring_reader import scheduler_shm_name
from models.demos.common.prefill.runners.layer_completion_ring_writer import LayerCompletionRingWriter
from models.demos.common.prefill.runners.migration import (
    migration_file_export_enabled,
    migration_publish_enabled,
    migration_table_path,
    setup_kv_chunk_table,
)
from models.demos.common.prefill.runners.runner_utils import (
    MTP_PAD_TOKEN_ID,
    activation_global_spec,
    build_h2d_service,
    compute_layer_split,
)
from models.demos.common.prefill.runners.runner_utils import d2d_activation_rows as d2d_rows_for
from models.demos.common.prefill.runners.runner_utils import d2d_activation_width as d2d_width
from models.demos.common.prefill.runners.runner_utils import (
    indexer_indices_global_spec,
    make_h2d_spec,
    num_mtp_tokens,
    open_mesh_device,
)
from models.demos.deepseek_v3_d_p.tt.mla.utils import mtp_lookahead_positions
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import CapturedCall


def _apply_manifest_env():
    manifest_path = os.environ.get("PREFILL_MANIFEST")
    if not manifest_path:
        return

    with open(manifest_path) as mp:
        manifest = json.load(mp)

    def sd(key, val):
        if val is not None:
            os.environ.setdefault(key, str(val))

    for key, val in manifest.get("env", {}).items():
        sd(key, val)


_apply_manifest_env()

SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))
METADATA_SIZE_BYTES = 12

# Worker grid for the socket copies on both ends of a stage boundary (activation <-> backing tensor).
# Each worker copies its page slice with one NoC round trip per page, so the copy time divides by the
# core count; 64 cores leave a fraction of a millisecond per copy and more would only chase that
# remainder. The H2D token stream and the D2H acks stay on SYNC_WORKER_CORES: they move a page or less
# per transfer, so extra workers would have nothing to do.
D2D_WORKER_GRID_MAX = 8


def d2d_worker_cores(mesh_device) -> ttnn.CoreRange:
    grid = mesh_device.compute_with_storage_grid_size()
    return ttnn.CoreRange(
        ttnn.CoreCoord(0, 0),
        ttnn.CoreCoord(min(D2D_WORKER_GRID_MAX, grid.x) - 1, min(D2D_WORKER_GRID_MAX, grid.y) - 1),
    )


LAYER_ACK_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_LAYER_ACK_FIFO_BYTES", 4 * 1024))
# One metadata-only D2H socket page: the ack record padded to the PCIe alignment.
LAYER_ACK_RECORD_BYTES = 64

SHUTDOWN_METADATA_WORD = -1
WARMUP_METADATA_WORD = -2

H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()])

D2D_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_PP_D2D_FIFO_BYTES", 256))

ADAPTER = get_adapter(os.environ.get("PREFILL_MODEL", DEFAULT_MODEL))
MODEL_CFG = ADAPTER.model_config

D2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[
        ttnn.PlacementShard(2),
        ttnn.PlacementShard(3) if ADAPTER.pipeline_activation_emb_tp_sharded else ttnn.PlacementReplicate(),
    ]
)
INDEX_MAPPER_CONFIG = ttnn.MeshMapperConfig(placements=[ttnn.PlacementShard(2), ttnn.PlacementShard(3)])

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", MODEL_CFG.NUM_LAYERS))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 16))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
DFLASH_MODEL = os.environ.get("DFLASH_HF_MODEL") or ADAPTER.dflash_model_default
DFLASH_ENABLED = ADAPTER.supports_dflash and os.environ.get("PREFILL_DFLASH", "0") == "1" and bool(DFLASH_MODEL)

MTP_LEVELS = int(os.environ.get("PREFILL_MTP_LEVELS", 0)) if ADAPTER.supports_mtp else 0
assert MTP_LEVELS >= 0, f"PREFILL_MTP_LEVELS must be >= 0, got {MTP_LEVELS}"
# Decode runs this many MTP levels on ONE shared KV cache, so prefill computes only the first (MTP_LEVELS = 1)
# and the KV table publishes the rest as aliases of its rows, acked with it.
MTP_DECODE_LEVELS = int(os.environ.get("PREFILL_MTP_DECODE_LEVELS", MTP_LEVELS)) if ADAPTER.supports_mtp else 0
assert MTP_DECODE_LEVELS == MTP_LEVELS or (
    MTP_LEVELS == 1 and MTP_DECODE_LEVELS > 1
), f"PREFILL_MTP_DECODE_LEVELS={MTP_DECODE_LEVELS} needs PREFILL_MTP_LEVELS=1 or equal to it, got {MTP_LEVELS}"
NUM_MTP_TOKENS = num_mtp_tokens(MTP_LEVELS)
TOKEN_ID_BYTES = 4
_MTP_PAD_AS_INT32 = MTP_PAD_TOKEN_ID - (1 << 32)
CHUNK_METADATA_SIZE_BYTES = METADATA_SIZE_BYTES

D2D_METADATA_SIZE_BYTES = METADATA_SIZE_BYTES + (4 if MTP_LEVELS else 0)
TP_SHARD_KV = os.environ.get("PREFILL_TP_SHARD_KV", str(int(ADAPTER.tp_shard_kv_default))) == "1"
assert (
    not TP_SHARD_KV or ADAPTER.supports_tp_shard_kv
), f"PREFILL_TP_SHARD_KV=1: model {ADAPTER.name!r} does not support a TP-deduped KV cache"
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
CHUNK_TIMEOUT_SECONDS = float(os.environ.get("TT_PREFILL_CHUNK_TIMEOUT_SECONDS", 0))
TIMING_DIR = os.environ.get("PREFILL_TIMING_DIR", "")
# Env-overridable so re-bisecting does not need a rebuild. #54834's fix removed the AttnRes floor
# that used to make this a narrow band; what is left is MLA's chunked-attention ceiling.
_L1_SMALL_SIZE = int(os.environ.get("PREFILL_L1_SMALL_SIZE", ADAPTER.l1_small_size))
if MTP_LEVELS:
    _L1_SMALL_SIZE += 512
USE_TRACE = os.environ.get("PREFILL_USE_TRACE", "0") == "1"
# GLM KVPE cache: scaled FP8 unless PREFILL_SCALED_FP8_KV_CACHE=0 (bf16). Dense-MLA models get None.
SCALED_FP8_KV_CACHE = os.environ.get("PREFILL_SCALED_FP8_KV_CACHE", "1") == "1"


def _sparse_kv_cache_format():
    if ADAPTER.default_sparse_kv_cache_format is None:
        return None
    return MlaKvCacheFormat.SCALED_FP8 if SCALED_FP8_KV_CACHE else MlaKvCacheFormat.BF16_RM


_TRACE_REGION_SIZE = int(os.environ.get("PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024)) if USE_TRACE else 0

assert not (MTP_LEVELS and DFLASH_ENABLED), "PREFILL_MTP_LEVELS>0 and PREFILL_DFLASH=1 are mutually exclusive"

# DFlash runs traced, and only traced. The eager drafter path is no longer a supported configuration:
# the tap fires from inside the verifier forward, so trace capture is what the wiring is built and
# validated against, and an untraced drafter is a second path nothing gates. The manifests pin
# PREFILL_USE_TRACE=1; this catches a run that overrides it back to 0 rather than letting it start and
# diverge silently. To re-measure traced-vs-eager equivalence, drop this assert in a scratch tree --
# deliberately not an env escape hatch, so a production run cannot reach the eager path by accident.
assert not (DFLASH_ENABLED and not USE_TRACE and not ADAPTER.dflash_allows_eager), (
    "PREFILL_DFLASH=1 requires PREFILL_USE_TRACE=1: the DFlash drafter is only supported on the "
    "traced path. Unset PREFILL_USE_TRACE (the dflash manifest pins it to 1) or drop PREFILL_DFLASH."
)

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _decode_metadata(metadata_msg: ttnn.Tensor, with_provided_levels: bool = False) -> dict:
    # One D2H read; the 4th word is MTP provided_levels.
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_msg)[0]).view(torch.int32).flatten()
    meta = {
        "slot_id": int(m[0]),
        "actual_start": int(m[1]),
        "actual_end": int(m[2]),
    }
    # Mirror _metadata_record_words: the fourth word exists only when MTP is on.
    if with_provided_levels and m.numel() > 3:
        meta["provided_levels"] = max(0, int(m[3]))
    return meta


def mtp_provided_levels(mtp_tokens, meta: dict) -> int:
    if not MTP_LEVELS:
        return 0
    if meta["actual_end"] < meta["actual_start"] + CHUNK_SIZE:
        return 0
    assert mtp_tokens is not None, "MTP is on but no lookahead tensor arrived with this chunk"
    lookahead = mtp_lookahead_positions(meta["actual_start"], _sp, CHUNK_SIZE // _sp, meta["actual_end"], MTP_LEVELS)
    chips = [c for c, slots in enumerate(lookahead) if slots[0] == meta["actual_end"]]
    assert len(chips) == 1, f"expected one chip whose lookahead starts at {meta['actual_end']}, got {chips}"
    device_tensors = ttnn.get_device_tensors(mtp_tokens)
    assert len(device_tensors) == _sp * _tp, f"got {len(device_tensors)} device tensors for a {_sp}x{_tp} mesh"
    # H2D_MAPPER_CONFIG shards SP over mesh axis 0 and device tensors are row-major, so chip c is device c * _tp
    ids = ttnn.to_torch(device_tensors[chips[0] * _tp]).view(torch.int32).flatten()
    assert ids.numel() >= MTP_LEVELS, f"lookahead row is {ids.numel()} ids, need at least {MTP_LEVELS}"
    provided = 0
    for tok in ids[:MTP_LEVELS].tolist():
        if tok == _MTP_PAD_AS_INT32:
            break
        provided += 1
    return provided


def _is_shutdown_sentinel(meta: dict) -> bool:
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _is_warmup_sentinel(meta: dict) -> bool:
    return (
        meta["slot_id"] == WARMUP_METADATA_WORD
        and meta["actual_start"] == WARMUP_METADATA_WORD
        and meta["actual_end"] == WARMUP_METADATA_WORD
    )


def _make_socket_metadata_buffer(mesh_device, metadata_size_bytes: int) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.zeros(1, 1, 1, metadata_size_bytes // 4, dtype=torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _h2d_drain_op(h2d_service, sockets, n_mtp: int):
    return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service,
        metadata_size_bytes=CHUNK_METADATA_SIZE_BYTES,
        overhang_size_bytes=n_mtp * TOKEN_ID_BYTES,
        metadata_out=sockets.persistent_metadata(),
        tokens_out=sockets.tokens,
        overhang_out=sockets.overhang,
    )


def _socket_next(h2d_service, sockets, n_mtp: int = 0) -> tuple:
    if sockets.drain_trace is not None:
        sockets.drain_trace.replay()
        return sockets.tokens, sockets.overhang, _decode_metadata(sockets.metadata)
    outs = _h2d_drain_op(h2d_service, sockets, n_mtp)
    tokens, overhang = (outs[0], outs[1]) if n_mtp else (outs[0], None)
    sockets.metadata = outs[-1]
    return tokens, overhang, _decode_metadata(sockets.metadata)


def build_d2d_pipeline_endpoints(
    mesh_device,
    rank: int,
    num_ranks: int,
    d2d_rows: int,
    d2d_width: int,
    inbound_planes: int = 1,
    outbound_planes: int = 1,
):
    # Separate specs per direction: a model whose boundary payload grows with depth (Kimi-K3 carries
    # one AttnRes snapshot per completed block) sends more planes than it received. This rank's
    # outbound_planes must equal the next rank's inbound_planes or the rendezvous rejects the pair.
    workers = d2d_worker_cores(mesh_device)

    def _common(planes):
        return dict(
            global_spec=activation_global_spec(d2d_rows, d2d_width, planes),
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=workers,
            receiver_worker_cores=workers,
            metadata_size_bytes=D2D_METADATA_SIZE_BYTES,
            share_fabric_links=True,
            socket_buffer_type=ttnn.BufferType.L1,
        )

    inbound = None
    if rank > 0:
        logger.info(f"[pp rank {rank}] [d2d] creating inbound receiver from rank {rank - 1}")
        inbound = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common(inbound_planes)
        )
    outbound = None
    if rank < num_ranks - 1:
        logger.info(f"[pp rank {rank}] [d2d] creating outbound sender to rank {rank + 1}")
        outbound = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common(outbound_planes)
        )
    logger.info(
        f"[pp rank {rank}] [d2d] endpoints up (inbound={'yes' if inbound else 'no'}/{inbound_planes}p "
        f"outbound={'yes' if outbound else 'no'}/{outbound_planes}p, workers={workers}, "
        f"fifo={D2D_FIFO_SIZE_BYTES}B)"
    )
    return inbound, outbound


def build_d2d_index_endpoints(mesh_device, rank: int, per_chip_shape, *, inbound: bool, outbound: bool):
    def _common():
        return dict(
            global_spec=indexer_indices_global_spec(per_chip_shape, GLOBAL_MESH_SHAPE),
            mapper=ttnn.create_mesh_mapper(mesh_device, INDEX_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
            socket_buffer_type=ttnn.BufferType.L1,
        )

    idx_in = None
    if inbound:
        logger.info(f"[pp rank {rank}] [d2d-idx] creating inbound index receiver from rank {rank - 1}")
        idx_in = ttnn.D2DStreamService.create_receiver(
            receiver_mesh=mesh_device, sender_rank=rank - 1, receiver_rank=rank, **_common()
        )
    idx_out = None
    if outbound:
        logger.info(f"[pp rank {rank}] [d2d-idx] creating outbound index sender to rank {rank + 1}")
        idx_out = ttnn.D2DStreamService.create_sender(
            sender_mesh=mesh_device, sender_rank=rank, receiver_rank=rank + 1, **_common()
        )
    logger.info(
        f"[pp rank {rank}] [d2d-idx] endpoints up (inbound={'yes' if idx_in else 'no'} "
        f"outbound={'yes' if idx_out else 'no'}, per-chip indices {tuple(per_chip_shape)})"
    )
    return idx_in, idx_out


def _d2d_recv_indices(d2d_in, idx_in) -> ttnn.Tensor:
    if d2d_in is not None:
        d2d_in.wait_for_fabric_links()
    idx_in.release_fabric_links()
    t0 = time.perf_counter()
    indices, metadata_msg = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        idx_in, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    ttnn.deallocate(metadata_msg)
    logger.info(f"[pp] RECV-d2d-idx [xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms")
    return indices


def _d2d_send_indices(
    d2d_out, idx_out, indices: ttnn.Tensor, rank: int, meta: Optional[dict], metadata_msg=None
) -> None:
    d2d_out.wait_for_fabric_links()
    backing_cfg = idx_out.get_backing_tensor().memory_config()
    staged = indices if indices.memory_config() == backing_cfg else ttnn.to_memory_config(indices, backing_cfg)
    _d2d_send(idx_out, staged, rank, meta, deallocate=staged is not indices, metadata_msg=metadata_msg)


def _d2d_drain_op(inbound, sockets):
    return ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        inbound,
        metadata_size_bytes=D2D_METADATA_SIZE_BYTES,
        metadata_out=sockets.persistent_metadata(),
        tokens_out=sockets.tokens,
    )


def _d2d_recv(inbound, sockets) -> tuple:
    t0 = time.perf_counter()
    if sockets.drain_trace is not None:
        sockets.drain_trace.replay()
        act = sockets.tokens
    else:
        act, sockets.metadata = _d2d_drain_op(inbound, sockets)
    meta = _decode_metadata(sockets.metadata, with_provided_levels=bool(MTP_LEVELS))
    where = f"[{meta['actual_start']},{meta['actual_end']}) slot={meta['slot_id']}"
    logger.info(f"[pp] RECV-d2d {where} [xfer] sync={(time.perf_counter() - t0) * 1000.0:.2f}ms")
    return act, meta


def _metadata_record_words(meta: dict) -> torch.Tensor:
    words = [meta["slot_id"], meta["actual_start"], meta["actual_end"]]
    if MTP_LEVELS:
        words.append(int(meta["provided_levels"]))
    return torch.tensor(words, dtype=torch.int32).reshape(1, 1, 1, -1)


def _send_metadata_record(outbound, meta: dict) -> ttnn.Tensor:
    backing = outbound.get_backing_tensor()
    return ttnn.from_torch(
        _metadata_record_words(meta),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=backing.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.create_mesh_mapper(
            backing.device(),
            ttnn.MeshMapperConfig(placements=[ttnn.PlacementReplicate(), ttnn.PlacementReplicate()]),
        ),
    )


def _fill_send_record(record: ttnn.Tensor, meta: dict) -> None:
    src = ttnn.from_torch(
        _metadata_record_words(meta),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(record.device()),
    )
    ttnn.copy_host_to_device_tensor(src, record)


def _d2d_send(
    outbound,
    activation: ttnn.Tensor,
    rank: int,
    meta: Optional[dict],
    *,
    deallocate: bool = True,
    metadata_msg=None,
    trace=None,
) -> None:
    t0 = time.perf_counter()

    if trace is not None:
        trace.replay()
    else:
        md_tensor = metadata_msg if metadata_msg is not None else _send_metadata_record(outbound, meta)
        ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(outbound, activation, metadata=md_tensor)
    if deallocate:
        ttnn.deallocate(activation)
    where = f"[{meta['actual_start']},{meta['actual_end']})" if meta is not None else "(on-device)"
    logger.info(f"[pp rank {rank}] SEND-d2d {where} [xfer] push={(time.perf_counter() - t0) * 1000.0:.2f}ms")


def _forward_shutdown(
    d2d_out,
    rank: int,
    d2d_rows: int,
    d2d_width: int,
    planes: int = 1,
    idx_out=None,
    idx_shape: Optional[tuple] = None,
) -> None:
    # `planes` must match this rank's OUTBOUND spec, not its inbound one.
    dev = d2d_out.get_backing_tensor().device()
    dummy = ttnn.from_torch(
        torch.zeros(1, planes, d2d_rows, d2d_width),
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
        "provided_levels": 0,
    }
    _d2d_send(d2d_out, dummy, rank, sentinel)
    d2d_out.release_fabric_links()
    if idx_out is not None:
        _, _, rows, k = idx_shape
        sp, tp = GLOBAL_MESH_SHAPE
        idx_dummy = ttnn.from_torch(
            torch.zeros(1, 1, sp * rows, tp * k, dtype=torch.int64),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=dev,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(dev, INDEX_MAPPER_CONFIG),
        )
        d2d_out.wait_for_fabric_links()
        _d2d_send(idx_out, idx_dummy, rank, sentinel)
        idx_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] forwarded SHUTDOWN sentinel to rank {rank + 1}")


def _forward_send_warmup(runtime, d2d_out, rank: int, idx_out=None) -> None:
    """Compile the traced send before this rank's capture; the next rank drops the record."""
    get_inputs = getattr(runtime, "send_warmup_inputs", None)
    words = (WARMUP_METADATA_WORD,) * 3 + ((0,) if MTP_LEVELS else ())
    inputs = get_inputs(words) if get_inputs is not None else None
    if inputs is None:
        return
    activation, md_tensor = inputs
    ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(d2d_out, activation, metadata=md_tensor)
    d2d_out.release_fabric_links()
    d2d_out.wait_for_fabric_links()
    if idx_out is not None:
        _d2d_send_indices(d2d_out, idx_out, runtime.outbound_indexer_indices, rank, None, metadata_msg=md_tensor)
        idx_out.release_fabric_links()
        idx_out.wait_for_fabric_links()
    ttnn.deallocate(activation)
    ttnn.deallocate(md_tensor)
    logger.info(f"[pp rank {rank}] forwarded send warm-up record to rank {rank + 1}")


def _lease_reclaim(d2d_in, d2d_out, idx_in=None, idx_out=None) -> None:
    for service in (d2d_in, idx_in, d2d_out, idx_out):
        if service is not None:
            service.wait_for_fabric_links()
    if d2d_in is not None:
        d2d_in.release_fabric_links()


class SocketBuffers:
    def __init__(self, metadata, tokens=None, overhang=None, send_record=None):
        self.metadata = metadata
        # Only a buffer handed in here is ours to reuse; `metadata` is rebound per chunk otherwise.
        self._metadata_is_persistent = metadata is not None
        self.tokens = tokens
        self.overhang = overhang
        self.send_record = send_record
        self.drain_trace = None
        self.send_trace = None

    def persistent_metadata(self):
        """The metadata destination for the drain, or None when it must allocate its own: outside the
        claim protocol `metadata` holds the previous chunk's tensor, which Gemma4 frees after use."""
        return self.metadata if self._metadata_is_persistent else None

    def release_traces(self) -> None:
        for trace in (self.drain_trace, self.send_trace):
            if trace is not None:
                trace.release()
        self.drain_trace = self.send_trace = None


def _capture_socket_ops(
    sockets: SocketBuffers, mesh_device, runtime, rank: int, *, h2d_service, d2d_in, d2d_out
) -> None:
    is_first = runtime.config.is_first_rank
    inbound = h2d_service if is_first else d2d_in
    n_mtp = NUM_MTP_TOKENS if is_first else 0

    if inbound is None:
        drain_skip = "no inbound socket"
    elif sockets.tokens is None or sockets.persistent_metadata() is None:
        drain_skip = "the drain is missing a persistent destination"
    elif n_mtp and sockets.overhang is None:
        drain_skip = "the MTP lookahead overhang has no persistent destination"
    else:
        drain_skip = None
    if drain_skip is None:
        call = CapturedCall(mesh_device)
        if is_first:
            call.capture(lambda: _h2d_drain_op(inbound, sockets, n_mtp))
        else:
            call.capture(lambda: _d2d_drain_op(inbound, sockets))
        sockets.drain_trace = call

    activation = getattr(runtime, "trace_output", None)
    forward_md = (
        sockets.send_record if sockets.send_record is not None else getattr(runtime, "trace_metadata_msg", None)
    )
    if d2d_out is None:
        send_skip = "last rank, nothing to forward"
    elif activation is None or forward_md is None:
        send_skip = "the forward keeps no persistent output/metadata pair"
    elif MTP_LEVELS and is_first and sockets.send_record is None:
        send_skip = "MTP's first rank has no persistent outgoing record"
    else:
        send_skip = None
    if send_skip is None:
        call = CapturedCall(mesh_device)
        call.capture(
            lambda: ttnn.experimental.deepseek_prefill.outbound_socket_service_sync(
                d2d_out, activation, metadata=forward_md
            )
        )
        sockets.send_trace = call

    def _how(trace, skip):
        return "traced" if trace is not None else f"eager ({skip})"

    logger.info(
        f"[pp rank {rank}] socket ops: drain={_how(sockets.drain_trace, drain_skip)} "
        f"send={_how(sockets.send_trace, send_skip)}"
    )


def _record_chunk_timing(rank: int, c: int, compute_start: float, compute_ms: float) -> None:
    if not TIMING_DIR:
        return
    try:
        # O_CREAT makes the file, not the parent. Without this the open raises ENOENT, the except
        # below swallows it, and the run silently records nothing -- which is how a 5-point sweep
        # finished with five empty timing dirs and no Gantt input.
        os.makedirs(TIMING_DIR, exist_ok=True)
        fd = os.open(os.path.join(TIMING_DIR, f"rank{rank}.csv"), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, f"{rank},{c},{compute_start:.6f},{compute_ms:.3f}\n".encode())
        finally:
            os.close(fd)
    except OSError:
        pass


@contextlib.contextmanager
def _with_hang_timeout():
    if not CHUNK_TIMEOUT_SECONDS:
        yield
        return
    saved = ttnn.device.GetOperationTimeoutSeconds()
    ttnn.device.SetOperationTimeoutSeconds(CHUNK_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        ttnn.device.SetOperationTimeoutSeconds(saved)


def _compute_and_send(
    runtime,
    kv_caches,
    rank: int,
    c: int,
    inp,
    meta: Optional[dict],
    d2d_out,
    sockets: SocketBuffers,
    d2h_service=None,
    mtp_tokens=None,
    indices=None,
    idx_out=None,
) -> float:
    if SYNC_PER_CHUNK:
        ttnn.synchronize_device(runtime.mesh_device)
    t_start = time.time()
    t_perf = time.perf_counter()
    where = f"slot={meta['slot_id']} [{meta['actual_start']},{meta['actual_end']})"
    logger.info(
        f"[pp rank {rank}] CHUNK_START c={c} compute_start={t_start:.6f} {where} "
        f"provided={meta.get('provided_levels')}"
    )

    mtp_kwargs = {"mtp_tokens": mtp_tokens, "provided_levels": meta["provided_levels"]} if MTP_LEVELS else {}
    with _with_hang_timeout():
        out = runtime.prefill_chunk(
            inp,
            kv_caches,
            slot_id=meta["slot_id"],
            actual_start=meta["actual_start"],
            actual_end=meta["actual_end"],
            request_id=c,
            d2h_service=d2h_service,
            metadata_msg=sockets.metadata,
            **mtp_kwargs,
            **({"indexer_indices": indices} if indices is not None else {}),
        )
    if SYNC_PER_CHUNK:
        ttnn.synchronize_device(runtime.mesh_device)
        compute_ms = (time.perf_counter() - t_perf) * 1000.0
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={compute_ms:.3f}")
        _record_chunk_timing(rank, c, t_start, compute_ms)
    forward_md = None
    if not runtime.config.is_last_rank:
        if sockets.send_trace is None:
            if runtime.config.use_trace and not (MTP_LEVELS and runtime.config.is_first_rank):
                persistent_md = getattr(runtime, "trace_metadata_msg", None)
                forward_md = persistent_md if persistent_md is not None else sockets.metadata
        elif sockets.send_record is not None:
            _fill_send_record(sockets.send_record, meta)
        _d2d_send(
            d2d_out,
            out,
            rank,
            meta,
            deallocate=not runtime.config.use_trace,
            metadata_msg=forward_md,
            trace=sockets.send_trace,
        )
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    if idx_out is not None:
        _d2d_send_indices(d2d_out, idx_out, runtime.outbound_indexer_indices, rank, meta, metadata_msg=forward_md)
        idx_out.release_fabric_links()
    return t_start


def _drain_and_log_e2e(runtime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float, idx_out=None) -> None:
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    if idx_out is not None:
        idx_out.wait_for_fabric_links()
    ttnn.synchronize_device(runtime.mesh_device)
    fcs = f"{first_compute_start:.6f}" if first_compute_start is not None else "n/a"
    logger.info(f"[pp rank {rank}] E2E_CLOCK first_compute_start={fcs} last_compute_end={time.time():.6f}")
    logger.info(f"[pp rank {rank}] processed {n_done} chunks in {(time.perf_counter() - t0) * 1000.0:.2f} ms")


def run_request_loop(
    runtime,
    kv_caches,
    rank: int,
    num_ranks: int,
    *,
    d2d_rows: int,
    d2d_width: int,
    outbound_planes: int = 1,
    h2d_service=None,
    d2d_in=None,
    d2d_out=None,
    d2h_service=None,
    layer_ack_service=None,
    before_first_chunk=None,
    idx_in=None,
    idx_out=None,
    sockets=None,
) -> None:
    cfg = runtime.config
    if cfg.is_first_rank and h2d_service is None:
        raise ValueError("request mode requires the H2D service on the first rank for input")
    if sockets is None:
        raise ValueError("request mode requires the SocketBuffers holding this rank's socket destinations")
    logger.info(
        f"[pp rank {rank}/{num_ranks}] request (unbounded) loop start "
        f"(is_first={cfg.is_first_rank} is_last={cfg.is_last_rank} input={'h2d' if cfg.is_first_rank else 'd2d'})"
    )
    t0 = time.perf_counter()
    c = 0
    first = None
    while not _shutdown:
        _lease_reclaim(d2d_in, d2d_out, idx_in, idx_out)
        if cfg.is_first_rank:
            inp, mtp_tokens, meta = _socket_next(h2d_service, sockets, NUM_MTP_TOKENS)
            meta["provided_levels"] = 0 if _is_shutdown_sentinel(meta) else mtp_provided_levels(mtp_tokens, meta)
        else:
            inp, meta = _d2d_recv(d2d_in, sockets)
            mtp_tokens = None
        indices = _d2d_recv_indices(d2d_in, idx_in) if idx_in is not None else None
        if _is_shutdown_sentinel(meta):
            logger.info(f"[pp rank {rank}] SHUTDOWN sentinel received after {c} chunks; exiting request loop")
            if sockets.tokens is None:
                ttnn.deallocate(inp)
            # A sentinel chunk runs no prefill_chunk, so a drain-allocated record has no other owner.
            if sockets.persistent_metadata() is None:
                ttnn.deallocate(sockets.metadata)
            if mtp_tokens is not None and sockets.overhang is None:
                ttnn.deallocate(mtp_tokens)
            if indices is not None:
                ttnn.deallocate(indices)
            if d2d_out is not None:
                _forward_shutdown(
                    d2d_out,
                    rank,
                    d2d_rows,
                    d2d_width,
                    outbound_planes,
                    idx_out,
                    runtime.indexer_indices_shape() if idx_out is not None else None,
                )
            break
        warmup = _is_warmup_sentinel(meta)
        if warmup:
            if sockets.tokens is None:
                ttnn.deallocate(inp)
            if sockets.persistent_metadata() is None:
                ttnn.deallocate(sockets.metadata)
            if indices is not None:
                ttnn.deallocate(indices)
        if before_first_chunk is not None:
            before_first_chunk()
            before_first_chunk = None
        if warmup:
            continue
        t = _compute_and_send(
            runtime,
            kv_caches,
            rank,
            c,
            inp,
            meta,
            d2d_out,
            sockets,
            d2h_service=d2h_service,
            mtp_tokens=mtp_tokens,
            indices=indices,
            idx_out=idx_out,
        )
        if first is None:
            first = t
        c += 1
        if layer_ack_service is not None:
            layer_ack_service.check()  # a lost D2H record is fatal for the run; surface it here, not at exit
    _drain_and_log_e2e(runtime, rank, d2d_out, first, c, t0, idx_out)


def _print_config() -> None:
    rows = [
        ("PREFILL_MODEL", ADAPTER.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", ADAPTER.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)),
        ("resolved weight_cache_path", str(ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_PP_LAYER_COUNTS", os.environ.get("PREFILL_PP_LAYER_COUNTS", "<even split>")),
        (
            "DFLASH_ENABLED",
            f"{DFLASH_ENABLED} (adapter.supports_dflash={ADAPTER.supports_dflash}, "
            f"drafter={DFLASH_MODEL or '<unset>'})",
        ),
        ("PREFILL_MTP_LEVELS", f"{MTP_LEVELS} (adapter.supports_mtp={ADAPTER.supports_mtp})"),
        ("PREFILL_MTP_DECODE_LEVELS", str(MTP_DECODE_LEVELS)),
        ("PREFILL_USE_TRACE", f"{USE_TRACE} (trace_region={_TRACE_REGION_SIZE >> 20} MB)"),
        ("PREFILL_TP_SHARD_KV", str(TP_SHARD_KV)),
        ("PREFILL_LAYER_ACK_D2H", os.environ.get("PREFILL_LAYER_ACK_D2H", "0")),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_FABRIC_MODE", os.environ.get("PREFILL_FABRIC_MODE", "2d_torus_xy")),
        ("PREFILL_PP_D2D_FIFO_BYTES", str(D2D_FIFO_SIZE_BYTES)),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", ADAPTER.prefill_trace_default)),
        ("PREFILL_ENABLE_MIGRATION", str(int(migration_publish_enabled()))),
        ("PREFILL_MIGRATION_TABLE_PATH", migration_table_path()),
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
        "mtp_levels": MTP_LEVELS,
        "mtp_decode_levels": MTP_DECODE_LEVELS,
        "sparse_kv_cache_format": _sparse_kv_cache_format(),
        "PREFILL_ENABLE_MIGRATION": migration_publish_enabled(),
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

    if not ttnn.distributed_context_is_initialized():
        ttnn.init_distributed_context()
    rank = int(ttnn.distributed_context_get_rank())
    num_ranks = int(ttnn.distributed_context_get_size())
    _assert_ranks_agree_on_config(rank, num_ranks)

    layer_split = compute_layer_split(
        NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS), MTP_LEVELS + int(DFLASH_ENABLED)
    )
    first_layer_idx, num_my_layers = layer_split[rank]
    is_first_rank = rank == 0
    is_last_rank = rank == num_ranks - 1
    logger.info(
        f"[pp rank {rank}/{num_ranks}] mesh={GLOBAL_MESH_SHAPE} layers=[{first_layer_idx}, "
        f"{first_layer_idx + num_my_layers}) is_first={is_first_rank} is_last={is_last_rank} "
        f"chunk_size={CHUNK_SIZE} max_seq_len={MAX_SEQ_LEN} num_users={NUM_USERS}"
    )

    mesh_device = open_mesh_device(
        GLOBAL_MESH_SHAPE,
        ADAPTER.fabric_payload_size(),
        l1_small_size=_L1_SMALL_SIZE,
        trace_region_size=_TRACE_REGION_SIZE,
    )

    if CHUNK_TIMEOUT_SECONDS:
        ttnn.device.SetOperationTimeoutSeconds(0)

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
        num_links=2 if is_blackhole() else 1,
        gate_mode_name=_gate_mode_name,
        # is_last_rank, not True: TtPrefillTransformer asserts kv_only_last_layer -> is_last_rank, since a
        kv_only_last_layer=is_last_rank and not MTP_LEVELS,
        dflash_enabled=DFLASH_ENABLED,
        dflash_checkpoint_path=DFLASH_MODEL,
        mtp_levels=MTP_LEVELS,
        mtp_decode_levels=MTP_DECODE_LEVELS,
        weight_cache_path=ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE),
        tp_shard_kv=TP_SHARD_KV,
        sparse_kv_cache_format=_sparse_kv_cache_format(),
        use_trace=USE_TRACE,
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
    if USE_TRACE and getattr(runtime, "capture_trace", None) is None:
        raise RuntimeError(
            f"PREFILL_USE_TRACE=1 but runtime {type(runtime).__name__} does not implement "
            "capture_trace(kv_caches); run with PREFILL_USE_TRACE=0."
        )
    kv_caches = ADAPTER.allocate_kv_cache(mesh_device=mesh_device, hf_config=hf_config, params=params)
    runtime.compile(kv_caches)

    _serve_request(runtime, kv_caches, mesh_device, hf_config, rank, num_ranks, is_first_rank)

    _release_trace = getattr(runtime, "release_trace", None)
    if _release_trace is not None:
        _release_trace()

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info(f"[pp rank {rank}] shutdown complete")


def _serve_request(runtime, kv_caches, mesh_device, hf_config, rank: int, num_ranks: int, is_first_rank: bool) -> None:
    d2d_activation_rows = d2d_rows_for(CHUNK_SIZE, sp_factor=GLOBAL_MESH_SHAPE[0], mtp_levels=MTP_LEVELS)
    # The drafter partial rides beside the hidden, as a 2H width or as its own plane (the adapter's choice).
    dflash_plane = int(DFLASH_ENABLED and ADAPTER.dflash_partial_separate_plane)
    d2d_activation_width = d2d_width(hf_config.hidden_size, dflash=DFLASH_ENABLED and not dflash_plane)
    # Planes on dim 1 of the D2D payload, evaluated at BOTH edges of this rank's slice: what it
    # receives and what it sends on. Read off the runtime's own config rather than recomputing the
    # split, so the socket cannot be sized for a range the model was not built with.
    d2d_in_planes = ADAPTER.pipeline_activation_planes(runtime.config.first_layer_idx) + dflash_plane
    d2d_out_planes = (
        ADAPTER.pipeline_activation_planes(runtime.config.first_layer_idx + runtime.config.num_layers) + dflash_plane
    )

    ttnn.distributed_context_barrier()

    h2d_service = None
    if is_first_rank:
        mesh_device.clear_loaded_sub_device_manager()

        def _open_h2d(global_spec, name: str):
            service = build_h2d_service(
                mesh_device,
                global_spec=global_spec,
                mapper_config=H2D_MAPPER_CONFIG,
                worker_cores=SYNC_WORKER_CORES,
                metadata_size_bytes=CHUNK_METADATA_SIZE_BYTES,
            )
            path = service.export_descriptor(name)
            logger.info(
                f"[pp rank {rank}] [h2d] descriptor service_id={name!r} -> {path}; "
                f"drive it with prefill_producer.py / the scheduler."
            )
            return service

        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        h2d_service = _open_h2d(make_h2d_spec(GLOBAL_MESH_SHAPE, CHUNK_SIZE, MTP_LEVELS), service_id)

    d2d_in = d2d_out = idx_in = idx_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(
            mesh_device, rank, num_ranks, d2d_activation_rows, d2d_activation_width, d2d_in_planes, d2d_out_planes
        )
        idx_inbound = getattr(runtime, "receives_indexer_indices", False)
        idx_outbound = getattr(runtime, "forwards_indexer_indices", False)
        if idx_inbound or idx_outbound:
            idx_in, idx_out = build_d2d_index_endpoints(
                mesh_device, rank, runtime.indexer_indices_shape(), inbound=idx_inbound, outbound=idx_outbound
            )
        ttnn.distributed_context_barrier()

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = scheduler_shm_name(service_id)
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    router = None
    d2h_service = None
    layer_ack_service = None
    producer = None

    def _unlink_stale_shm(name: str) -> None:
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    # The layer-ack channel goes up before the migration table is published: the source-side dgen
    # worker attaches to it during its pipeline bring-up and only then issues the KV-manager connect
    # that the migration layer needs before it can report WORKER_READY to wait_ready() below.
    use_d2h = os.environ.get("PREFILL_LAYER_ACK_D2H", "0") == "1"
    if use_d2h and getattr(runtime, "set_d2h_ack_service", None) is None:
        raise RuntimeError(
            f"PREFILL_LAYER_ACK_D2H=1 but runtime {type(runtime).__name__} does not implement "
            "set_d2h_ack_service(service); it reports layer completion through set_layer_completion_sink "
            "only, so run with PREFILL_LAYER_ACK_D2H=0."
        )

    from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

    ring_base = os.environ.get("PREFILL_LAYER_COMPLETION_RING", "/tt_prefill_layer_completion_ring")
    ring_shm_name = f"{ring_base}_{rank}"
    _unlink_stale_shm(ring_shm_name)
    if rank == master_rank:
        _unlink_stale_shm(ack_shm_name)
    # Record layout (layer_completion_layout.py); the migration block below uses layer space.
    layer_split = compute_layer_split(
        NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS), MTP_LEVELS + int(DFLASH_ENABLED)
    )
    first_layer_idx, num_my_layers = layer_split[rank]
    extra_layers = runtime.layer_ack_layers(0, 0)[0] if getattr(runtime, "layer_ack_layers", None) else 0
    layout = layer_completion_layout(
        layer_split,
        rank,
        extra_layers=extra_layers,
    )
    router = LayerCompletionRouter(
        rank=rank,
        world_size=num_ranks,
        master_rank=master_rank,
        ring_shm_name=ring_shm_name,
        scheduler_shm_name=ack_shm_name if rank == master_rank else "",
        completions_per_chunk=layout.completions_per_chunk,
        teardown_timeout_ms=30000,
    )
    if use_d2h:
        fifo_size_bytes = LAYER_ACK_FIFO_SIZE_BYTES
        if runtime.config.use_trace:
            # The capture's warm pass fires a whole chunk's acks before any host reader exists. Acks
            # that arrive while the FIFO is full overrun the reader kernel's exact-count write-ack
            # wait, which then never matches again, so the FIFO must hold all of them.
            ack_fifo_bytes = layout.rank_record_count * LAYER_ACK_RECORD_BYTES
            fifo_size_bytes = max(fifo_size_bytes, 1 << (ack_fifo_bytes - 1).bit_length())
        d2h_service = ttnn.D2HStreamService(
            mesh_device,
            global_spec=None,
            fifo_size_bytes=fifo_size_bytes,
            worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=(CHUNK_METADATA_SIZE_BYTES if rank == 0 else D2D_METADATA_SIZE_BYTES),
            metadata_ring=True,
        )
        # The D2H record is the chunk's metadata {slot_id, actual_start, actual_end}; LayerAckService labels it
        # and detects a dropped record from it.
        layer_ack_service = ttnn.LayerAckService(
            d2h_service,
            ring_shm_name,
            first_record=layout.rank_first_record,
            record_count=layout.rank_record_count,
            layer_ids=layout.rank_layer_ids,
        )
        if runtime.config.use_trace:
            runtime.set_d2h_ack_service(d2h_service)
        else:
            layer_ack_service.start()
        source_desc = "D2H device records"
    else:
        # Host-callback transport: the only one MiniMax-M3 and GPT-OSS have.
        if getattr(runtime, "set_layer_completion_sink", None) is None:
            raise RuntimeError(
                f"runtime {type(runtime).__name__} does not implement set_layer_completion_sink(sink), "
                "which the layer-ack path requires at every rank count "
                "(see docs/ADDING_A_PREFILL_MODEL.md)."
            )
        producer = LayerCompletionQueue.connect(ring_shm_name, connect_timeout_ms=30000)
        sink = LayerCompletionRingWriter(producer, is_shutdown=lambda: _shutdown)
        runtime.set_layer_completion_sink(sink)
        source_desc = "host on_layer_complete callback"
    logger.info(
        f"[migration] layer-completion routing up: "
        f"rank={rank}/{num_ranks} master={master_rank} "
        f"ring={ring_shm_name} source={source_desc} "
        + (f"(owns scheduler shm {ack_shm_name})" if rank == master_rank else "(subordinate -> master)")
    )

    migration_endpoint = setup_kv_chunk_table(
        runtime,
        kv_caches,
        mesh_device,
        GLOBAL_MESH_SHAPE,
        rank=rank,
        num_ranks=num_ranks,
        first_layer_idx=first_layer_idx,
        num_my_layers=num_my_layers,
    )

    traced = bool(getattr(runtime, "capture_trace", None)) and runtime.config.use_trace

    _claim_input = getattr(runtime, "claim_persistent_input", None) if traced else None
    _claim_overhang = getattr(runtime, "claim_persistent_overhang", None) if traced and NUM_MTP_TOKENS else None
    tokens = _claim_input() if _claim_input is not None else None
    sockets = SocketBuffers(
        # Keyed on the runtime having the claim protocol, not on what it claimed: the DFlash non-first
        # rank claims no input and still replays, and a claiming runtime never frees the metadata
        # message, so a drain-allocated one would outlive the capture and be live across the replay.
        metadata=(
            _make_socket_metadata_buffer(
                mesh_device, CHUNK_METADATA_SIZE_BYTES if is_first_rank else D2D_METADATA_SIZE_BYTES
            )
            if _claim_input is not None
            else None
        ),
        tokens=tokens,
        overhang=_claim_overhang() if _claim_overhang is not None else None,
        send_record=(
            _make_socket_metadata_buffer(mesh_device, D2D_METADATA_SIZE_BYTES)
            if traced and MTP_LEVELS and is_first_rank and d2d_out is not None
            else None
        ),
    )
    logger.info(
        f"[pp rank {rank}] input destination = "
        + ("runtime traced input (persistent, no per-chunk copy)" if sockets.tokens is not None else "per-chunk alloc")
    )

    def _capture_trace() -> None:
        # Runs after the first record was drained into the persistent buffers, which is safe: capture_trace()
        # only reads _trace_input (via _forward_traced) and _forward_send_warmup works on its own
        # send_warmup_inputs() tensors. The writers are _prepare_trace(), back in compile(), and prefill_chunk().
        if d2d_out is not None:
            _forward_send_warmup(runtime, d2d_out, rank, idx_out)
        runtime.capture_trace(kv_caches)
        _capture_socket_ops(
            sockets, mesh_device, runtime, rank, h2d_service=h2d_service, d2d_in=d2d_in, d2d_out=d2d_out
        )
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
            d2d_rows=d2d_activation_rows,
            d2d_width=d2d_activation_width,
            outbound_planes=d2d_out_planes,
            h2d_service=h2d_service,
            d2d_in=d2d_in,
            d2d_out=d2d_out,
            d2h_service=d2h_service,
            layer_ack_service=layer_ack_service,
            before_first_chunk=_capture_trace if traced else None,
            idx_in=idx_in,
            idx_out=idx_out,
            sockets=sockets,
        )
    finally:
        import gc

        sockets.release_traces()
        if layer_ack_service is not None:
            layer_ack_service.stop()
            layer_ack_service = None
        h2d_service = d2d_in = d2d_out = idx_in = idx_out = d2h_service = None
        gc.collect()
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()


if __name__ == "__main__":
    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (OSError, ValueError) as e:
        logger.warning(f"[prefill] could not raise RLIMIT_NPROC to the hard limit: {e}")

    main()
