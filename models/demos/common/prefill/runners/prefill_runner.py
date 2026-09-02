#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.


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


LAYER_ACK_FIFO_SIZE_BYTES = int(os.environ.get("PREFILL_LAYER_ACK_FIFO_BYTES", 4 * 1024))

SHUTDOWN_METADATA_WORD = -1

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

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", CHUNK_SIZE * 11))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", ADAPTER.default_gate_mode)
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
DFLASH_ENABLED = (
    ADAPTER.supports_dflash and os.environ.get("PREFILL_DFLASH", "0") == "1" and bool(os.environ.get("DFLASH_HF_MODEL"))
)
SYNC_PER_CHUNK = os.environ.get("PREFILL_SYNC_PER_CHUNK", "0") == "1"
TIMING_DIR = os.environ.get("PREFILL_TIMING_DIR", "")
_L1_SMALL_SIZE = ADAPTER.l1_small_size
USE_TRACE = os.environ.get("PREFILL_USE_TRACE", "0") == "1"
_TRACE_REGION_SIZE = int(os.environ.get("PREFILL_TRACE_REGION_SIZE", 256 * 1024 * 1024)) if USE_TRACE else 0
assert not (DFLASH_ENABLED and USE_TRACE), (
    "PREFILL_DFLASH=1 is incompatible with PREFILL_USE_TRACE=1: the DFlash drafter path is not "
    "trace-captured. Run DFlash with PREFILL_USE_TRACE=0."
)

os.environ.setdefault("PREFILL_TTNN_CACHE", ADAPTER.ttnn_cache_default)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


LAYER_COMPLETION_PUSH_SPIN_TIMEOUT_S = float(os.environ.get("PREFILL_LAYER_COMPLETION_PUSH_TIMEOUT_S", 30.0))
LAYER_COMPLETION_PUSH_SPIN_LOG_EVERY_S = 10.0
LAYER_COMPLETION_PUSH_SPIN_SLEEP_S = 0.001


def build_layer_completion_sink(producer, *, source_rank, num_layers):
    def on_layer_complete(layer_idx: int, request_id: int) -> None:
        seq = request_id * num_layers + layer_idx
        if producer.try_push(seq=seq, source_rank=source_rank, layer_idx=layer_idx, request_id=request_id):
            return

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


def _decode_metadata(metadata_msg) -> dict:
    m = ttnn.to_torch(ttnn.get_device_tensors(metadata_msg)[0]).view(torch.int32).flatten()
    return {"slot_id": int(m[0]), "actual_start": int(m[1]), "actual_end": int(m[2])}


def _is_shutdown_sentinel(meta: dict) -> bool:
    return (
        meta["slot_id"] == SHUTDOWN_METADATA_WORD
        and meta["actual_start"] == SHUTDOWN_METADATA_WORD
        and meta["actual_end"] == SHUTDOWN_METADATA_WORD
    )


def _socket_next(h2d_service) -> tuple:
    tt_tokens, metadata_msg = ttnn.experimental.deepseek_prefill.inbound_socket_service_sync(
        h2d_service, metadata_size_bytes=METADATA_SIZE_BYTES
    )
    return tt_tokens, _decode_metadata(metadata_msg), metadata_msg


def build_d2d_pipeline_endpoints(mesh_device, rank: int, num_ranks: int, chunk_size: int, hidden_size: int):
    global_spec = activation_global_spec(chunk_size, hidden_size)

    def _common():
        return dict(
            global_spec=global_spec,
            mapper=ttnn.create_mesh_mapper(mesh_device, D2D_MAPPER_CONFIG),
            fifo_size_bytes=D2D_FIFO_SIZE_BYTES,
            sender_worker_cores=SYNC_WORKER_CORES,
            receiver_worker_cores=SYNC_WORKER_CORES,
            metadata_size_bytes=METADATA_SIZE_BYTES,
            share_fabric_links=True,
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
    t0 = time.perf_counter()

    if metadata_msg is not None:
        md_tensor = metadata_msg
    else:
        backing = outbound.get_backing_tensor()
        words = [meta["slot_id"], meta["actual_start"], meta["actual_end"]]
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
    _d2d_send(d2d_out, dummy, rank, sentinel)
    d2d_out.release_fabric_links()
    logger.info(f"[pp rank {rank}] forwarded SHUTDOWN sentinel to rank {rank + 1}")


def _lease_reclaim(d2d_in, d2d_out) -> None:
    if d2d_in is not None:
        d2d_in.wait_for_fabric_links()
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
    if d2d_in is not None:
        d2d_in.release_fabric_links()


def _record_chunk_timing(rank: int, c: int, compute_start: float, compute_ms: float) -> None:
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
        ttnn.synchronize_device(runtime.mesh_device)
        compute_ms = (time.perf_counter() - t_perf) * 1000.0
        logger.info(f"[pp rank {rank}] CHUNK_COMPUTE c={c} compute_ms={compute_ms:.3f}")
        _record_chunk_timing(rank, c, t_start, compute_ms)
    if not runtime.config.is_last_rank:
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
        )
    if d2d_out is not None:
        d2d_out.release_fabric_links()
    return t_start


def _drain_and_log_e2e(runtime, rank: int, d2d_out, first_compute_start, n_done: int, t0: float) -> None:
    if d2d_out is not None:
        d2d_out.wait_for_fabric_links()
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
    hidden_size: int,
    h2d_service=None,
    d2d_in=None,
    d2d_out=None,
    d2h_service=None,
) -> None:
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
            inp, meta, metadata_msg = _socket_next(h2d_service)
        else:
            inp, meta, metadata_msg = _d2d_recv(d2d_in)
        if _is_shutdown_sentinel(meta):
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
        num_links=2 if is_blackhole() else 1,
        gate_mode_name=_gate_mode_name,
        kv_only_last_layer=is_last_rank and KV_ONLY_LAST_LAYER,
        dflash_enabled=DFLASH_ENABLED,
        weight_cache_path=ADAPTER.weight_cache_path(GLOBAL_MESH_SHAPE),
        sparse_kv_cache_format=ADAPTER.default_sparse_kv_cache_format,
        use_trace=USE_TRACE,
        overlap_shared_expert_with_dispatch=os.environ.get("PREFILL_OVERLAP_SHARED_EXPERT", "1") == "1",
    )

    runtime = ADAPTER.build_runtime(mesh_device=mesh_device, hf_config=hf_config, params=params)
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
    single_rank = num_ranks == 1
    text_config = getattr(hf_config, "text_config", hf_config)
    d2d_activation_width = text_config.hidden_size * (2 if DFLASH_ENABLED else 1)

    ttnn.distributed_context_barrier()

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

    d2d_in = d2d_out = None
    if num_ranks > 1:
        mesh_device.clear_loaded_sub_device_manager()
        d2d_in, d2d_out = build_d2d_pipeline_endpoints(mesh_device, rank, num_ranks, CHUNK_SIZE, d2d_activation_width)
        ttnn.distributed_context_barrier()

    service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
    ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
    master_rank = int(os.environ.get("PREFILL_MASTER_RANK", "0"))
    ack_channel = None
    router = None
    d2h_service = None
    layer_ack_service = None
    producer = None
    enable_layer_ack = (
        os.environ.get("PREFILL_ENABLE_LAYER_ACK", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")) == "1"
    )

    def _unlink_stale_shm(name: str) -> None:
        path = f"/dev/shm/{name.lstrip('/')}"
        if os.path.exists(path):
            logger.warning(f"[migration] removing stale shm {path} from a prior run")
            os.remove(path)

    migration_endpoint = None
    _mock_migration = os.environ.get("PREFILL_MOCK_MIGRATION", "0") == "1"
    _migration_enabled = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1"
    _file_export = migration_file_export_enabled()

    if _mock_migration and not _migration_enabled:
        _mock_table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        _mock_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        runtime.build_kv_chunk_table(kv_caches, path=_mock_table_path)
        remove_stale_device_map_sidecars(_mock_map_path)
        serialize_device_map(mesh_device, _mock_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {_mock_table_path}, device map -> {_mock_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    if _migration_enabled:
        from models.demos.common.prefill.runners.migration import (
            KvCacheStage,
            allgather_kv_stage_layouts,
            deliver_device_map_and_gather_stage_layouts,
            export_device_map_file_and_gather_stage_layouts,
            migration_device_map_file_path,
            publish_serialized_table_and_wait_ready,
            rank_scoped_device_map_path,
        )

        first_layer_idx, num_my_layers = compute_layer_split(
            NUM_LAYERS, num_ranks, ADAPTER.layer_split_boundaries(NUM_LAYERS)
        )[rank]
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))

        if num_ranks > 1:
            _abs_table = os.path.abspath(table_path)
            if any(_abs_table == p or _abs_table.startswith(p + "/") for p in ("/tmp", "/dev/shm", "/run", "/var/tmp")):
                raise ValueError(
                    f"PREFILL_MIGRATION_TABLE_PATH={_abs_table} is on per-host storage; with num_ranks="
                    f"{num_ranks} the table rank 0 writes is invisible to the other hosts' readers. Point "
                    "it at shared/NFS storage (e.g. /data/...)."
                )

        if is_first_rank and os.path.exists(table_path):
            logger.warning(f"[migration] removing stale KV chunk table {table_path} from a prior run")
            os.remove(table_path)

        if not _file_export:
            remove_stale_device_map_sidecars(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
            )

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

        _layout_kwarg = {"stage_layouts": stage_layouts} if _multi_cache_runtime else {"stage_layout": stage_layouts[0]}

        if _mock_migration:
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            if is_first_rank:
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
            device_map_path = rank_scoped_device_map_path(
                os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json"),
                rank,
                num_ranks,
            )
            serialize_device_map(mesh_device, device_map_path)
            logger.info(f"[migration] rank {rank}: local device map -> {device_map_path}")

            if is_first_rank:
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
        if not single_rank:
            raise ValueError(
                f"PREFILL_MOCK_MIGRATION=1 is unsupported for num_ranks={num_ranks} (each rank would "
                "publish a table covering only its own layer slice; a merged mock table is not "
                "implemented); run single-rank or unset PREFILL_MOCK_MIGRATION."
            )
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        runtime.build_kv_chunk_table(kv_caches, path=table_path)
        device_map_path = os.environ.get("PREFILL_MIGRATION_DEVICE_MAP_PATH", "/tmp/prefill_kv_device_map.json")
        serialize_device_map(mesh_device, device_map_path)
        logger.info(
            f"[mock-migration] KV chunk table -> {table_path}, device map -> {device_map_path} "
            f"(no migration worker); prefill_producer can import them"
        )

    use_d2h = os.environ.get("PREFILL_LAYER_ACK_D2H", "0") == "1"
    use_router = (not single_rank) or (use_d2h and enable_layer_ack)

    if use_router:
        from ttnn._experimental.layer_completion import LayerCompletionQueue, LayerCompletionRouter

        ring_base = os.environ.get("PREFILL_LAYER_COMPLETION_RING", "/tt_prefill_layer_completion_ring")
        ring_shm_name = f"{ring_base}_{rank}"
        _unlink_stale_shm(ring_shm_name)
        if rank == master_rank:
            _unlink_stale_shm(ack_shm_name)
        router = LayerCompletionRouter(
            rank=rank,
            world_size=num_ranks,
            master_rank=master_rank,
            ring_shm_name=ring_shm_name,
            scheduler_channel_shm_name=ack_shm_name if rank == master_rank else "",
            teardown_timeout_ms=30000,
        )
        if use_d2h:
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
                runtime.set_d2h_ack_service(d2h_service)
            else:
                layer_ack_service.start()
            source_desc = "D2H device records"
        else:
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
        _unlink_stale_shm(ack_shm_name)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        runtime.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")
    elif single_rank:
        logger.info("[migration] LayerAck channel disabled (set PREFILL_ENABLE_LAYER_ACK=1 to enable)")

    if getattr(runtime, "capture_trace", None) and runtime.config.use_trace:
        runtime.capture_trace(kv_caches)
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
        import gc

        if layer_ack_service is not None:
            layer_ack_service.stop()
            layer_ack_service = None
        h2d_service = d2d_in = d2d_out = d2h_service = None
        gc.collect()
        if producer is not None:
            producer.shutdown()
        if router is not None:
            router.stop()
        if ack_channel is not None:
            ack_channel.shutdown()
            ack_channel = None


if __name__ == "__main__":
    try:
        import resource

        _, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
    except (OSError, ValueError) as e:
        logger.warning(f"[prefill] could not raise RLIMIT_NPROC to the hard limit: {e}")

    main()
