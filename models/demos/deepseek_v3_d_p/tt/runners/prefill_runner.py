#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import signal

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
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
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)

# Sync-op worker core. Single core suffices: the kernel only copies the
# backing tensor's pages into a fresh output, no per-core parallelism needed.
H2D_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))

# Inline metadata payload — packed by the producer per-iter, surfaced by the
# kernel via h2d_socket_sync's optional metadata output. Matches the prefill
# scheduler's PrefillMetadata wire struct (12 bytes, 3 × uint32):
#   [0] slot_id        — which per-slot KV-cache buffer to write
#   [1] actual_start   — inclusive absolute KV pos of the first real token
#   [2] actual_end     — exclusive absolute KV pos past the last real token
# Trailing positions in the chunk past `actual_end` are PAD_ID. See
# include/tt_llm_engine/scheduler/prefill/prefill_metadata.hpp for the
# source-of-truth definition the scheduler builds.
H2D_METADATA_SIZE_BYTES = 12

# Per-iter mesh distribution for the token input. Used by the H2D service's
# internal mapper; the producer process builds an equivalent mapper from
# MeshShape on its side.
# `Shard(0)` shards the leading axis across mesh rows (SP); `Replicate()`
# duplicates across mesh cols (TP).
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)


VARIANT = get_variant(os.environ.get("PREFILL_MODEL_VARIANT", "deepseek_v3_d_p"))
MODEL_CFG = VARIANT.model_config

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
# Chunked prefill: tokens are streamed in CHUNK_SIZE chunks; the cache holds NUM_USERS independent
# per-user slots. MAX_SEQ_LEN is the per-user cache length: a multiple of CHUNK_SIZE and strictly
# greater than it (ring_joint_sdpa requires Q.seq < K.seq, so a single-chunk cache is invalid).
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 4 * CHUNK_SIZE))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
# Model-agnostic gate-mode default comes from the active variant (DEVICE_FP32 for the
# deepseek_v3_d_p variant; HOST_ALL source default is much slower).
GATE_FALLBACK_MODE = os.environ.get("PREFILL_GATE_FALLBACK_MODE", VARIANT.default_gate_mode)
# When on (default), the last transformer layer runs kv-only: it fills the KV
# cache for migration and skips its Q/SDPA/wo, FFN/MoE, final norm, and LM head.
KV_ONLY_LAST_LAYER = os.environ.get("PREFILL_KV_ONLY_LAST_LAYER", "1") == "1"
PREFILL_DEBUG = os.environ.get("PREFILL_DEBUG", "0") == "1"
# Kimi (single expert group, device gate) routes the MoE routing all-gather's global semaphores to
# L1_SMALL so they don't pin the main-L1 floor and clash with the next layer's MLA static CBs. This
# requires opening the mesh with an L1_SMALL region. Off for DeepSeek (semaphores stay in main L1).
_ROUTING_USE_L1_SMALL_SEMAPHORES = VARIANT.name == "kimi_k2_6"
_L1_SMALL_SIZE = 512 if _ROUTING_USE_L1_SMALL_SEMAPHORES else 0

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


os.environ.setdefault("PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)


def run_standalone_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Standalone chunked prefill: read the input token IDs from this variant's golden trace and drive
    them through the pipeline in chunk_size chunks (advancing kv_actual per chunk), filling slot_id's
    KV cache. No H2D socket, no SHM, no external producer — single process, for local bring-up / perf.
    Chunked prefill does not sample; the populated KV cache is the output. With PREFILL_STANDALONE_PCC
    the same trace supplies the golden kv_post_transform for validation.

    The pipeline runs the kv-only last layer (no first_token, no LM head); the runner just logs
    per-iter timing.

    Env:
      PREFILL_TRACE_DIR golden trace dir (default: this variant's prefill_trace_default)
      PREFILL_STANDALONE_INPUT   JSON file {"token_ids": [...]} overriding the trace's default input
      PREFILL_STANDALONE_SLOT    cache user slot to fill (default 0)
      PREFILL_STANDALONE_NCHUNKS chunks to run (default: ceil(len(token_ids)/chunk_size); pads the tail)
      PREFILL_STANDALONE_ITERS   repeat count for perf timing (default 1)
      PREFILL_STANDALONE_PCC     "1" -> PCC-check the KV cache vs the trace's golden (kv_cache_pcc_check)
    """
    import json
    import time as _time

    cfg = pipeline.config
    chunk_size = cfg.chunk_size
    trace_dir = resolve_trace_dir(os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default))
    # PREFILL_STANDALONE_INPUT overrides the trace's default token_ids with a user-supplied JSON file
    # ({"token_ids": [...]}); the trace is still used for the optional golden PCC check below.
    input_override = os.environ.get("PREFILL_STANDALONE_INPUT")
    if input_override:
        with open(input_override) as f:
            token_ids = list(json.load(f)["token_ids"])
        logger.info(f"[standalone] input override: {len(token_ids)} token_ids from {input_override}")
    else:
        logger.info(f"[standalone] reading input token_ids from {trace_dir}/metadata.json")
        token_ids = load_trace_token_ids(trace_dir)
    task_id = 0
    actual_isl = len(token_ids)
    slot_id = int(os.environ.get("PREFILL_STANDALONE_SLOT", "0")) % cfg.num_users

    # Chunk count: explicit env override, else round the prompt up to whole chunks. Pad/trim the
    # token list to exactly n_chunks * chunk_size (chunked prefill requires full chunks).
    nchunks_env = os.environ.get("PREFILL_STANDALONE_NCHUNKS")
    n_chunks = int(nchunks_env) if nchunks_env else ((actual_isl + chunk_size - 1) // chunk_size)
    total_len = n_chunks * chunk_size
    if total_len > cfg.max_seq_len:
        raise ValueError(
            f"task_id={task_id}: {n_chunks} chunks x {chunk_size} = {total_len} exceeds per-user "
            f"cache max_seq_len={cfg.max_seq_len}. Lower PREFILL_STANDALONE_NCHUNKS or bump PREFILL_MAX_SEQ_LEN."
        )
    token_ids = (token_ids + [1] * total_len)[:total_len]

    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))
    logger.info(f"[standalone] task_id={task_id} actual_isl={actual_isl} slot={slot_id} chunks={n_chunks}")
    iter_times_ms = []
    for it in range(num_iterations):
        _t0 = _time.perf_counter()
        # prefill() is single-chunk; drive the chunk loop here, advancing kv_actual per chunk. For
        # chunk-aligned offsets the block-cyclic layout degenerates to a plain per-chip reshape, so
        # prepare_prefill_input_tensor (is_balanced=False) produces the correct chip-major chunk.
        for c in range(n_chunks):
            kv_actual = c * chunk_size
            pipeline.prefill(
                prepare_prefill_input_tensor(
                    token_ids[kv_actual : kv_actual + chunk_size],
                    pipeline.mesh_device,
                    cfg.sp_factor,
                    False,  # chunked prefill is block-cyclic (non-balanced)
                    cfg.mesh_shape,
                    cfg.sp_axis,
                ),
                slot_id=slot_id,
                actual_start=kv_actual,
                actual_end=kv_actual + chunk_size,  # standalone drives full chunks (all positions real)
            )
        ttnn.synchronize_device(pipeline.mesh_device)
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        iter_times_ms.append(_dt_ms)
        logger.info(
            f"[prefill timing] task_id={task_id} iter={it} num_tokens={total_len} "
            f"chunks={n_chunks} slot={slot_id} prefill = {_dt_ms:.2f} ms"
        )
    logger.info(f"[iter timing summary] per-iter ms = {[round(t,2) for t in iter_times_ms]}")
    # stdout, not a log line: callers (tests / orchestrators) parse this. Chunked prefill fills the
    # KV cache and does not sample, so there is no first token to report.
    print(f"[standalone] prefill_complete task_id={task_id} slot={slot_id} isl={actual_isl}")

    if os.environ.get("PREFILL_STANDALONE_PCC", "0") == "1":
        from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import kv_cache_pcc_check

        kv_cache_pcc_check(pipeline, slot_id=slot_id, n_chunks=n_chunks, trace_dir=trace_dir)


def run_request_loop(pipeline: TtDeepSeekPrefillPipeline, h2d_service: ttnn.H2DStreamService) -> None:
    """Request loop: token IDs + per-iter control metadata arrive over the H2D
    socket service, pushed by a separate producer process (prefill_h2d_producer.py
    today; the inference-server / prefill scheduler in production).

    No SHM: the c2p token-input channel is replaced by the H2D socket service,
    and the p2c token write-back is removed (downstream consumption is via
    migration / layer-acks, not a host token hand-back).

    `h2d_socket_sync` returns (tokens, metadata); we decode the 3×uint32
    PrefillMetadata [slot_id, actual_start, actual_end] and pass them straight to
    `pipeline.prefill` (actual_start is the chunk's cache write offset; actual_end - actual_start is
    its real-token count). The loop is push-driven: it has no notion of how many chunks a request
    spans — the producer/scheduler decides that.
    """
    import time as _time

    logger.info(
        "[request] entering request loop — blocks on h2d_socket_sync for each push, "
        "runs until SIGTERM/SIGINT (Ctrl-C). Drive it with prefill_h2d_producer.py / the scheduler."
    )

    i = 0
    while not _shutdown:
        # Device-side sync: workers block on data_ready_sem (set by the service
        # core after a producer push lands), copy backing -> fresh output, ack
        # consumed_counter. Returns tensors independent of the backing. This call
        # blocks until the next push arrives, so the loop is naturally idle-waiting.
        tt_tokens, tt_metadata = ttnn.experimental.deepseek_prefill.h2d_socket_sync(
            h2d_service, metadata_size_bytes=H2D_METADATA_SIZE_BYTES
        )
        # Decode per-iter PrefillMetadata (replicated across the mesh — first device view).
        meta_host = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
        slot_id = int(meta_host[0])
        actual_start = int(meta_host[1])
        actual_end = int(meta_host[2])
        actual_isl = actual_end - actual_start
        logger.info(
            f"[request] iter={i} metadata: slot_id={slot_id} "
            f"actual_start={actual_start} actual_end={actual_end} actual_isl={actual_isl}"
        )
        # Time ONLY the prefill compute, not the idle h2d_socket_sync wait above.
        _t0 = _time.perf_counter()
        pipeline.prefill(
            tt_tokens,
            slot_id=slot_id,
            actual_start=actual_start,
            actual_end=actual_end,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        logger.info(f"[request] iter={i} chunk slot={slot_id} [{actual_start},{actual_end}) prefill = {_dt_ms:.2f} ms")
        i += 1
    logger.info(f"[request] loop exited after {i} requests")


def _print_config() -> None:
    """Print all env var values at startup so the config is visible in logs."""
    rows = [
        ("PREFILL_MODEL_VARIANT", VARIANT.name),
        ("PREFILL_HF_MODEL", os.environ.get("PREFILL_HF_MODEL", VARIANT.hf_model_default)),
        ("PREFILL_TTNN_CACHE", os.environ.get("PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)),
        ("resolved weight_cache_path", str(resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE))),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_CHUNK_SIZE", str(CHUNK_SIZE)),
        ("PREFILL_NUM_USERS", str(NUM_USERS)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", GATE_FALLBACK_MODE),
        ("PREFILL_KV_ONLY_LAST_LAYER", str(KV_ONLY_LAST_LAYER)),
        ("PREFILL_STANDALONE", os.environ.get("PREFILL_STANDALONE", "0")),
        ("PREFILL_TRACE_DIR", os.environ.get("PREFILL_TRACE_DIR", VARIANT.prefill_trace_default)),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<trace default>")),
        ("PREFILL_STANDALONE_ITERS", os.environ.get("PREFILL_STANDALONE_ITERS", "1")),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_CLIENT_DIR", os.environ.get("PREFILL_MIGRATION_CLIENT_DIR", "<unset; uses PYTHONPATH>")),
        ("PREFILL_MIGRATION_CMD_QUEUE", os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", "/prefill_mig_cmd_1")),
        ("PREFILL_MIGRATION_TABLE_QUEUE", os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", "/prefill_mig_tbl_1")),
        ("PREFILL_MIGRATION_RESP_QUEUE", os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", "/prefill_mig_rsp_1")),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")),
    ]

    sep = "=" * 70
    config_lines = [sep, "prefill_runner configuration", sep]
    for label, val in rows:
        config_lines.append(f"  {label:<35} = {val}")
    config_lines.append(sep)
    logger.info("\n" + "\n".join(config_lines))


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    enable_migration = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1"
    logger.info(
        f"prefill_runner mesh={GLOBAL_MESH_SHAPE} "
        f"migration={'ON (KV chunk table publish)' if enable_migration else 'OFF'}"
    )

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG, l1_small_size=_L1_SMALL_SIZE)

    hf_config = load_hf_config(VARIANT)
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE)
    pipeline_config = TtPrefillPipelineConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
        num_links=2,
        capacity_factor=CAPACITY_FACTOR,
        gate_fallback_mode=GateComputeMode[GATE_FALLBACK_MODE],
        weight_cache_path=cache_path,
        model_cfg=MODEL_CFG,
        kv_only_last_layer=KV_ONLY_LAST_LAYER,
        routing_use_l1_small_for_semaphores=_ROUTING_USE_L1_SMALL_SEMAPHORES,
    )

    pipeline = TtDeepSeekPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={},
        config=pipeline_config,
    )
    pipeline.compile()

    ack_channel = None
    if enable_migration:
        # Standalone-worker model: build the KV chunk address table from the device
        # KV layout and serialize it to a .pb file. The inference server / orchestrator
        # forwards this path to the migration_worker via
        # MigrationLayerClient.send_kv_chunk_table(path). The runner owns the device,
        # so only it knows the KV cache NoC addresses; it has no IPC with the worker.
        from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import (
            build_and_serialize_kv_chunk_table,
            send_kv_chunk_table,
        )

        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        build_and_serialize_kv_chunk_table(
            mesh_device=mesh_device,
            kvpe_cache=pipeline.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
            sp_axis=0,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=NUM_USERS,
            path=table_path,
        )
        send_kv_chunk_table(table_path)

    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        # Truly standalone: file input, no H2D socket service at all.
        logger.info("Setup complete, running standalone loop (file input, no socket)")
        run_standalone_loop(pipeline)
    else:
        # Request mode: input arrives over the H2D socket service. Build it,
        # export the descriptor so a producer can connect, then read pushes.
        #
        # The model's compile() leaves a custom sub-device manager loaded (CCL /
        # MoE row-split grids). H2DStreamService's constructor enqueues an init
        # program and tt-metal validates its kernel cores against the active
        # sub-device — which fails (num_intersections == num_cores) because the
        # service cores don't fit a single custom sub-device. Revert to the
        # default whole-chip sub-device so the service program validates.
        mesh_device.clear_loaded_sub_device_manager()
        # The service's per-push token buffer holds one chunk (the request loop streams one chunk per push).
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
            f"run prefill_h2d_producer.py (or the scheduler) in another process to drive token pushes."
        )

        # Per-layer LayerAck: the runner bumps a counter once per layer; the scheduler
        # reads the delta (the ack carries no payload) and drives the migration worker.
        # ttnn.InterProcessCounterChannel owns a named POSIX shm segment the scheduler
        # connects to via shm_layer_ack_name(service_id).
        ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
        # A prior run that didn't tear down cleanly leaves this segment behind, so
        # InterProcessCounterChannel's shm_open(O_CREAT|O_EXCL) fails with EEXIST.
        # Unlink any stale segment first (POSIX shm lives at /dev/shm/<name minus '/'>).
        _stale_ack_shm = f"/dev/shm/{ack_shm_name.lstrip('/')}"
        if os.path.exists(_stale_ack_shm):
            logger.warning(f"[migration] removing stale LayerAck shm {_stale_ack_shm} from a prior run")
            os.remove(_stale_ack_shm)
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        pipeline.set_layer_ack_channel(ack_channel)
        logger.info(f"[migration] LayerAck channel ready at {ack_shm_name}; runner emits one ack per layer")

        logger.info("Setup complete, entering request loop")
        run_request_loop(pipeline, h2d_service)

        # Release the H2D service while the mesh + command queues + service core
        # are still alive. Its dtor frees a command queue and the service-core L1;
        # if that runs AFTER close_mesh_device (at interpreter exit) it aborts with
        # "cq_id 0 out of range" / "deallocate_l1 on unclaimed core".
        import gc

        del h2d_service
        gc.collect()

    if ack_channel is not None:
        # munmap + shm_unlink; doesn't need the mesh, but tear down here for symmetry.
        ack_channel.shutdown()
        del ack_channel

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
