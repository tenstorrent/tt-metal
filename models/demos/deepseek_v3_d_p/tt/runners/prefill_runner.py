#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import json
import os
import signal
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import prepare_prefill_input_tensor
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
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
    sd("PREFILL_MIGRATE_GOLDEN_PTS", ",".join(u["kv_cache"] for u in users))

    # Mode: explicit must match user count; otherwise derive (1 user => burst, else pairwise).
    mode = mig.get("mode")
    if mode in ("burst", "pairwise"):
        if mode == "burst" and N != 1:
            raise ValueError(f"manifest migration.mode 'burst' requires exactly 1 user, got {N}")
        if mode == "pairwise" and N < 2:
            raise ValueError(f"manifest migration.mode 'pairwise' requires more than 1 user, got {N}")
    elif mode is None:
        mode = "burst" if N == 1 else "pairwise"
    else:
        raise ValueError(f"manifest migration.mode must be 'pairwise' or 'burst', got: {mode}")
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

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "1") == "1"
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", "DEVICE_FP32")

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _load_hf_config():
    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL") or "models/demos/deepseek_v3/reference"
    logger.info(f"Loading HF config from {model_path}")
    return AutoConfig.from_pretrained(model_path, trust_remote_code=True)


def _open_mesh_device():
    sp = GLOBAL_MESH_SHAPE[0]
    fabric_config = ttnn.FabricConfig.FABRIC_1D if sp <= 8 else ttnn.FabricConfig.FABRIC_2D

    fabric_router_config = create_fabric_router_config(
        max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE,
    )

    ttnn.set_fabric_config(
        fabric_config,
        ttnn.FabricReliabilityMode.RELAXED_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
        fabric_router_config,
    )
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*GLOBAL_MESH_SHAPE))


DEFAULT_TTNN_CACHE = "/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure"
DEFAULT_HOST_REF_CACHE = "/tmp/prefill_ref_cache"

# TtPrefillTransformer reads these env vars directly and aborts if unset.
# Export defaults here so the runner works without the caller having to set them.
os.environ.setdefault("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE)
os.environ.setdefault("TT_DS_PREFILL_HOST_REF_CACHE", DEFAULT_HOST_REF_CACHE)


def _resolve_weight_cache_path() -> Path | None:
    """Mirror the layout produced by the pytest weight_cache_path fixture so
    we read the same files the cache-populate run wrote:
      $TT_DS_PREFILL_TTNN_CACHE / deepseek_v3_d_p_{arch}_{N}dev / {sp}x{tp}
    Defaults to DEFAULT_TTNN_CACHE; returns None only if explicitly set empty."""
    env_cache = os.environ.get("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE)
    if not env_cache:
        return None
    arch = "bh" if is_blackhole() else "wh"
    num_devices = ttnn.get_num_devices()
    sp, tp = GLOBAL_MESH_SHAPE
    path = Path(env_cache) / f"deepseek_v3_d_p_{arch}_{num_devices}dev" / f"{sp}x{tp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_h2d_service(mesh_device: ttnn.MeshDevice) -> ttnn.H2DStreamService:
    """Construct an H2DStreamService whose per-shard backing tensor matches
    what `prepare_prefill_input_tensor` would have produced.

    Per-shard target: `(1, 1, isl_per_chip)` uint32 ROW_MAJOR DRAM.
    Achieved by setting global_spec.shape = `(sp_factor, 1, isl_per_chip)` and
    mapping `[Shard(0), Replicate]` on a `(sp, tp)` mesh — first axis of the
    tensor is sharded across mesh rows (sp), nothing else is split.

    No `worker_cores`, no metadata: Mode 1 only. Per-iter usage is
    `forward_to_tensor_bytes(...) -> barrier() -> consume the backing tensor
    via the standard FD-dispatched embedding op` (see run_standalone_loop).
    """
    sp_factor, tp_factor = GLOBAL_MESH_SHAPE
    assert MAX_SEQ_LEN % sp_factor == 0, f"MAX_SEQ_LEN={MAX_SEQ_LEN} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = MAX_SEQ_LEN // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = _make_global_spec()
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        H2D_MAPPER_CONFIG,
    )
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; h2d_socket_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter control bytes
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=8 * per_chip_bytes,  # 8 in-flight pages of headroom
        scratch_cb_size_bytes=per_chip_bytes,  # one page; service requires >= page_size
        mapper=mapper,
        worker_cores=H2D_SYNC_WORKER_CORES,
        metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
    )
    logger.info(
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{isl_per_chip}) "
        f"uint32 ROW_MAJOR DRAM, per_chip_bytes={per_chip_bytes}, "
        f"worker_cores={H2D_SYNC_WORKER_CORES}"
    )
    return service


def _make_global_spec() -> ttnn.TensorSpec:
    """Per-iter input spec used by `_build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side).
    Shape `(sp_factor, 1, isl_per_chip)` uint32 ROW_MAJOR DRAM."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    isl_per_chip = MAX_SEQ_LEN // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def run_standalone_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Truly standalone: read token IDs from a JSON file, shard them via
    `prepare_prefill_input_tensor`, and feed `pipeline.prefill`. No H2D socket,
    no SHM, no external producer — single process, for local bring-up / perf.

    Reads PREFILL_STANDALONE_INPUT (default: standalone_input.json next to this
    script). File format: {"task_id": <int>, "token_ids": [<int>, ...]}.
    """
    import json
    import time as _time

    cfg = pipeline.config
    default_path = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", default_path))
    logger.info(f"[standalone] reading input from {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    task_id = data["task_id"]
    token_ids = list(data["token_ids"])

    if len(token_ids) > MAX_SEQ_LEN:
        raise ValueError(
            f"task_id={task_id} prompt has {len(token_ids)} tokens but MAX_SEQ_LEN={MAX_SEQ_LEN}. "
            f"Bump PREFILL_MAX_SEQ_LEN."
        )
    actual_isl = len(token_ids)
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))

    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))
    logger.info(f"[standalone] task_id={task_id} actual_isl={actual_isl} iters={num_iterations}")
    iter_times_ms = []
    first_token = None
    for i in range(num_iterations):
        _t0 = _time.perf_counter()
        first_token = pipeline.prefill(
            prepare_prefill_input_tensor(
                token_ids,
                pipeline.mesh_device,
                cfg.sp_factor,
                cfg.is_balanced,
                cfg.mesh_shape,
                cfg.sp_axis,
            ),
            actual_isl=actual_isl,
            slot_id=0,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        iter_times_ms.append(_dt_ms)
        logger.info(
            f"[prefill timing] task_id={task_id} iter={i} num_tokens={MAX_SEQ_LEN} "
            f"pipeline.prefill() = {_dt_ms:.2f} ms first_token={first_token}"
        )
    logger.info(f"[iter timing summary] per-iter ms = {[round(t,2) for t in iter_times_ms]}")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(f"[standalone] task_id={task_id} first_token={first_token}")


def run_request_loop(pipeline: TtDeepSeekPrefillPipeline, h2d_service: ttnn.H2DStreamService) -> None:
    """Request loop: token IDs + per-iter control metadata arrive over the H2D
    socket service, pushed by a separate producer process (prefill_h2d_producer.py
    today; the inference-server / prefill scheduler in production).

    No SHM: the c2p token-input channel is replaced by the H2D socket service,
    and the p2c token write-back is removed (downstream consumption is via
    migration / layer-acks, not a host token hand-back).

    `h2d_socket_sync` returns (tokens, metadata); we decode the 3×uint32
    PrefillMetadata [slot_id, actual_start, actual_end], derive
    actual_isl = actual_end - actual_start, and pass them into `pipeline.prefill`.
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
        tt_tokens, tt_metadata = h2d_socket_sync(
            h2d_service,
            H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
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
        first_token = pipeline.prefill(
            tt_tokens,
            actual_isl=actual_isl,
            slot_id=slot_id,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        logger.info(
            f"[request] iter={i} num_tokens={MAX_SEQ_LEN} "
            f"pipeline.prefill() = {_dt_ms:.2f} ms first_token={first_token}"
        )
        i += 1
    logger.info(f"[request] loop exited after {i} requests")


def _print_config() -> None:
    """Print all env var values at startup so the config is visible in logs."""
    UNSET = "<NOT SET>"
    rows = [
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", UNSET)),
        ("TT_DS_PREFILL_TTNN_CACHE", os.environ.get("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE)),
        ("PREFILL_SP", str(_sp)),
        ("PREFILL_TP", str(_tp)),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS)),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN)),
        ("PREFILL_IS_BALANCED", str(IS_BALANCED)),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR)),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name),
        ("PREFILL_STANDALONE", os.environ.get("PREFILL_STANDALONE", "0")),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<default>")),
        ("PREFILL_STANDALONE_ITERS", os.environ.get("PREFILL_STANDALONE_ITERS", "1")),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
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

    mesh_device = _open_mesh_device()

    hf_config = _load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = _resolve_weight_cache_path()
    pipeline_config = TtPrefillPipelineConfig(
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        mesh_shape=GLOBAL_MESH_SHAPE,
        is_balanced=IS_BALANCED,
        num_links=2,
        capacity_factor=CAPACITY_FACTOR,
        gate_fallback_mode=GateComputeMode[_gate_mode_name],
        weight_cache_path=cache_path,
    )

    pipeline = TtDeepSeekPrefillPipeline(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict={},
        config=pipeline_config,
    )
    pipeline.compile()

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
        h2d_service = _build_h2d_service(mesh_device)
        service_id = os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill")
        descriptor_path = h2d_service.export_descriptor(service_id)
        logger.info(
            f"[h2d] exported descriptor service_id={service_id!r} -> {descriptor_path}; "
            f"run prefill_h2d_producer.py (or the scheduler) in another process to drive token pushes."
        )

        logger.info("Setup complete, entering request loop")
        run_request_loop(pipeline, h2d_service)

        # Release the H2D service while the mesh + command queues + service core
        # are still alive. Its dtor frees a command queue and the service-core L1;
        # if that runs AFTER close_mesh_device (at interpreter exit) it aborts with
        # "cq_id 0 out of range" / "deallocate_l1 on unclaimed core".
        import gc

        del h2d_service
        gc.collect()

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
