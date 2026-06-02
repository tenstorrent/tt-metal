#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import signal
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import INVALID_SLOT_ID
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)

# Sync-op worker core. Single core suffices: the kernel only copies the
# backing tensor's pages into a fresh output, no per-core parallelism needed.
H2D_SYNC_WORKER_CORES = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))

# Inline metadata payload — packed by the producer per-iter, surfaced by the
# kernel via h2d_socket_sync's optional metadata output. Layout (4 × int32):
#   [0] actual_isl
#   [1] slot_id
#   [2] dst_slot
#   [3] reserved (currently 0)
H2D_METADATA_SIZE_BYTES = 16

# Per-iter mesh distribution for the token input. Used by both the H2D service
# (its internal mapper) and any host-side `_tokens_to_host_tensor()` callers
# (the producer process, which builds an equivalent mapper from MeshShape).
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
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", "HOST_ALL")
PREFILL_DEBUG = os.environ.get("PREFILL_DEBUG", "0") == "1"
PREFILL_TRACE_SYNCS = os.environ.get("PREFILL_TRACE_SYNCS", "0") == "1"

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


def _is_shutdown() -> bool:
    return _shutdown


def _load_hf_config():
    model_path = os.environ.get("DEEPSEEK_V3_HF_MODEL")
    if not model_path:
        raise RuntimeError("DEEPSEEK_V3_HF_MODEL must be set")
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
    what `TtDeepSeekPrefillPipeline._prepare_input_tensor` would have produced.

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
    # (actual_isl, slot_id, dst_slot) inline with the token push.
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
    """Per-iter input spec shared by `_build_h2d_service` (sets the service's
    global tensor shape) and `_tokens_to_host_tensor` (matches the host
    tensor's distributed-spec to the service's per-shard expectation).
    Shape `(sp_factor, 1, isl_per_chip)` uint32 ROW_MAJOR DRAM."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    isl_per_chip = MAX_SEQ_LEN // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, isl_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def _tokens_to_host_tensor(token_ids: list[int], mapper) -> ttnn.Tensor:
    """Build the pre-distributed host tensor consumed by `service.forward_to_tensor`.

    Applies the `is_balanced` chunk reorder host-side, reshapes to
    `(sp_factor, 1, isl_per_chip)`, then runs `ttnn.from_torch` with the
    supplied mapper to produce a multi-device-host tensor whose per-shard
    spec equals `H2DStreamService.get_per_shard_spec()`. The service streams
    each shard's bytes directly to its target coord — no bytes round-trip,
    no internal mapper invocation.
    """
    sp_factor = GLOBAL_MESH_SHAPE[0]
    assert len(token_ids) == MAX_SEQ_LEN, f"token_ids must be padded to MAX_SEQ_LEN={MAX_SEQ_LEN}, got {len(token_ids)}"
    isl_per_chip = MAX_SEQ_LEN // sp_factor

    if IS_BALANCED:
        chunk_order = create_balanced_chunk_order(sp_factor)
        t = torch.tensor(token_ids, dtype=torch.int64).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        t = reorder_tensor_chunks(t, chunk_order, seq_dim=2)
        token_ids_sharded = t.squeeze(0).squeeze(-1).reshape(sp_factor, 1, isl_per_chip)
    else:
        token_ids_sharded = torch.tensor(token_ids, dtype=torch.int64).reshape(sp_factor, 1, isl_per_chip)

    # uint32 bit pattern is what the device sees; int32 carries the same bits
    # for any non-negative token id (DeepSeek vocab fits in 18 bits anyway).
    return ttnn.from_torch(
        token_ids_sharded.to(torch.int32),
        spec=_make_global_spec(),
        mesh_mapper=mapper,
    )


def run_standalone_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Truly standalone: read token IDs from a JSON file and push them straight
    through `pipeline.prefill(token_ids=...)`. No H2D socket, no SHM, no external
    producer — single process, for local bring-up / perf.

    Reads PREFILL_STANDALONE_INPUT (default: standalone_input.json next to this
    script). File format: {"task_id": <int>, "token_ids": [<int>, ...]}.
    """
    import json
    import time as _time

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
        first_token = pipeline.prefill(token_ids=token_ids, slot_id=0, actual_isl=actual_isl)
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

    `h2d_socket_sync` returns (tokens, metadata); we decode the 4×int32 metadata
    [actual_isl, slot_id, dst_slot, reserved] and pass the real values into
    `pipeline.prefill`.
    """
    import time as _time

    logger.info(
        "[request] entering request loop — blocks on h2d_socket_sync for each push, "
        "runs until SIGTERM/SIGINT (Ctrl-C). Drive it with prefill_h2d_producer.py / the scheduler."
    )

    i = 0
    while not _shutdown:
        _t0 = _time.perf_counter()
        # Device-side sync: workers block on data_ready_sem (set by the service
        # core after a producer push lands), copy backing -> fresh output, ack
        # consumed_counter. Returns tensors independent of the backing. This call
        # blocks until the next push arrives, so the loop is naturally idle-waiting.
        tt_tokens, tt_metadata = h2d_socket_sync(
            h2d_service,
            H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
        )
        # Decode per-iter metadata (replicated across the mesh — first device view).
        meta_host = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
        actual_isl = int(meta_host[0])
        slot_id = int(meta_host[1])
        dst_slot = int(meta_host[2])
        logger.info(f"[request] iter={i} metadata: actual_isl={actual_isl} slot_id={slot_id} dst_slot={dst_slot}")
        first_token = pipeline.prefill(
            input_tensor=tt_tokens,
            slot_id=slot_id,
            actual_isl=actual_isl,
            dst_slot=dst_slot if dst_slot != INVALID_SLOT_ID else None,
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
        # (label, value, required)
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", UNSET), True),
        ("TT_DS_PREFILL_TTNN_CACHE", os.environ.get("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE), False),
        ("PREFILL_SP", str(_sp), False),
        ("PREFILL_TP", str(_tp), False),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS), False),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN), False),
        ("PREFILL_IS_BALANCED", str(IS_BALANCED), False),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR), False),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name, False),
        ("PREFILL_STANDALONE", os.environ.get("PREFILL_STANDALONE", "0"), False),
        ("PREFILL_STANDALONE_ITERS", os.environ.get("PREFILL_STANDALONE_ITERS", "5"), False),
        ("PREFILL_H2D_SERVICE_ID", os.environ.get("PREFILL_H2D_SERVICE_ID", "ds_prefill"), False),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0"), False),
        ("PREFILL_DEBUG", os.environ.get("PREFILL_DEBUG", "0"), False),
        ("PREFILL_TRACE_SYNCS", os.environ.get("PREFILL_TRACE_SYNCS", "0"), False),
    ]

    missing_required = [label for label, val, req in rows if req and val == UNSET]

    sep = "=" * 70
    config_lines = [sep, "prefill_runner configuration", sep]
    for label, val, req in rows:
        flag = " [REQUIRED]" if req else ""
        warn = " *** MISSING ***" if val == UNSET and req else ""
        config_lines.append(f"  {label:<35} = {val}{flag}{warn}")
    config_lines.append(sep)
    logger.info("\n" + "\n".join(config_lines))

    if missing_required:
        raise RuntimeError(
            f"Missing required environment variables: {missing_required}. "
            f"See the module docstring for descriptions."
        )


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    _print_config()

    enable_migration = os.environ.get("PREFILL_ENABLE_MIGRATION", "0") == "1"

    if enable_migration:
        # tt-run's --rank-bindings-mapping has set up sub-contexts at the C++
        # level; lazy-init the Python handle and assert we landed in prefill.
        from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import (
            PREFILL_SUBCTX_ID,
            ensure_distributed_context,
            get_distributed_info,
        )

        ensure_distributed_context()
        subctx_id, local_rank, local_size, world_rank, world_size = get_distributed_info()
        assert subctx_id == PREFILL_SUBCTX_ID, (
            f"prefill_runner expects PREFILL_SUBCTX_ID={PREFILL_SUBCTX_ID}, "
            f"got subctx_id={subctx_id}. Wrong rank-bindings-mapping or layout."
        )
        logger.info(
            f"prefill_runner subctx={subctx_id} local={local_rank}/{local_size} "
            f"world={world_rank}/{world_size} mesh={GLOBAL_MESH_SHAPE} migration=ON"
        )
    else:
        logger.info(f"prefill_runner standalone mesh={GLOBAL_MESH_SHAPE} migration=OFF")

    if PREFILL_DEBUG:
        from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
            probe_dram_allocatable_base,
            verify_kvpe_cache_layout,
        )

    mesh_device = _open_mesh_device()
    if PREFILL_DEBUG:
        probe_dram_allocatable_base(mesh_device, "after-mesh-open")

    hf_config = _load_hf_config()
    hf_config.max_seq_len = MAX_SEQ_LEN
    if PREFILL_DEBUG:
        probe_dram_allocatable_base(mesh_device, "after-hf-config")

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
    if PREFILL_DEBUG:
        probe_dram_allocatable_base(mesh_device, "after-pipeline-build")
        verify_kvpe_cache_layout(mesh_device, pipeline.kvpe_cache)
    pipeline.compile()
    if PREFILL_DEBUG:
        probe_dram_allocatable_base(mesh_device, "after-compile")

    if enable_migration:
        from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import DECODE_EP_ID, setup_prefill_migration

        endpoint = setup_prefill_migration(
            mesh_device=mesh_device,
            kvpe_cache=pipeline.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
        )
        pipeline.setup_migration(endpoint, DECODE_EP_ID)
        logger.info("[migration] pipeline.setup_migration() done; per-layer migrations fire on every prefill request")

    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        # Truly standalone: file input, no H2D socket service at all.
        logger.info("Setup complete, running standalone loop (file input, no socket)")
        run_standalone_loop(pipeline)
    else:
        # Request mode: input arrives over the H2D socket service. Build it,
        # export the descriptor so a producer can connect, then read pushes.
        h2d_service = _build_h2d_service(mesh_device)
        if PREFILL_DEBUG:
            probe_dram_allocatable_base(mesh_device, "after-h2d-service")
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
