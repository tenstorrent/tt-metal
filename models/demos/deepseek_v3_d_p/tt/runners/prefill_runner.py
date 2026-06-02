#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import signal
from pathlib import Path

from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.migration_setup import INVALID_SLOT_ID
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
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


def _install_sync_tracer() -> None:
    """Monkey-patch ttnn.synchronize_device + ttnn.to_torch to print where they
    are called from and how long each call took. First occurrence at each user
    call site prints a stack snippet; subsequent calls just increment counters
    that we dump at exit.

    Enabled via PREFILL_TRACE_SYNCS=1. The goal is to expose any 'hidden' sync
    points inside ttnn ops that we don't see by grepping for synchronize_device
    in the model python code.
    """
    import atexit
    import time as _time
    import traceback

    seen: dict = {}  # key -> {"count": int, "total_us": float, "max_us": float}

    def _site_key():
        # Skip the wrapper frame + this frame; find the first frame that
        # isn't inside ttnn's own python package.
        stack = traceback.extract_stack()[:-2]
        user = [f for f in stack if "/ttnn/ttnn/" not in f.filename]
        f = user[-1] if user else stack[-1]
        return (f.filename, f.lineno, f.name)

    def _record(name, key, duration_us):
        full_key = (name, *key)
        slot = seen.get(full_key)
        if slot is None:
            seen[full_key] = {"count": 1, "total_us": duration_us, "max_us": duration_us}
            logger.info(f"[sync-trace] FIRST {name} from {key[0]}:{key[1]} in {key[2]}  ({duration_us:.1f} us)")
        else:
            slot["count"] += 1
            slot["total_us"] += duration_us
            if duration_us > slot["max_us"]:
                slot["max_us"] = duration_us

    _orig_sync = ttnn.synchronize_device

    def _patched_sync(*args, **kwargs):
        key = _site_key()
        t0 = _time.perf_counter()
        ret = _orig_sync(*args, **kwargs)
        dt_us = (_time.perf_counter() - t0) * 1e6
        _record("synchronize_device", key, dt_us)
        return ret

    ttnn.synchronize_device = _patched_sync

    _orig_to_torch = ttnn.to_torch

    def _patched_to_torch(*args, **kwargs):
        key = _site_key()
        t0 = _time.perf_counter()
        ret = _orig_to_torch(*args, **kwargs)
        dt_us = (_time.perf_counter() - t0) * 1e6
        _record("to_torch", key, dt_us)
        return ret

    ttnn.to_torch = _patched_to_torch

    def _dump_summary():
        logger.info("=" * 80)
        logger.info("[sync-trace] summary (per call site)")
        logger.info("=" * 80)
        # sort by total_us descending — the expensive ones first
        rows = sorted(seen.items(), key=lambda kv: -kv[1]["total_us"])
        for (name, fn, ln, func), s in rows:
            avg = s["total_us"] / max(1, s["count"])
            logger.info(
                f"  {name:<22} {fn}:{ln} in {func}: "
                f"count={s['count']} total={s['total_us']/1e3:.1f}ms "
                f"avg={avg:.1f}us max={s['max_us']:.1f}us"
            )
        logger.info("=" * 80)

    atexit.register(_dump_summary)
    logger.info("[sync-trace] installed monkey-patches on ttnn.synchronize_device and ttnn.to_torch")


if PREFILL_TRACE_SYNCS:
    _install_sync_tracer()

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


def run_standalone_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Run a single prefill from a JSON file (no C++ server / SHM required).

    Reads PREFILL_STANDALONE_INPUT (default: standalone_input.json next to this
    script).  File format:
        {
            "task_id": <int>,
            "token_ids": [<int>, ...],
            // Optional ground-truth check; both fields are optional. If
            // expected_token_id is present and doesn't match the produced
            // first_token, the runner exits non-zero. expected_token is just
            // a human-readable hint for log messages (e.g., "lo").
            "expected_token_id": <int>,
            "expected_token": "<str>"
        }
    Prints the first generated token to stdout.
    """
    import json
    import time as _time

    default_path = Path(__file__).parent / "standalone_input.json"
    input_path = Path(os.environ.get("PREFILL_STANDALONE_INPUT", default_path))

    logger.info(f"[standalone] Reading input from {input_path}")
    with open(input_path) as f:
        data = json.load(f)

    task_id = data["task_id"]
    token_ids = list(data["token_ids"])
    expected_token_id = data.get("expected_token_id")
    expected_token = data.get("expected_token")

    logger.info(
        f"[standalone] task_id={task_id} num_tokens={len(token_ids)} " f"first5={token_ids[:5]} last5={token_ids[-5:]}"
    )

    if len(token_ids) > MAX_SEQ_LEN:
        tail_preview = token_ids[MAX_SEQ_LEN : MAX_SEQ_LEN + 10]
        tail_suffix = "..." if len(token_ids) > MAX_SEQ_LEN + 10 else ""
        raise ValueError(
            f"task_id={task_id} prompt has {len(token_ids)} tokens but "
            f"MAX_SEQ_LEN={MAX_SEQ_LEN}. Bump SEQ_LEN in the launcher. "
            f"Dropped tail tokens would have been: {tail_preview}{tail_suffix}"
        )

    actual_isl = len(token_ids)
    if len(token_ids) < MAX_SEQ_LEN:
        token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))

    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "5"))
    iter_times_ms = []
    first_token = None
    for i in range(num_iterations):
        _t0 = _time.perf_counter()
        first_token = pipeline.prefill(token_ids=token_ids, slot_id=0, actual_isl=actual_isl)
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        iter_times_ms.append(_dt_ms)
        logger.info(
            f"[prefill timing] task_id={task_id} iter={i} num_tokens={len(token_ids)} "
            f"pipeline.prefill() = {_dt_ms:.2f} ms"
        )
    logger.info(f"[iter timing summary] per-iter ms = {[round(t,2) for t in iter_times_ms]}")

    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(f"[standalone] first_token={first_token}")
    logger.info(f"Sent token {first_token} for task {task_id}")

    if expected_token_id is not None:
        hint = f" ({expected_token!r})" if expected_token is not None else ""
        if first_token == expected_token_id:
            logger.info(
                f"[standalone] ground-truth check OK: first_token={first_token} matches "
                f"expected_token_id={expected_token_id}{hint}"
            )
        else:
            raise AssertionError(
                f"[standalone] ground-truth mismatch: produced first_token={first_token}, "
                f"expected expected_token_id={expected_token_id}{hint}"
            )


def run_request_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Read prefill requests from SHM, run pipeline.prefill, write tokens back.

    SharedMemory lives in the C++ inference server's tree at
    cpp_server/src/runners/shared_memory.py. The launcher (the C++ server's
    Python child-process wrapper) must put cpp_server/src on PYTHONPATH; this
    runner imports it lazily so standalone mode (which doesn't need SHM) works
    regardless of that PYTHONPATH entry.
    """
    try:
        from runners.shared_memory import PREFILL_MAX_TOKEN_IDS, SharedMemory
    except ImportError as exc:
        raise ImportError(
            "Cannot import runners.shared_memory. SHM mode requires "
            "<tt-inference-server>/tt-media-server/cpp_server/src on PYTHONPATH. "
            "(Standalone mode does not need it; set PREFILL_STANDALONE=1 instead.)"
        ) from exc

    c2p_name = os.environ.get("TT_IPC_SHM_C2P")
    p2c_name = os.environ.get("TT_IPC_SHM_P2C")
    if not (c2p_name and p2c_name):
        raise RuntimeError("TT_IPC_SHM_C2P / TT_IPC_SHM_P2C must be set")

    logger.info(f"Opening SHM C2P={c2p_name} P2C={p2c_name}")
    import time as _time

    with SharedMemory(c2p_name, max_token_ids=PREFILL_MAX_TOKEN_IDS, is_shutdown=_is_shutdown) as c2p, SharedMemory(
        p2c_name, max_token_ids=1, is_shutdown=_is_shutdown
    ) as p2c:
        logger.info("SHM bridge started, waiting for prefill requests...")

        while not _shutdown:
            msg = c2p.read()
            if msg is None:
                break

            task_id = msg.task_id
            token_ids = list(msg.token_ids)
            dst_slot = msg.slot_id

            logger.info(
                f"Received task_id={task_id} num_tokens={len(token_ids)} "
                f"first5={token_ids[:5]} last5={token_ids[-5:]} "
                f"dst_slot={dst_slot if dst_slot != INVALID_SLOT_ID else 'INVALID(skip-migration)'}"
            )

            if len(token_ids) > MAX_SEQ_LEN:
                tail_preview = token_ids[MAX_SEQ_LEN : MAX_SEQ_LEN + 10]
                tail_suffix = "..." if len(token_ids) > MAX_SEQ_LEN + 10 else ""
                raise ValueError(
                    f"task_id={task_id} prompt has {len(token_ids)} tokens but "
                    f"MAX_SEQ_LEN={MAX_SEQ_LEN}. Bump SEQ_LEN in the launcher. "
                    f"Dropped tail tokens would have been: {tail_preview}{tail_suffix}"
                )
            actual_isl = len(token_ids)

            if len(token_ids) < MAX_SEQ_LEN:
                token_ids = token_ids + [1] * (MAX_SEQ_LEN - len(token_ids))

            _t0 = _time.perf_counter()
            first_token = pipeline.prefill(
                token_ids=token_ids,
                slot_id=0,
                actual_isl=actual_isl,
                dst_slot=dst_slot,
            )
            _dt_ms = (_time.perf_counter() - _t0) * 1000.0
            logger.info(
                f"[prefill timing] task_id={task_id} num_tokens={len(token_ids)} "
                f"dst_slot={dst_slot if dst_slot != INVALID_SLOT_ID else '-'} "
                f"pipeline.prefill() = {_dt_ms:.2f} ms"
            )

            p2c.write_token(task_id, first_token)
            logger.info(f"Sent token {first_token} for task {task_id}")

    logger.info("Request loop exited")


def _print_config() -> None:
    """Print all env var values at startup so the config is visible in logs."""
    UNSET = "<NOT SET>"
    rows = [
        # (label, value, required)
        ("DEEPSEEK_V3_HF_MODEL", os.environ.get("DEEPSEEK_V3_HF_MODEL", UNSET), True),
        ("TT_DS_PREFILL_TTNN_CACHE", os.environ.get("TT_DS_PREFILL_TTNN_CACHE", DEFAULT_TTNN_CACHE), False),
        ("TT_IPC_SHM_C2P", os.environ.get("TT_IPC_SHM_C2P", UNSET), False),
        ("TT_IPC_SHM_P2C", os.environ.get("TT_IPC_SHM_P2C", UNSET), False),
        ("PREFILL_SP", str(_sp), False),
        ("PREFILL_TP", str(_tp), False),
        ("PREFILL_NUM_LAYERS", str(NUM_LAYERS), False),
        ("PREFILL_MAX_SEQ_LEN", str(MAX_SEQ_LEN), False),
        ("PREFILL_IS_BALANCED", str(IS_BALANCED), False),
        ("PREFILL_CAPACITY_FACTOR", str(CAPACITY_FACTOR), False),
        ("PREFILL_GATE_FALLBACK_MODE", _gate_mode_name, False),
        ("PREFILL_STANDALONE", os.environ.get("PREFILL_STANDALONE", "0"), False),
        ("PREFILL_STANDALONE_INPUT", os.environ.get("PREFILL_STANDALONE_INPUT", "<default>"), False),
        ("PREFILL_STANDALONE_ITERS", os.environ.get("PREFILL_STANDALONE_ITERS", "5"), False),
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

    logger.info("Setup complete, entering request loop")
    if os.environ.get("PREFILL_STANDALONE", "0") == "1":
        run_standalone_loop(pipeline)
    else:
        run_request_loop(pipeline)

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)
    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
