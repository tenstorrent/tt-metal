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
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import create_balanced_chunk_order, reverse_reorder_tensor_chunks
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import PREFILL_EP_ID
from models.demos.deepseek_v3_d_p.tt.tt_deepseek_prefill_pipeline import (
    TtDeepSeekPrefillPipeline,
    TtPrefillPipelineConfig,
)
from models.demos.deepseek_v3_d_p.utils.transformer_helpers import (
    ReferenceCacheKey,
    check_reference_cache_exists,
    load_reference_cache,
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
# Trailing positions in the chunk past `actual_end` are PAD_ID; the kernel
# skips compute on them. See
# include/tt_llm_engine/scheduler/prefill/prefill_metadata.hpp for the
# source-of-truth definition the scheduler builds.
H2D_METADATA_SIZE_BYTES = 12

# Per-iter mesh distribution for the token input. Used by both the H2D service
# (its internal mapper) and any host-side `_tokens_to_host_tensor()` callers
# (the producer process, which builds an equivalent mapper from MeshShape).
# `Shard(0)` shards the leading axis across mesh rows (SP); `Replicate()`
# duplicates across mesh cols (TP).
H2D_MAPPER_CONFIG = ttnn.MeshMapperConfig(
    placements=[ttnn.PlacementShard(0), ttnn.PlacementReplicate()],
)

# ---------------------------------------------------------------------------
# KV-cache PCC validation knobs (optional). When PREFILL_KV_VALIDATE=1, after
# each prefill the runner reads pipeline.kvpe_cache back to host, looks up the
# matching golden via the ReferenceCacheKey, and logs per-layer KV/PE PCC.
# Skips with a single WARNING (does NOT raise) if no golden file exists for
# the current config — use the offline golden-generator script to populate.
PREFILL_KV_VALIDATE = os.environ.get("PREFILL_KV_VALIDATE", "0") == "1"
PREFILL_KV_PCC_THRESHOLD = float(os.environ.get("PREFILL_KV_PCC_THRESHOLD", "0.99"))
# The runner is config-agnostic about what prompt is fed; the cache_key needs
# input_source to match what the golden was built from. Override only if you
# fed a different input source than the JSON's default.
PREFILL_KV_GOLDEN_INPUT_SOURCE = os.environ.get("PREFILL_KV_GOLDEN_INPUT_SOURCE", "longbook_qa_eng")
PREFILL_KV_GOLDEN_PAD_SIDE = os.environ.get("PREFILL_KV_GOLDEN_PAD_SIDE", "right")

# Cross-process layer-ack channel (scheduler-facing). After each prefill chunk
# completes, the runner injects this many acks back to the scheduler over the
# InterProcessCounterChannel owner-side SHM segment. Must equal the
# scheduler's SchedulerParams::layers_per_chunk on the IS side; default 1
# (one ack per chunk — coarsest granularity, defers per-layer pipelining).
PREFILL_LAYERS_PER_CHUNK = int(os.environ.get("PREFILL_LAYERS_PER_CHUNK", "1"))


def _validate_kv_against_golden(
    pipeline: TtDeepSeekPrefillPipeline,
    actual_start: int,
    actual_end: int,
) -> None:
    """Read pipeline.kvpe_cache back, compare per-layer K/V + PE against the
    golden over the absolute slot window [actual_start, actual_end). For
    single-chunk-per-slot the window is [0, actual_isl); for multi-chunk-per-slot
    this validates the chunk that was just written.

    Logs per-layer PCC and a pass/fail summary at PREFILL_KV_PCC_THRESHOLD.
    Skips with WARNING if no golden exists for the current config — never raises.
    """
    cfg = pipeline.config
    hf = pipeline.hf_config
    cache_key = ReferenceCacheKey(
        weight_type="pretrained",  # standalone runner always loads pretrained
        input_source=PREFILL_KV_GOLDEN_INPUT_SOURCE,
        isl_total=cfg.max_seq_len,
        num_layers=cfg.num_layers,
        n_routed_experts=hf.n_routed_experts,
        padding_side=PREFILL_KV_GOLDEN_PAD_SIDE,
    )
    if not check_reference_cache_exists(cache_key):
        logger.warning(
            f"[kv-validate] no golden for {cache_key}; skipping. Generate via the "
            f"offline golden-generator script to enable PCC validation."
        )
        return

    logger.info(f"[kv-validate] loading golden: {cache_key} (window=[{actual_start}, {actual_end}))")
    _ref_snapshots, ref_kvpe_list = load_reference_cache(cache_key)

    # Read device KV back. Mirrors test_prefill_transformer.py:606-613.
    # Shape after concat: [num_layers, tp_factor, seq_total, head_dim].
    tt_kvpe_all = ttnn.to_torch(
        pipeline.kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            pipeline.mesh_device, dims=(2, 1), mesh_shape=pipeline.mesh_device.shape
        ),
    ).to(torch.bfloat16)
    tt_kvpe_all_layers = tt_kvpe_all[:, :1, :, :]  # take first TP replica

    kv_lora_rank = hf.kv_lora_rank
    n_validated = min(cfg.num_layers, len(ref_kvpe_list))
    failures: list[str] = []
    chunk_order = create_balanced_chunk_order(cfg.sp_factor) if cfg.is_balanced else None

    # Cache layout: each device has seq_local rows; ConcatMesh on seq dim
    # produces [device0_rows, device1_rows, ..., device(sp-1)_rows]. The
    # chunk at slot positions [actual_start, actual_end) lives at PER-DEVICE
    # rows [actual_start//sp, actual_end//sp) on *every* device. Re-view the
    # global seq dim as (sp_factor, seq_local) to extract chunk slices
    # per-device, then flatten back to a chunk_size-long tensor (still in
    # per-chunk balanced order across devices). For single-chunk-per-slot
    # actual_start//sp == 0 and actual_end//sp == seq_local, so this is a
    # no-op view + flatten that reproduces the pre-fix behavior.
    sp_factor = cfg.sp_factor
    seq_total = tt_kvpe_all_layers.shape[2]
    seq_local = seq_total // sp_factor
    chunk_start_local = actual_start // sp_factor
    chunk_end_local = actual_end // sp_factor
    chunk_size_global = (chunk_end_local - chunk_start_local) * sp_factor
    # [num_layers, 1, sp_factor, seq_local, head_dim]
    tt_kvpe_per_device = tt_kvpe_all_layers.reshape(
        tt_kvpe_all_layers.shape[0],
        tt_kvpe_all_layers.shape[1],
        sp_factor,
        seq_local,
        tt_kvpe_all_layers.shape[-1],
    )

    for i in range(n_validated):
        ref_layer = ref_kvpe_list[i]
        try:
            # Extract this layer's chunk window via per-device slicing.
            # Shape: [1, 1, sp_factor, chunk_local, head_dim] →
            #        [1, 1, chunk_size_global, head_dim] (still chunk-balanced).
            tt_window = tt_kvpe_per_device[i : i + 1, :, :, chunk_start_local:chunk_end_local, :].reshape(
                1, 1, chunk_size_global, tt_kvpe_per_device.shape[-1]
            )
            if chunk_order is not None:
                tt_window = reverse_reorder_tensor_chunks(tt_window, chunk_order, seq_dim=2)
            ref_window = ref_layer[..., actual_start:actual_end, :]
            _, kv_pcc = comp_pcc(
                ref_window[..., :kv_lora_rank].float(),
                tt_window[..., :kv_lora_rank].float(),
            )
            _, pe_pcc = comp_pcc(
                ref_window[..., kv_lora_rank:].float(),
                tt_window[..., kv_lora_rank:].float(),
            )
            logger.info(f"[kv-validate] layer_{i:<3d}  KV PCC = {kv_pcc:.6f}, PE PCC = {pe_pcc:.6f}")
            if kv_pcc < PREFILL_KV_PCC_THRESHOLD:
                failures.append(f"layer_{i}_kv={kv_pcc:.6f}")
            if pe_pcc < PREFILL_KV_PCC_THRESHOLD:
                failures.append(f"layer_{i}_pe={pe_pcc:.6f}")
        except Exception as e:
            logger.error(f"[kv-validate] layer_{i} PCC compare failed: {e}")
            failures.append(f"layer_{i}_exc")

    if failures:
        logger.error(
            f"[kv-validate] FAIL: {len(failures)} layer(s)/component(s) below threshold "
            f"{PREFILL_KV_PCC_THRESHOLD}: {', '.join(failures)}"
        )
    else:
        logger.success(f"[kv-validate] PASS: {n_validated} layer(s) all >= {PREFILL_KV_PCC_THRESHOLD}")


_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 3200 * _sp))
# Per-chunk H2D push size. For single-chunk-per-slot this equals MAX_SEQ_LEN.
# For multi-chunk-per-slot the scheduler sends chunk_size tokens per push, so
# the H2D service must be sized to chunk_size (not max_seq_len). Defaults to
# MAX_SEQ_LEN for backward compat.
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", MAX_SEQ_LEN))
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
    assert CHUNK_SIZE % sp_factor == 0, f"CHUNK_SIZE={CHUNK_SIZE} must be divisible by sp_factor={sp_factor}"
    isl_per_chip = CHUNK_SIZE // sp_factor
    per_chip_bytes = isl_per_chip * 4  # uint32

    global_spec = _make_global_spec()
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        H2D_MAPPER_CONFIG,
    )
    # worker_cores set so the service-core kernel multicasts a data-ready inc
    # after each transfer; h2d_socket_sync() waits on that on-device, which
    # avoids the host-side barrier() round-trip per iteration.
    # metadata_size_bytes set so the producer can ship per-iter PrefillMetadata
    # (slot_id, actual_start, actual_end) inline with the token push.
    service = ttnn.H2DStreamService(
        mesh_device=mesh_device,
        global_spec=global_spec,
        fifo_size_bytes=1 * per_chip_bytes,  # DEBUG: serialize at H2D level (1 in-flight) to test multi-chunk race
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
    global tensor shape: `(sp_factor, 1, chunk_per_chip)` uint32 ROW_MAJOR DRAM.
    The H2D payload size is per-CHUNK (what the scheduler pushes per iter), not
    per-slot — so this scales with CHUNK_SIZE, not MAX_SEQ_LEN."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    chunk_per_chip = CHUNK_SIZE // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, chunk_per_chip]),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_type=ttnn.BufferType.DRAM,
    )


def run_standalone_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Truly standalone: read token IDs from a JSON file and push them straight
    through `pipeline.prefill(token_ids=...)`. No H2D socket, no SHM, no external
    producer — single process, for local bring-up / perf.

    Two modes, chosen by PREFILL_STANDALONE_CHUNK_SIZE:
      * Single-chunk (chunk_size == MAX_SEQ_LEN, the default): one forward call
        covering the whole prompt. Validation slices match padding side.
      * Multi-chunk (chunk_size < MAX_SEQ_LEN): the prompt is partitioned into
        chunks of CHUNK_SIZE tokens, one pipeline.prefill() per chunk with
        actual_start = chunk_idx * CHUNK_SIZE. Validation slices at the
        chunk's absolute window. Padding-side is right-pad implicit (chunks
        cover [0, MAX_SEQ_LEN)).

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

    chunk_size = int(os.environ.get("PREFILL_STANDALONE_CHUNK_SIZE", MAX_SEQ_LEN))
    if chunk_size > MAX_SEQ_LEN or MAX_SEQ_LEN % chunk_size != 0:
        raise ValueError(f"PREFILL_STANDALONE_CHUNK_SIZE={chunk_size} must divide MAX_SEQ_LEN={MAX_SEQ_LEN}")
    multi_chunk = chunk_size < MAX_SEQ_LEN

    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))

    if not multi_chunk:
        # Single-chunk path. Identical to the pre-multi-chunk standalone behavior.
        logger.info(f"[standalone] task_id={task_id} actual_isl={actual_isl} iters={num_iterations} (single-chunk)")
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

        if PREFILL_KV_VALIDATE:
            if PREFILL_KV_GOLDEN_PAD_SIDE == "right":
                val_start, val_end = 0, actual_isl
            else:
                val_start, val_end = MAX_SEQ_LEN - actual_isl, MAX_SEQ_LEN
            _validate_kv_against_golden(pipeline, actual_start=val_start, actual_end=val_end)
        print(f"[standalone] task_id={task_id} first_token={first_token}")
        return

    # Multi-chunk path. Process the prompt in chunk_size-sized pieces.
    num_chunks = MAX_SEQ_LEN // chunk_size
    logger.info(
        f"[standalone-multi] task_id={task_id} actual_isl={actual_isl} "
        f"chunk_size={chunk_size} num_chunks={num_chunks} iters={num_iterations}"
    )
    iter_times_ms = []
    first_token = None
    for iter_idx in range(num_iterations):
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = chunk_start + chunk_size
            chunk_tokens = token_ids[chunk_start:chunk_end]
            _t0 = _time.perf_counter()
            first_token = pipeline.prefill(
                token_ids=chunk_tokens,
                slot_id=0,
                actual_isl=chunk_size,
                actual_start=chunk_start,
            )
            _dt_ms = (_time.perf_counter() - _t0) * 1000.0
            iter_times_ms.append(_dt_ms)
            logger.info(
                f"[prefill timing] task_id={task_id} iter={iter_idx} chunk={chunk_idx} "
                f"window=[{chunk_start}, {chunk_end}) pipeline.prefill() = {_dt_ms:.2f} ms "
                f"first_token={first_token}"
            )
            # Per-chunk KV validation against this chunk's slice of the golden.
            # NOTE: with Step 4 (chunked SDPA) deferred, attention for chunks 1+
            # ignores prior K/V context, so:
            #   * Layer 0 KV/PE PCC should match golden for ALL chunks (K/V at
            #     layer 0 is a pure projection of the chunk's input embeddings,
            #     independent of attention output).
            #   * Layer 1+ PCC will degrade for chunks 1+ (cascading wrongness
            #     from incorrect attention output at layer 0).
            if PREFILL_KV_VALIDATE:
                _validate_kv_against_golden(pipeline, actual_start=chunk_start, actual_end=chunk_end)

    logger.info(f"[iter timing summary] per-chunk ms = {[round(t,2) for t in iter_times_ms]}")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(f"[standalone] task_id={task_id} first_token={first_token}")


def run_request_loop(
    pipeline: TtDeepSeekPrefillPipeline,
    h2d_service: ttnn.H2DStreamService,
    ack_channel: ttnn.InterProcessCounterChannel,
) -> None:
    """Request loop: token IDs + per-iter control metadata arrive over the H2D
    socket service, pushed by a separate producer process (prefill_h2d_producer.py
    today; the inference-server / prefill scheduler in production).

    Per chunk:
      1. block on h2d_socket_sync for the next (tokens, metadata) push
      2. decode the 3×uint32 PrefillMetadata [slot_id, actual_start, actual_end]
      3. derive actual_isl = actual_end - actual_start
      4. call pipeline.prefill(...) → first_token + per-layer KV writes
      5. inject PREFILL_LAYERS_PER_CHUNK acks back to the scheduler over
         the InterProcessCounterChannel owner segment
      6. (optional, when PREFILL_KV_VALIDATE=1) read back kvpe_cache and PCC
         vs the offline golden

    `dst_slot` is not in PrefillMetadata so this loop passes None (no inline
    migration — the scheduler plumbs dst via a separate MigrationCmd channel
    when migration is enabled).
    """
    import time as _time

    logger.info(
        f"[request] entering request loop — blocks on h2d_socket_sync for each push; "
        f"injects {PREFILL_LAYERS_PER_CHUNK} ack(s) per chunk to the scheduler. "
        f"Runs until SIGTERM/SIGINT (Ctrl-C). Drive it with prefill_h2d_producer.py / the scheduler."
    )

    i = 0
    while not _shutdown:
        tt_tokens, tt_metadata = h2d_socket_sync(
            h2d_service,
            H2D_SYNC_WORKER_CORES,
            metadata_size_bytes=H2D_METADATA_SIZE_BYTES,
        )
        meta_host = ttnn.to_torch(ttnn.get_device_tensors(tt_metadata)[0]).view(torch.int32).flatten()
        slot_id = int(meta_host[0])
        actual_start = int(meta_host[1])
        actual_end = int(meta_host[2])
        actual_isl = actual_end - actual_start
        logger.info(
            f"[request] iter={i} metadata: slot_id={slot_id} "
            f"actual_start={actual_start} actual_end={actual_end} actual_isl={actual_isl}"
        )
        # TEMPORARY DEBUG: dump per-device token shards. Sync first so we read
        # the actual H2D push, not stale scratch.
        try:
            ttnn.synchronize_device(pipeline.mesh_device)
            _shards = ttnn.get_device_tensors(tt_tokens)
            _all = []
            for _d, _sh in enumerate(_shards[:8]):
                _t = ttnn.to_torch(_sh).view(torch.int32).flatten()
                _all.append(_t)
                _last = _t[-1].item()
                _first = _t[0].item()
                _hash = int(_t.sum().item()) & 0xFFFFFFFF
                logger.info(
                    f"[request] iter={i} dev{_d}: first={_first} last={_last} " f"len={_t.numel()} sum_xor={_hash:08x}"
                )
        except Exception as _e:
            logger.warning(f"[request] iter={i} token-shard dump failed: {_e}")
        # Time ONLY the prefill compute, not the h2d_socket_sync wait above.
        _t0 = _time.perf_counter()
        first_token = pipeline.prefill(
            input_tensor=tt_tokens,
            slot_id=slot_id,
            actual_isl=actual_isl,
            dst_slot=None,
            actual_start=actual_start,
        )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        # Scheduler-facing ack: the scheduler's ack_reader_thread counts off
        # exactly SchedulerParams::layers_per_chunk acks per pushed chunk
        # before retiring the InFlightChunkFIFO head. With layers_per_chunk=1
        # (our bring-up cadence) we inject once per chunk; production
        # per-layer cadence would inject(1) inside a per-layer loop instead.
        ack_channel.inject(PREFILL_LAYERS_PER_CHUNK)
        logger.info(
            f"[request] iter={i} num_tokens={MAX_SEQ_LEN} "
            f"pipeline.prefill() = {_dt_ms:.2f} ms first_token={first_token} "
            f"acks_injected={PREFILL_LAYERS_PER_CHUNK}"
        )

        # Per-chunk KV-PCC validation against the offline golden. Same hook
        # as run_standalone_loop. Single readback per iter — cheap relative
        # to the prefill compute. Logs WARNING + skips if no golden exists
        # for the current config; doesn't raise. Window slices at
        # [actual_start, actual_end) so this works for multi-chunk-per-slot.
        if PREFILL_KV_VALIDATE:
            _validate_kv_against_golden(pipeline, actual_start=actual_start, actual_end=actual_end)

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
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
            False,
        ),
        (
            "PREFILL_LAYER_ACK_SHM",
            os.environ.get("PREFILL_LAYER_ACK_SHM", f"/prefill_{PREFILL_EP_ID}_layer_ack"),
            False,
        ),
        ("PREFILL_DEBUG", os.environ.get("PREFILL_DEBUG", "0"), False),
        ("PREFILL_TRACE_SYNCS", os.environ.get("PREFILL_TRACE_SYNCS", "0"), False),
        ("PREFILL_KV_VALIDATE", str(int(PREFILL_KV_VALIDATE)), False),
        ("PREFILL_KV_PCC_THRESHOLD", str(PREFILL_KV_PCC_THRESHOLD), False),
        ("PREFILL_KV_GOLDEN_INPUT_SOURCE", PREFILL_KV_GOLDEN_INPUT_SOURCE, False),
        ("PREFILL_KV_GOLDEN_PAD_SIDE", PREFILL_KV_GOLDEN_PAD_SIDE, False),
        ("PREFILL_LAYERS_PER_CHUNK", str(PREFILL_LAYERS_PER_CHUNK), False),
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

    # In the disaggregated worker model the migration_worker is a separate prun
    # job; the runner does no MPI rank translation and needs no tt-run sub-context.
    logger.info(f"prefill_runner mesh={GLOBAL_MESH_SHAPE} migration={'ON' if enable_migration else 'OFF'}")

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
        from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import (
            ShmLayerAckChannel,
            build_and_serialize_kv_chunk_table,
        )

        # The runner's whole migration footprint (docs/scheduler/prefill.md): it
        # has NO IPC with the migration_worker. It only —
        #   (1) builds + serializes the KV chunk address table at startup (it owns
        #       the device, so only it knows the KV cache NoC addresses). The IS
        #       forwards this path to the worker via SET_TABLE.
        #   (2) emits a LayerAck (one shm counter bump) per layer; the scheduler
        #       reads the delta and drives the worker.
        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        build_and_serialize_kv_chunk_table(
            mesh_device=mesh_device,
            kvpe_cache=pipeline.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
            path=table_path,
        )

        ack_shm_name = os.environ.get("PREFILL_LAYER_ACK_SHM", f"/prefill_{PREFILL_EP_ID}_layer_ack")
        pipeline.set_layer_ack_channel(ShmLayerAckChannel(ack_shm_name, create=True))
        logger.info("[migration] table published + LayerAck channel set; runner emits one ack per layer")

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

        # Layer-ack channel (scheduler-facing). The runner is OWNER; the IS-side
        # scheduler attaches as CONNECTOR via the matching shm_name. Match the
        # naming helper in tt-llm-engine's prefill_shm_names.hpp:
        #     /tt_prefill_layer_acks_<service_id>
        # Pre-unlink in case a prior run crashed without cleanup (owner ctor
        # uses O_CREAT|O_EXCL and would throw on a leftover segment).
        import os as _os

        ack_shm_name = f"/tt_prefill_layer_acks_{service_id}"
        try:
            _os.unlink("/dev/shm" + ack_shm_name)
        except FileNotFoundError:
            pass
        ack_channel = ttnn.InterProcessCounterChannel(ack_shm_name)
        logger.info(
            f"[h2d] layer-ack channel constructed: shm={ack_shm_name} "
            f"(scheduler attaches via shm_layer_ack_name('{service_id}'))"
        )

        logger.info("Setup complete, entering request loop")
        run_request_loop(pipeline, h2d_service, ack_channel)

        # Release everything while the mesh + command queues + service core
        # are still alive. H2D service's dtor frees a command queue + service-core
        # L1; if that runs AFTER close_mesh_device (at interpreter exit) it aborts
        # with "cq_id 0 out of range" / "deallocate_l1 on unclaimed core".
        # ack_channel teardown is munmap + shm_unlink — does not need the mesh.
        import gc

        ack_channel.shutdown()
        del ack_channel
        del h2d_service
        gc.collect()

    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
    ttnn.close_mesh_device(mesh_device)

    logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
