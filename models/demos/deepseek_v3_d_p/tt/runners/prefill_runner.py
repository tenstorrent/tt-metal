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
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import prepare_prefill_input_tensor
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

_sp = int(os.environ.get("PREFILL_SP", 8))
_tp = int(os.environ.get("PREFILL_TP", 4))
GLOBAL_MESH_SHAPE = (_sp, _tp)
NUM_LAYERS = int(os.environ.get("PREFILL_NUM_LAYERS", 61))
# Chunked prefill: per-user KV-cache length (60*1024), streamed in CHUNK_SIZE chunks, NUM_USERS slots.
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 60 * 1024))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
# Chunked / indexed-RoPE path is non-balanced (block-cyclic).
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "0") == "1"
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

# Golden DeepSeek-R1 chunked-prefill trace for the longbook_qa prompt (56320 = 11 * 5120 tokens).
# Same default as tests/test_prefill_transformer_chunked.py; holds metadata.json (token_ids) plus
# kv_cache/layer_*.safetensors (kv_post_transform_layer_*). Override with DEEPSEEK_PREFILL_TRACE_DIR.
DEFAULT_PREFILL_TRACE_DIR = (
    "/mnt/models/deepseek-prefill-cache/golden/kimi-26/debug_trace/longbook_qa_eng_prefill_56320_nopad"
)

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
    """Construct an H2DStreamService whose per-shard backing tensor matches one
    chunk of what `prepare_prefill_input_tensor` would have produced.

    Chunked prefill streams ONE CHUNK_SIZE-wide chunk per push, so the per-shard
    target is `(1, 1, chunk_local)` uint32 ROW_MAJOR DRAM (chunk_local =
    CHUNK_SIZE // sp_factor). Achieved by setting global_spec.shape =
    `(sp_factor, 1, chunk_local)` and mapping `[Shard(0), Replicate]` on a
    `(sp, tp)` mesh — first axis of the tensor is sharded across mesh rows (sp),
    nothing else is split.

    worker_cores + metadata are set: the service core multicasts a data-ready inc
    (h2d_socket_sync waits on it on-device) and carries the per-iter
    PrefillMetadata (slot_id, actual_start, actual_end) inline with the push.
    """
    sp_factor, tp_factor = GLOBAL_MESH_SHAPE
    assert CHUNK_SIZE % sp_factor == 0, f"CHUNK_SIZE={CHUNK_SIZE} must be divisible by sp_factor={sp_factor}"
    chunk_local = CHUNK_SIZE // sp_factor
    per_chip_bytes = chunk_local * 4  # uint32

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
        f"[h2d] H2DStreamService built: global_shape=({sp_factor},1,{chunk_local}) "
        f"uint32 ROW_MAJOR DRAM (one chunk), per_chip_bytes={per_chip_bytes}, "
        f"worker_cores={H2D_SYNC_WORKER_CORES}"
    )
    return service


def _make_global_spec() -> ttnn.TensorSpec:
    """Per-iter input spec used by `_build_h2d_service` to set the service's
    global tensor shape (the producer matches it on the host side). One chunk:
    shape `(sp_factor, 1, chunk_local)` uint32 ROW_MAJOR DRAM."""
    sp_factor = GLOBAL_MESH_SHAPE[0]
    chunk_local = CHUNK_SIZE // sp_factor
    return ttnn.TensorSpec(
        shape=ttnn.Shape([sp_factor, 1, chunk_local]),
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
    # Chunked prefill needs a whole number of chunks; pad the tail up to the next chunk boundary.
    pad_to = ((len(token_ids) + CHUNK_SIZE - 1) // CHUNK_SIZE) * CHUNK_SIZE
    if len(token_ids) < pad_to:
        token_ids = token_ids + [1] * (pad_to - len(token_ids))

    slot_id = int(data.get("slot_id", 0)) % NUM_USERS

    n_chunks = len(token_ids) // CHUNK_SIZE
    num_iterations = int(os.environ.get("PREFILL_STANDALONE_ITERS", "1"))
    logger.info(f"[standalone] task_id={task_id} actual_isl={actual_isl} iters={num_iterations}")
    iter_times_ms = []
    for it in range(num_iterations):
        _t0 = _time.perf_counter()
        # prefill() is single-chunk; drive the chunk loop here, advancing kv_actual per chunk. For
        # chunk-aligned offsets the block-cyclic layout degenerates to a plain per-chip reshape, so
        # prepare_prefill_input_tensor (is_balanced=False) produces the correct chip-major chunk.
        for c in range(n_chunks):
            kv_actual = c * CHUNK_SIZE
            pipeline.prefill(
                prepare_prefill_input_tensor(
                    token_ids[kv_actual : kv_actual + CHUNK_SIZE],
                    pipeline.mesh_device,
                    cfg.sp_factor,
                    cfg.is_balanced,
                    cfg.mesh_shape,
                    cfg.sp_axis,
                ),
                slot_id=slot_id,
                kv_actual_isl=kv_actual,
            )
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        iter_times_ms.append(_dt_ms)
        logger.info(
            f"[prefill timing] task_id={task_id} iter={it} num_tokens={len(token_ids)} "
            f"chunks={n_chunks} slot={slot_id} pipeline.prefill() = {_dt_ms:.2f} ms"
        )
    logger.info(f"[iter timing summary] per-iter ms = {[round(t,2) for t in iter_times_ms]}")

    # stdout, not a log line: callers (tests / orchestrators) parse this. Chunked prefill fills the
    # KV cache and does not sample, so there is no first token to report.
    print(f"[standalone] prefill_complete task_id={task_id} slot={slot_id} isl={actual_isl}")
    logger.info(f"Prefill complete for task {task_id} (slot={slot_id}, isl={actual_isl})")


def _kv_cache_pcc_check(pipeline: TtDeepSeekPrefillPipeline, slot_id: int, n_chunks: int) -> float:
    """Gather the device KV cache for `slot_id`, un-rotate the block-cyclic layout to natural order,
    and PCC-compare each layer against the golden DeepSeek-R1 `kv_post_transform` trace.

    Shared by `run_standalone_chunked_prefill_loop` (single-process) and the request-loop PCC mode
    (driven by the external producer over the H2D socket). Returns the min per-layer PCC and asserts
    (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1) when any layer is below threshold.

    Env:
      DEEPSEEK_PREFILL_TRACE_DIR              golden trace dir (default: the longbook_qa 56320 trace)
      PREFILL_STANDALONE_CHUNKED_PCC          min per-layer KV-cache PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC only, do not assert
    """

    import torch
    from safetensors import safe_open

    from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    sp = cfg.sp_factor
    chunk_size = cfg.chunk_size
    num_layers = cfg.num_layers
    seq_len_cache = cfg.max_seq_len
    total_len = n_chunks * chunk_size

    trace_dir = Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR))
    if not trace_dir.exists():
        raise FileNotFoundError(f"golden trace dir not found: {trace_dir} (set DEEPSEEK_PREFILL_TRACE_DIR)")

    threshold = float(os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88"))
    record_only = os.environ.get("PREFILL_STANDALONE_CHUNKED_RECORD_ONLY", "0") == "1"
    kv_lora = pipeline.hf_config.kv_lora_rank
    kvpe_dim = pipeline.hf_config.qk_rope_head_dim + kv_lora

    # One gather: [num_users*num_layers, tp_replicas, seq_len_cache, kvpe] -> collapse TP via [:, :1].
    cache_full = ttnn.to_torch(
        pipeline.kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[
        :, :1
    ]  # [num_users*num_layers, 1, seq_len_cache, kvpe]

    p = blockcyclic_positions(sp, chunk_size, seq_len_cache)
    logger.info(f"[kv-pcc] device KV cache vs golden kv_post_transform (slot={slot_id}, per layer):")
    min_pcc = 1.0
    failures = []
    for i in range(num_layers):
        # user-major slot layout: cache batch index = slot_id * num_layers + layer_idx
        batch_idx = slot_id * num_layers + i
        nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
        nat[p] = cache_full[batch_idx, 0]  # un-rotate block-cyclic -> natural order
        dev_cache = nat[:total_len]

        with safe_open(trace_dir / "kv_cache" / f"layer_{i}.safetensors", framework="pt") as fsafe:
            g_post = fsafe.get_slice(f"kv_post_transform_layer_{i}")[:total_len].to(torch.float32)
        # nope (kv_lora) compares directly; the RoPE (pe) slice uses the Meta-interleaved basis while
        # the golden stores the HF half-split, so re-interleave the golden before comparing.
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        d = ref_pe.shape[-1]
        ref_pe_int = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe_int, dev_cache[:, kv_lora:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f} -> {layer_pcc:.6f}")
        if layer_pcc < threshold:
            failures.append((i, layer_pcc))

    logger.info(f"[kv-pcc] KV cache min PCC across {num_layers} layers: {min_pcc:.6f} (threshold {threshold})")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(
        f"[standalone-chunked] kv_cache_pcc_complete slot={slot_id} n_chunks={n_chunks} "
        f"total_len={total_len} min_pcc={min_pcc:.6f}"
    )
    if failures:
        msg = "; ".join(f"layer {layer} PCC {pcc:.6f} < {threshold}" for layer, pcc in failures)
        if record_only:
            logger.warning(f"[kv-pcc] sub-threshold PCC (record-only, not asserted): {msg}")
        else:
            raise AssertionError(f"[kv-pcc] KV cache PCC below {threshold}: {msg}")
    else:
        logger.success(f"[kv-pcc] KV cache PCC PASSED (min {min_pcc:.6f} >= {threshold})")
    return min_pcc


def run_standalone_chunked_prefill_loop(pipeline: TtDeepSeekPrefillPipeline) -> None:
    """Standalone chunked-prefill *validation* loop: drive the golden longbook_qa prompt through the
    pipeline in CHUNK_SIZE chunks (exactly as `run_standalone_loop` drives the chunk loop), then PCC
    the resulting device KV cache against the golden DeepSeek-R1 `kv_post_transform` trace.

    This exercises the FULL runner machinery — main() opens the mesh, builds the pipeline, and
    compiles before calling us; we then push N chunks through `pipeline.prefill` (forward_chunk +
    the shared KV cache) — but instead of file/socket input it reads the golden prompt and asserts
    correctness, mirroring `tests/test_prefill_transformer_chunked.py::run_chunked_transformer`.

    Token input is sharded with `prepare_prefill_input_tensor` (is_balanced=False), same as
    `run_standalone_loop`. For chunk-aligned offsets the block-cyclic gather degenerates to a plain
    per-chip reshape, so this matches the test's `rotated_chip_positions` gather exactly.

    The KV cache is gathered and un-rotated (block-cyclic -> natural) and PCC-compared per layer
    against the trace. The cache is allocated user-major (slot index = user_id * num_layers +
    layer_idx), so layer i of slot `slot_id` lives at batch index `slot_id * num_layers + i`.

    Env (in addition to the standard runner env — see _print_config):
      DEEPSEEK_PREFILL_TRACE_DIR              golden trace dir (default: the longbook_qa 56320 trace)
      PREFILL_STANDALONE_CHUNKED_NCHUNKS      chunks to run (default 11 -> 56320 tokens)
      PREFILL_STANDALONE_CHUNKED_SLOT         KV-cache user slot to fill (default 0)
      PREFILL_STANDALONE_CHUNKED_PCC          min per-layer KV-cache PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC only, do not assert
    """
    import json
    import time as _time

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    sp = cfg.sp_factor
    chunk_size = cfg.chunk_size
    num_layers = cfg.num_layers
    seq_len_cache = cfg.max_seq_len  # allocated cache seq dim — drives the block-cyclic inversion below

    n_chunks = int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "11"))
    slot_id = int(os.environ.get("PREFILL_STANDALONE_CHUNKED_SLOT", "0")) % NUM_USERS
    total_len = n_chunks * chunk_size
    assert total_len <= seq_len_cache, (
        f"{n_chunks} chunks x {chunk_size} = {total_len} exceeds per-user cache max_seq_len={seq_len_cache}; "
        f"bump PREFILL_MAX_SEQ_LEN or lower PREFILL_STANDALONE_CHUNKED_NCHUNKS"
    )

    trace_dir = Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR))
    if not trace_dir.exists():
        raise FileNotFoundError(f"golden trace dir not found: {trace_dir} (set DEEPSEEK_PREFILL_TRACE_DIR)")

    logger.info(
        f"[standalone-chunked] trace={trace_dir} n_chunks={n_chunks} chunk_size={chunk_size} "
        f"total_len={total_len} slot={slot_id} cache={seq_len_cache} sp={sp} layers={num_layers}"
    )

    with open(trace_dir / "metadata.json") as f:
        md = json.load(f)
    token_ids_full = list(md["token_ids"])[:total_len]
    assert (
        len(token_ids_full) == total_len
    ), f"trace metadata has {len(token_ids_full)} tokens but need {total_len}; lower PREFILL_STANDALONE_CHUNKED_NCHUNKS"

    # --- Drive chunked prefill: one CHUNK_SIZE chunk per pipeline.prefill, advancing kv_actual. ---
    _t0 = _time.perf_counter()
    for c in range(n_chunks):
        kv_actual = c * chunk_size
        pipeline.prefill(
            prepare_prefill_input_tensor(
                token_ids_full[kv_actual : kv_actual + chunk_size],
                mesh_device,
                cfg.sp_factor,
                cfg.is_balanced,
                cfg.mesh_shape,
                cfg.sp_axis,
            ),
            slot_id=slot_id,
            kv_actual_isl=kv_actual,
        )
        logger.info(f"[standalone-chunked] prefilled chunk {c + 1}/{n_chunks} (kv_actual={kv_actual})")
    ttnn.synchronize_device(mesh_device)
    dt_ms = (_time.perf_counter() - _t0) * 1000.0
    logger.info(f"[standalone-chunked] {n_chunks} chunks prefilled in {dt_ms:.2f} ms")

    # --- PCC the device KV cache against the golden kv_post_transform trace (per layer). ---
    _kv_cache_pcc_check(pipeline, slot_id, n_chunks)


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

    # Bounded PCC mode: when PREFILL_REQUEST_LOOP_PCC=1 the runner expects a finite stream of
    # PREFILL_STANDALONE_CHUNKED_NCHUNKS chunks (the producer reads the golden longbook_qa trace and
    # pushes exactly that many), then exits the loop so main() can PCC the resulting KV cache against
    # the golden trace — the socket-driven analogue of run_standalone_chunked_prefill_loop.
    pcc_mode = os.environ.get("PREFILL_REQUEST_LOOP_PCC", "0") == "1"
    expected_chunks = int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "11")) if pcc_mode else None

    logger.info(
        "[request] entering request loop — blocks on h2d_socket_sync for each push, "
        + (
            f"runs for {expected_chunks} chunks then PCC-checks the KV cache (PREFILL_REQUEST_LOOP_PCC=1)."
            if pcc_mode
            else "runs until SIGTERM/SIGINT (Ctrl-C). Drive it with prefill_h2d_producer.py / the scheduler."
        )
    )

    last_slot_id = int(os.environ.get("PREFILL_STANDALONE_CHUNKED_SLOT", "0")) % NUM_USERS
    i = 0
    while not _shutdown:
        if expected_chunks is not None and i >= expected_chunks:
            logger.info(f"[request] received all {expected_chunks} expected chunks; exiting loop for PCC check")
            break
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
        slot_id = int(meta_host[0]) % NUM_USERS
        last_slot_id = slot_id
        actual_start = int(meta_host[1])
        actual_end = int(meta_host[2])
        actual_isl = actual_end - actual_start
        # Chunked prefill: each push is one CHUNK_SIZE-wide chunk; actual_start is the absolute KV
        # position where this chunk begins (cumulative valid tokens before it) -> kv_actual_isl. The
        # real-token count (actual_isl) is informational — padding is handled by causality + the
        # caller advancing actual_start by the real count for the next chunk.
        logger.info(
            f"[request] iter={i} metadata: slot_id={slot_id} actual_start={actual_start} "
            f"actual_end={actual_end} actual_isl={actual_isl}"
        )
        # Time ONLY the prefill compute, not the idle h2d_socket_sync wait above. prefill() consumes
        # (deallocates) tt_tokens; free the metadata tensor here. No token is returned — chunked
        # prefill fills the KV cache and the decode stage reads it directly.
        _t0 = _time.perf_counter()
        pipeline.prefill(
            tt_tokens,
            slot_id=slot_id,
            kv_actual_isl=actual_start,
        )
        ttnn.deallocate(tt_metadata)
        _dt_ms = (_time.perf_counter() - _t0) * 1000.0
        logger.info(
            f"[request] iter={i} chunk_tokens={CHUNK_SIZE} kv_actual_isl={actual_start} "
            f"actual_isl={actual_isl} slot={slot_id} pipeline.prefill() = {_dt_ms:.2f} ms"
        )
        i += 1
    logger.info(f"[request] loop exited after {i} requests")

    if pcc_mode:
        # Drain the device, then PCC the KV cache filled over the socket against the golden trace.
        ttnn.synchronize_device(pipeline.mesh_device)
        logger.info(f"[request] running KV-cache PCC check (slot={last_slot_id}, n_chunks={i})")
        _kv_cache_pcc_check(pipeline, last_slot_id, i)


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
        ("PREFILL_STANDALONE_CHUNKED", os.environ.get("PREFILL_STANDALONE_CHUNKED", "0")),
        ("DEEPSEEK_PREFILL_TRACE_DIR", os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR)),
        ("PREFILL_STANDALONE_CHUNKED_NCHUNKS", os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "11")),
        ("PREFILL_STANDALONE_CHUNKED_SLOT", os.environ.get("PREFILL_STANDALONE_CHUNKED_SLOT", "0")),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        ("PREFILL_REQUEST_LOOP_PCC", os.environ.get("PREFILL_REQUEST_LOOP_PCC", "0")),
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
        chunk_size=CHUNK_SIZE,
        num_users=NUM_USERS,
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
            num_users=NUM_USERS,
            chunk_size_global=CHUNK_SIZE,  # block-cyclic period of the non-balanced prefill cache
            path=table_path,
        )
        send_kv_chunk_table(table_path)

    if os.environ.get("PREFILL_STANDALONE_CHUNKED", "0") == "1":
        # Standalone validation: golden longbook_qa input, chunked prefill, KV-cache PCC vs trace.
        logger.info("Setup complete, running standalone chunked-prefill loop (golden KV-cache PCC check)")
        run_standalone_chunked_prefill_loop(pipeline)
    elif os.environ.get("PREFILL_STANDALONE", "0") == "1":
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
