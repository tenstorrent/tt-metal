#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import os
import signal
from pathlib import Path

import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.runners.h2d_socket_sync_op import h2d_socket_sync
from models.demos.deepseek_v3_d_p.tt.runners.runner_utils import (
    build_h2d_service,
    get_variant,
    load_hf_config,
    open_mesh_device,
    prepare_prefill_input_tensor,
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
# Chunked prefill: per-user KV-cache length (60*1024), streamed in CHUNK_SIZE chunks, NUM_USERS slots.
MAX_SEQ_LEN = int(os.environ.get("PREFILL_MAX_SEQ_LEN", 60 * 1024))
CHUNK_SIZE = int(os.environ.get("PREFILL_CHUNK_SIZE", 5 * 1024))
NUM_USERS = int(os.environ.get("PREFILL_NUM_USERS", 2))
# Chunked / indexed-RoPE path is non-balanced (block-cyclic), regardless of variant default.
IS_BALANCED = os.environ.get("PREFILL_IS_BALANCED", "0") == "1"
CAPACITY_FACTOR = int(os.environ.get("PREFILL_CAPACITY_FACTOR", 8))
_gate_mode_name = os.environ.get("PREFILL_GATE_FALLBACK_MODE", VARIANT.default_gate_mode)

_shutdown = False


def _handle_sigterm(signum, frame):
    global _shutdown
    _shutdown = True


DEFAULT_HOST_REF_CACHE = "/tmp/prefill_ref_cache"

# Golden DeepSeek-R1 chunked-prefill trace for the longbook_qa prompt (56320 = 11 * 5120 tokens).
# Same default as tests/test_prefill_transformer_chunked.py; holds metadata.json (token_ids) plus
# kv_cache/layer_*.safetensors (kv_post_transform_layer_*). Override with DEEPSEEK_PREFILL_TRACE_DIR.
DEFAULT_PREFILL_TRACE_DIR = (
    "/mnt/models/deepseek-prefill-cache/golden/kimi-26/debug_trace/longbook_qa_eng_prefill_56320_nopad"
)

# Default the variant's TTNN-cache env so resolve_weight_cache_path works
# without the caller exporting anything. TtPrefillTransformer additionally
# *guards* on TT_DS_PREFILL_{TTNN,HOST_REF}_CACHE being set (it only validates
# they're non-None — the real weights come from `weight_cache_path`), so seed
# those too. For DeepSeek the variant env IS TT_DS_PREFILL_TTNN_CACHE.
os.environ.setdefault(VARIANT.ttnn_cache_env, VARIANT.ttnn_cache_default)
os.environ.setdefault("TT_DS_PREFILL_TTNN_CACHE", VARIANT.ttnn_cache_default)
os.environ.setdefault("TT_DS_PREFILL_HOST_REF_CACHE", DEFAULT_HOST_REF_CACHE)


_kv_pt_trace_cache: dict[str, dict] = {}


def _load_kv_pt_trace(pt_path: str) -> dict:
    """Load (and memoize) a `.pt` reference produced by `save_reference_cache`
    (see `utils.transformer_helpers`). Holds `ref_snapshots` + `ref_kvpe_list`;
    we only consume `ref_kvpe_list[i]` shape `[1, 1, seq, kv_lora + qk_rope_head_dim]`
    here. `mmap=True` keeps it lazy on first touch; subsequent layers are zero-copy
    slices into the same backing storage.

    Both `validate_migration_kv` PCC calls (BEFORE/AFTER) reuse one load via the
    module-level cache; the cache lives for the runner's lifetime.
    """
    import torch

    cached = _kv_pt_trace_cache.get(pt_path)
    if cached is not None:
        return cached
    cached = torch.load(pt_path, map_location="cpu", weights_only=True, mmap=True)
    if "ref_kvpe_list" not in cached:
        raise KeyError(
            f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} missing 'ref_kvpe_list'. "
            f"Got keys: {list(cached.keys())}. Expected a save_reference_cache .pt."
        )
    _kv_pt_trace_cache[pt_path] = cached
    return cached


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


def _read_sharded_rows(tensor_dir: Path, key: str, start: int, end: int) -> "torch.Tensor":
    """Read rows [start:end] of `key` from a chunked_group_a_v1 shard directory (Kimi), concatenating
    the rows_<s>_<e>.safetensors shards that overlap the range. Mirrors the test-side reader in
    tests/test_prefill_transformer_chunked.py."""
    from safetensors import safe_open

    parts = []
    for shard in sorted(tensor_dir.glob("rows_*.safetensors")):
        s, e = (int(x) for x in shard.stem.split("_")[1:3])
        if e <= start or s >= end:
            continue
        with safe_open(shard, framework="pt") as f:
            parts.append(f.get_slice(key)[max(start, s) - s : min(end, e) - s].to(torch.float32))
    assert parts, f"no shards overlap rows [{start}:{end}] in {tensor_dir}"
    return torch.cat(parts, dim=0)


def _load_kv_post_transform(trace_dir: Path, layer: int, total_len: int) -> "torch.Tensor":
    """Load kv_post_transform_layer_{layer}[:total_len] (float32), auto-detecting the trace layout:
    single_file (DeepSeek: kv_cache/layer_N.safetensors with the tensor as a key) vs chunked_group_a_v1
    (Kimi: kv_cache/layer_N/ directory of row-sharded rows_<s>_<e>.safetensors)."""
    from safetensors import safe_open

    key = f"kv_post_transform_layer_{layer}"
    sharded_dir = trace_dir / "kv_cache" / f"layer_{layer}"
    if sharded_dir.is_dir():
        return _read_sharded_rows(sharded_dir, key, 0, total_len)
    with safe_open(trace_dir / "kv_cache" / f"layer_{layer}.safetensors", framework="pt") as fsafe:
        return fsafe.get_slice(key)[:total_len].to(torch.float32)


def _kv_cache_pcc_check(
    pipeline: TtDeepSeekPrefillPipeline,
    slot_id: int,
    n_chunks: int,
    pt_path_override: str | None = None,
    real_len: int | None = None,
) -> float:
    """Gather the device KV cache for `slot_id`, un-rotate the block-cyclic layout to natural order,
    and PCC-compare each layer against the golden DeepSeek-R1 `kv_post_transform` trace.

    Shared by `run_standalone_chunked_prefill_loop` (single-process) and the request-loop PCC mode
    (driven by the external producer over the H2D socket). Returns the min per-layer PCC and asserts
    (unless PREFILL_STANDALONE_CHUNKED_RECORD_ONLY=1) when any layer is below threshold.

    Env:
      DEEPSEEK_PREFILL_TRACE_PT               golden reference .pt produced by save_reference_cache
                                              (carries ref_kvpe_list[layer] of shape
                                              [1, 1, seq, kv_lora + qk_rope_head_dim]). Preferred
                                              when set — covers ISL/layer configs without a
                                              standalone safetensors trace dir.
      DEEPSEEK_PREFILL_TRACE_DIR              golden trace dir (default: the longbook_qa 56320 trace).
                                              Used only if DEEPSEEK_PREFILL_TRACE_PT is unset.
                                              Holds kv_cache/layer_*.safetensors keyed by
                                              kv_post_transform_layer_<i>.
      PREFILL_STANDALONE_CHUNKED_PCC          min per-layer KV-cache PCC threshold (default 0.88)
      PREFILL_STANDALONE_CHUNKED_RECORD_ONLY  "1" -> log PCC only, do not assert
    """

    import torch

    from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions
    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    sp = cfg.sp_factor
    chunk_size = cfg.chunk_size
    num_layers = cfg.num_layers
    seq_len_cache = cfg.max_seq_len
    total_len = n_chunks * chunk_size
    # Compare only the REAL tokens. With a partial last chunk (prompt not a multiple of chunk_size),
    # total_len = n_chunks*chunk_size overshoots the prompt by the padding; real_len (the last chunk's
    # actual_end) bounds the compare to written, non-pad positions. Falls back to total_len for the
    # exact-multiple case (real_len unset).
    compare_len = real_len if real_len is not None else total_len

    pt_path = (pt_path_override or os.environ.get("DEEPSEEK_PREFILL_TRACE_PT", "")).strip()
    if pt_path:
        if not Path(pt_path).is_file():
            raise FileNotFoundError(f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} does not exist or is not a file")
        kv_pt = _load_kv_pt_trace(pt_path)["ref_kvpe_list"]
        if len(kv_pt) < num_layers:
            raise RuntimeError(
                f"DEEPSEEK_PREFILL_TRACE_PT={pt_path} has {len(kv_pt)} layers in ref_kvpe_list "
                f"but pipeline.num_layers={num_layers}; pick a .pt that matches the runner's layer count."
            )
        trace_dir = None
    else:
        trace_dir = Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR))
        if not trace_dir.exists():
            raise FileNotFoundError(
                f"golden trace dir not found: {trace_dir} "
                "(set DEEPSEEK_PREFILL_TRACE_DIR or DEEPSEEK_PREFILL_TRACE_PT)"
            )
        kv_pt = None

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
        dev_cache = nat[:compare_len]

        if kv_pt is not None:
            # ref_kvpe_list[i] = ref_cache.key_cache[i] from HF's DynamicCache after the HF model's MLA
            # forward. Per tests/test_prefill_transformer.py (canonical KVPE PCC, lines ~664-671),
            # this tensor is ALREADY in the device's rotary basis — pe is compared directly with no
            # re-interleave. Applying HF->Meta to it produces noise (see _ref_pe_for_comp below).
            g_post = kv_pt[i][0, 0, :compare_len].to(torch.float32)
        else:
            # The safetensors trace stores `kv_post_transform_layer_<i>` in HF half-split layout
            # (single-file DeepSeek or Kimi's row-sharded dir — _load_kv_post_transform auto-detects);
            # nope (kv_lora) compares directly, the pe slice is re-interleaved to Meta below. Load only
            # the populated [:compare_len] positions.
            g_post = _load_kv_post_transform(trace_dir, i, compare_len)
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        if kv_pt is not None:
            ref_pe_for_comp = ref_pe  # already Meta-interleaved (HF DynamicCache from save_reference_cache)
            basis_tag = "direct"
        else:
            d = ref_pe.shape[-1]
            ref_pe_for_comp = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(
                -1, d
            )  # HF -> Meta
            basis_tag = "interleaved"
        _, pcc_pe = comp_pcc(ref_pe_for_comp, dev_cache[:, kv_lora:])
        layer_pcc = min(pcc_nope, pcc_pe)
        min_pcc = min(min_pcc, layer_pcc)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe({basis_tag})={pcc_pe:.6f} -> {layer_pcc:.6f}")
        if layer_pcc < threshold:
            failures.append((i, layer_pcc))

    logger.info(f"[kv-pcc] KV cache min PCC across {num_layers} layers: {min_pcc:.6f} (threshold {threshold})")
    # stdout, not a log line: callers (tests / orchestrators) parse this.
    print(
        f"[standalone-chunked] kv_cache_pcc_complete slot={slot_id} n_chunks={n_chunks} "
        f"total_len={total_len} compare_len={compare_len} min_pcc={min_pcc:.6f}"
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


def validate_migration_kv(
    pipeline: TtDeepSeekPrefillPipeline, src_slot: int, dst_slot: int, n_chunks: int, real_len: int | None = None
):
    """Validate the KV cache BEFORE and AFTER a slot->slot migration.

    The migration (src_slot -> dst_slot) is driven by tt-llm-engine (the prefill
    scheduler / driver over the migration layer) — the runner never issues migrate
    itself (see integration_setup; the runner only publishes the table via SET_TABLE).
    This reuses `_kv_cache_pcc_check` to PCC BOTH endpoints against the SAME golden
    `kv_post_transform` trace:

      * BEFORE: the SRC slot — the model-produced KV that tt-llm-engine migrates.
      * AFTER:  the DST slot — the migrated copy tt-llm-engine wrote.

    A correct migration => the DST slot PCCs to golden exactly as the SRC slot does, so
    a drop (or an empty / 0-PCC dst) flags a broken or absent migration. Emits
    `[kv-migrate-validate] BEFORE/AFTER` lines (orchestrators parse these).
    """
    logger.info(f"[kv-migrate-validate] BEFORE migration: validating SRC slot {src_slot} (real_len={real_len})")
    src_pcc = _kv_cache_pcc_check(pipeline, src_slot, n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] BEFORE src_slot={src_slot} min_pcc={src_pcc:.6f}")

    logger.info(f"[kv-migrate-validate] AFTER migration: validating DST slot {dst_slot} (real_len={real_len})")
    dst_pcc = _kv_cache_pcc_check(pipeline, dst_slot, n_chunks, real_len=real_len)
    print(f"[kv-migrate-validate] AFTER dst_slot={dst_slot} min_pcc={dst_pcc:.6f}")

    logger.success(
        f"[kv-migrate-validate] slot {src_slot} -> {dst_slot}: "
        f"BEFORE(src) min_pcc={src_pcc:.6f}, AFTER(dst) min_pcc={dst_pcc:.6f}"
    )
    return src_pcc, dst_pcc


def validate_migrations_pairwise(pipeline: TtDeepSeekPrefillPipeline, pairs):
    """Validate N concurrent slot->slot migrations of distinct prompts.

    Asserts each dst slot's KV equals its own src slot's (migration fidelity + cross-talk detection),
    via one host-side compare of the raw stored cache. Then golden-anchors the slots configured by
    PREFILL_MIGRATE_GOLDEN_PTS: a positional comma list of .pt paths indexed by slot (same order as
    the driver's --token-json; empty entry skips that slot), confirming each prefill is model-correct.
    Raises AssertionError on any failure.
    """
    import torch

    from tests.ttnn.utils_for_testing import comp_pcc

    cfg = pipeline.config
    mesh_device = pipeline.mesh_device
    num_layers = cfg.num_layers
    thr = float(os.environ.get("PREFILL_MIGRATE_PAIRWISE_PCC", "0.99"))

    # Single gather of the whole cache: [num_users*num_layers, 1, seq_len_cache, kvpe] (TP via [:, :1]).
    cache_full = ttnn.to_torch(
        pipeline.kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1]

    failures = []
    for src, dst in pairs:
        min_pcc = 1.0
        for layer in range(num_layers):
            sb = src * num_layers + layer
            db = dst * num_layers + layer
            _, pcc = comp_pcc(cache_full[sb, 0], cache_full[db, 0])
            min_pcc = min(min_pcc, pcc)
        status = "PASS" if min_pcc >= thr else "FAIL"
        logger.info(f"[kv-migrate-validate] pairwise src_slot={src} dst_slot={dst} min_pcc={min_pcc:.6f} -> {status}")
        print(f"[kv-migrate-validate] AFTER pairwise src={src} dst={dst} min_pcc={min_pcc:.6f}")
        if min_pcc < thr:
            failures.append((src, dst, min_pcc))

    if failures:
        msg = "; ".join(f"{s}->{d} pcc={p:.6f}" for s, d, p in failures)
        raise AssertionError(f"[kv-migrate-validate] {len(failures)} pair(s) dst!=src below {thr}: {msg}")
    logger.success(f"[kv-migrate-validate] ALL {len(pairs)} pair(s) dst==src PASSED (>= {thr})")

    # Golden anchor config. One knob: PREFILL_MIGRATE_GOLDEN_PTS = positional comma list of .pt
    # paths (entry i anchors slot i, same order as --token-json; empty entry skips that slot).
    # Back-compat: PREFILL_MIGRATE_GOLDEN_SLOT + per-slot PREFILL_MIGRATE_GOLDEN_PT_<slot>.
    golden_pt = {}
    pts = os.environ.get("PREFILL_MIGRATE_GOLDEN_PTS", "").strip()
    if pts:
        for slot, path in enumerate(pts.split(",")):
            if path.strip():
                golden_pt[slot] = path.strip()
    else:
        for tok in os.environ.get("PREFILL_MIGRATE_GOLDEN_SLOT", "").split(","):
            if tok.strip().isdigit():
                golden_pt[int(tok)] = os.environ.get(f"PREFILL_MIGRATE_GOLDEN_PT_{int(tok)}", "").strip() or None

    n_pairs = max(1, len(pairs))
    gchunks = max(1, int(os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "1")) // n_pairs)
    for s in sorted(golden_pt):
        d = next((dd for ss, dd in pairs if ss == s), None)
        gpt = golden_pt[s]
        logger.info(f"[kv-migrate-validate] golden anchor: src slot {s} (n_chunks={gchunks}) pt={gpt or 'global'}")
        sp = _kv_cache_pcc_check(pipeline, s, gchunks, pt_path_override=gpt)
        print(f"[kv-migrate-validate] GOLDEN src_slot={s} min_pcc={sp:.6f}")
        if d is not None:
            dp = _kv_cache_pcc_check(pipeline, d, gchunks, pt_path_override=gpt)
            print(f"[kv-migrate-validate] GOLDEN dst_slot={d} min_pcc={dp:.6f}")


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
    # All-slots mode: prefill + PCC-validate the golden prompt into EVERY user slot
    # (PREFILL_STANDALONE_CHUNKED_ALL_SLOTS=1). Otherwise fill the single
    # PREFILL_STANDALONE_CHUNKED_SLOT (default 0), preserving the original behavior.
    if os.environ.get("PREFILL_STANDALONE_CHUNKED_ALL_SLOTS", "0") == "1":
        slot_ids = list(range(NUM_USERS))
    else:
        slot_ids = [int(os.environ.get("PREFILL_STANDALONE_CHUNKED_SLOT", "0")) % NUM_USERS]
    total_len = n_chunks * chunk_size
    assert total_len <= seq_len_cache, (
        f"{n_chunks} chunks x {chunk_size} = {total_len} exceeds per-user cache max_seq_len={seq_len_cache}; "
        f"bump PREFILL_MAX_SEQ_LEN or lower PREFILL_STANDALONE_CHUNKED_NCHUNKS"
    )

    # Input tokens: when a .pt golden is in use (DEEPSEEK_PREFILL_TRACE_PT), the runner's KV would
    # diverge from the .pt's ref_kvpe_list unless we drive the same prompt that produced the .pt.
    # Allow DEEPSEEK_PREFILL_TOKENS_JSON (or PREFILL_STANDALONE_INPUT as a fallback) to point at the
    # matching standalone_input_*.json shipped next to the .pt; otherwise fall back to the legacy
    # safetensors-trace path that reads metadata.json from DEEPSEEK_PREFILL_TRACE_DIR.
    tokens_json = (
        os.environ.get("DEEPSEEK_PREFILL_TOKENS_JSON", "").strip()
        or os.environ.get("PREFILL_STANDALONE_INPUT", "").strip()
    )
    pt_path_present = bool(os.environ.get("DEEPSEEK_PREFILL_TRACE_PT", "").strip())
    if tokens_json or pt_path_present:
        if not tokens_json:
            raise RuntimeError(
                "DEEPSEEK_PREFILL_TRACE_PT is set but no input tokens were provided. "
                "Point DEEPSEEK_PREFILL_TOKENS_JSON (or PREFILL_STANDALONE_INPUT) at the "
                "standalone_input_*.json that produced the .pt."
            )
        if not Path(tokens_json).is_file():
            raise FileNotFoundError(f"DEEPSEEK_PREFILL_TOKENS_JSON={tokens_json} does not exist")
        logger.info(
            f"[standalone-chunked] tokens_json={tokens_json} n_chunks={n_chunks} chunk_size={chunk_size} "
            f"total_len={total_len} slots={slot_ids} num_users={NUM_USERS} cache={seq_len_cache} sp={sp} layers={num_layers}"
        )
        with open(tokens_json) as f:
            md = json.load(f)
    else:
        trace_dir = Path(os.environ.get("DEEPSEEK_PREFILL_TRACE_DIR", DEFAULT_PREFILL_TRACE_DIR))
        if not trace_dir.exists():
            raise FileNotFoundError(
                f"golden trace dir not found: {trace_dir} "
                "(set DEEPSEEK_PREFILL_TRACE_DIR, or use DEEPSEEK_PREFILL_TRACE_PT + DEEPSEEK_PREFILL_TOKENS_JSON)"
            )
        logger.info(
            f"[standalone-chunked] trace={trace_dir} n_chunks={n_chunks} chunk_size={chunk_size} "
            f"total_len={total_len} slots={slot_ids} num_users={NUM_USERS} cache={seq_len_cache} sp={sp} layers={num_layers}"
        )
        with open(trace_dir / "metadata.json") as f:
            md = json.load(f)
    token_ids_full = list(md["token_ids"])[:total_len]
    assert len(token_ids_full) == total_len, (
        f"input tokens have {len(token_ids_full)} entries but need {total_len} "
        f"(n_chunks={n_chunks} * chunk_size={chunk_size}); "
        "lower PREFILL_STANDALONE_CHUNKED_NCHUNKS, lower PREFILL_CHUNK_SIZE, or pick a longer prompt"
    )

    # Prefill the golden prompt into each requested slot, then PCC-validate that slot. Run every slot
    # before failing so all results are visible (collect failures, assert once at the end).
    slot_min_pcc: dict[int, float] = {}
    slot_failures: dict[int, str] = {}
    for slot_id in slot_ids:
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
            logger.info(
                f"[standalone-chunked] slot {slot_id}/{NUM_USERS}: prefilled chunk {c + 1}/{n_chunks} "
                f"(kv_actual={kv_actual})"
            )
        ttnn.synchronize_device(mesh_device)
        dt_ms = (_time.perf_counter() - _t0) * 1000.0
        logger.info(f"[standalone-chunked] slot {slot_id}: {n_chunks} chunks prefilled in {dt_ms:.2f} ms")

        # --- PCC the device KV cache against the golden kv_post_transform trace (per layer). ---
        try:
            slot_min_pcc[slot_id] = _kv_cache_pcc_check(pipeline, slot_id, n_chunks)
        except AssertionError as e:
            slot_failures[slot_id] = str(e)
            logger.error(f"[standalone-chunked] slot {slot_id} FAILED PCC: {e}")

    summary = ", ".join(f"slot{s}={slot_min_pcc.get(s, float('nan')):.6f}" for s in slot_ids)
    logger.info(f"[standalone-chunked] per-slot min PCC over {len(slot_ids)} slot(s): {summary}")
    if slot_failures:
        raise AssertionError(
            f"[standalone-chunked] {len(slot_failures)}/{len(slot_ids)} slots failed PCC: "
            + "; ".join(f"slot {s}: {m}" for s, m in slot_failures.items())
        )
    logger.success(
        f"[standalone-chunked] ALL {len(slot_ids)} slot(s) PASSED (min PCC {min(slot_min_pcc.values()):.6f})"
    )


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
    chunks_per_slot: dict[int, int] = {}  # per-slot chunk count (concurrent multi-slot migration)
    real_end_per_slot: dict[int, int] = {}  # per-slot real prompt length (max actual_end; excludes padding)

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
        chunks_per_slot[slot_id] = chunks_per_slot.get(slot_id, 0) + 1
        actual_start = int(meta_host[1])
        actual_end = int(meta_host[2])
        actual_isl = actual_end - actual_start
        # Real prompt length for this slot (excludes the padded tail of a partial last chunk): the
        # last chunk's actual_end == the true token count. Bounds the KV-PCC compare below.
        real_end_per_slot[slot_id] = max(real_end_per_slot.get(slot_id, 0), actual_end)
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
        # Drain the device, then validate the KV cache against the golden trace.
        ttnn.synchronize_device(pipeline.mesh_device)

        if os.environ.get("PREFILL_VALIDATE_MIGRATION", "0") == "1":
            # Migration mode: tt-llm-engine (the prefill scheduler/driver) migrates N
            # (src -> dst) pairs over the migration layer and writes a DONE sentinel when
            # prefill + all migrations finish. The sentinel CONTENT is the machine-readable
            # pair list ("src dst" per line) the driver wrote, so we validate exactly the
            # pairs that migrated (BEFORE=src, AFTER=dst) against the same golden. Each src
            # slot is validated with ITS OWN chunk count (not the loop total), since with
            # concurrent migrations the chunks are spread across N slots. Falls back to the
            # single PREFILL_MIGRATE_SRC/DST_SLOT env pair if the sentinel carries no pairs.
            done_file = os.environ.get("MIGRATION_DONE_FILE", "/tmp/migration_done.sentinel")
            wait_s = int(os.environ.get("PREFILL_MIGRATE_WAIT_S", "1200"))
            logger.info(f"[kv-migrate-validate] waiting for DONE sentinel {done_file} (<= {wait_s}s)")
            deadline = _time.time() + wait_s
            while not os.path.exists(done_file):
                if _time.time() >= deadline:
                    raise TimeoutError(
                        f"[kv-migrate-validate] sentinel {done_file} never appeared after {wait_s}s "
                        "(did the prefill driver finish prefill + migration?)"
                    )
                _time.sleep(0.5)
            # Parse "src dst" pairs from the sentinel; fall back to the env single pair.
            pairs = []
            try:
                for line in open(done_file).read().splitlines():
                    parts = line.split()
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        pairs.append((int(parts[0]) % NUM_USERS, int(parts[1]) % NUM_USERS))
            except OSError:
                pass
            if not pairs:
                pairs = [
                    (
                        int(os.environ.get("PREFILL_MIGRATE_SRC_SLOT", "0")) % NUM_USERS,
                        int(os.environ.get("PREFILL_MIGRATE_DST_SLOT", "1")) % NUM_USERS,
                    )
                ]
            logger.success(f"[kv-migrate-validate] sentinel found — validating {len(pairs)} pair(s): {pairs}")
            ttnn.synchronize_device(pipeline.mesh_device)
            if os.environ.get("PREFILL_MIGRATE_PAIRWISE", "0") == "1":
                # N distinct prompts: dst==src fidelity + optional per-slot golden anchor.
                validate_migrations_pairwise(pipeline, pairs)
            else:
                # Same prompt across slots: PCC each (src, dst) against the shared golden.
                for src_slot, dst_slot in pairs:
                    n_src = chunks_per_slot.get(src_slot, i)  # per-slot chunk count (NOT the loop total)
                    rl_src = real_end_per_slot.get(src_slot)  # real ISL (excludes pad); dst copies the same range
                    validate_migration_kv(pipeline, src_slot, dst_slot, n_src, real_len=rl_src)
            logger.success(f"[kv-migrate-validate] ALL {len(pairs)} migrated pair(s) PASSED")
        else:
            # Validate EVERY populated slot, each over its own populated range: chunks_per_slot[s]
            # chunks and real_len = that slot's actual_end (excludes padding). Multi-user prefill
            # fills several slots with different-length prompts; each is PCC'd against the shared
            # golden over only its real (non-pad) positions. _kv_cache_pcc_check raises on a
            # sub-threshold slot (unless RECORD_ONLY), so any failure aborts here.
            slots = sorted(chunks_per_slot)
            logger.info(f"[request] running KV-cache PCC check for {len(slots)} slot(s): {slots}")
            slot_pccs = {}
            for s in slots:
                real_len = real_end_per_slot.get(s)
                n_chunks_s = chunks_per_slot[s]
                logger.info(f"[request]  -> slot={s} n_chunks={n_chunks_s} real_len={real_len}")
                slot_pccs[s] = _kv_cache_pcc_check(pipeline, s, n_chunks_s, real_len=real_len)
            logger.success(
                f"[request] all {len(slots)} slot(s) PASSED KV-cache PCC: "
                + ", ".join(f"slot{s}={p:.6f}" for s, p in sorted(slot_pccs.items()))
            )


def _print_config() -> None:
    """Print all env var values at startup so the config is visible in logs."""
    UNSET = "<NOT SET>"
    rows = [
        ("PREFILL_MODEL_VARIANT", VARIANT.name),
        (VARIANT.hf_model_env, os.environ.get(VARIANT.hf_model_env, VARIANT.hf_model_default)),
        (VARIANT.ttnn_cache_env, os.environ.get(VARIANT.ttnn_cache_env, VARIANT.ttnn_cache_default)),
        ("resolved weight_cache_path", str(resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE))),
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
        ("DEEPSEEK_PREFILL_TRACE_PT", os.environ.get("DEEPSEEK_PREFILL_TRACE_PT", "<NOT SET>")),
        ("DEEPSEEK_PREFILL_TOKENS_JSON", os.environ.get("DEEPSEEK_PREFILL_TOKENS_JSON", "<NOT SET>")),
        ("PREFILL_STANDALONE_CHUNKED_NCHUNKS", os.environ.get("PREFILL_STANDALONE_CHUNKED_NCHUNKS", "11")),
        ("PREFILL_STANDALONE_CHUNKED_SLOT", os.environ.get("PREFILL_STANDALONE_CHUNKED_SLOT", "0")),
        ("PREFILL_STANDALONE_CHUNKED_PCC", os.environ.get("PREFILL_STANDALONE_CHUNKED_PCC", "0.88")),
        ("PREFILL_REQUEST_LOOP_PCC", os.environ.get("PREFILL_REQUEST_LOOP_PCC", "0")),
        ("PREFILL_ENABLE_MIGRATION", os.environ.get("PREFILL_ENABLE_MIGRATION", "0")),
        (
            "PREFILL_MIGRATION_TABLE_PATH",
            os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb"),
        ),
        ("PREFILL_MIGRATION_CMD_QUEUE", os.environ.get("PREFILL_MIGRATION_CMD_QUEUE", "/prefill_mig_cmd_1")),
        ("PREFILL_MIGRATION_TABLE_QUEUE", os.environ.get("PREFILL_MIGRATION_TABLE_QUEUE", "/prefill_mig_tbl_1")),
        ("PREFILL_MIGRATION_RESP_QUEUE", os.environ.get("PREFILL_MIGRATION_RESP_QUEUE", "/prefill_mig_rsp_1")),
        ("PREFILL_MIGRATION_WAIT_READY_MS", os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000")),
        ("PREFILL_MIGRATION_CLIENT_DIR", os.environ.get("PREFILL_MIGRATION_CLIENT_DIR", "<NOT SET>")),
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

    mesh_device = open_mesh_device(GLOBAL_MESH_SHAPE, MODEL_CFG)

    hf_config = load_hf_config(VARIANT)
    hf_config.max_seq_len = MAX_SEQ_LEN

    cache_path = resolve_weight_cache_path(VARIANT, GLOBAL_MESH_SHAPE)
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
        model_cfg=MODEL_CFG,
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
        # Full migration bring-up. The runner owns the device, so it is the ONLY
        # component that knows both the KV cache NoC addresses
        # (kvpe_cache.buffer_address()) and the local mesh's UMD chip ids — and
        # the worker needs BOTH to reach WORKER_READY (it gates on SetTable +
        # AssignDevMap; see control_thread.cpp::maybe_emit_worker_ready). The
        # call serializes the table, sends SET_TABLE + AssignDevMap, then blocks
        # until WORKER_READY so the scheduler can safely start migrations as
        # soon as the request loop opens.
        from models.demos.deepseek_v3_d_p.tt.runners.integration_setup import publish_kv_chunk_table_and_wait_ready

        table_path = os.environ.get("PREFILL_MIGRATION_TABLE_PATH", "/tmp/prefill_kv_chunk_table.pb")
        wait_ready_ms = int(os.environ.get("PREFILL_MIGRATION_WAIT_READY_MS", "120000"))
        publish_kv_chunk_table_and_wait_ready(
            mesh_device=mesh_device,
            kvpe_cache=pipeline.kvpe_cache,
            seq_len=MAX_SEQ_LEN,
            num_layers=NUM_LAYERS,
            mesh_shape=GLOBAL_MESH_SHAPE,
            sp_axis=0,  # GLOBAL_MESH_SHAPE = (sp, tp) — SP is axis 0
            num_users=NUM_USERS,
            chunk_size_global=CHUNK_SIZE,  # block-cyclic period (prefill chunk size)
            path=table_path,
            wait_ready_timeout_ms=wait_ready_ms,
        )

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
        # Chunked prefill streams ONE CHUNK_SIZE-wide chunk per push, so the
        # service's global tensor is sized to a chunk, not the full sequence.
        h2d_service = build_h2d_service(
            mesh_device,
            mesh_shape=GLOBAL_MESH_SHAPE,
            max_seq_len=CHUNK_SIZE,
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
