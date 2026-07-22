# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Chunked-prefill test for TtPrefillTransformer (DeepSeek V3, N layers).

Runs chunked prefill through a specified number of layers (5, 10, or 61 = full model), processing
the sequence in 5*1024 = 5120-token chunks into a KV cache for ONE user of length 55*1024 = 56320.
Each layer attends to its own cache slot; chunks are driven in order so a layer's KV is populated
before the next chunk reads it. The MoE path is orthogonal to attention; only the MLA path
exercises the chunked-prefill code.

Per-layer decoder outputs are PCC-compared against the precomputed golden DeepSeek-R1 trace. To keep
host memory bounded (61 full-sequence tensors would be ~100GB) the comparison is done per chunk per
layer, reading only each layer's chunk slice from the trace.

Requires an 8x4 Blackhole mesh and (env from the task):
    TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
    DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
    TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/goldened
Override the trace dir with PREFILL_TRACE_DIR.
"""

import gc
import json
import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open
from tracy import signpost

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import get_block_timings, reset_block_timings
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import (
    cache_half_pccs,
    gather_cache_tp0,
    interleave_pe,
    read_sharded_rows,
    unrotate_cache_layer,
)
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = 5 * 1024  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (1 user)
# Larger KV cache for the no-PCC perf sweep only (up to 100k ISL = 20 chunks). Kept separate from
# SEQ_CACHE so the PCC tests and the _PADDED_FULL_55K split (which assert against 55*1024) are untouched.
SEQ_CACHE_NOPCC = 100 * 1024  # 102400 KV cache length (1 user)


def _resolve_trace_dir(variant) -> Path:
    """Golden chunked-prefill trace dir for `variant`: the model-agnostic PREFILL_TRACE_DIR env
    overrides the variant's prefill_trace_default. A vllm trace nests metadata.json + kv_cache under
    one run-hash subdir, so if the dir itself has no metadata.json, descend into the sole subdir that
    does."""
    path = Path(os.environ.get("PREFILL_TRACE_DIR", variant.test_prefill_trace_default))
    if (path / "metadata.json").exists():
        return path
    subs = [d for d in sorted(path.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]
    if len(subs) != 1:
        raise FileNotFoundError(f"no metadata.json in {path} or a unique subdir (found {len(subs)} candidates)")
    return subs[0]


# Full 55k (56320) sequence in varied chunks (same split as test_prefill_block_chunked's full55k):
# requested prefix [1k,2k,3k,4k,5k,3k,2k,5k] (=25600) + a varied tail (=30720) of non-1024-aligned
# sizes that exercise mid-tile rotation offsets. Every split is a multiple of 32 and <= CHUNK.
_PADDED_FULL_55K = [
    1024,
    2048,
    3072,
    4096,
    5120,
    3072,
    2048,
    5120,
    2592,
    1568,
    4608,
    800,
    5120,
    3360,
    4640,
    2048,
    1536,
    4448,
]  # sum == 55 * 1024
assert sum(_PADDED_FULL_55K) == SEQ_CACHE and all(v % 32 == 0 and 0 < v <= CHUNK for v in _PADDED_FULL_55K)

# Per-chunk per-layer threshold; error accumulates with depth, so this matches the single-shot
# transformer's device-gate trace bar (TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88). Calibrate + tighten.
LAYER_PCC_THRESHOLD = 0.88
# Deepest config whose per-layer PCC is asserted; deeper runs (L61) stay record-only until their
# accumulation headroom is pinned.
GATED_LAYER_DEPTH = 10
# Record-only warn floors for the deep KV / indexer-K cache PCC: a WARNING, never a test stop. Set at the
# observed L78 minimum (not below it) so a future regression trips the warning. KVPE nope bottoms ~0.86
# (glm_5_2 @L75); indexer-K nope 0.952 (glm_5_1 @L52; glm_5_1 captures all 78 layers, glm_5_2's
# 0-2+every-4th subsample only reaches 0.980).
KV_CACHE_PCC_WARN_THRESHOLD = 0.85
INDEXER_K_PCC_WARN_THRESHOLD = 0.95

# Per-chunk baseline medians (seconds) for the no-PCC perf gate, pulled from a known-good CI run. Keyed
# by (num_layers, n_chunks, num_iters) so only the exact config we have a CI number for is gated; every
# other combo in the no-PCC sweep stays record-only. Each list has one entry per chunk (index c ==
# chunk c). A single margin (the perf_margin pytest arg) is applied to every chunk. Recalibrate by
# re-reading the "chunk timing stats" table from a fresh green CI run.
KIMI_NO_PCC_BASELINE_CHUNK_TIMES_S = {
    # test_kimi_prefill_transformer_chunked_no_pcc[...-L61-chunks_eleven-ten_iters] (55k / code_debug).
    # TODO: populate the 11 per-chunk medians from the first green code_debug 55k CI run's
    # "chunk timing stats" table; until then this config runs record-only (no perf gate). The old 5-chunk
    # longbook baseline was [1.330, 1.326, 1.326, 1.340, 1.369] (run 28753487696) -- chunks 0-4 should be
    # ~unchanged (chunk c attends to KV[0:c*CHUNK] regardless of n_chunks); chunks 5-10 are new.
    # (61, 11, 10): [...],
}
# Default +/- tolerance band around each baseline chunk median (fraction). Overridable per test via the
# perf_margin pytest argument (see test_prefill_block_perf.py's `margin` column for the design).
DEFAULT_PERF_MARGIN = 0.05


def _load_metadata_token_ids(trace_dir: Path, total_len: int) -> torch.Tensor:
    with open(trace_dir / "metadata.json") as f:
        md = json.load(f)
    return torch.tensor(md["token_ids"][:total_len], dtype=torch.int64)


# Trace layouts: DeepSeek ("single_file") writes one safetensors file per layer with every tensor
# as a key; Kimi ("chunked_group_a_v1") writes each tensor as a directory of row-sharded files
# (rows_<start>_<end>.safetensors, chunk_rows each) and renames hidden_states/ -> decoder_io/. Both
# capture decoder_output + kv_post_transform (all this test needs), so only the reader differs.
_LAYOUT_SINGLE_FILE = "single_file"
_LAYOUT_CHUNKED_GROUP_A = "chunked_group_a_v1"


def _load_layer_rows(
    trace_dir: Path, layout: str, group: str, layer: int, key: str, start: int, end: int
) -> torch.Tensor:
    """Read trace tensor `key` rows [start:end] (float32) for `layer`, handling both layouts. `group`
    is the logical bucket: "hidden_states" (decoder_io for Kimi), "kv_cache", or "dsa" (indexer-K, stored
    as dsa/<key>/ where key == indexer_k_layer_<layer>, like decoder_io)."""
    if layout == _LAYOUT_CHUNKED_GROUP_A:
        if group == "hidden_states":
            tensor_dir = trace_dir / "decoder_io" / key
        elif group == "dsa":
            tensor_dir = trace_dir / "dsa" / key
        else:
            tensor_dir = trace_dir / "kv_cache" / f"layer_{layer}"
        return read_sharded_rows(tensor_dir, key, start, end)
    path = trace_dir / group / f"layer_{layer}.safetensors"
    with safe_open(path, framework="pt") as f:
        return f.get_slice(key)[start:end].to(torch.float32)


def _ref_layer_slice(trace_dir: Path, layout: str, layer: int, start: int, end: int) -> torch.Tensor:
    """Read decoder_output_layer_{layer}[start:end] from the trace (partial read, natural order)."""
    return _load_layer_rows(trace_dir, layout, "hidden_states", layer, f"decoder_output_layer_{layer}", start, end)


def _record_kv_cache_pcc(
    trace_dir, layout, tt_kvpe_cache, mesh_device, sp, num_layers, seq_len_cache, total_len, kv_lora
):
    """Record-only: gather the device KV cache, un-rotate the block-cyclic layout, and PCC each layer's
    valid region [:total_len] against the golden kv_post_transform trace ([nope | pe], the pe half re-based
    to the device Meta interleave via cache_half_pccs). Per-layer cache — slot == layer. Does NOT assert."""
    logger.info("Device KV cache vs golden kv_post_transform (record-only):")
    cache_full = gather_cache_tp0(tt_kvpe_cache, mesh_device)  # [num_layers, seq_len_cache, kvpe]
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    cache_min_pcc = {}
    for i in range(num_layers):
        dev_cache = unrotate_cache_layer(cache_full[i], p, total_len)
        g_post = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, total_len)
        pcc_nope, pcc_pe = cache_half_pccs(g_post, dev_cache, kv_lora, pe_interleave=True)
        cache_min_pcc[i] = min(pcc_nope, pcc_pe)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f}")
        if cache_min_pcc[i] < KV_CACHE_PCC_WARN_THRESHOLD:
            logger.warning(
                f"  KV cache layer {i} PCC {cache_min_pcc[i]:.6f} below warn floor "
                f"{KV_CACHE_PCC_WARN_THRESHOLD} (record-only, not asserted)"
            )
    kv_min = min(cache_min_pcc.values())
    logger.info(f"KV cache min PCC across layers: {kv_min:.6f}")
    if kv_min < KV_CACHE_PCC_WARN_THRESHOLD:
        logger.warning(f"KV cache min PCC {kv_min:.6f} below warn floor {KV_CACHE_PCC_WARN_THRESHOLD} (record-only)")


def _record_indexer_k_cache_pcc(
    trace_dir, layout, tt_index_kv_cache, mesh_device, sp, num_layers, seq_len_cache, total_len, config
):
    """Record-only: gather the device DSA indexer-K cache, un-rotate the block-cyclic layout, and PCC
    each captured layer's valid region [:total_len] against the golden dsa/indexer_k trace. The
    index_head_dim key is [rope | nope] (rope = first half, indexed-RoPE; nope = second half, no rope);
    BOTH compare directly because GLM's indexer RoPE is natively interleaved and the vLLM golden stores
    that same basis (verified on device: the half-split reindex gives ~0 PCC, direct gives ~0.9999).
    Same gather/un-rotate as the KVPE cache (caller-owned tensor, ConcatMesh2dToTensor dims=(2,1),
    blockcyclic_positions). indexer_k is captured for a subset of layers (glm_5_1: all; glm_5_2: 0-2 +
    every 4th) — layers without a golden are skipped. Does NOT assert; GLM DSA variants only."""
    logger.info("Device indexer-K cache vs golden dsa/indexer_k (record-only):")
    cache_full = gather_cache_tp0(tt_index_kv_cache, mesh_device)  # [num_full_indexer_layers or num_layers, T, D]
    layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    if not layers:
        logger.info("  (no indexer_k golden layers present -- skipping)")
        return
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    rope = config.index_head_dim // 2  # [rope | nope]
    idx_min_pcc = {}
    for i in layers:
        # Compact index cache (GLM-5.2 cross-layer reuse): layer i's slot is its full-indexer rank, not i
        # (rank == i for glm_5_1, where every layer is full). Matches the indexer's own write addressing.
        dev_cache = unrotate_cache_layer(cache_full[full_indexer_rank(config, i)], p, total_len)
        g = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, total_len)
        pcc_rope, pcc_nope = cache_half_pccs(g, dev_cache, rope, pe_interleave=False)
        idx_min_pcc[i] = min(pcc_nope, pcc_rope)
        logger.info(f"  indexer cache layer {i} PCC: nope={pcc_nope:.6f} rope={pcc_rope:.6f}")
        if idx_min_pcc[i] < INDEXER_K_PCC_WARN_THRESHOLD:
            logger.warning(
                f"  indexer-K cache layer {i} PCC {idx_min_pcc[i]:.6f} below warn floor "
                f"{INDEXER_K_PCC_WARN_THRESHOLD} (record-only, not asserted)"
            )
    idx_min = min(idx_min_pcc.values())
    logger.info(f"Indexer-K cache min PCC across {len(layers)} captured layers: {idx_min:.6f}")
    if idx_min < INDEXER_K_PCC_WARN_THRESHOLD:
        logger.warning(
            f"Indexer-K cache min PCC {idx_min:.6f} below warn floor {INDEXER_K_PCC_WARN_THRESHOLD} (record-only)"
        )


def _preload_kvpe_prefix_from_trace(
    tt_kvpe_cache,
    trace_dir,
    layout,
    num_layers,
    preload_isl,
    trace_len,
    sp,
    seq_len_cache,
    kvpe_dim,
    kv_lora,
    mesh_device,
    sp_axis,
    host_dtype,
    host_layout,
):
    """Preload the first `preload_isl` tokens of each layer's prior KV into the block-cyclic device KVPE
    cache, so a chunk measured at KV depth preload_isl attends to REAL prior KV (representative MoE routing)
    instead of the zero-init prefix. Mirrors test_mla.py's chunked preload: per layer read the natural-order
    prior KV from the trace's kv_post_transform, re-interleave the k_pe slice (the trace stores HF half-split;
    the device cache is Meta interleaved), lay it out block-cyclic, and copy host->device.

    Rows past `trace_len` (the trace only goes so deep) are filled with RANDOM KV so we can still exercise
    KV depths beyond the trace — random keeps the MoE gate non-degenerate (unlike a zero prefix), which is
    all this no-PCC timing run needs. The indexer key cache is preloaded separately from the golden
    dsa/indexer_k trace (see _preload_indexer_k_prefix_from_trace)."""
    real_len = min(preload_isl, trace_len)
    rand_len = preload_isl - real_len
    logger.info(
        f"Preloading {preload_isl}-token KV prefix into {num_layers} layer slot(s) "
        f"({real_len} real from trace {trace_dir}, {rand_len} random beyond the trace)"
    )
    # Build the replicated host cache in bf16 (the sparse KVPE cache dtype), not float32: at num_layers=78
    # x SEQ_CACHE_NOPCC the float32 tensor would be ~19 GB. Per-layer transients (randn/blockcyclic) are
    # freed each iteration, so the peak is this one bf16 tensor plus a single layer's working set.
    cache_host = torch.zeros(num_layers, 1, seq_len_cache, kvpe_dim, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(1234)  # deterministic random tail
    for i in range(num_layers):
        kv_prior = torch.randn(preload_isl, kvpe_dim, generator=gen).to(torch.bfloat16)
        if real_len > 0:
            real = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, real_len)
            real[:, kv_lora:] = interleave_pe(real[:, kv_lora:])
            kv_prior[:real_len] = real.to(torch.bfloat16)
        cache_host[i, 0] = blockcyclic_cache_host(kv_prior, sp, CHUNK, seq_len_cache, kvpe_dim)[0, 0]
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2  # SP-shard the cache seq dim; TP-replicate (matches init_kvpe_cache)
    cache_host_tt = ttnn.from_torch(
        cache_host,
        dtype=host_dtype,
        layout=host_layout,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
    )
    ttnn.copy_host_to_device_tensor(cache_host_tt, tt_kvpe_cache)
    ttnn.synchronize_device(mesh_device)


def _preload_indexer_k_prefix_from_trace(
    tt_index_kv_cache,
    trace_dir,
    layout,
    config,
    num_layers,
    preload_isl,
    trace_len,
    sp,
    seq_len_cache,
    index_head_dim,
    mesh_device,
    sp_axis,
):
    """Preload the first `preload_isl` tokens of the DSA indexer key cache from the golden dsa/indexer_k
    trace, so a measured chunk at KV depth preload_isl has a REAL indexer prefix (representative top-k
    selection at depth) rather than a zero prior. Only "full" indexer layers own a cache slot / have a
    golden (glm_5_1: all; glm_5_2: 0-2 + every 4th); layer i is written to its compacted slot
    full_indexer_rank(config, i), and the cache is strided by num_full_indexer_layers. The golden
    index_head_dim key is [rope | nope] already in the device's interleaved basis (GLM), so it is written
    verbatim -- no re-interleave (unlike KVPE's k_pe half-split -> interleaved). Rows past the trace are
    random (timing-representative). Mirrors _preload_kvpe_prefix_from_trace otherwise."""
    full_layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    if not full_layers:
        logger.info(f"no indexer_k golden in trace {trace_dir}; leaving the indexer prefix zero")
        return
    num_slots = num_full_indexer_layers(config) or num_layers
    real_len = min(preload_isl, trace_len)
    rand_len = preload_isl - real_len
    logger.info(
        f"Preloading {preload_isl}-token indexer-K prefix into {len(full_layers)} full-indexer slot(s) "
        f"({real_len} real from trace, {rand_len} random beyond the trace)"
    )
    cache_host = torch.zeros(num_slots, 1, seq_len_cache, index_head_dim, dtype=torch.bfloat16)
    gen = torch.Generator().manual_seed(2345)  # deterministic random tail (distinct from the KVPE seed)
    for i in full_layers:
        idx_prior = torch.randn(preload_isl, index_head_dim, generator=gen).to(torch.bfloat16)
        if real_len > 0:
            real = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, real_len)
            idx_prior[:real_len] = real.to(torch.bfloat16)
        slot = full_indexer_rank(config, i)
        cache_host[slot, 0] = blockcyclic_cache_host(idx_prior, sp, CHUNK, seq_len_cache, index_head_dim)[0, 0]
    cache_shard_dims = [None, None]
    cache_shard_dims[sp_axis] = 2  # SP-shard the cache seq dim; TP-replicate (matches init_kvpe_cache)
    cache_host_tt = ttnn.from_torch(
        cache_host,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=cache_shard_dims),
    )
    ttnn.copy_host_to_device_tensor(cache_host_tt, tt_index_kv_cache)
    ttnn.synchronize_device(mesh_device)


def run_chunked_transformer_padded(
    variant,
    config,
    mesh_device,
    weight_cache_path,
    num_layers,
    splits,
    gate_fallback_mode,
    num_links,
    topology,
    routing_use_l1_small_for_semaphores=False,
):
    """Chunked prefill through num_layers with VARIABLE/partial chunks `splits` (each run as a full
    CHUNK-wide tile padded with a pad token). Exercises the rotated + partial MLA path across the full
    55k sequence. One shared num_layers-slot KV cache; per chunk per layer we un-rotate, scatter the
    VALID rows, and PCC the valid region [kv_actual:valid_end) against the trace."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")
    layout = variant.prefill_trace_layout

    profiler.clear()
    profiler.start("total_test_time")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"
    tile = ttnn.TILE_SIZE

    chunk_local = CHUNK // sp
    total_len = sum(splits)
    for v in splits:
        assert 0 < v <= CHUNK and v % tile == 0, f"split {v} must be tile-aligned and <= {CHUNK}"

    # Slab-aligned cache covering the largest rotated write (kv_actual + CHUNK), >= 2 slabs.
    max_window = CHUNK * 2
    ka = 0
    for v in splits:
        max_window = max(max_window, ka + CHUNK)
        ka += v
    seq_len_cache = ((max_window + CHUNK - 1) // CHUNK) * CHUNK

    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = seq_len_cache

    logger.info(
        f"chunked-padded transformer: num_layers={num_layers} mesh={mesh_shape} splits={splits} "
        f"total_len={total_len} cache={seq_len_cache} chunk={CHUNK}"
    )

    token_ids_full = _load_metadata_token_ids(trace_dir, total_len)

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=seq_len_cache,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    mesh_device.enable_program_cache()
    layer_min_pcc = {i: 1.0 for i in range(num_layers)}

    profiler.start("tt_forward")
    ka = 0
    for c, isl in enumerate(splits):
        kv_actual = ka
        valid_end = kv_actual + isl
        ka += isl

        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = [positions[ch][r] for ch in range(sp) for r in range(chunk_local)]  # global pos, len CHUNK
        gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
        chunk_tok = token_ids_full[gather_idx].clone()  # REORDER (block-cyclic gather)
        chunk_tok[torch.tensor([gp >= valid_end for gp in flat])] = 0  # PAD positions -> pad token (masked)

        tt_tokens = ttnn.from_torch(
            chunk_tok.reshape(sp, 1, chunk_local),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        # forward (not a separate forward_chunk) drives the chunked path via actual_start/actual_end;
        # it uses self.indexed_rope, runs the norm/lm_head/sample tail (token ignored), and with
        # return_intermediates snapshots each layer to host as intermediates["layer_i"].
        _, _, layer_outputs = transformer.forward(
            tt_tokens,
            tt_kvpe_cache,
            actual_isl=isl,
            actual_start=kv_actual,
            actual_end=valid_end,
            cache_user_id=0,
            return_intermediates=True,
        )
        ttnn.synchronize_device(mesh_device)

        valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < valid_end]
        src = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
        dst = torch.tensor([gp - kv_actual for _, gp in valid_pairs], dtype=torch.long)  # 0..isl-1

        for i in range(num_layers):
            # host snapshot [1, CHUNK, emb] (SP-seq + TP-hidden concatenated); index [0] -> [CHUNK, emb].
            out_flat = layer_outputs[f"layer_{i}"][0].to(torch.float32)

            natural = torch.zeros(isl, emb_dim, dtype=torch.float32)
            natural[dst] = out_flat[src]  # un-rotate valid rows -> natural order [kv_actual, valid_end)
            ref = _ref_layer_slice(trace_dir, layout, i, kv_actual, valid_end)
            _, pcc = comp_pcc(ref, natural)
            layer_min_pcc[i] = min(layer_min_pcc[i], pcc)
            # Record-only mode: log every per-layer/per-chunk PCC, do not assert (deep-layer
            # accumulation profiling). Flag sub-threshold values as warnings instead of failing.
            logger.info(f"  chunk {c} (kv_actual={kv_actual} isl={isl}) layer {i} PCC: {pcc:.6f}")
            if pcc < LAYER_PCC_THRESHOLD:
                logger.warning(f"  chunk {c} layer {i} PCC {pcc:.6f} below {LAYER_PCC_THRESHOLD} (not asserted)")
        logger.info(f"  chunk {c} done (kv_actual={kv_actual} isl={isl}, {num_layers} layers)")
    profiler.end("tt_forward")

    logger.info("Per-layer min PCC across chunks:")
    for i in range(num_layers):
        logger.info(f"  layer {i}: {layer_min_pcc[i]:.6f}")

    # Gate the shallow configs (measured >=0.99) so a real regression fails CI; the full-depth run's
    # accumulation headroom is not yet pinned, so it stays record-only.
    overall_min = min(layer_min_pcc.values())
    if num_layers <= GATED_LAYER_DEPTH:
        assert overall_min >= LAYER_PCC_THRESHOLD, f"min per-layer PCC {overall_min:.6f} < {LAYER_PCC_THRESHOLD}"

    _record_kv_cache_pcc(
        trace_dir,
        layout,
        tt_kvpe_cache,
        mesh_device,
        sp,
        num_layers,
        seq_len_cache,
        total_len,
        config.kv_lora_rank,
    )

    profiler.end("total_test_time")
    logger.success(
        f"Chunked-padded transformer passed (num_layers={num_layers}, {len(splits)} chunks, "
        f"min PCC {overall_min:.6f})"
    )
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


def run_chunked_transformer(
    variant,
    config,
    mesh_device,
    weight_cache_path,
    num_layers,
    n_chunks,
    gate_fallback_mode,
    num_links,
    topology,
    routing_use_l1_small_for_semaphores=False,
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")
    layout = variant.prefill_trace_layout

    profiler.clear()
    profiler.start("total_test_time")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"

    chunk_local = CHUNK // sp  # 640
    total_len = n_chunks * CHUNK
    assert total_len <= SEQ_CACHE, f"{n_chunks} chunks ({total_len}) exceed cache {SEQ_CACHE}"

    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = SEQ_CACHE

    logger.info(
        f"chunked transformer: num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"total_len={total_len} cache={SEQ_CACHE} chunk={CHUNK}"
    )

    token_ids_full = _load_metadata_token_ids(trace_dir, total_len)

    # --- Weights from the prebuilt TTNN cache (empty state_dict when complete). ---
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,  # per-chunk size -> MoE/FFN dispatch buffers
        max_seq_len=SEQ_CACHE,  # KV ring buffer = full cache
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    # ONE shared KV cache holding all layers' slots [num_layers, 1, seq_local, 576]; layer i uses
    # cache_layer_idx=i. The ring scratch buffer is shared across layers inside TtPrefillTransformer.
    # Sparse (DSA) requires an UNCOMPRESSED bf16/fp8_e4m3 ROW_MAJOR KVPE cache (sparse_sdpa reads it
    # natively; mla.forward asserts) — NOT the init_kvpe_cache bfloat8_b/TILE default that dense
    # ring_mla wants. Match the cache format to the path.
    has_indexer = resolve_has_indexer(config)
    kvpe_dtype_layout = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) if has_indexer else {}
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
        **kvpe_dtype_layout,
    )

    # Sparse (DSA) layers read a block-cyclic indexer key cache that is caller-owned and passed into
    # forward, exactly like the KVPE cache. It is user-major layer-stacked
    # [num_users*index_cache_layers, 1, T, D_idx], so the indexer addresses slot
    # user*index_cache_layers + cache_layer_idx. Unlike the per-layer KVPE cache, the indexer stride is the
    # COMPACTED full-indexer count (num_full_indexer_layers) for GLM-5.2 cross-layer reuse — "shared" layers
    # reuse a "full" layer's cache and get no slot of their own — falling back to num_layers when there is no
    # indexer_types map. bf8 (half the memory, top-k within bf16 noise). Dense variants get None.
    tt_index_kv_cache = None
    if has_indexer:
        # A sparse config must carry index_head_dim; assert rather than silently defaulting so a
        # misconfigured (missing-field) sparse setup fails loudly with a clear message.
        assert getattr(config, "index_head_dim", None) is not None, "sparse config must provide index_head_dim"
        index_cache_layers = num_full_indexer_layers(config) or num_layers
        tt_index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=SEQ_CACHE,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=index_cache_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
        )

    mesh_device.enable_program_cache()

    # min PCC per layer across chunks (for the summary)
    layer_min_pcc = {i: 1.0 for i in range(num_layers)}

    profiler.start("tt_forward")
    for c in range(n_chunks):
        kv_actual = c * CHUNK  # chunk-aligned -> rotation degenerates
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        local_pos = flat - kv_actual  # permutation of [0, CHUNK)

        # token_ids in block-cyclic chip-major order -> [sp, 1, chunk_local], SP-sharded on dim 0.
        chunk_tok = token_ids_full[flat].reshape(sp, 1, chunk_local)
        tt_tokens = ttnn.from_torch(
            chunk_tok,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        # forward (not a separate forward_chunk): full chunk, all positions real, so actual_end is
        # kv_actual + CHUNK. With return_intermediates it snapshots each layer to host as
        # intermediates["layer_i"]; forward uses self.indexed_rope.
        _, _, layer_outputs = transformer.forward(
            tt_tokens,
            tt_kvpe_cache,
            actual_isl=CHUNK,
            actual_start=kv_actual,
            actual_end=kv_actual + CHUNK,
            cache_user_id=0,
            return_intermediates=True,
            index_kv_cache=tt_index_kv_cache,
        )
        ttnn.synchronize_device(mesh_device)

        for i in range(num_layers):
            # host snapshot [1, CHUNK, emb] (SP-seq + TP-hidden concatenated); index [0] -> [CHUNK, emb].
            out_flat = layer_outputs[f"layer_{i}"][0].to(torch.float32)

            natural = torch.zeros(CHUNK, emb_dim, dtype=torch.float32)
            natural[local_pos] = out_flat  # un-rotate block-cyclic -> natural chunk order
            ref = _ref_layer_slice(trace_dir, layout, i, kv_actual, kv_actual + CHUNK)
            _, pcc = comp_pcc(ref, natural)
            layer_min_pcc[i] = min(layer_min_pcc[i], pcc)
            # Record-only mode: log every per-layer/per-chunk PCC, do not assert (deep-layer
            # accumulation profiling). Flag sub-threshold values as warnings instead of failing.
            logger.info(f"  chunk {c} layer {i} PCC: {pcc:.6f}")
            if pcc < LAYER_PCC_THRESHOLD:
                logger.warning(f"  chunk {c} layer {i} PCC {pcc:.6f} below {LAYER_PCC_THRESHOLD} (not asserted)")
        logger.info(f"  chunk {c} done ({num_layers} layers)")
    profiler.end("tt_forward")

    logger.info("Per-layer min PCC across chunks:")
    for i in range(num_layers):
        logger.info(f"  layer {i}: {layer_min_pcc[i]:.6f}")

    # Gate the shallow configs (measured >=0.99) so a real regression fails CI; the full-depth run's
    # accumulation headroom is not yet pinned, so it stays record-only.
    overall_min = min(layer_min_pcc.values())
    if num_layers <= GATED_LAYER_DEPTH:
        assert overall_min >= LAYER_PCC_THRESHOLD, f"min per-layer PCC {overall_min:.6f} < {LAYER_PCC_THRESHOLD}"

    _record_kv_cache_pcc(
        trace_dir,
        layout,
        tt_kvpe_cache,
        mesh_device,
        sp,
        num_layers,
        SEQ_CACHE,
        total_len,
        config.kv_lora_rank,
    )
    if tt_index_kv_cache is not None and (trace_dir / "dsa" / "indexer_k_layer_0").exists():
        _record_indexer_k_cache_pcc(
            trace_dir,
            layout,
            tt_index_kv_cache,
            mesh_device,
            sp,
            num_layers,
            SEQ_CACHE,
            total_len,
            config,
        )

    profiler.end("total_test_time")
    logger.success(
        f"Chunked prefill transformer passed (num_layers={num_layers}, n_chunks={n_chunks}, "
        f"min PCC {overall_min:.6f})"
    )
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.skipif(not is_blackhole(), reason="DeepSeek prefill requires Blackhole")
@pytest.mark.timeout(0)
def test_ds_prefill_transformer_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_links,
    topology,
):
    run_chunked_transformer(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE,
        num_links,
        topology,
    )


@pytest.mark.parametrize("splits", [_PADDED_FULL_55K], ids=["full55k"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.skipif(not is_blackhole(), reason="DeepSeek prefill requires Blackhole")
@pytest.mark.timeout(0)
def test_ds_prefill_transformer_chunked_padded(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    splits,
    num_links,
    topology,
):
    run_chunked_transformer_padded(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        splits,
        GateComputeMode.DEVICE,
        num_links,
        topology,
    )


# ---------------------------------------------------------------------------
# Kimi K2.6 variants
# ---------------------------------------------------------------------------
# Same chunked-prefill validation as the DeepSeek tests, with the kimi_k2_6 variant and the on-device
# gate (GateComputeMode.DEVICE_FP32 — Kimi has a single expert group, so it uses the grouped-topk
# fp32 device path) + KimiK26Config fabric payload. These skip until the Kimi golden trace lands (set
# PREFILL_TRACE_DIR; see tt/runners/adapters/).


@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                # Carve a small L1_SMALL region so the MoE routing all-gather can place its global
                # semaphores there (use_l1_small_for_semaphores) instead of pinning the main-L1 floor.
                # Kept minimal: L1_SMALL is carved from the top of L1, so a large value would shift the
                # main-L1 buffer floor down and could re-introduce the clash.
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_prefill_transformer_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_links,
    topology,
):
    run_chunked_transformer(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        routing_use_l1_small_for_semaphores=True,
    )


@pytest.mark.parametrize("splits", [_PADDED_FULL_55K], ids=["full55k"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                # Carve a small L1_SMALL region so the MoE routing all-gather can place its global
                # semaphores there (use_l1_small_for_semaphores) instead of pinning the main-L1 floor.
                # Kept minimal: L1_SMALL is carved from the top of L1, so a large value would shift the
                # main-L1 buffer floor down and could re-introduce the clash.
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_prefill_transformer_chunked_padded(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    splits,
    num_links,
    topology,
):
    run_chunked_transformer_padded(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        splits,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        routing_use_l1_small_for_semaphores=True,
    )


# GLM-5.1 variants
# ---------------------------------------------------------------------------
# Same chunked-prefill validation as the DeepSeek/Kimi tests, for the glm_5_1 / glm_5_2 variants and the
# on-device gate (GateComputeMode.DEVICE_FP32 — GLM's noaux_tc gate uses the grouped-topk fp32 device path)
# + GLM fabric payload (5.1 == 5.2 dims). glm_5_2 additionally exercises DSA indexer reuse per chunk: each
# chunk is one forward, so full layers recompute that chunk's top-k and shared layers reuse it within the
# chunk. Golden = each variant's vLLM 55k structured trace (chunked_group_a_v1; via test_prefill_trace_default,
# override with PREFILL_TRACE_DIR).


@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [1, 10, 78], ids=["L1", "L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE),
                # Small L1_SMALL region for the MoE routing all-gather's global semaphores
                # (use_l1_small_for_semaphores); see the Kimi chunked test for the rationale.
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_1", "glm_5_2"], indirect=True, ids=["glm51", "glm52"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_links,
    topology,
):
    run_chunked_transformer(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        routing_use_l1_small_for_semaphores=True,
    )


def run_chunked_transformer_no_pcc(
    variant,
    config,
    mesh_device,
    weight_cache_path,
    num_layers,
    n_chunks,
    gate_fallback_mode,
    num_links,
    topology,
    num_iters,
    routing_use_l1_small_for_semaphores=False,
    baseline_chunk_times_s=None,
    perf_margin=None,
    preload_isl=0,
):
    """No-PCC perf/smoke variant of run_chunked_transformer: build the transformer ONCE, then drive the
    full n_chunks-chunk prefill `num_iters` times with return_intermediates=False (no per-layer host
    readback, no PCC). Tokens are the real (longbook) ids from the golden trace when present, else a
    deterministic in-vocab pattern, so this is trace-optional. The KV cache is reused across iterations
    (each chunk overwrites the same [preload_isl, preload_isl + measured_len) region in order).

    `preload_isl` (must be a multiple of CHUNK): treat the cache as already holding this many prior KV
    tokens, so the measured chunks run at absolute KV positions [preload_isl, preload_isl + n_chunks*CHUNK)
    instead of starting at 0. This lets the MLA-vs-FFN ratio be measured at a target KV depth WITHOUT first
    running chunked prefill up to that point. The prior [0, preload_isl) KVPE is preloaded from the golden
    trace (kv_post_transform, block-cyclic; see _preload_kvpe_prefix_from_trace) so the measured chunk
    attends to REAL prior KV — the MoE gate then sees realistic hidden states and routes representatively
    (a zero prefix would degrade routing). preload_isl>0 therefore requires a trace (set PREFILL_TRACE_DIR);
    KV depths beyond the trace length are filled with random KV (still non-degenerate for routing) so larger
    ISLs than the trace can be exercised. NOTE: the sparse KVPE gather trims to the populated KV depth
    (preload_isl + measured chunks, rounded up to whole block-cyclic slabs), so the measured gather cost
    tracks the real valid length and grows with preload_isl (realistic per-depth perf).

    Perf gate: when `baseline_chunk_times_s` is provided (a per-chunk list of baseline medians pulled
    from a known-good CI run), each chunk's measured median must stay within +/- `perf_margin` of its
    baseline; a single `perf_margin` covers every chunk. The table appends the baseline, tolerance band,
    and PASS/FAIL per chunk, and the run fails if any chunk is out of band. When no baseline is given the
    table is record-only (perf-exploration combos)."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")

    def format_duration(seconds: float) -> str:
        return f"{seconds:7.3f}s"

    def print_duration_table(iteration_chunk_times: list[list[float]]) -> list[str]:
        """Log the per-chunk median/stddev table (and, when a baseline is set, the tolerance band +
        PASS/FAIL). Returns the list of human-readable failure messages (empty if all chunks pass or if
        there is no baseline) so the caller can assert after the table has been printed."""
        # Iteration 0 includes compile/JIT effects; exclude it from perf stats.
        samples = iteration_chunk_times[1:]
        if not samples:
            logger.warning("No post-warmup iterations available for chunk timing stats (need num_iters >= 2)")
            return []

        gated = baseline_chunk_times_s is not None
        if gated and len(baseline_chunk_times_s) != n_chunks:
            raise ValueError(
                f"baseline_chunk_times_s has {len(baseline_chunk_times_s)} entries but n_chunks={n_chunks}"
            )
        margin = perf_margin if perf_margin is not None else 0.0

        headers = ["chunk", "median_time", "stddev"]
        if gated:
            headers += ["baseline", "low", "high", "status"]
        rows = []
        failures: list[str] = []
        for chunk_idx in range(n_chunks):
            chunk_samples = [row[chunk_idx] for row in samples]
            median_time = statistics.median(chunk_samples)
            stddev_time = statistics.stdev(chunk_samples) if len(chunk_samples) >= 2 else 0.0
            row = [f"chunk {chunk_idx}", format_duration(median_time), format_duration(stddev_time)]
            if gated:
                baseline = baseline_chunk_times_s[chunk_idx]
                low = baseline * (1.0 - margin)
                high = baseline * (1.0 + margin)
                ok = low <= median_time <= high
                row += [
                    format_duration(baseline),
                    format_duration(low),
                    format_duration(high),
                    "PASS" if ok else "FAIL",
                ]
                if not ok:
                    failures.append(
                        f"chunk {chunk_idx} median {median_time:.3f}s outside "
                        f"baseline {baseline:.3f}s +/- {margin * 100:.1f}% band [{low:.3f}s, {high:.3f}s]"
                    )
            rows.append(row)

        widths = [len(header) for header in headers]
        for row in rows:
            for idx, cell in enumerate(row):
                widths[idx] = max(widths[idx], len(cell))

        def render_row(values: list[str]) -> str:
            return "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(values)) + " |"

        separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

        margin_note = f", baseline gate +/- {margin * 100:.1f}%" if gated else ", record-only (no baseline)"
        logger.info(f"chunk timing stats computed over {len(samples)} iterations (iter 0 omitted){margin_note}")
        logger.info("\n" + separator)
        logger.info(render_row(headers))
        logger.info(separator)
        for row in rows:
            logger.info(render_row(row))
        logger.info(separator)
        return failures

    profiler.clear()
    profiler.start("total_test_time")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"

    chunk_local = CHUNK // sp  # 640
    assert preload_isl % CHUNK == 0, f"preload_isl ({preload_isl}) must be a multiple of CHUNK ({CHUNK})"
    measured_len = n_chunks * CHUNK  # tokens actually run this call
    total_len = preload_isl + measured_len  # logical seq length: preloaded prefix + measured chunks
    assert (
        total_len <= SEQ_CACHE_NOPCC
    ), f"preload_isl {preload_isl} + {n_chunks} chunks ({measured_len}) = {total_len} exceed cache {SEQ_CACHE_NOPCC}"

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = SEQ_CACHE_NOPCC

    logger.info(
        f"chunked transformer (no-PCC): num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"preload_isl={preload_isl} total_len={total_len} cache={SEQ_CACHE_NOPCC} chunk={CHUNK} "
        f"num_iters={num_iters}"
    )

    iteration_chunk_times: list[list[float]] = []

    # Token ids: prefer the real (code_debug/longbook) ids from the golden trace (same source as the PCC
    # test) but never compared here; fall back to a deterministic in-vocab pattern so this stays
    # trace-optional. _resolve_trace_dir raises when the base dir is absent (e.g. the code_debug dataset
    # isn't staged on this host), so swallow that too and fall back -- the verdict is trace-independent.
    vocab_size = config.vocab_size
    try:
        trace_dir = _resolve_trace_dir(variant)
    except FileNotFoundError:
        trace_dir = None
    # preload_isl needs REAL prior KV (for representative MoE routing), which only the golden trace has.
    # KV depths beyond the trace are filled with random KV in the preload (see helper), so a trace is
    # required for preload_isl>0 but preload_isl may exceed the trace length.
    if preload_isl > 0 and (trace_dir is None or not trace_dir.exists()):
        pytest.skip(f"preload_isl={preload_isl} requires a golden trace; none found (set PREFILL_TRACE_DIR)")
    trace_native_len = 0  # native token/KV length of the trace; 0 when no trace
    if trace_dir is not None and trace_dir.exists():
        trace_tokens = _load_metadata_token_ids(trace_dir, total_len)
        trace_native_len = trace_tokens.numel()
        if trace_tokens.numel() < total_len:
            reps = (total_len + trace_tokens.numel() - 1) // trace_tokens.numel()
            trace_tokens = trace_tokens.repeat(reps)[:total_len]
        token_ids_full = trace_tokens % vocab_size
        logger.info(f"no-PCC: loaded {trace_native_len} token ids from trace {trace_dir}")
    else:
        token_ids_full = torch.arange(total_len, dtype=torch.int64) % vocab_size
        logger.info(f"no-PCC: trace not found ({trace_dir}); using synthetic token ids")

    # --- Weights from the prebuilt TTNN cache (empty state_dict when complete). ---
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,  # per-chunk size -> MoE/FFN dispatch buffers
        max_seq_len=SEQ_CACHE_NOPCC,  # KV ring buffer + RoPE cos/sin cache = full no-PCC cache (up to 100k)
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    # Sparse (DSA: glm_5_1 / glm_5_2) requires an UNCOMPRESSED bf16/fp8_e4m3 ROW_MAJOR KVPE cache
    # (sparse_sdpa reads it natively; mla.forward asserts) — NOT the init_kvpe_cache bfloat8_b/TILE
    # default that dense ring_mla wants. Match the cache format to the path (dense variants keep the
    # default). Same distinction as run_chunked_transformer.
    kvpe_dtype_layout = dict(dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) if resolve_has_indexer(config) else {}
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE_NOPCC,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
        **kvpe_dtype_layout,
    )

    # Sparse (DSA) layers read a block-cyclic indexer key cache that is caller-owned and passed into
    # forward, exactly like the KVPE cache. Strided by the compacted full-indexer count
    # (num_full_indexer_layers, >1 for glm_5_2 cross-layer reuse; num_layers without an indexer_types map)
    # so it matches the indexer's cache_batch stride. bf8 TILE. Dense variants get None.
    tt_index_kv_cache = None
    if resolve_has_indexer(config):
        assert getattr(config, "index_head_dim", None) is not None, "sparse config must provide index_head_dim"
        tt_index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=SEQ_CACHE_NOPCC,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_full_indexer_layers(config) or num_layers,
            num_users=1,
            dtype=ttnn.bfloat8_b,
        )

    # preload_isl > 0: seed the prior [0, preload_isl) KVPE + indexer-K from the golden trace so the measured
    # chunk attends to real KV (representative MoE routing) and the indexer scores a real prefix.
    if preload_isl > 0:
        _preload_kvpe_prefix_from_trace(
            tt_kvpe_cache,
            trace_dir,
            variant.prefill_trace_layout,
            num_layers,
            preload_isl,
            trace_native_len,
            sp,
            SEQ_CACHE_NOPCC,
            kvpe_dim,
            config.kv_lora_rank,
            mesh_device,
            sp_axis,
            kvpe_dtype_layout.get("dtype", ttnn.bfloat8_b),
            kvpe_dtype_layout.get("layout", ttnn.TILE_LAYOUT),
        )
        if tt_index_kv_cache is not None:
            _preload_indexer_k_prefix_from_trace(
                tt_index_kv_cache,
                trace_dir,
                variant.prefill_trace_layout,
                config,
                num_layers,
                preload_isl,
                trace_native_len,
                sp,
                SEQ_CACHE_NOPCC,
                config.index_head_dim,
                mesh_device,
                sp_axis,
            )

    # Precompute per-chunk SP-sharded token tiles once (reused across iterations). Chunk-aligned offsets
    # make the block-cyclic rotation degenerate to a plain per-chip reshape.
    chunk_tok_host = []
    for c in range(n_chunks):
        kv_actual = preload_isl + c * CHUNK
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        chunk_tok_host.append(token_ids_full[flat].reshape(sp, 1, chunk_local))

    mesh_device.enable_program_cache()

    # Profiling warmup: run chunk 0 once through all layers so every kernel is JIT-compiled and the
    # program cache is populated BEFORE the measured region. Gated by TT_PREFILL_PROFILE_WARMUP so
    # normal runs are unaffected. Bracketed by PROFILE_WARMUP_START / PROFILE_MEASURE_START signposts;
    # the per-layer post-processor keeps only ops AFTER PROFILE_MEASURE_START, so this compile pass is
    # excluded from the device-time / op2op breakdown.
    if os.environ.get("TT_PREFILL_PROFILE_WARMUP", "0") == "1":
        signpost("PROFILE_WARMUP_START")
        warm_tokens = ttnn.from_torch(
            chunk_tok_host[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )
        # Warm at the measured region's first offset (preload_isl) so the JITted programs match exactly.
        transformer.forward(
            warm_tokens,
            tt_kvpe_cache,
            actual_isl=CHUNK,
            actual_start=preload_isl,
            actual_end=preload_isl + CHUNK,
            cache_user_id=0,
            return_intermediates=False,
            index_kv_cache=tt_index_kv_cache,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(warm_tokens)
        logger.info("[profile] warmup chunk 0 complete (kernels JITted); measured region begins")
        signpost("PROFILE_MEASURE_START")

    profiler.start("tt_forward")
    for it in range(num_iters):
        iter_start = time.time()
        chunk_times: list[float] = []
        for c in range(n_chunks):
            kv_actual = preload_isl + c * CHUNK
            tt_tokens = ttnn.from_torch(
                chunk_tok_host[c],
                device=mesh_device,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
            )
            chunk_start = time.time()
            # forward with return_intermediates=False: nothing is cloned to host, no PCC. Chunked
            # prefill is full-chunk (all positions real) so actual_end is kv_actual + CHUNK; forward
            # uses self.indexed_rope. The small (first_token) return is discarded.
            transformer.forward(
                tt_tokens,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=kv_actual,
                actual_end=kv_actual + CHUNK,
                cache_user_id=0,
                return_intermediates=False,
                index_kv_cache=tt_index_kv_cache,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(tt_tokens)
            chunk_times.append(time.time() - chunk_start)
        iter_total = time.time() - iter_start
        iteration_chunk_times.append(chunk_times)
        logger.info(f"iter {it} done ({n_chunks} chunks) in {iter_total:.3f} seconds")
        # Drop iter 0's per-layer MLA/FFN samples (the compile iteration), same as the chunk-time table.
        if it == 0:
            reset_block_timings()
    profiler.end("tt_forward")

    profiler.end("total_test_time")
    logger.success(
        f"Chunked prefill no-PCC run done (num_layers={num_layers}, n_chunks={n_chunks}, " f"num_iters={num_iters})"
    )
    perf_failures = print_duration_table(iteration_chunk_times)
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")

    # Rough per-layer MLA-vs-FFN split (TT_PREFILL_BLOCK_TIMING=1): host wall-clock, sync-bracketed, so
    # absolutes inflate (syncs serialize) — read the RATIO and per-layer shape, not the totals. Mean +/- std
    # over all recorded forward calls (iter 0 dropped above). Each layer sample = one chunk's block call.
    timings = get_block_timings()
    if timings:

        def _mean_std_ms(samples):
            mean_ms = statistics.mean(samples) * 1000.0 if samples else 0.0
            std_ms = statistics.stdev(samples) * 1000.0 if len(samples) >= 2 else 0.0
            return mean_ms, std_ms

        headers = ["layer", "mla_mean_ms", "mla_std_ms", "moe_mean_ms", "moe_std_ms", "mla%", "moe%"]
        rows = []
        tot_mla = tot_moe = 0.0
        for layer_idx in sorted(timings):
            rec = timings[layer_idx]
            mla_mean, mla_std = _mean_std_ms(rec["mla"])
            moe_mean, moe_std = _mean_std_ms(rec["ffn"])
            both = mla_mean + moe_mean
            tot_mla += mla_mean
            tot_moe += moe_mean
            rows.append(
                [
                    f"{layer_idx}",
                    f"{mla_mean:.2f}",
                    f"{mla_std:.2f}",
                    f"{moe_mean:.2f}",
                    f"{moe_std:.2f}",
                    f"{100.0 * mla_mean / both:.1f}" if both else "-",
                    f"{100.0 * moe_mean / both:.1f}" if both else "-",
                ]
            )
        tot_both = tot_mla + tot_moe
        rows.append(
            [
                "ALL",
                f"{tot_mla:.2f}",
                "-",
                f"{tot_moe:.2f}",
                "-",
                f"{100.0 * tot_mla / tot_both:.1f}" if tot_both else "-",
                f"{100.0 * tot_moe / tot_both:.1f}" if tot_both else "-",
            ]
        )
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
        sep = "-+-".join("-" * w for w in widths)

        def render(vals):
            return " | ".join(v.rjust(widths[i]) for i, v in enumerate(vals))

        n_samples = len(next(iter(timings.values()))["mla"])
        logger.info(
            f"per-layer MLA-vs-MoE mean +/- std host time over {n_samples} chunk-calls (sync-bracketed, iter 0 "
            f"dropped; moe column is the FFN region -- MoE on MoE layers, dense FFN on the first_k_dense layers) "
            f"-- ratio is the signal, absolutes are inflated by the syncs:"
        )
        logger.info("\n" + sep)
        logger.info(render(headers))
        logger.info(sep)
        for r in rows[:-1]:
            logger.info(render(r))
        logger.info(sep)
        logger.info(render(rows[-1]))
        logger.info(sep)

    # Assert AFTER the table is logged so the full per-chunk breakdown is always visible on failure.
    assert not perf_failures, "chunk timing out of baseline tolerance:\n  " + "\n  ".join(perf_failures)


# No-PCC perf/smoke variant: runs the full n_chunks-chunk prefill `num_iters` times with no golden
# trace dependency, no intermediate readback, and no PCC. Requires only the Kimi TTNN weight cache (set
# TT_KIMI_PREFILL_TTNN_CACHE + KIMI_K2_6_HF_MODEL); the golden trace is optional.
@pytest.mark.parametrize("perf_margin", [DEFAULT_PERF_MARGIN], ids=["margin5pct"])
@pytest.mark.parametrize(
    "num_iters", [1, 2, 10, 20, 25], ids=["iters1", "two_iters", "ten_iters", "iters20", "iters25"]
)
@pytest.mark.parametrize(
    "n_chunks",
    [1, 2, 5, 10, 11, 20],
    ids=["chunks1", "chunks2", "chunks5", "chunks10", "chunks_eleven", "chunks20"],
)
# preload_isl (multiple of CHUNK): pretend the cache already holds this many prior KV tokens so the
# measured chunks run at KV depth [preload_isl, preload_isl + n_chunks*CHUNK) WITHOUT first running prefill
# up to that point. Pair with n_chunks=1 to sweep the single-chunk MLA/MoE ratio vs depth. 0 = start from
# empty cache. Depths within the golden trace use real KV; the rest fill random KV beyond the trace (still
# non-degenerate for routing) so larger ISLs than the trace can be measured. Requires a golden trace when >0.
@pytest.mark.parametrize(
    "preload_isl",
    [0, 5 * CHUNK, 10 * CHUNK, 19 * CHUNK],
    ids=["preload0", "preload25k", "preload50k", "preload95k"],
)
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                # L1_SMALL region for the MoE routing all-gather's semaphores (see TtMoERoutingSetup).
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.timeout(0)
def test_kimi_prefill_transformer_chunked_no_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    num_links,
    topology,
    perf_margin,
    preload_isl,
):
    if preload_isl + n_chunks * CHUNK > SEQ_CACHE_NOPCC:
        pytest.skip(f"preload_isl {preload_isl} + {n_chunks} chunks exceeds the {SEQ_CACHE_NOPCC}-token cache")
    # Gate against the CI baseline only for the exact config we have a recorded number for; every other
    # combo in the sweep stays record-only (baseline None -> print_duration_table does not assert). The
    # baseline is only meaningful at preload_isl=0 (the recorded runs started from an empty cache).
    baseline_chunk_times_s = (
        KIMI_NO_PCC_BASELINE_CHUNK_TIMES_S.get((num_layers, n_chunks, num_iters)) if preload_isl == 0 else None
    )
    run_chunked_transformer_no_pcc(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        num_iters,
        routing_use_l1_small_for_semaphores=True,
        baseline_chunk_times_s=baseline_chunk_times_s,
        perf_margin=perf_margin,
        preload_isl=preload_isl,
    )


# DeepSeek counterpart of the no-PCC perf sweep above: same chunked driver, deepseek_v3_d_p variant
# (DeepSeekV3Config fabric payload, no L1_SMALL routing semaphores). Used to compare DeepSeek vs Kimi
# chunked-prefill perf at matched ISL (n_chunks x CHUNK) and num_layers.
@pytest.mark.parametrize(
    "num_iters", [1, 2, 10, 20, 25], ids=["iters1", "two_iters", "ten_iters", "iters20", "iters25"]
)
@pytest.mark.parametrize(
    "n_chunks",
    [1, 2, 5, 10, 11, 20],
    ids=["chunks1", "chunks2", "chunks5", "chunks10", "chunks_eleven", "chunks20"],
)
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(
                    max_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE
                ),
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["deepseek_v3_d_p"], indirect=True, ids=["deepseek_v3"])
@pytest.mark.skipif(not is_blackhole(), reason="DeepSeek prefill requires Blackhole")
@pytest.mark.timeout(0)
def test_ds_prefill_transformer_chunked_no_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    num_links,
    topology,
):
    run_chunked_transformer_no_pcc(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        num_iters,
        routing_use_l1_small_for_semaphores=False,
    )


# GLM (glm_5_1 / glm_5_2) counterpart of the no-PCC perf sweep: same chunked driver, sparse (DSA) path.
# Reports end-to-end prefill time — the per-iteration total ("iter {it} done ... in Xs") and the per-chunk
# median/stddev table — for the two GLM variants at matched ISL (n_chunks x CHUNK) and num_layers. No PCC;
# the empty-cache case (preload0) needs no golden trace, preload_isl > 0 requires it. Uses the GLM fabric payload + on-device fp32 gate + L1_SMALL
# routing semaphores, exactly like test_glm_prefill_transformer_chunked. glm_5_2 additionally exercises the
# DSA cross-layer indexer reuse per chunk. Requires the GLM TTNN weight cache (set the variant's cache env).
@pytest.mark.parametrize(
    "num_iters", [1, 2, 10, 20, 25], ids=["iters1", "two_iters", "ten_iters", "iters20", "iters25"]
)
@pytest.mark.parametrize(
    "n_chunks",
    [1, 2, 5, 10, 11, 20],
    ids=["chunks1", "chunks2", "chunks5", "chunks10", "chunks_eleven", "chunks20"],
)
# preload_isl (multiple of CHUNK): pretend the cache already holds this many prior KV tokens so the
# measured chunks run at KV depth [preload_isl, preload_isl + n_chunks*CHUNK) WITHOUT first running prefill
# up to that point. Pair with n_chunks=1 to sweep single-chunk ratio vs depth. 0 = start from empty cache.
# Chunk-multiple KV depths to seed before measuring. Depths within the ~55k golden trace use real KV; the
# rest (e.g. preload95k) fills random KV beyond the trace so larger ISLs than the trace can be measured.
# Requires a golden trace when >0.
@pytest.mark.parametrize(
    "preload_isl",
    [0, 5 * CHUNK, 10 * CHUNK, 19 * CHUNK],
    ids=["preload0", "preload25k", "preload50k", "preload95k"],
)
@pytest.mark.parametrize("num_layers", [1, 10, 78], ids=["L1", "L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                # FABRIC_2D + Topology.Linear: the production transport for the GLM chunked-prefill
                # perf measurement. RELAXED_INIT is required for FABRIC_2D bring-up on BH Galaxy.
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
                # Small L1_SMALL region for the MoE routing all-gather's global semaphores
                # (use_l1_small_for_semaphores); see test_glm_prefill_transformer_chunked.
                "l1_small_size": 512,
            },
            2,
            ttnn.Topology.Linear,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="mesh-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_1", "glm_5_2"], indirect=True, ids=["glm51", "glm52"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked_no_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    num_links,
    topology,
    preload_isl,
):
    if preload_isl + n_chunks * CHUNK > SEQ_CACHE_NOPCC:
        pytest.skip(f"preload_isl {preload_isl} + {n_chunks} chunks exceeds the {SEQ_CACHE_NOPCC}-token cache")
    run_chunked_transformer_no_pcc(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        num_iters,
        routing_use_l1_small_for_semaphores=True,
        preload_isl=preload_isl,
    )
