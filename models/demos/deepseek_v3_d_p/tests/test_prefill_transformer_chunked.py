# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Chunked-prefill test for TtPrefillTransformer (DeepSeek V3, N layers).

Runs chunked prefill through a specified number of layers (5, 10, or 61 = full model), processing
the sequence in 5*1024 = 5120-token chunks into a KV cache for ONE user of length 55*1024 = 56320.
Each layer attends to its own cache slot; chunks are driven in order so a layer's KV is populated
before the next chunk reads it. The MoE path is unchanged from the single-shot transformer; only
the MLA path runs chunked (is_chunked=True).

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
from models.demos.common.prefill.runners.runner_utils import resolve_trace_dir
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_1_config import GLM51Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.reference.mistral_small_4_config import MistralSmall4Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tt.mla.indexer import (
    full_indexer_rank,
    get_fused_ring_host_timing,
    normalized_hadamard_matrix,
    reset_fused_ring_host_timing,
    resolve_has_indexer,
)
from models.demos.deepseek_v3_d_p.tt.mla.rope import ChunkMetadata, write_chunk_metadata
from models.demos.deepseek_v3_d_p.tt.mla.utils import (
    blockcyclic_cache_host,
    blockcyclic_positions,
    rotated_chip_positions,
)
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import get_block_timings, reset_block_timings
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.prefill_summary_utils import emit_summary, render_table
from models.demos.deepseek_v3_d_p.utils.smbus_telemetry import is_high_power
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from models.demos.deepseek_v3_d_p.utils.test_utils import (
    cache_half_pccs,
    gather_cache_tp0,
    interleave_pe,
    read_sharded_rows,
    token_normalized,
    unrotate_cache_layer,
)
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = PREFILL_CHUNK_TOKENS  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (1 user)
# Default KV cache for the no-PCC perf sweeps; callers needing a longer one pass run_chunked_transformer_
# updated(seq_cache=...). Kept separate from SEQ_CACHE so the PCC tests (which assert against 55*1024)
# are untouched.
SEQ_CACHE_NOPCC = 100 * 1024  # 102400 KV cache length (1 user)


def _resolve_trace_dir(variant) -> Path:
    """Golden chunked-prefill trace dir for `variant`: PREFILL_TRACE_DIR overrides the variant's
    prefill_trace_default."""
    configured = os.environ.get("PREFILL_TRACE_DIR") or variant.test_prefill_trace_default
    if not configured:
        # No golden recorded (adapter default None): a path that cannot exist, so callers' existing
        # `if not trace_dir.exists(): pytest.skip(...)` reports it instead of TypeError from Path(None).
        return Path(f"<unset PREFILL_TRACE_DIR for {variant.name}>")
    return resolve_trace_dir(configured)


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

# 15k (15360) CI-sized variant of the above. Cost here tracks the CHUNK count, not the token count --
# every chunk is a full CHUNK-wide padded tile regardless of its isl -- so 6 chunks runs ~3x faster
# than full55k's 18 while keeping the properties that make the rotated path fail:
#   * cumulative starts 0, 2592, 4160, 9280, 10080, 13440 -- only chunk 0 is slab-aligned, so 5 of 6
#     chunks have the rotation actually active (it degenerates to the identity when slab-aligned);
#   * starts land in 3 different slabs (<5120, 5120-10239, >=10240), so the rotation's slab-step term
#     is exercised, not just the within-slab offset;
#   * 2592 / 1568 / 800 / 3360 / 1920 are non-1024-aligned -> mid-tile rotation offsets;
#   * 5120 keeps one exactly-full (unpadded) chunk in the mix.
_PADDED_MID_15K = [2592, 1568, 5120, 800, 3360, 1920]  # sum == 15 * 1024
assert sum(_PADDED_MID_15K) == 15 * 1024 and all(v % 32 == 0 and 0 < v <= CHUNK for v in _PADDED_MID_15K)

# Per-chunk per-layer threshold; error accumulates with depth, so this matches the single-shot
# transformer's device-gate trace bar (TRACE_PCC_THRESHOLD_DEVICE_BF16 = 0.88). Calibrate + tighten.
LAYER_PCC_THRESHOLD = 0.88
# Variants gating hidden states on the RMS-normalised PCC: Mistral's massive activation channels
# make the raw score a measurement of outliers (see test_prefill_transformer._compare_intermediate_pcc).
_NPCC_GATED_VARIANTS = {"mistral_small_4"}


# Floors for the deep KV / indexer-K cache PCC. Set at the observed L78 minimum (not below it) so a
# future regression fails the test. KVPE nope bottoms ~0.86 (glm_5_2 @L75); indexer-K nope 0.952
# (glm_5_1 @L52; glm_5_1 captures all 78 layers, glm_5_2's 0-2+every-4th subsample only reaches 0.980).
KV_CACHE_PCC_THRESHOLD = 0.85
INDEXER_K_PCC_THRESHOLD = 0.95

# Per-chunk baseline medians (seconds) for the perf gate, derived from completed CI runs. Keyed by
# (num_layers, n_chunks, num_iters) so only exact configs with CI numbers are gated; every other combo
# in the sweep stays record-only. Each list has one entry per chunk (index c == chunk c). Recalibrate
# from the per-chunk median across multiple independent Galaxy runs rather than copying one run's
# measurements directly.
#
# Traced and untraced get SEPARATE tables and SEPARATE margins, selected by mode in
# `kimi_chunked_perf_gate` -- a traced baseline can never gate an untraced run or vice versa. The two
# are different regimes, not a small delta: traced measures 0.6-0.95 s/chunk (a ramp, since chunk c
# attends to KV[0:c*CHUNK]) while untraced is a flat ~1.04 s/chunk, host-dispatch bound so the op2op
# gap swamps the depth ramp entirely.
KIMI_TRACED_BASELINE_CHUNK_TIMES_S = {
    # test_kimi_prefill_transformer_chunked_perf[...-L61-preload0-chunks_eleven-ten_iters-traced]
    # (55k / code_debug). Re-centered 2026-08-29: 2D matmul program configs on the shared expert and
    # the latent projections took 2-5% off every chunk and 7/11 fell out the bottom of the old band --
    # the baseline was stale, not the margin too tight. The saving is device-side, so it lands here and
    # nowhere else: the untraced twin is host-dispatch bound at ~1.04 s/chunk and did not move, passing
    # its own gate in the same run.
    #
    # Per-chunk medians of run 33251442925/job 99098593625 verbatim. ONE run, where the superseded
    # value carried a second run agreeing to <=0.010 s -- traced replay has the device as its only
    # noise source and per-chunk stddev here is 0.000-0.003 s, but cross-check the next green run.
    (61, 11, 10): [
        0.519,
        0.521,
        0.569,
        0.597,
        0.631,
        0.665,
        0.683,
        0.725,
        0.777,
        0.816,
        0.855,
    ],
}
KIMI_UNTRACED_BASELINE_CHUNK_TIMES_S = {
    # test_kimi_prefill_transformer_chunked_perf[...-L61-preload0-chunks_eleven-ten_iters-notrace]
    # (55k / code_debug), 2026-08-21 on an 8x4 galaxy, with the routed experts folded into one program
    # on a Fabric2D mesh. One run's per-chunk medians -- unlike the traced twin, a single run is thin
    # evidence here, so re-cut this from a set of green runs the first time it disagrees.
    #
    # WITHIN a run the untraced spread is huge -- per-chunk stddev reaches 0.33 s (~30%), because every
    # iteration re-dispatches every op from host and pays a fresh, variable op2op gap. The MEDIAN of the
    # 9 post-warmup iterations is not: on the previous baseline no chunk median across 32 recorded runs
    # landed further than 3.2% from its median-of-runs value. So the gate is on the median, with
    # UNTRACED_PERF_MARGIN rather than the traced 3%.
    #
    # If this goes flaky, re-center on the median over several runs before widening. Widening to 10% is
    # the fallback after that -- a band that needs more than 10% is a regression, not noise.
    (61, 11, 10): [1.043, 1.036, 1.037, 1.047, 1.044, 1.034, 1.034, 1.034, 1.046, 1.041, 1.044],
}
# Per-mode +/- tolerance band around each baseline chunk median (fraction). Traced replays a captured
# program, so the device is its only noise source; untraced re-dispatches from host every iteration and
# carries the op2op gap, so it needs the wider band. Overridable per test via the perf_margin pytest
# argument (None = use the mode default; see test_prefill_block_perf.py's `margin` column).
#
# 5% untraced is deliberately tight: the worst per-chunk deviation over all 32 recorded runs is 3.2%,
# so CI noise already spends ~2/3 of the band. That is the intended bar -- catch a >5% eager-dispatch
# regression -- but it leaves little slack, so triage a failure as noise-vs-regression (compare the
# other chunks and the traced twin from the same run) before touching the number.
TRACED_PERF_MARGIN = 0.03
UNTRACED_PERF_MARGIN = 0.05

# Deepest config whose per-layer PCC is asserted; deeper runs (L61) stay record-only until their
# accumulation headroom is pinned.
GATED_LAYER_DEPTH = 10
# KV-cache PCC bar for the trace-correctness tests (_record_kv_cache_pcc with
# assert_layer_depth=GATED_LAYER_DEPTH): the KV the traced forward wrote vs the golden
# kv_post_transform, over the SHALLOW gated layers only. Distinct from KV_CACHE_PCC_THRESHOLD above,
# which is the full-depth floor (deep bf8_b layers drift well below this bar).
TRACE_KV_CACHE_PCC_THRESHOLD = 0.96


def _load_metadata_token_ids(trace_dir: Path, total_len: int, require_full: bool = False) -> torch.Tensor:
    """Input token ids from a golden, truncated to total_len. `require_full` for the PCC callers: a
    golden shorter than the row indexes past every golden tensor, not just this one (Mistral's is
    15,360, so full55k raises IndexError), and a missing prerequisite should skip rather than fail.
    The no-PCC caller tiles a short golden instead -- it compares nothing."""
    with open(trace_dir / "metadata.json") as f:
        md = json.load(f)
    ids = md["token_ids"][:total_len]
    if require_full and len(ids) < total_len:
        pytest.skip(f"golden {trace_dir} has {len(ids)} tokens; this row needs {total_len}")
    return torch.tensor(ids, dtype=torch.int64)


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
    trace_dir,
    layout,
    tt_kvpe_cache,
    mesh_device,
    sp,
    num_layers,
    seq_len_cache,
    total_len,
    kv_lora,
    assert_threshold=KV_CACHE_PCC_THRESHOLD,
    assert_layer_depth=None,
    return_per_layer=False,
):
    """Gather the device KV cache, un-rotate the block-cyclic layout, and PCC each layer's valid region
    [:total_len] against the golden kv_post_transform trace ([nope | pe], the pe half re-based to the
    device Meta interleave via cache_half_pccs). Per-layer cache — slot == layer.

    Returns the min PCC across all layers (or `(min, per_layer_dict)` with return_per_layer). The min
    is asserted >= `assert_threshold`; pass None to make the check record-only. With
    `assert_layer_depth` set, only layers 0..assert_layer_depth (inclusive) are asserted — deeper
    layers are recorded only, mirroring the decoder-output GATED_LAYER_DEPTH policy (deep KV PCC
    drifts under bf8_b)."""
    logger.info("Device KV cache vs golden kv_post_transform:")
    cache_full = gather_cache_tp0(tt_kvpe_cache.storage, mesh_device)  # [num_layers, seq_len_cache, kvpe]
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    cache_min_pcc = {}
    for i in range(num_layers):
        dev_cache = unrotate_cache_layer(cache_full[i], p, total_len)
        g_post = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, total_len)
        pcc_nope, pcc_pe = cache_half_pccs(g_post, dev_cache, kv_lora, pe_interleave=True)
        cache_min_pcc[i] = min(pcc_nope, pcc_pe)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f}")
        if assert_threshold is not None and cache_min_pcc[i] < assert_threshold:
            logger.warning(f"  KV cache layer {i} PCC {cache_min_pcc[i]:.6f} below {assert_threshold}")
    kv_min = min(cache_min_pcc.values())
    logger.info(f"KV cache min PCC across layers: {kv_min:.6f}")
    if assert_threshold is not None:
        if assert_layer_depth is not None:
            gated_min = min(v for i, v in cache_min_pcc.items() if i <= assert_layer_depth)
            logger.info(
                f"KV cache min PCC over asserted layers 0..{assert_layer_depth}: {gated_min:.6f} "
                f"(layers >{assert_layer_depth} recorded only)"
            )
            assert (
                gated_min >= assert_threshold
            ), f"KV cache min PCC {gated_min:.6f} (layers 0..{assert_layer_depth}) < {assert_threshold}"
        else:
            assert kv_min >= assert_threshold, f"KV cache min PCC {kv_min:.6f} < {assert_threshold}"
    if return_per_layer:
        return kv_min, cache_min_pcc
    return kv_min


def _record_indexer_k_cache_pcc(
    trace_dir, layout, tt_index_kv_cache, mesh_device, sp, num_layers, seq_len_cache, total_len, config
):
    """Gather the device DSA indexer-K cache, un-rotate the block-cyclic layout, and PCC each captured
    layer's valid region [:total_len] against the golden dsa/indexer_k trace. The index_head_dim key is
    [rope | nope] (rope = first half, indexed-RoPE; nope = second half, no rope); BOTH compare directly
    because GLM's indexer RoPE is natively interleaved and the vLLM golden stores that same basis
    (verified on device: the half-split reindex gives ~0 PCC, direct gives ~0.9999). Same gather/un-rotate
    as the KVPE cache (caller-owned tensor, ConcatMesh2dToTensor dims=(2,1), blockcyclic_positions).
    indexer_k is captured for a subset of layers (glm_5_1: all; glm_5_2: 0-2 + every 4th) — layers without
    a golden are skipped. GLM DSA variants only."""
    logger.info("Device indexer-K cache vs golden dsa/indexer_k:")
    cache_full = gather_cache_tp0(tt_index_kv_cache, mesh_device)  # [full layers built, T, D]
    layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    if not layers:
        logger.info("  (no indexer_k golden layers present -- skipping)")
        return
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    rope = config.index_head_dim // 2  # [rope | nope]
    index_hadamard = normalized_hadamard_matrix(config.index_head_dim).float()
    idx_min_pcc = {}
    for i in layers:
        # Compact index cache (GLM-5.2 cross-layer reuse): layer i's slot is its full-indexer rank, not i
        # (rank == i for glm_5_1, where every layer is full). Matches the indexer's own write addressing.
        dev_cache = unrotate_cache_layer(cache_full[full_indexer_rank(config, i)], p, total_len)
        dev_cache = (dev_cache.float() @ index_hadamard).to(torch.bfloat16)
        g = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, total_len)
        pcc_rope, pcc_nope = cache_half_pccs(g, dev_cache, rope, pe_interleave=False)
        idx_min_pcc[i] = min(pcc_nope, pcc_rope)
        logger.info(f"  indexer cache layer {i} PCC: nope={pcc_nope:.6f} rope={pcc_rope:.6f}")
        if idx_min_pcc[i] < INDEXER_K_PCC_THRESHOLD:
            logger.warning(f"  indexer-K cache layer {i} PCC {idx_min_pcc[i]:.6f} below {INDEXER_K_PCC_THRESHOLD}")
    idx_min = min(idx_min_pcc.values())
    logger.info(f"Indexer-K cache min PCC across {len(layers)} captured layers: {idx_min:.6f}")
    assert idx_min >= INDEXER_K_PCC_THRESHOLD, f"Indexer-K cache min PCC {idx_min:.6f} < {INDEXER_K_PCC_THRESHOLD}"


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
    ttnn.copy_host_to_device_tensor(cache_host_tt, tt_kvpe_cache.storage)
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
    full_indexer_rank(config, i), and the cache is strided by the full-layer count over the built
    layers. The golden
    index_head_dim key is [rope | nope] already in the device's interleaved RoPE basis (GLM). Apply the
    decode-compatible Hadamard before writing it; no RoPE re-interleave is needed (unlike KVPE's k_pe
    half-split -> interleaved). Rows past the trace are random (timing-representative). Mirrors
    _preload_kvpe_prefix_from_trace otherwise."""
    full_layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    if not full_layers:
        logger.info(f"no indexer_k golden in trace {trace_dir}; leaving the indexer prefix zero")
        return
    num_slots = full_indexer_rank(config, num_layers)
    real_len = min(preload_isl, trace_len)
    rand_len = preload_isl - real_len
    logger.info(
        f"Preloading {preload_isl}-token indexer-K prefix into {len(full_layers)} full-indexer slot(s) "
        f"({real_len} real from trace, {rand_len} random beyond the trace)"
    )
    cache_host = torch.zeros(num_slots, 1, seq_len_cache, index_head_dim, dtype=torch.bfloat16)
    index_hadamard = normalized_hadamard_matrix(index_head_dim).float()
    gen = torch.Generator().manual_seed(2345)  # deterministic random tail (distinct from the KVPE seed)
    for i in full_layers:
        idx_prior = torch.randn(preload_isl, index_head_dim, generator=gen).to(torch.bfloat16)
        if real_len > 0:
            real = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, real_len)
            idx_prior[:real_len] = real.to(torch.bfloat16)
        idx_prior = (idx_prior.float() @ index_hadamard).to(torch.bfloat16)
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

    token_ids_full = _load_metadata_token_ids(trace_dir, total_len, require_full=True)

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
        is_chunked=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BFP8_TILE,
        hf_config=config,
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
            # Log every per-layer/per-chunk PCC so a failure localises to the first bad (chunk, layer);
            # the run-wide minimum is asserted after the loop.
            logger.info(f"  chunk {c} (kv_actual={kv_actual} isl={isl}) layer {i} PCC: {pcc:.6f}")
            if pcc < LAYER_PCC_THRESHOLD:
                logger.warning(f"  chunk {c} layer {i} PCC {pcc:.6f} below {LAYER_PCC_THRESHOLD}")
        logger.info(f"  chunk {c} done (kv_actual={kv_actual} isl={isl}, {num_layers} layers)")
    profiler.end("tt_forward")

    logger.info("Per-layer min PCC across chunks:")
    for i in range(num_layers):
        logger.info(f"  layer {i}: {layer_min_pcc[i]:.6f}")

    # Asserted at every depth, full-depth included: a routing/layout regression here (e.g. a wrong
    # per-chip real-token split under rotation) collapses PCC, and a record-only run would go green.
    overall_min = min(layer_min_pcc.values())
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
    preload_isl=0,
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
    assert preload_isl % CHUNK == 0, f"preload_isl ({preload_isl}) must be a multiple of CHUNK ({CHUNK})"
    measured_len = n_chunks * CHUNK
    total_len = preload_isl + measured_len
    assert (
        total_len <= SEQ_CACHE
    ), f"preload_isl {preload_isl} + {n_chunks} chunks ({measured_len}) = {total_len} exceed cache {SEQ_CACHE}"

    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = SEQ_CACHE

    logger.info(
        f"chunked transformer: num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"preload_isl={preload_isl} total_len={total_len} cache={SEQ_CACHE} chunk={CHUNK}"
    )

    token_ids_full = _load_metadata_token_ids(trace_dir, total_len, require_full=True)

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
        is_chunked=True,
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
    cache_format = MlaKvCacheFormat.BF16_RM if has_indexer else MlaKvCacheFormat.BFP8_TILE
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    # Sparse (DSA) layers read a block-cyclic indexer key cache that is caller-owned and passed into
    # forward, exactly like the KVPE cache. It is user-major layer-stacked
    # [num_users*index_cache_layers, 1, T, D_idx], so the indexer addresses slot
    # user*index_cache_layers + cache_layer_idx. Unlike the per-layer KVPE cache, the indexer stride is the
    # COMPACTED full-indexer count over the layers this instance builds — GLM-5.2 "shared" layers reuse a
    # "full" layer's cache and get no slot of their own, and full_indexer_rank returns num_layers unchanged
    # without an indexer_types map. bf8 (half the memory, top-k within bf16 noise). Dense variants get None.
    tt_index_kv_cache = None
    if has_indexer:
        # A sparse config must carry index_head_dim; assert rather than silently defaulting so a
        # misconfigured (missing-field) sparse setup fails loudly with a clear message.
        assert getattr(config, "index_head_dim", None) is not None, "sparse config must provide index_head_dim"
        index_cache_layers = full_indexer_rank(config, num_layers)
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

    if preload_isl > 0:
        trace_native_len = token_ids_full.numel()
        _preload_kvpe_prefix_from_trace(
            tt_kvpe_cache,
            trace_dir,
            layout,
            num_layers,
            preload_isl,
            trace_native_len,
            sp,
            SEQ_CACHE,
            kvpe_dim,
            config.kv_lora_rank,
            mesh_device,
            sp_axis,
            cache_format.storage_dtype,
            cache_format.storage_layout,
        )
        if tt_index_kv_cache is not None:
            _preload_indexer_k_prefix_from_trace(
                tt_index_kv_cache,
                trace_dir,
                layout,
                config,
                num_layers,
                preload_isl,
                trace_native_len,
                sp,
                SEQ_CACHE,
                config.index_head_dim,
                mesh_device,
                sp_axis,
            )

    mesh_device.enable_program_cache()

    # min PCC per layer across chunks (for the summary)
    layer_min_pcc = {i: 1.0 for i in range(num_layers)}

    profiler.start("tt_forward")
    for c in range(n_chunks):
        kv_actual = preload_isl + c * CHUNK  # chunk-aligned -> rotation degenerates
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
            logger.info(f"  chunk {c} layer {i} PCC: {pcc:.6f}")
            if pcc < LAYER_PCC_THRESHOLD:
                logger.warning(f"  chunk {c} layer {i} PCC {pcc:.6f} below {LAYER_PCC_THRESHOLD}")
        logger.info(f"  chunk {c} done ({num_layers} layers)")
    profiler.end("tt_forward")

    logger.info("Per-layer min PCC across chunks:")
    for i in range(num_layers):
        logger.info(f"  layer {i}: {layer_min_pcc[i]:.6f}")

    overall_min = min(layer_min_pcc.values())
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
):
    topology = per_axis_topology(device_params["fabric_config"])
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
):
    topology = per_axis_topology(device_params["fabric_config"])
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


# Execution modes for the padded chunked test (pytest param `mode`). Both run the SAME splits through
# the SAME harness (run_chunked_transformer_padded_trace) and assert the same thing — per-layer
# KV-cache PCC vs the golden — so the pair is a direct trace-vs-no-trace comparison:
#   notrace — the harness's "scalar" variant: per-chunk actual_start/actual_end passed as host
#             scalars (metadata=None), forward run eagerly per split.
#   traced  — the metadata variant: per-chunk scalars read on-device from the metadata tensors, and
#             the forward captured ONCE as a ttnn trace and replayed per split.
#
# NOTE when selecting with -k: "notrace" CONTAINS the substring "trace", so `-k trace` matches BOTH
# modes. Use `-k notrace` / `-k traced` to pin one.
_PADDED_MODES = ["notrace", "traced"]


@pytest.mark.parametrize("mode", _PADDED_MODES, ids=_PADDED_MODES)
@pytest.mark.parametrize("splits", [_PADDED_MID_15K, _PADDED_FULL_55K], ids=["mid15k", "full55k"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            # L1_SMALL holds the routing semaphores plus sparse-MLA high-bandwidth-gather semaphores.
            torus_xy_device_params(
                fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi_k2_6"])
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
    mode,
):
    """Padded/rotated chunked prefill, traced vs untraced (see _PADDED_MODES). Both modes exercise
    padding-aware MoE over the same `splits` and assert per-layer KV-cache PCC against the golden, so
    a traced-vs-notrace diff isolates the trace/metadata path itself rather than harness differences."""
    topology = per_axis_topology(device_params["fabric_config"])
    common = (
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        splits,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
    )
    # The harness names its untraced variant "scalar"; here the only question is trace or no trace.
    run_chunked_transformer_padded_trace(
        *common,
        routing_use_l1_small_for_semaphores=True,
        mode="traced" if mode == "traced" else "scalar",
    )


# Mistral counterpart of the padded/rotated chunked row above. Three deviations, all forced by the
# config rather than chosen:
#   * GPT_DEVICE, not DEVICE_FP32. moe_grouped_topk.cpp's parse_score_func takes only sigmoid and
#     sqrtsoftplus, so the sigmoid device gate cannot express Mistral's softmax -> top-4 ->
#     renormalize router. DEVICE_FP32 would apply a sigmoid affinity and produce wrong routing
#     weights without failing -- no crash, and invisible to a KV-only assertion.
#   * L1/L36, since the model is 36 layers deep (Kimi is 61).
#   * The golden is host-generated (adapter's prefill_trace_default; PREFILL_TRACE_DIR overrides).
#     It runs the same torch path the device is compared against, so it localises per layer but is
#     NOT independent of the reference.
# It also needs a TTNN weight cache for `num_layers`; the row asserts completeness rather than
# building one, so stage the cache first (TT_MISTRAL4_PREFILL_TTNN_CACHE).
@pytest.mark.parametrize("mode", _PADDED_MODES, ids=_PADDED_MODES)
@pytest.mark.parametrize("splits", [_PADDED_MID_15K, _PADDED_FULL_55K], ids=["mid15k", "full55k"])
@pytest.mark.parametrize("num_layers", [1, 36], ids=["L1", "L36"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 targets the Blackhole galaxy")
@pytest.mark.timeout(0)
def test_mistral4_prefill_transformer_chunked_padded(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    splits,
    num_links,
    mode,
):
    """Padded/rotated chunked prefill for Mistral, traced vs untraced, asserted per-layer against the
    golden KV cache.

    The llama4 query temperature is NOT exercised at L1: the transformer is built kv_only_last_layer,
    so at one layer the only layer is the kv-only one -- it runs attn_norm and the KV branch, never
    _q_stem, and the temperature scales Q alone. L36 does reach it (layers 0..34 are full), above
    8192. test_mla.py and tests/torch/test_mistral_small_4_mla_reference.py cover the term itself."""
    topology = per_axis_topology(device_params["fabric_config"])
    run_chunked_transformer_padded_trace(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        splits,
        GateComputeMode.GPT_DEVICE,
        num_links,
        topology,
        routing_use_l1_small_for_semaphores=True,
        mode="traced" if mode == "traced" else "scalar",
    )


@pytest.mark.parametrize("use_trace", [False, True], ids=["notrace", "traced"])
@pytest.mark.parametrize("num_iters", [2], ids=["two_iters"])
# Zero-padded: `-k chunks5` would substring-match chunks51 (the rows below hack around the same
# collision with the ad-hoc id `chunks_eleven`).
@pytest.mark.parametrize(
    "n_chunks",
    [1, 2, 5, 10, 20, 51],
    ids=["chunks01", "chunks02", "chunks05", "chunks10", "chunks20", "chunks51"],
)
# L1 is the perf-analysis lever, not a smoke row: it keeps a Tracy capture of one layer tractable
# (0.09 MB / 1 segment, against 17 MB / 71 for L36).
@pytest.mark.parametrize("num_layers", [1, 36], ids=["L1", "L36"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            # trace_region_size: without it the captured buffers fall back to general DRAM and
            # trace_bytes() reads 0.
            torus_xy_device_params(
                fabric_payload_size=MistralSmall4Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["mistral_small_4"], indirect=True, ids=["mistral4"])
@pytest.mark.skipif(not is_blackhole(), reason="Mistral Small 4 prefill requires Blackhole")
@pytest.mark.timeout(0)
def test_mistral4_prefill_transformer_chunked_no_pcc(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    use_trace,
    num_links,
):
    """Long-context chunked-prefill TIMING at n_chunks x CHUNK tokens: no PCC and no golden trace
    (synthetic in-vocab ids at preload_isl=0), which is what makes rows past Mistral's 15,360-token
    golden reachable. Accuracy lives in test_mistral4_prefill_transformer_chunked_padded.

    Two traps. Run `traced`: untraced, per-chunk time is flat (~0.67 s/chunk) at any KV depth because
    it measures host dispatch, which hides attention growth entirely. And read the PER-CHUNK MEDIAN
    from the rendered table, not `iter N done ... in Xs` -- the iteration total carries fixed overhead
    that does not scale with the window, so window/iter_total understates throughput by 17-30%.
    """
    run_chunked_transformer_updated(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.GPT_DEVICE,
        num_links,
        per_axis_topology(device_params["fabric_config"]),
        num_iters,
        routing_use_l1_small_for_semaphores=True,
        use_trace=use_trace,
        # chunks51 is 261,120 tokens; sized per-row so the longest sweep needs no env var and the
        # other variants' baselines keep the 100k default.
        seq_cache=max(SEQ_CACHE_NOPCC, n_chunks * CHUNK),
    )


# GLM-5.1 variants
# ---------------------------------------------------------------------------
# Same chunked-prefill validation as the DeepSeek/Kimi tests, for the glm_5_1 / glm_5_2 variants and the
# on-device gate (GateComputeMode.DEVICE_FP32 — GLM's noaux_tc gate uses the grouped-topk fp32 device path)
# + GLM fabric payload (5.1 == 5.2 dims). glm_5_2 additionally exercises DSA indexer reuse per chunk: each
# chunk is one forward, so full layers recompute that chunk's top-k and shared layers reuse it within the
# chunk. Golden = each variant's vLLM 55k structured trace (chunked_group_a_v1; via test_prefill_trace_default,
# override with PREFILL_TRACE_DIR).


@pytest.mark.parametrize(
    "n_chunks, preload_isl",
    [(1, 10 * CHUNK), (11, 0)],
    ids=["warm_cache", "cold_cache"],
)
@pytest.mark.parametrize("num_layers", [1, 10, 78], ids=["L1", "L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            # Routing consumes 512 B; leave 256 B for sparse-MLA high-bandwidth-gather semaphores
            # and retain the existing reserve for other needs.
            torus_xy_device_params(fabric_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE, l1_small_size=1152),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
    preload_isl,
    num_links,
):
    topology = per_axis_topology(device_params["fabric_config"])
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
        preload_isl=preload_isl,
    )


def run_chunked_transformer_updated(
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
    check_pcc=False,
    use_trace=False,
    seq_cache=None,
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

    def print_duration_table(iteration_chunk_times: list[list[float]]) -> tuple[list[str], list[str]]:
        """Log the per-chunk median/stddev table (and, when a baseline is set, the tolerance band +
        PASS/FAIL). Returns (failures, table_lines): failures are the human-readable out-of-band messages
        (empty if all chunks pass or there is no baseline) so the caller can assert after the table is
        printed; table_lines is the rendered table for the caller to emit as a summary."""
        # Iteration 0 includes compile/JIT effects; exclude it from perf stats.
        samples = iteration_chunk_times[1:]
        if not samples:
            logger.warning("No post-warmup iterations available for chunk timing stats (need num_iters >= 2)")
            return [], []

        gated = baseline_chunk_times_s is not None
        if gated and len(baseline_chunk_times_s) != n_chunks:
            raise ValueError(
                f"baseline_chunk_times_s has {len(baseline_chunk_times_s)} entries but n_chunks={n_chunks}"
            )
        if gated and perf_margin is None:
            # perf_margin defaults to None ("use the mode default", resolved by kimi_chunked_perf_gate).
            # Reaching the gate with it still None means a caller wired up a baseline without a band;
            # falling through to 0.0 would collapse the band to zero width and fail every chunk.
            raise ValueError("baseline_chunk_times_s was given without a perf_margin")
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

        margin_note = f", baseline gate +/- {margin * 100:.1f}%" if gated else ", record-only (no baseline)"
        logger.info(f"chunk timing stats computed over {len(samples)} iterations (iter 0 omitted){margin_note}")
        return failures, render_table(headers, rows)

    profiler.clear()
    profiler.start("total_test_time")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"

    chunk_local = CHUNK // sp  # 640
    seq_cache = seq_cache or SEQ_CACHE_NOPCC
    assert preload_isl % CHUNK == 0, f"preload_isl ({preload_isl}) must be a multiple of CHUNK ({CHUNK})"
    measured_len = n_chunks * CHUNK  # tokens actually run this call
    total_len = preload_isl + measured_len  # logical seq length: preloaded prefix + measured chunks
    assert (
        total_len <= seq_cache
    ), f"preload_isl {preload_isl} + {n_chunks} chunks ({measured_len}) = {total_len} exceed cache {seq_cache}"

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = seq_cache

    logger.info(
        f"chunked transformer (no-PCC): num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"preload_isl={preload_isl} total_len={total_len} cache={seq_cache} chunk={CHUNK} "
        f"num_iters={num_iters}"
    )

    # Trace is DENSE-MLA ONLY. Asserted here (not just deep in set_trace_controller) so a sparse model
    # fails before the multi-minute weight load. GLM (glm_5_1 / glm_5_2) carries the DSA indexer fields,
    # whose ops have no per-element-tensor metadata overload yet — and the captured forward never threads
    # index_kv_cache, so a traced sparse run would silently skip the indexer and write wrong KV.
    if use_trace:
        assert not resolve_has_indexer(config), (
            "use_trace=True is not supported for sparse/DSA (indexer) attention — GLM (glm_5_1 / "
            "glm_5_2) and friends. Supported traced models are the dense-MLA ones (deepseek_v3, "
            "kimi_k2_6, kimi_k2_7); port the indexer ops to the metadata form first."
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
        max_seq_len=seq_cache,  # KV ring buffer + RoPE cos/sin cache = full no-PCC cache (up to 100k)
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=gate_fallback_mode,
        weight_cache_path=effective_cache_path,
        lm_head_is_column_parallel=True,
        is_chunked=True,
        slot_num=1,
        # Strip the tail (LM head + final norm + sampling): the populated KV cache is this runner's
        # output, so the tail is dead work that would otherwise land inside the measured per-chunk
        # time. It is also what makes the forward DEVICE-ONLY and therefore capturable — the LM head
        # does an all-gather + a host read, and a read inside begin_capture() is a hard TT_FATAL
        # ("Reads are not supported during trace capture"). Set for BOTH modes, not just use_trace,
        # so traced and untraced timings measure the same work and stay comparable.
        kv_only_last_layer=True,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    # Sparse (DSA: glm_5_1 / glm_5_2) requires an UNCOMPRESSED bf16/fp8_e4m3 ROW_MAJOR KVPE cache
    # (sparse_sdpa reads it natively; mla.forward asserts) — NOT the init_kvpe_cache bfloat8_b/TILE
    # default that dense ring_mla wants. Match the cache format to the path (dense variants keep the
    # default). Same distinction as run_chunked_transformer.
    cache_format = MlaKvCacheFormat.BF16_RM if resolve_has_indexer(config) else MlaKvCacheFormat.BFP8_TILE
    tt_kvpe_cache = init_mla_kv_cache(
        cache_format=cache_format,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    # Sparse (DSA) layers read a block-cyclic indexer key cache that is caller-owned and passed into
    # forward, exactly like the KVPE cache. Strided by the compacted full-indexer count over the built
    # layers (>1 for glm_5_2 cross-layer reuse; num_layers without an indexer_types map) so it matches the
    # indexer's cache_batch stride. bf8 TILE. Dense variants get None.
    tt_index_kv_cache = None
    if resolve_has_indexer(config):
        assert getattr(config, "index_head_dim", None) is not None, "sparse config must provide index_head_dim"
        tt_index_kv_cache = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=seq_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=full_indexer_rank(config, num_layers),
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
            seq_cache,
            kvpe_dim,
            config.kv_lora_rank,
            mesh_device,
            sp_axis,
            cache_format.storage_dtype,
            cache_format.storage_layout,
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
                seq_cache,
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

    # use_trace: capture the chunk forward ONCE, then replay it per chunk. The per-chunk scalars
    # (slot_id / actual_start / actual_end) cannot be host arguments on a captured program, so they move
    # into 1-element uint32 DRAM tensors the metadata ops read on-device; the token input moves into a
    # persistent buffer refreshed in place. kv_only_last_layer=True (set on the transformer above) is what makes the forward
    # device-only and therefore capturable at all.
    trace_controller = None
    trace_input = None
    trace_metadata = None
    if use_trace:
        rep = ttnn.ReplicateTensorToMesh(mesh_device)

        def _meta1(val, on_device=True):
            t = torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1)
            kw = dict(dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=rep)
            if on_device:
                kw.update(device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            return ttnn.from_torch(t, **kw)

        trace_input = ttnn.from_torch(
            chunk_tok_host[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )
        # ChunkMetadata, not a bare 3-tuple: the replay reads llama4_scale at a captured address
        # (mirrors TtPrefillRuntime._setup_trace). None for every non-Mistral variant.
        trace_metadata = ChunkMetadata(
            _meta1(0),
            _meta1(preload_isl),
            _meta1(preload_isl + CHUNK),
            transformer.rope_setup.make_llama4_scale_buffer(CHUNK),
        )
        host_tok = [
            ttnn.from_torch(
                chunk_tok_host[c],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
            )
            for c in range(n_chunks)
        ]

        def _fwd_meta():
            return transformer.forward(
                trace_input,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                return_intermediates=False,
                metadata=trace_metadata,
            )

        trace_controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(trace_controller)
        _fwd_meta()  # warm/compile the metadata-variant programs before recording
        ttnn.synchronize_device(mesh_device)
        trace_controller.begin_capture()
        _fwd_meta()
        trace_controller.end_capture()
        ttnn.synchronize_device(mesh_device)
        logger.info(
            f"[no-pcc trace] captured {num_layers}-layer forward = {trace_controller.num_segments} segments, "
            f"{trace_controller.trace_bytes() / 1024 / 1024:.2f} MB"
        )
        # A zero-segment capture would make every replay a no-op: the timings would collapse and any
        # check_pcc would compare the warm pass's (correct) KV. Fail instead of reporting that.
        assert trace_controller.num_segments > 0, "use_trace captured 0 segments — nothing to replay"

    profiler.start("tt_forward")
    for it in range(num_iters):
        iter_start = time.time()
        chunk_times: list[float] = []
        for c in range(n_chunks):
            kv_actual = preload_isl + c * CHUNK
            if use_trace:
                ttnn.copy_host_to_device_tensor(host_tok[c], trace_input)
                # One call: on Mistral this also refreshes the llama4 scale buffer the replay reads.
                # Writing the scalars alone leaves it at ones -- no temperature, and no PCC gate here
                # would see it. No-op for variants without one.
                write_chunk_metadata(
                    trace_metadata,
                    (0, kv_actual, kv_actual + CHUNK),
                    hf_config=config,
                    mesh_device=mesh_device,
                    chunk_size_global=CHUNK,
                    sp_axis=sp_axis,
                )
                chunk_start = time.time()
                trace_controller.replay()
                ttnn.synchronize_device(mesh_device)
                chunk_times.append(time.time() - chunk_start)
                continue
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
            reset_fused_ring_host_timing()
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
            fused_host_calls, fused_host_seconds = get_fused_ring_host_timing()
            ttnn.deallocate(tt_tokens)
            chunk_seconds = time.time() - chunk_start
            chunk_times.append(chunk_seconds)
            if fused_host_calls:
                logger.info(
                    f"[fused-indexer host timing] iter={it} chunk={c} calls={fused_host_calls} "
                    f"host_submit={fused_host_seconds * 1000:.2f} ms "
                    f"({fused_host_seconds / chunk_seconds * 100:.1f}% of {chunk_seconds * 1000:.2f} ms chunk)"
                )
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
    perf_failures, perf_table_lines = print_duration_table(iteration_chunk_times)
    timing_lines = [f"  {key}: {profiler.get(key) * 1000:.2f} ms" for key in profiler.times]
    if perf_table_lines:
        emit_summary(
            "perf",
            f"{variant.name}_L{num_layers}_c{n_chunks}_i{num_iters}_p{preload_isl}",
            f"Chunk timing — {variant.name} (L{num_layers}, {n_chunks} chunks, {num_iters} iters, preload {preload_isl})",
            perf_table_lines + ["", "phase timings:"] + timing_lines,
        )
    for line in timing_lines:
        logger.info(line)

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

    # Optional accuracy check on the same run that produced the timings, so a perf number is never
    # reported for a functionally wrong prefill. Off by default: this runner's whole point is to need no
    # golden trace, and enabling it makes one mandatory.
    if check_pcc:
        assert (
            trace_dir is not None and trace_dir.exists()
        ), "check_pcc=True needs the golden trace (set PREFILL_TRACE_DIR); this runner does not require it otherwise"
        # This runner only ever reads token_ids from the trace, so it never resolved the tensor layout;
        # the KV comparison needs it.
        layout = variant.prefill_trace_layout
        _record_kv_cache_pcc(
            trace_dir,
            layout,
            tt_kvpe_cache,
            mesh_device,
            sp,
            num_layers,
            # SEQ_CACHE_NOPCC, not SEQ_CACHE: this runner allocates the 100k cache, and
            # _record_kv_cache_pcc derives blockcyclic_positions from this length — passing the 55k
            # constant made the un-rotate scatter a [102400, 576] gather into a [56320, 576] index.
            seq_cache,
            total_len,
            config.kv_lora_rank,
            assert_threshold=TRACE_KV_CACHE_PCC_THRESHOLD,
            assert_layer_depth=GATED_LAYER_DEPTH,
        )

    # Release the captured trace + the sub-device managers that own its buffers BEFORE the mesh
    # closes. Without this the MeshTraceBuffers are freed via SubDeviceManager's dtor after the
    # allocator is gone and teardown segfaults in BankManager::deallocate_buffer — the test itself
    # PASSES and the process still exits 139, which reads as a red CI job. Same failure the runner
    # hit (see TtPrefillRuntime.release_trace); the trace path needs the release wherever it lives.
    if trace_controller is not None:
        trace_controller.release()
        transformer.set_trace_controller(None)
    transformer.release_sub_device_managers()

    # Assert AFTER the table is logged so the full per-chunk breakdown is always visible on failure.
    assert not perf_failures, "chunk timing out of baseline tolerance:\n  " + "\n  ".join(perf_failures)


def kimi_chunked_perf_gate(use_trace, num_layers, n_chunks, num_iters, preload_isl, perf_margin=None):
    """Resolve the chunked-Kimi perf gate for one parametrization: returns
    ``(baseline_chunk_times_s, margin)`` for run_chunked_transformer_updated.

    A baseline of None leaves the run record-only (print_duration_table prints the table and does not
    assert), which is what every combo outside the two calibrated CI configs gets. Two conditions on
    top of the table lookup:

      use_trace   -- picks WHICH table and WHICH default margin. The two modes are separate regimes
                     (see the tables), so this is a hard fork, not a shared table with a mode column:
                     a traced baseline can never arm an untraced run, or the reverse.
      preload_isl -- the baselines only mean anything at 0 (the recorded runs started from an empty
                     cache); any preload depth stays record-only.

    `perf_margin` overrides the mode default when not None, so a caller can still pin a band explicitly.
    """
    table, default_margin = (
        (KIMI_TRACED_BASELINE_CHUNK_TIMES_S, TRACED_PERF_MARGIN)
        if use_trace
        else (KIMI_UNTRACED_BASELINE_CHUNK_TIMES_S, UNTRACED_PERF_MARGIN)
    )
    baseline = table.get((num_layers, n_chunks, num_iters)) if preload_isl == 0 else None
    return baseline, (default_margin if perf_margin is None else perf_margin)


# No-PCC perf/smoke variant: runs the full n_chunks-chunk prefill `num_iters` times with no golden
# trace dependency, no intermediate readback, and no PCC. Requires only the Kimi TTNN weight cache (set
# TT_KIMI_PREFILL_TTNN_CACHE + KIMI_K2_6_HF_MODEL); the golden trace is optional.
# Two independent axes on top of the existing perf sweep:
#   check_pcc — also PCC the populated KV against the golden (needs PREFILL_TRACE_DIR)
#   use_trace — capture the chunk forward once and replay it per chunk
# The nopcc x {trace, notrace} pair is what CI runs: it times the real path and passes on
# completion, with no golden dependency. Renamed from *_no_pcc now that PCC is optional, so the
# name no longer contradicts the flag.
# ids: "traced" not "trace" — "notrace" CONTAINS "trace", so a -k "trace" term would match BOTH
# modes and silently double a CI job. Matches the padded test's convention.
@pytest.mark.parametrize("use_trace", [False, True], ids=["notrace", "traced"])
# None = take the band from the mode (TRACED_PERF_MARGIN / UNTRACED_PERF_MARGIN, resolved by
# kimi_chunked_perf_gate). The two bands differ by more than 3x, so no single literal serves both.
@pytest.mark.parametrize("perf_margin", [None], ids=["margin_auto"])
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi_k2_6"])
@pytest.mark.skipif(not is_blackhole(), reason="Kimi requires Blackhole")
@pytest.mark.skipif(
    not is_high_power(),
    reason="perf job requires a high-power (>=130W TDP) galaxy; guards the exabox.tenstorrent.com/power=14kw label",
)
@pytest.mark.timeout(0)
def test_kimi_prefill_transformer_chunked_perf(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_layers,
    n_chunks,
    num_iters,
    num_links,
    perf_margin,
    use_trace,
    preload_isl,
):
    topology = per_axis_topology(device_params["fabric_config"])
    if preload_isl + n_chunks * CHUNK > SEQ_CACHE_NOPCC:
        pytest.skip(f"preload_isl {preload_isl} + {n_chunks} chunks exceeds the {SEQ_CACHE_NOPCC}-token cache")
    # Gate against the CI baseline only for the exact configs we have recorded numbers for; every other
    # combo in the sweep stays record-only (baseline None -> print_duration_table does not assert). Both
    # modes are gated, each against its own table and its own band -- see kimi_chunked_perf_gate.
    baseline_chunk_times_s, perf_margin = kimi_chunked_perf_gate(
        use_trace, num_layers, n_chunks, num_iters, preload_isl, perf_margin
    )
    run_chunked_transformer_updated(
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
        check_pcc=False,  # timing only — accuracy lives in test_kimi_prefill_transformer_chunked
        use_trace=use_trace,
    )


# ids: "traced" not "trace" — "notrace" CONTAINS "trace", so a -k "trace" term would match BOTH
# modes and silently double a CI job. Matches the padded test's convention.
@pytest.mark.parametrize("use_trace", [False, True], ids=["notrace", "traced"])
# None = take the band from the mode (TRACED_PERF_MARGIN / UNTRACED_PERF_MARGIN, resolved by
# kimi_chunked_perf_gate). The two bands differ by more than 3x, so no single literal serves both.
@pytest.mark.parametrize("perf_margin", [None], ids=["margin_auto"])
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(
                fabric_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE,
                l1_small_size=768,
                trace_region_size=256 * 1024 * 1024,
            ),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["kimi_k2_6"], indirect=True, ids=["kimi_k2_6"])
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
    num_iters,
    num_links,
    perf_margin,
    use_trace,
    preload_isl,
):
    topology = per_axis_topology(device_params["fabric_config"])
    if preload_isl + n_chunks * CHUNK > SEQ_CACHE_NOPCC:
        pytest.skip(f"preload_isl {preload_isl} + {n_chunks} chunks exceeds the {SEQ_CACHE_NOPCC}-token cache")
    # ALWAYS record-only -- this accuracy test is never perf-gated. It shares the perf test's
    # parametrization but NOT its is_high_power() skipif, so it can run on a standard-power Blackhole,
    # where the baseline tables -- measured on a >=130W galaxy -- describe nothing. Gating here would fail
    # an accuracy run for a timing reason, on hardware the baseline never covered, and the timing table is
    # incidental to this test anyway (see check_pcc below). The perf gate belongs to, and stays in,
    # test_kimi_prefill_transformer_chunked_perf.
    baseline_chunk_times_s = None
    run_chunked_transformer_updated(
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
        # Inert while baseline_chunk_times_s is None (print_duration_table only uses the margin when it
        # has a baseline); kept for id/signature symmetry with the perf test.
        perf_margin=perf_margin,
        preload_isl=preload_isl,
        check_pcc=True,  # this test exists for the KV PCC; the timing table is incidental
        use_trace=use_trace,
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=DeepSeekV3Config.FABRIC_PAYLOAD_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
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
):
    topology = per_axis_topology(device_params["fabric_config"])
    run_chunked_transformer_updated(
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
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            # Routing consumes 512 B; leave 256 B for sparse-MLA high-bandwidth-gather semaphores
            # and retain the existing reserve for other needs.
            torus_xy_device_params(fabric_payload_size=GLM51Config.FABRIC_PAYLOAD_SIZE, l1_small_size=1152),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_1", "glm_5_2"], indirect=True, ids=["glm51", "glm52"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.skipif(
    not is_high_power(),
    reason="perf job requires a high-power (>=130W TDP) galaxy; guards the exabox.tenstorrent.com/power=14kw label",
)
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
    preload_isl,
):
    topology = per_axis_topology(device_params["fabric_config"])
    if preload_isl + n_chunks * CHUNK > SEQ_CACHE_NOPCC:
        pytest.skip(f"preload_isl {preload_isl} + {n_chunks} chunks exceeds the {SEQ_CACHE_NOPCC}-token cache")
    run_chunked_transformer_updated(
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


def run_chunked_transformer_padded_trace(
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
    mode="traced",
):
    """VARIABLE/partial-chunk prefill on ONE kv_only build, in one of three independent modes (pytest
    param `mode`), each asserted ONLY against the golden kv_post_transform (no cross-path comparison):
      - "scalar":  untraced scalar path (host actual_start/actual_end, metadata=None);
      - "eager":   metadata path run eagerly per split (per-chunk scalars read on-device from a
                   persistent metadata tensor), no capture;
      - "traced":  the SAME metadata forward captured once as a ttnn trace and replayed per split.
    Each mode runs a single pass and asserts its per-layer KV-cache PCC (vs the golden) meets
    LAYER_PCC_THRESHOLD for the asserted layer depth.

    All three modes DO exercise padding-aware MoE (they did not while the trace layer force-disabled it
    by clearing actual_isl), and they cover both producers of the padding config:
      * "scalar" builds it on HOST (build_padding_config, rotated branch: metadata=None with a real
        partial actual_isl and actual_start). `splits` repeats ISLs at different starts (e.g. 5120 at
        10240/20480/35168), so this also covers the build_padding_config memo being keyed on
        actual_start -- keying on actual_isl alone silently reused the first chunk's rotated config.
      * "eager"/"traced" build it ON DEVICE (moe_padding_config) from the actual_start/actual_end
        metadata tensors. They pass actual_isl=CHUNK, which now only flags padding awareness as ON --
        the real per-chunk bound comes from those tensors, which is what makes ONE capture replay
        correctly across chunks."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")
    layout = variant.prefill_trace_layout

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"
    tile = ttnn.TILE_SIZE
    chunk_local = CHUNK // sp
    total_len = sum(splits)
    for v in splits:
        assert 0 < v <= CHUNK and v % tile == 0, f"split {v} must be tile-aligned and <= {CHUNK}"

    # Slab-aligned cache covering the largest rotated write (mirror run_chunked_transformer_padded).
    max_window, ka = CHUNK * 2, 0
    for v in splits:
        max_window = max(max_window, ka + CHUNK)
        ka += v
    seq_len_cache = ((max_window + CHUNK - 1) // CHUNK) * CHUNK

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = seq_len_cache
    logger.info(
        f"chunked-padded TRACE: num_layers={num_layers} mesh={mesh_shape} splits={splits} "
        f"total_len={total_len} cache={seq_len_cache} chunk={CHUNK}"
    )
    token_ids_full = _load_metadata_token_ids(trace_dir, total_len, require_full=True)

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

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
        is_chunked=True,
        slot_num=1,
        # kv_only_last_layer -> device-only forward (no host readback) so ttnn trace can capture it.
        kv_only_last_layer=True,
        overlap_shared_expert_with_dispatch=True,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    mesh_device.enable_program_cache()

    # Per-split padded token tile (block-cyclic gather; positions >= valid_end -> pad token 0) + scalars.
    def _padded_chunk_tok(kv_actual, isl):
        valid_end = kv_actual + isl
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = [positions[ch][r] for ch in range(sp) for r in range(chunk_local)]
        gather_idx = torch.tensor([min(gp, total_len - 1) for gp in flat], dtype=torch.long)
        tok = token_ids_full[gather_idx].clone()
        tok[torch.tensor([gp >= valid_end for gp in flat])] = 0
        return tok.reshape(sp, 1, chunk_local)

    starts, ka = [], 0
    for isl in splits:
        starts.append((ka, ka + isl))  # (kv_actual, valid_end)
        ka += isl
    chunk_tok_host = [_padded_chunk_tok(ks, e - ks) for (ks, e) in starts]

    def _make_cache():
        return init_mla_kv_cache(
            cache_format=MlaKvCacheFormat.BFP8_TILE,
            hf_config=config,
            mesh_device=mesh_device,
            seq_len=seq_len_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_layers,
            num_users=1,
        )

    sp_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None))
    rep_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # Each mode runs a SINGLE pass and asserts its own per-layer KV-cache PCC against the GOLDEN trace.
    # There is no scalar-vs-metadata cross-comparison: the scalar, metadata(eager) and trace(replay)
    # paths are independent tests, each validated only against the golden.

    # ---- SCALAR mode: untraced scalar path (host actual_start/actual_end), asserted vs GOLDEN. ----
    if mode == "scalar":
        cache = _make_cache()
        # UNTRACED ONLY: also PCC each layer's DECODER OUTPUT, not just the KV cache. This needs
        # return_intermediates=True, whose _to_host(h) + synchronize_device is a host readback and so
        # cannot live inside a trace capture — which is why the metadata/traced modes below assert KV
        # PCC alone. Running it here keeps the stronger check on the untraced reference rather than
        # losing it entirely.
        #
        # Layers 0..num_layers-2 only: the transformer is built kv_only_last_layer=True, so the LAST
        # layer is kv_only (attn_norm + the KV branch) and never produces a hidden state — the loop
        # returns before snapshotting it (tt_prefill_transformer: `if self.kv_only_last_layer and
        # i == len(self.layers)-1: return None, None, intermediates`). Its correctness is covered by
        # the KV PCC below. The norm / LM head / logits tail is likewise not built in this config.
        emb_dim = config.hidden_size
        n_decoder_layers = num_layers - 1
        layer_min_pcc = {i: 1.0 for i in range(n_decoder_layers)}
        gate_on_npcc = variant.name in _NPCC_GATED_VARIANTS
        for c, ((ks, e), tok) in enumerate(zip(starts, chunk_tok_host)):
            isl = e - ks
            # Same rotated layout _padded_chunk_tok gathered with: recomputed here (cheap, host-side)
            # so the valid rows can be un-rotated back to natural order for the golden comparison.
            positions = rotated_chip_positions(ks, sp, chunk_local)
            flat = [positions[ch][r] for ch in range(sp) for r in range(chunk_local)]
            valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < e]
            src = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
            dst = torch.tensor([gp - ks for _, gp in valid_pairs], dtype=torch.long)  # 0..isl-1
            tt_tokens = ttnn.from_torch(
                tok,
                device=mesh_device,
                dtype=ttnn.uint32,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=sp_mapper,
            )
            _, _, layer_outputs = transformer.forward(
                tt_tokens,
                cache,
                actual_isl=isl,
                actual_start=ks,
                actual_end=e,
                cache_user_id=0,
                metadata=None,
                return_intermediates=True,
            )
            ttnn.deallocate(tt_tokens)
            ttnn.synchronize_device(mesh_device)
            for i in range(n_decoder_layers):
                # host snapshot [1, CHUNK, emb] (SP-seq + TP-hidden concatenated); [0] -> [CHUNK, emb].
                out_flat = layer_outputs[f"layer_{i}"][0].to(torch.float32)
                natural = torch.zeros(isl, emb_dim, dtype=torch.float32)
                natural[dst] = out_flat[src]  # un-rotate valid rows -> natural [ks, e)
                ref = _ref_layer_slice(trace_dir, layout, i, ks, e)
                _, pcc = comp_pcc(ref, natural)
                # Both scores always: the gated one is asserted, the other is context in the log.
                # Row-subsampled when it is only context -- at full resolution this second comp_pcc
                # costs ~75 s per row, which the non-gated variants (Kimi/DeepSeek/GLM) would pay for
                # a number nothing asserts. Stride 8 still correlates 1.6M elements.
                nstride = 1 if gate_on_npcc else 8
                _, npcc = comp_pcc(token_normalized(ref[::nstride]), token_normalized(natural[::nstride]))
                score = npcc if gate_on_npcc else pcc
                layer_min_pcc[i] = min(layer_min_pcc[i], score)
                logger.info(
                    f"  chunk {c} (kv_actual={ks} isl={isl}) layer {i} decoder "
                    f"PCC: {pcc:.6f} nPCC: {npcc:.6f} ({'npcc' if gate_on_npcc else 'pcc'} gated)"
                )
                if score < LAYER_PCC_THRESHOLD:
                    logger.warning(f"  chunk {c} layer {i} decoder score {score:.6f} below {LAYER_PCC_THRESHOLD}")
        ttnn.synchronize_device(mesh_device)

        logger.info("[padded-trace] NOTRACE per-layer min decoder-output PCC across chunks:")
        for i in range(n_decoder_layers):
            logger.info(f"  layer {i}: {layer_min_pcc[i]:.6f}")
        if layer_min_pcc:
            decoder_min = min(layer_min_pcc.values())
            logger.info(f"[padded-trace] NOTRACE decoder-output min PCC across all layers = {decoder_min:.6f}")
            # FULL DEPTH, not gated to GATED_LAYER_DEPTH: this mirrors run_chunked_transformer_padded,
            # which asserts min decoder PCC over every layer and runs at L61 in the Blaze CI job — so a
            # full-depth decoder bar is known stable. The KV check below deliberately stays gated: deep
            # KV PCC is the quantity with unpinned accumulation headroom (and a known L61 run-to-run
            # spread), not the decoder output.
            assert (
                decoder_min >= LAYER_PCC_THRESHOLD
            ), f"decoder-output min PCC {decoder_min:.6f} < {LAYER_PCC_THRESHOLD}"

        logger.info("[padded-trace] SCALAR path done; recording per-layer KV PCC vs GOLDEN")
        _record_kv_cache_pcc(
            trace_dir,
            layout,
            cache,
            mesh_device,
            sp,
            num_layers,
            seq_len_cache,
            total_len,
            config.kv_lora_rank,
            assert_threshold=LAYER_PCC_THRESHOLD,
            assert_layer_depth=(GATED_LAYER_DEPTH if num_layers > GATED_LAYER_DEPTH else None),
        )
        ttnn.deallocate(cache.storage)
        transformer.release_sub_device_managers()
        logger.success("[padded-trace] SCALAR run complete (asserted vs golden)")
        return

    # ---- METADATA modes (eager / traced): on-device per-split scalars, asserted vs GOLDEN. ----
    cache_B = _make_cache()  # persistent (captured) cache
    trace_input = ttnn.from_torch(
        chunk_tok_host[0],
        device=mesh_device,
        dtype=ttnn.uint32,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=sp_mapper,
    )

    # Per-element-tensor metadata: 3 persistent 1-element tensors (slot_id, actual_start, actual_end).
    def _meta1_dev(val):
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep_mapper,
        )

    def _meta1_host(val):
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=rep_mapper,
        )

    # ChunkMetadata, not a bare 3-tuple: the replay reads llama4_scale at a captured address
    # (mirrors TtPrefillRuntime._setup_trace). None for every non-Mistral variant.
    trace_metadata = ChunkMetadata(
        _meta1_dev(0),
        _meta1_dev(starts[0][0]),
        _meta1_dev(starts[0][1]),
        transformer.rope_setup.make_llama4_scale_buffer(CHUNK),
    )
    tok_host_tt = [
        ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=sp_mapper)
        for t in chunk_tok_host
    ]

    def _fwd_meta():
        transformer.forward(
            trace_input,
            cache_B,
            actual_isl=CHUNK,
            actual_start=None,
            actual_end=None,
            cache_user_id=0,
            metadata=trace_metadata,
        )

    # Metadata execution mode (pytest param `mode`). "traced" captures the metadata forward ONCE and
    # replays it per split (the traced-metadata path). "eager" runs the SAME metadata forward EAGERLY
    # per split (no capture/replay). ("scalar" mode returned above and never reaches here.)
    _PADDED_TRACE = mode == "traced"
    if _PADDED_TRACE:
        controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(controller)
        _fwd_meta()  # warmup (compile metadata program variants)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[padded-trace] TRACED metadata: capturing {num_layers}-layer forward...")
        controller.begin_capture()
        _fwd_meta()
        controller.end_capture()
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[padded-trace] {controller.num_segments} segments, {controller.trace_bytes()/1024/1024:.2f} MB")
        # Prove we are actually testing the traced path. Without this, a capture that recorded nothing
        # would still "pass": every replay would be a no-op, the KV cache would keep whatever the warm
        # pass left in it, and the KV PCC below would compare that stale-but-correct data against the
        # golden and succeed — a green run that exercised no trace at all.
        assert controller.num_segments > 0, "traced mode captured 0 segments — nothing was recorded to replay"

        for c, (ks, e) in enumerate(starts):
            ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
            write_chunk_metadata(
                trace_metadata,
                (0, ks, e),
                hf_config=config,
                mesh_device=mesh_device,
                chunk_size_global=CHUNK,
                sp_axis=sp_axis,
            )
            controller.replay()
        ttnn.synchronize_device(mesh_device)
        controller.release()
        transformer.set_trace_controller(None)
    else:
        logger.info("[padded-trace] EAGER metadata (eager mode): per-split forward, no capture")
        for c, (ks, e) in enumerate(starts):
            ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
            write_chunk_metadata(
                trace_metadata,
                (0, ks, e),
                hf_config=config,
                mesh_device=mesh_device,
                chunk_size_global=CHUNK,
                sp_axis=sp_axis,
            )
            _fwd_meta()
        ttnn.synchronize_device(mesh_device)
    ttnn.deallocate(trace_input)
    # [:3]: field 3 is the persistent llama4 buffer, not per-chunk state (see ChunkMetadata).
    for t in trace_metadata[:3]:
        ttnn.deallocate(t)
    logger.info(f"[padded-trace] {mode} metadata path done; recording per-layer KV PCC vs GOLDEN")
    _record_kv_cache_pcc(
        trace_dir,
        layout,
        cache_B,
        mesh_device,
        sp,
        num_layers,
        seq_len_cache,
        total_len,
        config.kv_lora_rank,
        assert_threshold=LAYER_PCC_THRESHOLD,
        assert_layer_depth=(GATED_LAYER_DEPTH if num_layers > GATED_LAYER_DEPTH else None),
    )
    transformer.release_sub_device_managers()
    logger.success(f"[padded-trace] {mode} metadata run complete (asserted vs golden)")
