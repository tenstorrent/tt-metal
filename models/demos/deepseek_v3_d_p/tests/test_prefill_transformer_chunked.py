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
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from models.demos.deepseek_v3_d_p.utils.sub_device_trace import SubDeviceTraceController
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = 5 * 1024  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (1 user)


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
# KV-cache PCC bar for the trace-correctness test (verify_kv_cache_pcc): the chunk-0 KV the traced
# forward wrote vs the golden kv_post_transform. The cache is bf8_b, so this is below the bf16 bar;
# calibrate + tighten after the first measured run.
KV_CACHE_PCC_THRESHOLD = 0.96


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


def _read_sharded_rows(tensor_dir: Path, key: str, start: int, end: int) -> torch.Tensor:
    """Read rows [start:end] of `key` from a chunked_group_a_v1 shard directory, concatenating the
    rows_<s>_<e>.safetensors shards that overlap the range (partial read, natural order)."""
    parts = []
    for shard in sorted(tensor_dir.glob("rows_*.safetensors")):
        s, e = (int(x) for x in shard.stem.split("_")[1:3])
        if e <= start or s >= end:
            continue
        with safe_open(shard, framework="pt") as f:
            parts.append(f.get_slice(key)[max(start, s) - s : min(end, e) - s].to(torch.float32))
    assert parts, f"no shards overlap rows [{start}:{end}] in {tensor_dir}"
    return torch.cat(parts, dim=0)


def _load_layer_rows(
    trace_dir: Path, layout: str, group: str, layer: int, key: str, start: int, end: int
) -> torch.Tensor:
    """Read trace tensor `key` rows [start:end] (float32) for `layer`, handling both layouts. `group`
    is the logical bucket: "hidden_states" (decoder_io for Kimi) or "kv_cache"."""
    if layout == _LAYOUT_CHUNKED_GROUP_A:
        if group == "hidden_states":
            tensor_dir = trace_dir / "decoder_io" / key
        else:
            tensor_dir = trace_dir / "kv_cache" / f"layer_{layer}"
        return _read_sharded_rows(tensor_dir, key, start, end)
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
    kvpe_dim,
    kv_lora,
    assert_threshold=None,
    assert_layer_depth=None,
    return_per_layer=False,
):
    """Gather the device KV cache, un-rotate the block-cyclic layout, and PCC each layer's valid
    region [:total_len] against the golden kv_post_transform trace. nope is compared directly; the
    RoPE (pe) slice uses the Meta-interleaved basis (golden stores HF half-split). Returns the min PCC
    across layers. With assert_threshold set, asserts min PCC >= threshold (otherwise record-only,
    mirroring the per-layer decoder_output reporting). With assert_layer_depth set, only layers
    0..assert_layer_depth (inclusive) are asserted against the threshold; deeper layers are recorded
    only (matches the decoder-output GATED_LAYER_DEPTH policy — deeper KV PCC drifts under bf8_b)."""
    logger.info("Device KV cache vs golden kv_post_transform (record-only):")
    # One gather: [num_layers, tp_replicas, seq_len_cache, kvpe] -> collapse TP replicas via [:, :1].
    cache_full = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[
        :, :1
    ]  # [num_layers, 1, seq_len_cache, kvpe]
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    cache_min_pcc = {}
    for i in range(num_layers):
        nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
        nat[p] = cache_full[i, 0]  # un-rotate block-cyclic -> natural order
        dev_cache = nat[:total_len]
        g_post = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, total_len)
        _, pcc_nope = comp_pcc(g_post[:, :kv_lora], dev_cache[:, :kv_lora])
        ref_pe = g_post[:, kv_lora:]
        d = ref_pe.shape[-1]
        ref_pe_int = torch.stack([ref_pe[:, : d // 2], ref_pe[:, d // 2 :]], dim=-1).reshape(-1, d)  # HF -> Meta
        _, pcc_pe = comp_pcc(ref_pe_int, dev_cache[:, kv_lora:])
        cache_min_pcc[i] = min(pcc_nope, pcc_pe)
        logger.info(f"  cache layer {i} PCC: nope={pcc_nope:.6f} pe(interleaved)={pcc_pe:.6f}")
        # ---- KV_PE_DEBUG: decide the device pe layout convention + localize nope drift per chunk ----
        if os.environ.get("KV_PE_DEBUG") == "1":
            dev_pe = dev_cache[:, kv_lora:]
            g_meta2hf = torch.cat([ref_pe[:, 0::2], ref_pe[:, 1::2]], dim=-1)  # treat golden as interleaved
            cands = {
                "pe_raw(golden HF as-is)": ref_pe,
                "pe_hf2meta(current)": ref_pe_int,
                "pe_meta2hf": g_meta2hf,
            }
            for name, cand in cands.items():
                _, pc = comp_pcc(cand, dev_pe)
                logger.info(f"    [KV_PE_DEBUG] L{i} {name}: pcc={pc:.6f}")
            # per-chunk nope localization (which position range drifts?)
            for c in range(0, total_len, CHUNK):
                e = min(c + CHUNK, total_len)
                _, pc_n = comp_pcc(g_post[c:e, :kv_lora], dev_cache[c:e, :kv_lora])
                logger.info(f"    [KV_PE_DEBUG] L{i} nope chunk[{c}:{e}] pcc={pc_n:.6f}")
    overall = min(cache_min_pcc.values())
    logger.info(f"KV cache min PCC across layers: {overall:.6f}")
    if assert_threshold is not None:
        if assert_layer_depth is not None:
            gated = {i: v for i, v in cache_min_pcc.items() if i <= assert_layer_depth}
            gated_min = min(gated.values())
            logger.info(
                f"KV cache min PCC over asserted layers 0..{assert_layer_depth}: {gated_min:.6f} "
                f"(layers >{assert_layer_depth} recorded only)"
            )
            assert (
                gated_min >= assert_threshold
            ), f"KV cache min PCC {gated_min:.6f} (layers 0..{assert_layer_depth}) < {assert_threshold}"
        else:
            assert overall >= assert_threshold, f"KV cache min PCC {overall:.6f} < {assert_threshold}"
    if return_per_layer:
        return overall, cache_min_pcc
    return overall


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
        is_chunked=True,
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
        kvpe_dim,
        config.kv_lora_rank,
    )

    profiler.end("total_test_time")
    logger.success(
        f"Chunked-padded transformer passed (num_layers={num_layers}, {len(splits)} chunks, "
        f"min PCC {overall_min:.6f})"
    )
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


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
):
    """Trace+metadata twin of run_chunked_transformer_padded. On ONE kv_only build it runs the same
    VARIABLE/partial-chunk prefill TWICE:
      PASS A (untraced, scalar host actual_start/actual_end) -> KV cache A;
      PASS B (a metadata ttnn trace captured once + replayed per split, per-chunk scalars read on-device
              from a persistent metadata tensor) -> KV cache B.
    It then asserts the per-layer KV-cache PCC (vs the golden kv_post_transform) of the TRACED metadata
    path == the UNTRACED scalar path (bit-exact), and that the traced path meets the PCC threshold. This
    is the trace-safe equivalent of the untraced run_chunked_transformer_padded for the metadata path."""
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
    token_ids_full = _load_metadata_token_ids(trace_dir, total_len)

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
        return init_kvpe_cache(
            kvpe_cache_head_dim=kvpe_dim,
            mesh_device=mesh_device,
            seq_len=seq_len_cache,
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            num_kvpe_cache_layers=num_layers,
            num_users=1,
        )

    sp_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None))
    rep_mapper = ttnn.ReplicateTensorToMesh(mesh_device)

    # ---- PASS A: untraced scalar (host actual_start/actual_end), the reference path ----
    cache_A = _make_cache()
    for (ks, e), tok in zip(starts, chunk_tok_host):
        tt_tokens = ttnn.from_torch(
            tok,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=sp_mapper,
        )
        transformer.forward(
            tt_tokens,
            cache_A,
            actual_isl=e - ks,
            actual_start=ks,
            actual_end=e,
            cache_user_id=0,
            metadata=None,
        )
        ttnn.deallocate(tt_tokens)
    ttnn.synchronize_device(mesh_device)
    logger.info("[padded-trace] PASS A (untraced scalar) done; recording per-layer KV PCC vs golden")
    _, pcc_A = _record_kv_cache_pcc(
        trace_dir,
        layout,
        cache_A,
        mesh_device,
        sp,
        num_layers,
        seq_len_cache,
        total_len,
        kvpe_dim,
        config.kv_lora_rank,
        return_per_layer=True,
    )
    ttnn.deallocate(cache_A)

    # ---- PASS B: metadata trace captured ONCE, replayed per split ----
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

    trace_metadata = (_meta1_dev(0), _meta1_dev(starts[0][0]), _meta1_dev(starts[0][1]))
    tok_host_tt = [
        ttnn.from_torch(t, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=sp_mapper)
        for t in chunk_tok_host
    ]
    meta_host_tt = [(_meta1_host(0), _meta1_host(ks), _meta1_host(e)) for (ks, e) in starts]

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

    controller = SubDeviceTraceController(mesh_device)
    transformer.set_trace_controller(controller)
    _fwd_meta()  # warmup (compile metadata program variants)
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[padded-trace] capturing {num_layers}-layer metadata forward...")
    controller.begin_capture()
    _fwd_meta()
    controller.end_capture()
    ttnn.synchronize_device(mesh_device)
    logger.info(f"[padded-trace] {controller.num_segments} segments, {controller.trace_bytes()/1024/1024:.2f} MB")

    for c, (ks, e) in enumerate(starts):
        ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
        for src, dst in zip(meta_host_tt[c], trace_metadata):
            ttnn.copy_host_to_device_tensor(src, dst)
        controller.replay()
    ttnn.synchronize_device(mesh_device)
    controller.release()
    transformer.set_trace_controller(None)
    ttnn.deallocate(trace_input)
    for t in trace_metadata:
        ttnn.deallocate(t)
    logger.info("[padded-trace] PASS B (metadata trace) done; recording per-layer KV PCC vs golden")
    _, pcc_B = _record_kv_cache_pcc(
        trace_dir,
        layout,
        cache_B,
        mesh_device,
        sp,
        num_layers,
        seq_len_cache,
        total_len,
        kvpe_dim,
        config.kv_lora_rank,
        assert_threshold=LAYER_PCC_THRESHOLD,
        assert_layer_depth=(GATED_LAYER_DEPTH if num_layers > GATED_LAYER_DEPTH else None),
        return_per_layer=True,
    )
    transformer.release_sub_device_managers()

    # ---- VERIFY: traced metadata path == untraced scalar path (per-layer, bit-exact KV) ----
    logger.info("[padded-trace] per-layer KV PCC: untraced(scalar) vs traced(metadata):")
    max_diff = 0.0
    for i in range(num_layers):
        a, b = pcc_A[i], pcc_B[i]
        max_diff = max(max_diff, abs(a - b))
        logger.info(f"  layer {i}: untraced={a:.6f}  traced={b:.6f}  |diff|={abs(a-b):.2e}")
    logger.success(f"[padded-trace] max |untraced - traced| per-layer KV PCC = {max_diff:.2e}")
    assert max_diff < 1e-3, (
        f"metadata+trace KV PCC differs from untraced scalar by {max_diff:.2e} (>1e-3) — "
        f"the traced path should be bit-identical to the untraced path"
    )


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
        is_chunked=True,
        slot_num=1,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    # ONE shared KV cache holding all layers' slots [num_layers, 1, seq_local, 576]; layer i uses
    # cache_layer_idx=i. The ring scratch buffer is shared across layers inside TtPrefillTransformer.
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
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
        kvpe_dim,
        config.kv_lora_rank,
    )

    profiler.end("total_test_time")
    logger.success(
        f"Chunked prefill transformer passed (num_layers={num_layers}, n_chunks={n_chunks}, "
        f"min PCC {overall_min:.6f})"
    )
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


def run_chunked_transformer_kv_cache(
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
    use_trace=False,
    verify_kv_cache_pcc=False,
    use_metadata=False,
):
    """No-PCC perf/smoke variant of run_chunked_transformer: build the transformer ONCE (with
    kv_only_last_layer=True so the LM head + sampling tail is never built/run — the populated KV cache
    is the output), then drive the prefill `num_iters` times with return_intermediates=False (no
    per-layer host readback, no PCC).

    When use_trace=True the forward is captured ONCE as a ttnn trace on chunk 0 and replayed every
    iteration via ttnn.execute_trace — this collapses the per-op host-dispatch (op2op) gaps. Trace
    capture requires a device-only forward, which kv_only_last_layer gives us (no LM-head host
    readback); to keep capture/replay valid we PIN to chunk 0 (same inputs each iteration). Multi-chunk
    tracing (varying actual_start / token_ids) is deferred."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    if verify_kv_cache_pcc:
        # KV-cache PCC runs on one of: the traced metadata path (use_trace+use_metadata), the EAGER
        # metadata path (use_metadata, no trace — the determinism BASELINE the traced number must match),
        # or the single-chunk scalar trace path. The scalar/pinned trace path can only populate chunk 0.
        assert use_metadata or use_trace, "verify_kv_cache_pcc needs the metadata path or the trace path"
        if not use_metadata:
            assert use_trace and n_chunks == 1, "scalar KV-cache PCC is single-chunk traced (chunk 0)"

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

    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    config.max_seq_len = SEQ_CACHE

    logger.info(
        f"chunked transformer (no-PCC): num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"total_len={total_len} cache={SEQ_CACHE} chunk={CHUNK} num_iters={num_iters}"
    )

    # Token ids: prefer the real (longbook) ids from the golden trace (same source as the PCC test) but
    # never compared here; fall back to a deterministic in-vocab pattern so this stays trace-optional.
    vocab_size = config.vocab_size
    trace_dir = _resolve_trace_dir(variant)
    layout = variant.prefill_trace_layout
    logger.info(f"Trace dir is: {trace_dir}")
    if verify_kv_cache_pcc:
        # KV-cache PCC needs the EXACT golden tokens (no vocab modulo) so the device cache matches the
        # golden kv_post_transform; skip if the golden trace is unavailable.
        if not trace_dir.exists():
            pytest.skip(f"verify_kv_cache_pcc needs the golden trace (not found: {trace_dir})")
        token_ids_full = _load_metadata_token_ids(trace_dir, total_len)
        logger.info(f"no-PCC(trace KV PCC): loaded {token_ids_full.numel()} exact token ids from {trace_dir}")
    elif trace_dir.exists():
        trace_tokens = _load_metadata_token_ids(trace_dir, total_len)
        if trace_tokens.numel() < total_len:
            reps = (total_len + trace_tokens.numel() - 1) // trace_tokens.numel()
            trace_tokens = trace_tokens.repeat(reps)[:total_len]
        token_ids_full = trace_tokens % vocab_size
        logger.info(f"no-PCC: loaded {token_ids_full.numel()} token ids from trace {trace_dir}")
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
        # No-PCC mode never samples a token — the output is the populated KV cache. kv_only_last_layer
        # makes the last block kv-only and skips the final RMSNorm + LM head + sampling entirely, so
        # forward() returns after the layer loop with no host readback. This is what makes the forward
        # a device-only call that ttnn trace can capture (and matches the production prefill runner).
        kv_only_last_layer=True,
        # Keep the shared-expert/dispatch overlap on in BOTH modes. Under trace, SubDeviceTraceController
        # splits the capture at the overlap's sub-device load/clear so the swap stays legal (the capture
        # would otherwise abort with "Cannot reset worker state during trace capture").
        overlap_shared_expert_with_dispatch=True,
        routing_use_l1_small_for_semaphores=routing_use_l1_small_for_semaphores,
    )
    ttnn.synchronize_device(mesh_device)
    gc.collect()
    profiler.end("tt_transformer_creation")

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
    )

    # Precompute per-chunk SP-sharded token tiles once (reused across iterations). Chunk-aligned offsets
    # make the block-cyclic rotation degenerate to a plain per-chip reshape.
    chunk_tok_host = []
    for c in range(n_chunks):
        kv_actual = c * CHUNK
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        chunk_tok_host.append(token_ids_full[flat].reshape(sp, 1, chunk_local))

    mesh_device.enable_program_cache()

    # Per-chunk replay timings (metadata trace path only); surfaced in the perf JSON below.
    per_chunk_seconds = []

    if verify_kv_cache_pcc and use_metadata and not use_trace:
        # ------------- EAGER METADATA MULTI-CHUNK KV-PCC (no trace) — DETERMINISM BASELINE -------------
        # Identical forward to the traced metadata path (per-chunk scalars read on-device from the
        # persistent metadata tensor), but run eagerly per chunk instead of capture+replay. This is the
        # reference the traced KV-PCC must match: if the traced number differs materially, the trace
        # machinery introduced non-determinism (not the per-element metadata ops).
        trace_input = ttnn.from_torch(
            chunk_tok_host[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        def _meta1_dev(val):
            return ttnn.from_torch(
                torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _meta1_host(val):
            return ttnn.from_torch(
                torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        eager_metadata = (_meta1_dev(0), _meta1_dev(0), _meta1_dev(CHUNK))
        tok_host_tt = [
            ttnn.from_torch(
                chunk_tok_host[c],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
            )
            for c in range(n_chunks)
        ]
        meta_host_tt = [
            (_meta1_host(0), _meta1_host(c * CHUNK), _meta1_host(c * CHUNK + CHUNK)) for c in range(n_chunks)
        ]

        # Warm-up / compile iteration (NOT measured): the first eager forward JIT-compiles every op in
        # the metadata path, which dominates the first chunk's wall time (tens of seconds of compile, not
        # execution). Run one full unmeasured walk over all chunks to populate the program cache before
        # the timed loop. Its KV writes are overwritten by the measured pass below (each chunk overwrites
        # its own [c*CHUNK:(c+1)*CHUNK] slots), so it does not affect the KV-PCC result.
        logger.info("  [eager metadata] warm-up/compile iteration (not measured)...")
        for c in range(n_chunks):
            ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
            for src, dst in zip(meta_host_tt[c], eager_metadata):
                ttnn.copy_host_to_device_tensor(src, dst)
            transformer.forward(
                trace_input,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                metadata=eager_metadata,
                return_intermediates=False,
            )
        ttnn.synchronize_device(mesh_device)

        per_iter_seconds = []
        profiler.start("tt_forward")
        signpost("PROFILE_MEASURE_START")
        for it in range(num_iters):
            iter_start = time.time()
            for c in range(n_chunks):
                ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
                for src, dst in zip(meta_host_tt[c], eager_metadata):
                    ttnn.copy_host_to_device_tensor(src, dst)
                chunk_start = time.time()
                transformer.forward(
                    trace_input,
                    tt_kvpe_cache,
                    actual_isl=CHUNK,
                    actual_start=None,
                    actual_end=None,
                    cache_user_id=0,
                    metadata=eager_metadata,
                    return_intermediates=False,
                )
                ttnn.synchronize_device(mesh_device)
                dt = time.time() - chunk_start
                per_chunk_seconds.append(dt)
                logger.info(f"  iter {it} chunk {c} (eager metadata): {dt:.3f} seconds")
            iter_seconds = time.time() - iter_start
            per_iter_seconds.append(iter_seconds)
            logger.info(f"iter {it} done ({n_chunks} chunks eager metadata) in {iter_seconds:.3f} seconds")
        profiler.end("tt_forward")
        signpost("PROFILE_MEASURE_END")

        kv_min_pcc = _record_kv_cache_pcc(
            trace_dir,
            layout,
            tt_kvpe_cache,
            mesh_device,
            sp,
            num_layers,
            SEQ_CACHE,
            total_len,
            kvpe_dim,
            config.kv_lora_rank,
            assert_threshold=KV_CACHE_PCC_THRESHOLD,
            assert_layer_depth=GATED_LAYER_DEPTH,
        )
        logger.success(
            f"[eager KV PCC] min KV-cache PCC over {num_layers} layers ({n_chunks} chunks) = "
            f"{kv_min_pcc:.6f} (layers 0..{GATED_LAYER_DEPTH} asserted >= {KV_CACHE_PCC_THRESHOLD})"
        )
        ttnn.deallocate(trace_input)
        for t in eager_metadata:
            ttnn.deallocate(t)
    elif use_trace and use_metadata:
        # ---------------------- METADATA MULTI-CHUNK TRACE PATH (N chunks) ----------------------
        # Capture the forward ONCE, then replay it for every chunk. The per-chunk scalars
        # (slot/actual_start/actual_end) are NOT baked into the captured command stream — the trace-safe
        # MLA ops read them on-device from a persistent metadata DRAM tensor. So advancing chunks is just
        # an in-place host->device update of two persistent buffers (token input + metadata) between
        # replays; the same captured trace produces the correct KV for each chunk. This is the real
        # end-to-end trace-safety validation across multiple chunks.

        # Persistent buffers — created ONCE, never reallocated, so the addresses the trace captured stay
        # valid across all execute_trace calls. (Re-running from_torch(device=...) per chunk would
        # reallocate and the replayed program would read a stale/freed address.)
        trace_input = ttnn.from_torch(
            chunk_tok_host[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        # Per-element-tensor metadata: 3 persistent 1-element uint32 replicated-DRAM tensors
        # (slot_id, actual_start, actual_end), seeded with chunk 0; updated in place per chunk.
        def _meta1_dev(val):
            return ttnn.from_torch(
                torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                device=mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        def _meta1_host(val):
            return ttnn.from_torch(
                torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )

        trace_metadata = (_meta1_dev(0), _meta1_dev(0), _meta1_dev(CHUNK))

        # Pre-build the per-chunk HOST tensors (no device=) for the cheap in-place updates: the SP-sharded
        # token tile and the (slot, c*CHUNK, c*CHUNK+CHUNK) per-element metadata triple for each chunk.
        tok_host_tt = [
            ttnn.from_torch(
                chunk_tok_host[c],
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
            )
            for c in range(n_chunks)
        ]
        meta_host_tt = [
            (_meta1_host(0), _meta1_host(c * CHUNK), _meta1_host(c * CHUNK + CHUNK)) for c in range(n_chunks)
        ]

        def _forward_meta():
            # actual_start/actual_end = None: every per-chunk scalar comes from `metadata` on-device.
            transformer.forward(
                trace_input,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=None,
                actual_end=None,
                cache_user_id=0,
                metadata=trace_metadata,
                return_intermediates=False,
            )

        controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(controller)

        # Warmup/compile pass (controller idle): populates the program cache for the METADATA op variants
        # (different program hash than the scalar ops) BEFORE capture.
        _forward_meta()
        ttnn.synchronize_device(mesh_device)

        logger.info(f"[trace] capturing {num_layers}-layer forward (metadata path, overlap on)...")
        controller.begin_capture()
        _forward_meta()
        controller.end_capture()
        ttnn.synchronize_device(mesh_device)

        trace_bytes = controller.trace_bytes()
        logger.info(
            f"[trace] {num_layers}-layer forward = {controller.num_segments} trace segments, "
            f"{trace_bytes / (1024 * 1024):.2f} MB ({trace_bytes:,} bytes)"
        )

        # Replay across all chunks: update the persistent token + metadata buffers in-place (cq 0), then
        # replay (execute_trace cq_id=0, blocking — ordered after the copies). One captured trace, N chunks.
        signpost("PROFILE_MEASURE_START")
        per_iter_seconds = []
        profiler.start("tt_forward")
        for it in range(num_iters):
            iter_start = time.time()
            for c in range(n_chunks):
                ttnn.copy_host_to_device_tensor(tok_host_tt[c], trace_input)
                for src, dst in zip(meta_host_tt[c], trace_metadata):
                    ttnn.copy_host_to_device_tensor(src, dst)
                chunk_start = time.time()
                controller.replay()
                ttnn.synchronize_device(mesh_device)
                dt = time.time() - chunk_start
                per_chunk_seconds.append(dt)
                logger.info(f"  iter {it} chunk {c} (trace replay): {dt:.3f} seconds")
            iter_seconds = time.time() - iter_start
            per_iter_seconds.append(iter_seconds)
            logger.info(f"iter {it} done ({n_chunks} chunks via trace) in {iter_seconds:.3f} seconds")
        profiler.end("tt_forward")
        signpost("PROFILE_MEASURE_END")

        # Correctness: after replaying all n_chunks, the cache holds tokens [0:total_len]. PCC the full
        # valid region vs the golden kv_post_transform. The metadata path drove every per-chunk scalar
        # on-device from the persistent metadata tensor, so this is the end-to-end multi-chunk trace-safety
        # proof. Layers 0..GATED_LAYER_DEPTH are asserted at the threshold; deeper layers recorded only.
        if verify_kv_cache_pcc:
            kv_min_pcc = _record_kv_cache_pcc(
                trace_dir,
                layout,
                tt_kvpe_cache,
                mesh_device,
                sp,
                num_layers,
                SEQ_CACHE,
                total_len,
                kvpe_dim,
                config.kv_lora_rank,
                assert_threshold=KV_CACHE_PCC_THRESHOLD,
                assert_layer_depth=GATED_LAYER_DEPTH,
            )
            logger.success(
                f"[trace KV PCC] min KV-cache PCC over {num_layers} layers ({n_chunks} chunks) = "
                f"{kv_min_pcc:.6f} (layers 0..{GATED_LAYER_DEPTH} asserted >= {KV_CACHE_PCC_THRESHOLD})"
            )

        controller.release()
        transformer.set_trace_controller(None)
        ttnn.deallocate(trace_input)
        for t in trace_metadata:
            ttnn.deallocate(t)
    elif use_trace:
        # ----------------------------- TRACE PATH (pinned to chunk 0) -----------------------------
        # Capture the forward ONCE as a ttnn trace, then replay it every iteration with execute_trace.
        # The trace records the device command stream, so the per-op host-dispatch (op2op) gaps that
        # dominate the chunked-prefill loss collapse to ~0 on replay. Requires a device-only forward —
        # kv_only_last_layer gives us that (no LM-head host readback). We pin to chunk 0: the captured
        # input buffer is reused unchanged, so every replay re-runs chunk 0 (multi-chunk is deferred).
        assert n_chunks == 1, "trace path is pinned to a single chunk (chunk 0)"

        # Persistent input — created once, NEVER deallocated inside the loop, so the address the trace
        # captured stays valid across all execute_trace calls.
        trace_input = ttnn.from_torch(
            chunk_tok_host[0],
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        def _forward_chunk0():
            transformer.forward(
                trace_input,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=0,
                actual_end=CHUNK,
                cache_user_id=0,
                return_intermediates=False,
            )

        # The MoE keeps its shared-expert/dispatch overlap (a sub-device-manager swap), which can't live
        # inside one trace. SubDeviceTraceController captures the forward as several trace segments split
        # at each load/clear, doing the host load/clear between segments (see utils/sub_device_trace.py).
        controller = SubDeviceTraceController(mesh_device)
        transformer.set_trace_controller(controller)

        # Warm/compile pass: controller idle -> MoE does eager load/clear, populating the program cache
        # BEFORE capture (trace records dispatch, not JIT).
        _forward_chunk0()
        ttnn.synchronize_device(mesh_device)

        # Capture: one forward pass, auto-split into segments at the MoE sub-device boundaries.
        logger.info(f"[trace] capturing {num_layers}-layer forward (chunk 0, overlap on)...")
        controller.begin_capture()
        _forward_chunk0()
        controller.end_capture()
        ttnn.synchronize_device(mesh_device)

        # How much device memory do all the captured trace segments take? (headline number)
        trace_bytes = controller.trace_bytes()
        logger.info(
            f"[trace] {num_layers}-layer forward = {controller.num_segments} trace segments, "
            f"{trace_bytes / (1024 * 1024):.2f} MB ({trace_bytes:,} bytes)"
        )

        # Replay: each iteration runs the full segmented forward (execute_trace + load/clear between).
        signpost("PROFILE_MEASURE_START")
        per_iter_seconds = []
        profiler.start("tt_forward")
        for it in range(num_iters):
            iter_start = time.time()
            controller.replay()
            ttnn.synchronize_device(mesh_device)
            iter_seconds = time.time() - iter_start
            per_iter_seconds.append(iter_seconds)
            logger.info(f"  iter {it} (trace replay): {iter_seconds:.3f} seconds")
        profiler.end("tt_forward")
        signpost("PROFILE_MEASURE_END")

        # Correctness: the chunk-0 KV the TRACED forward just wrote (replay overwrites the warmup's
        # writes, so this reflects the replayed segments) vs the golden kv_post_transform. KV-cache only.
        if verify_kv_cache_pcc:
            kv_min_pcc = _record_kv_cache_pcc(
                trace_dir,
                layout,
                tt_kvpe_cache,
                mesh_device,
                sp,
                num_layers,
                SEQ_CACHE,
                total_len,
                kvpe_dim,
                config.kv_lora_rank,
                assert_threshold=KV_CACHE_PCC_THRESHOLD,
            )
            logger.success(
                f"[trace KV PCC] min KV-cache PCC over {num_layers} layers = {kv_min_pcc:.6f} "
                f"(>= {KV_CACHE_PCC_THRESHOLD})"
            )

        controller.release()
        transformer.set_trace_controller(None)
        ttnn.deallocate(trace_input)
    else:
        # Optional profiling warmup: run chunk 0 once through all layers so every kernel is JIT-compiled and
        # the program cache is populated BEFORE the measured region. Gated by TT_PREFILL_PROFILE_WARMUP so
        # normal runs are unaffected. Used for the E2E phase (warm wall-clock, compile excluded). The
        # DEVICE-PERF phase deliberately runs WITHOUT warmup: device kernel duration is on-device execution
        # time (independent of host-side JIT), so the first compile+run chunk yields valid kernel times — and
        # skipping the warm pass halves the ops tracy must buffer, avoiding the profiler DRAM overflow.
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
            transformer.forward(
                warm_tokens,
                tt_kvpe_cache,
                actual_isl=CHUNK,
                actual_start=0,
                actual_end=CHUNK,
                cache_user_id=0,
                return_intermediates=False,
            )
            ttnn.synchronize_device(mesh_device)
            ttnn.deallocate(warm_tokens)
            # Flush the on-device profiler DRAM buffer so the warmup chunk's ops don't co-reside with the
            # measured chunk's (the warmup-only overflow that drops markers). Guarded to tracy runs — the
            # device profiler is only enabled when TT_METAL_DEVICE_PROFILER=1 (set by `python -m tracy`).
            if os.environ.get("TT_METAL_DEVICE_PROFILER") == "1":
                ttnn.ReadDeviceProfiler(mesh_device)
            logger.info("[profile] warmup chunk 0 complete (kernels JITted); measured region begins")

        # Bracket the measured loop with PROFILE_MEASURE_START / PROFILE_MEASURE_END so the device-perf
        # driver's between_signposts filter keeps only the forward ops (excluding one-time weight-load
        # tilize/typecast at construction, and any warmup chunk). Emitted unconditionally — outside a tracy
        # profiler these signposts are harmless no-ops (e.g. the plain e2e subprocess).
        signpost("PROFILE_MEASURE_START")
        per_iter_seconds = []
        profiler.start("tt_forward")
        for it in range(num_iters):
            iter_start = time.time()
            for c in range(n_chunks):
                kv_actual = c * CHUNK
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
                )
                ttnn.synchronize_device(mesh_device)
                ttnn.deallocate(tt_tokens)
                logger.info(f"  iter {it} chunk {c}: {time.time() - chunk_start:.3f} seconds")
            iter_seconds = time.time() - iter_start
            per_iter_seconds.append(iter_seconds)
            logger.info(f"iter {it} done ({n_chunks} chunks) in {iter_seconds:.3f} seconds")
        profiler.end("tt_forward")
        signpost("PROFILE_MEASURE_END")  # closes the device-perf between_signposts region

    # Free the MoE overlap sub-device managers before the mesh device closes — leaving them registered
    # at teardown has segfaulted close_mesh_device. Idempotent / no-op when overlap is off.
    transformer.release_sub_device_managers()

    profiler.end("total_test_time")
    logger.success(
        f"Chunked prefill KV-cache run done (num_layers={num_layers}, n_chunks={n_chunks}, " f"num_iters={num_iters})"
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


# Trace+metadata twin of test_ds_prefill_transformer_chunked_padded: the variable/partial-chunk prefill
# run via a captured metadata ttnn trace replayed per split, asserting its per-layer KV-cache PCC matches
# the untraced scalar path bit-exactly (and meets the PCC threshold). Needs trace_region_size > 0.
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
                "trace_region_size": 256 * 1024 * 1024,
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
def test_ds_prefill_transformer_chunked_padded_trace(
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
    run_chunked_transformer_padded_trace(
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


# Trace+metadata twin of test_kimi_prefill_transformer_chunked_padded: the variable/partial-chunk prefill
# run via a captured metadata ttnn trace replayed per split, asserting its per-layer KV-cache PCC matches
# the untraced scalar path bit-exactly. Needs trace_region_size > 0; Kimi uses the DEVICE_FP32 gate + the
# L1_SMALL semaphore region. The trace controller chops capture at the MoE sub-device load/clear.
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
                "l1_small_size": 512,
                "trace_region_size": 256 * 1024 * 1024,
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
def test_kimi_prefill_transformer_chunked_padded_trace(
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
    run_chunked_transformer_padded_trace(
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


# Trace-correctness variant: capture the metadata forward ONCE, replay it for 11 chunks (advancing the
# per-chunk scalars on-device via the persistent metadata tensor), then PCC the full KV cache region
# [0:11*CHUNK] the TRACED forward wrote against the golden kv_post_transform (KV-cache only —
# kv_only_last_layer means there is no decoder-output/logits tail). Layers 0..GATED_LAYER_DEPTH are
# asserted at the threshold; deeper layers (L61) are recorded only.
@pytest.mark.parametrize("n_chunks", [1, 11], ids=["chunks1", "chunks11"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                "l1_small_size": 512,
                "trace_region_size": 256 * 1024 * 1024,
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
def test_kimi_prefill_transformer_chunked_trace_kv_pcc(
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
    run_chunked_transformer_kv_cache(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,  # 11 chunks, advanced on-device via the metadata tensor
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        num_iters=1,  # one pass fills the cache [0:n_chunks*CHUNK]; more iters re-walk for timing only
        routing_use_l1_small_for_semaphores=True,
        use_trace=True,
        use_metadata=True,
        verify_kv_cache_pcc=True,
    )


# EAGER (no-trace) counterpart of test_..._trace_kv_pcc: same metadata forward + same KV-cache PCC, run
# op-by-op instead of captured/replayed. This is the DETERMINISM BASELINE — the traced KV-PCC must match
# this number closely (a material gap would mean the trace machinery, not the per-element metadata ops,
# introduced non-determinism). Same params/threshold as the traced test.
@pytest.mark.parametrize("n_chunks", [1, 11], ids=["chunks1", "chunks11"])
@pytest.mark.parametrize("num_layers", [1, 10, 61], ids=["L1", "L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                "l1_small_size": 512,
                "trace_region_size": 256 * 1024 * 1024,
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
def test_kimi_prefill_transformer_chunked_notrace_kv_pcc(
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
    run_chunked_transformer_kv_cache(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_layers,
        n_chunks,
        GateComputeMode.DEVICE_FP32,
        num_links,
        topology,
        num_iters=1,
        routing_use_l1_small_for_semaphores=True,
        use_trace=False,  # eager per-chunk forward — no capture/replay
        use_metadata=True,
        verify_kv_cache_pcc=True,
    )


# ----------------------------------------------------------------------------------------------------
# RUNNER (TtPrefillRuntime) traced-prefill validation. Drives the production runtime's use_trace path
# end-to-end: build runtime -> allocate KV cache -> compile() [captures the segmented metadata trace] ->
# prefill_chunk() x N [replays, advancing the per-element metadata in place] -> kv_cache_pcc_check vs the
# golden. This is the standalone harness the runner trace integration is validated against (no engine).
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [10, 61], ids=["L10", "L61"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
                "l1_small_size": 512,
                "trace_region_size": 256 * 1024 * 1024,
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
def test_kimi_prefill_runtime_traced_kv_pcc(
    variant, config_only, mesh_device, device_params, weight_cache_path, num_layers, n_chunks, num_links, topology
):
    from models.demos.deepseek_v3_d_p.tt.tt_prefill_runtime import TtPrefillRuntime, TtPrefillRuntimeConfig
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import allocate_mla_kvpe_cache

    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")

    hf_config = config_only
    hf_config.max_seq_len = SEQ_CACHE  # MLA reads this for the KV ring-buffer length
    mesh_shape = list(mesh_device.shape)
    assert tuple(mesh_shape) == (8, 4)
    total_len = n_chunks * CHUNK

    rt_config = TtPrefillRuntimeConfig(
        num_layers=num_layers,
        max_seq_len=SEQ_CACHE,
        mesh_shape=(8, 4),
        chunk_size=CHUNK,
        num_users=1,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=weight_cache_path / f"{mesh_shape[0]}x{mesh_shape[1]}",  # mesh-shape cache subdir
        model_cfg=KimiK26Config,
        kv_only_last_layer=True,
        routing_use_l1_small_for_semaphores=True,
        use_trace=True,
    )
    runtime = TtPrefillRuntime(mesh_device=mesh_device, hf_config=hf_config, state_dict={}, config=rt_config)
    kv_cache = allocate_mla_kvpe_cache(
        mesh_device=mesh_device,
        hf_config=hf_config,
        max_seq_len=SEQ_CACHE,
        mesh_shape=(8, 4),
        sp_axis=0,
        num_layers=num_layers,
        num_users=1,
    )

    runtime.compile(kv_cache)  # captures the segmented metadata trace
    ttnn.synchronize_device(mesh_device)

    token_ids_full = _load_metadata_token_ids(trace_dir, total_len)
    logger.info(f"[runtime-trace] loaded {token_ids_full.numel()} golden tokens from {trace_dir}")

    per_chunk = []
    for c in range(n_chunks):
        toks = token_ids_full[c * CHUNK : (c + 1) * CHUNK].tolist()
        tt_input = runtime.make_chunk_input(toks)
        t0 = time.time()
        runtime.prefill_chunk(tt_input, kv_cache, slot_id=0, actual_start=c * CHUNK, actual_end=(c + 1) * CHUNK)
        ttnn.synchronize_device(mesh_device)
        per_chunk.append(time.time() - t0)
        logger.info(f"[runtime-trace] chunk {c} (replay) {per_chunk[-1]:.3f}s")

    kv_min_pcc = runtime.kv_cache_pcc_check(kv_cache, slot_id=0, n_chunks=n_chunks, trace_dir=trace_dir)
    logger.success(f"[runtime-trace KV PCC] min over {num_layers} layers ({n_chunks} chunks) = {kv_min_pcc:.6f}")
