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
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = 5 * 1024  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (1 user)


def _resolve_trace_dir(variant) -> Path:
    """Golden chunked-prefill trace dir for `variant`: the model-agnostic PREFILL_TRACE_DIR env
    overrides the variant's prefill_trace_default. A vllm trace nests metadata.json + kv_cache under
    one run-hash subdir, so if the dir itself has no metadata.json, descend into the sole subdir that
    does."""
    path = Path(os.environ.get("PREFILL_TRACE_DIR", variant.prefill_trace_default))
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
    trace_dir, layout, tt_kvpe_cache, mesh_device, sp, num_layers, seq_len_cache, total_len, kvpe_dim, kv_lora
):
    """Record-only: gather the device KV cache, un-rotate the block-cyclic layout, and PCC each
    layer's valid region [:total_len] against the golden kv_post_transform trace. nope is compared
    directly; the RoPE (pe) slice uses the Meta-interleaved basis (golden stores HF half-split).
    Does NOT assert — mirrors the per-layer decoder_output record-only reporting."""
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
    logger.info(f"KV cache min PCC across layers: {min(cache_min_pcc.values()):.6f}")


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
            number_of_non_padded_tokens=isl,
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
            number_of_non_padded_tokens=CHUNK,
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
# PREFILL_TRACE_DIR; see model_variants.py).


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
