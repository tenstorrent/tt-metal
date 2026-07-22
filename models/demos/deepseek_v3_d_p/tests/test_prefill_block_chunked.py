# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Chunked-prefill test for TtPrefillBlock (DeepSeek V3, first MoE layer = layer 3).

Processes a long sequence in chunks of 5*1024 = 5120 tokens, writing into a KV cache
allocated for ONE user of sequence length 55*1024 = 56320. The MoE path is orthogonal to
attention; only the MLA path exercises the chunked-prefill code.

Validates against the precomputed golden DeepSeek-R1 trace (variant.prefill_trace_default; override
with PREFILL_TRACE_DIR). For an N-chunk run we compare the first N*5120 tokens of the trace.

Weights come from the prebuilt TTNN cache ($TT_DS_PREFILL_TTNN_CACHE); the block builds with an
empty state_dict when the cache is complete. Requires an 8x4 Blackhole mesh and:

    TT_DS_PREFILL_TTNN_CACHE=/mnt/models/DeepSeek-R1-0528-Cache/DeepSeek-R1-0528-Cache-prefill_secure
    DEEPSEEK_V3_HF_MODEL=/mnt/models/deepseek-ai/DeepSeek-R1-0528
    TT_DS_PREFILL_HOST_REF_CACHE=/mnt/models/deepseek-prefill-cache/goldened
"""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.reference.glm_5_2_config import GLM52Config
from models.demos.deepseek_v3_d_p.reference.kimi_k2_6_config import KimiK26Config
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, num_full_indexer_layers, resolve_has_indexer
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
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
# Full 55k (56320) sequence in varied chunks: the requested prefix [1k,2k,3k,4k,5k,3k,2k,5k] (=25600),
# then a varied tail (=30720) of non-1024-aligned sizes that exercise mid-tile rotation offsets (e.g.
# 2592 -> 1-tile straddle, like test_mla's padded_partial). Every split is a multiple of 32 and <= CHUNK.
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


@dataclass(frozen=True)
class ChunkedThresholds:
    # Pinned from the 8x4 calibration run (all stages observed >= 0.9997 across 1..11 chunks).
    output: float = 0.99
    kv_nope: float = 0.99
    kv_pe: float = 0.99


THRESHOLDS = ChunkedThresholds()


def _load_trace_tensor(trace_dir: Path, subdir: str, layer: int, key: str, total_len: int) -> torch.Tensor:
    """Load `key` from trace_dir/<subdir>/layer_<layer>.safetensors, sliced to [:total_len]."""
    path = trace_dir / subdir / f"layer_{layer}.safetensors"
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(key)[:total_len].to(torch.float32)


def _pcc(label: str, ref: torch.Tensor, dev: torch.Tensor, thr: float) -> float:
    _, pcc = comp_pcc(ref.float(), dev.float())
    logger.info(f"  {label} PCC: {pcc:.6f} (threshold {thr})")
    assert pcc > thr, f"{label} PCC {pcc:.6f} below threshold {thr}"
    return pcc


def _pcc_pe(label: str, ref_pe: torch.Tensor, dev_pe: torch.Tensor, thr: float) -> float:
    """Compare a RoPE (pe) slice, trying both bases (the golden trace stores HF half-split, the
    device uses Meta interleave). Logs both and asserts on the better one."""
    _, direct = comp_pcc(ref_pe.float(), dev_pe.float())
    _, interleaved = comp_pcc(interleave_pe(ref_pe).float(), dev_pe.float())
    best = max(direct, interleaved)
    basis = "interleaved" if interleaved >= direct else "direct"
    logger.info(f"  {label} PCC: direct={direct:.6f} interleaved={interleaved:.6f} -> {best:.6f} [{basis}]")
    assert best > thr, f"{label} PCC {best:.6f} below threshold {thr}"
    return best


def _gather_kv(tt: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """SP-sharded (dim 2), TP-replicated KV intermediate -> natural [chunk, D] block-cyclic order."""
    full = ttnn.to_torch(
        tt,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)
    return full[:, :1][0, 0]  # TP replica 0 -> [chunk_size_global, D]


def run_chunked_block(
    variant, config, mesh_device, weight_cache_path, n_chunks, layer_idx, gate_fallback_mode, num_links, topology
):
    is_dense = layer_idx < variant.model_config.NUM_DENSE_LAYERS
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")

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
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank  # 576
    kv_lora = config.kv_lora_rank  # 512
    config.max_seq_len = SEQ_CACHE

    logger.info(
        f"chunked block: layer={layer_idx} ({'dense' if is_dense else 'moe'}) mesh={mesh_shape} "
        f"n_chunks={n_chunks} total_len={total_len} cache={SEQ_CACHE} chunk={CHUNK}"
    )

    # --- Golden trace: layer L input is layer L-1 decoder output; references for layer L. ---
    profiler.start("trace_loading")
    input_hidden = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx - 1, f"decoder_output_layer_{layer_idx - 1}", total_len
    )
    ref_out = _load_trace_tensor(trace_dir, "hidden_states", layer_idx, f"decoder_output_layer_{layer_idx}", total_len)
    ref_post_attn_norm = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx, f"post_attn_norm_layer_{layer_idx}", total_len
    )
    ref_post_mla_residual = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx, f"post_mla_residual_layer_{layer_idx}", total_len
    )
    g_compressed = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"compressed_kv_layer_{layer_idx}", total_len)
    g_nope = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_latent_normed_layer_{layer_idx}", total_len)
    g_rope = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_kpe_roped_layer_{layer_idx}", total_len)
    g_post = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_post_transform_layer_{layer_idx}", total_len)
    profiler.end("trace_loading")
    logger.info(f"loaded trace: input {tuple(input_hidden.shape)}, ref_out {tuple(ref_out.shape)}")

    # --- Weights from the prebuilt TTNN cache (empty state_dict when complete). ---
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    init_checker(effective_cache_path)
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillBlock.check_cache_complete(
        effective_cache_path, layer_idx, is_dense=is_dense, experts_per_chip=experts_per_chip
    ), f"TTNN cache incomplete for layer {layer_idx} at {effective_cache_path}"

    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        layer_idx=layer_idx,
        seq_len=CHUNK,  # per-chunk size -> MoE/FFN dispatch buffers
        max_seq_len=SEQ_CACHE,  # KV ring buffer = full cache
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        weight_cache_path=effective_cache_path,
        slot_num=1,
        layer_num=1,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    profiler.start("tt_block_creation")
    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_block_creation")

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=SEQ_CACHE, chunk_size_global=CHUNK)

    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=1,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[sp_axis] = -2
    hidden_shard_dims[tp_axis] = -1
    out_concat_dims = [None, None]
    out_concat_dims[sp_axis] = -2
    out_concat_dims[tp_axis] = -1

    # natural-order accumulators
    out_accum = torch.zeros(total_len, emb_dim, dtype=torch.float32)
    hidden_accum = {
        "post_attn_norm": torch.zeros(total_len, emb_dim, dtype=torch.float32),
        "post_mla_residual": torch.zeros(total_len, emb_dim, dtype=torch.float32),
    }
    kv_accum = {
        "tt_kv": torch.zeros(total_len, kvpe_dim, dtype=torch.float32),
        "tt_kv_nope": torch.zeros(total_len, kv_lora, dtype=torch.float32),
        "tt_kv_rope": torch.zeros(total_len, kvpe_dim - kv_lora, dtype=torch.float32),
        "tt_kvpe": torch.zeros(total_len, kvpe_dim, dtype=torch.float32),
    }

    mesh_device.enable_program_cache()

    profiler.start("tt_forward")
    for c in range(n_chunks):
        kv_actual = c * CHUNK  # chunk-aligned -> rotation degenerates, no pad masking needed
        valid_end = kv_actual + CHUNK  # full chunk (all positions real)
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        assert flat.min() >= kv_actual and flat.max() < kv_actual + CHUNK, "unexpected rotation for aligned chunk"

        chunk_in = input_hidden[flat].reshape(1, 1, CHUNK, emb_dim)
        tt_h = ttnn.from_torch(
            chunk_in,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=hidden_shard_dims),
        )

        tt_out, kvi = block.forward(
            tt_h,
            indexed_rope,
            tt_kvpe_cache,
            cache_layer_idx=0,
            actual_start=kv_actual,
            actual_end=valid_end,
            cache_user_id=0,
            return_kv_intermediates=True,
        )

        out_flat = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.float32)[0, 0]
        out_accum[flat] = out_flat

        for name in kv_accum:
            kv_accum[name][flat] = _gather_kv(kvi[name], mesh_device)

        # hidden intermediates: SP-sharded seq + TP-sharded hidden (same layout as the output).
        for name in hidden_accum:
            hidden_accum[name][flat] = ttnn.to_torch(
                kvi[name],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                ),
            ).to(torch.float32)[0, 0]

        ttnn.synchronize_device(mesh_device)
        logger.info(f"  chunk {c} done (kv_actual={kv_actual})")
    profiler.end("tt_forward")

    # --- PCC comparisons over [:total_len] ---
    profiler.start("pcc_validation")
    logger.info("Comparing KV intermediates vs golden trace:")
    _pcc("compressed_kv[nope]", g_compressed[:, :kv_lora], kv_accum["tt_kv"][:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("compressed_kv[pe]", g_compressed[:, kv_lora:], kv_accum["tt_kv"][:, kv_lora:], THRESHOLDS.kv_pe)
    _pcc("kv_latent_normed", g_nope, kv_accum["tt_kv_nope"], THRESHOLDS.kv_nope)
    _pcc_pe("kv_kpe_roped", g_rope, kv_accum["tt_kv_rope"], THRESHOLDS.kv_pe)
    _pcc("kv_post_transform[nope]", g_post[:, :kv_lora], kv_accum["tt_kvpe"][:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("kv_post_transform[pe]", g_post[:, kv_lora:], kv_accum["tt_kvpe"][:, kv_lora:], THRESHOLDS.kv_pe)

    logger.info("Comparing hidden intermediates vs golden trace:")
    _pcc("post_attn_norm", ref_post_attn_norm, hidden_accum["post_attn_norm"], THRESHOLDS.output)
    _pcc("post_mla_residual", ref_post_mla_residual, hidden_accum["post_mla_residual"], THRESHOLDS.output)

    logger.info("Comparing layer output vs golden decoder_output:")
    _pcc("layer_output", ref_out, out_accum, THRESHOLDS.output)

    # --- Independent sanity check: un-rotate the final device cache and compare to kv_post_transform. ---
    cache_sr = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1][
        0, 0
    ]  # [SEQ_CACHE, kvpe]
    p = blockcyclic_positions(sp, CHUNK, SEQ_CACHE)
    nat = torch.empty(SEQ_CACHE, kvpe_dim, dtype=torch.float32)
    nat[p] = cache_sr
    dev_cache = nat[:total_len]
    logger.info("Device cache sanity vs kv_post_transform:")
    _pcc("cache[nope]", g_post[:, :kv_lora], dev_cache[:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("cache[pe]", g_post[:, kv_lora:], dev_cache[:, kv_lora:], THRESHOLDS.kv_pe)
    profiler.end("pcc_validation")

    profiler.end("total_test_time")
    logger.success(f"Chunked prefill block test passed (n_chunks={n_chunks}, total_len={total_len})")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


@pytest.mark.parametrize("n_chunks", [1, 2, 5, 10, 11], ids=["chunks1", "chunks2", "chunks5", "chunks10", "chunks11"])
@pytest.mark.parametrize(
    "layer_idx, gate_fallback_mode",
    [(2, None), (3, GateComputeMode.DEVICE)],
    ids=["dense", "moe-gate_device"],
)
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
@pytest.mark.timeout(1800)
def test_ds_prefill_block_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    n_chunks,
    layer_idx,
    gate_fallback_mode,
    num_links,
    topology,
):
    run_chunked_block(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        n_chunks,
        layer_idx,
        gate_fallback_mode,
        num_links,
        topology,
    )


def run_chunked_block_multiuser(
    variant, config, mesh_device, weight_cache_path, num_users, layer_idx, gate_fallback_mode, num_links, topology
):
    """Multi-user slot routing: prefill ONE chunk into a single target slot of an num_users-slot KV
    cache, then assert (a) that slot un-rotates to the golden kv_post_transform and (b) every other
    slot stays zero. This is the cheapest exercise of the user-major flat-slot index
    (cache_user_id * layer_num + cache_layer_idx); a wrong mapping writes the wrong slot, so the
    target-slot PCC drops and an untouched slot becomes non-zero. The target is the LAST slot, the one
    a mapping that ignores cache_user_id would leave empty."""
    is_dense = layer_idx < variant.model_config.NUM_DENSE_LAYERS
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"

    chunk_local = CHUNK // sp
    target_slot = num_users - 1
    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    kv_lora = config.kv_lora_rank
    config.max_seq_len = SEQ_CACHE
    logger.info(f"multiuser block: layer={layer_idx} num_users={num_users} target_slot={target_slot}")

    input_hidden = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx - 1, f"decoder_output_layer_{layer_idx - 1}", CHUNK
    )
    g_post = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_post_transform_layer_{layer_idx}", CHUNK)

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    init_checker(effective_cache_path)
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillBlock.check_cache_complete(
        effective_cache_path, layer_idx, is_dense=is_dense, experts_per_chip=experts_per_chip
    ), f"TTNN cache incomplete for layer {layer_idx} at {effective_cache_path}"

    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        layer_idx=layer_idx,
        seq_len=CHUNK,
        max_seq_len=SEQ_CACHE,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        weight_cache_path=effective_cache_path,
        slot_num=num_users,
        layer_num=1,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode
    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=SEQ_CACHE, chunk_size_global=CHUNK)

    # num_users-slot cache (batch = num_users * 1 layer); only target_slot is written below.
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=num_users,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[sp_axis] = -2
    hidden_shard_dims[tp_axis] = -1

    mesh_device.enable_program_cache()

    positions = rotated_chip_positions(0, sp, chunk_local)  # chunk-aligned -> rotation degenerate
    flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
    chunk_in = input_hidden[flat].reshape(1, 1, CHUNK, emb_dim)
    tt_h = ttnn.from_torch(
        chunk_in,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=hidden_shard_dims),
    )
    block.forward(
        tt_h,
        indexed_rope,
        tt_kvpe_cache,
        cache_layer_idx=0,
        actual_start=0,
        actual_end=CHUNK,
        cache_user_id=target_slot,
    )
    ttnn.synchronize_device(mesh_device)

    # Read all slots: gather SP (dim2) + TP (dim1), keep TP replica 0 -> [num_users, 1, SEQ_CACHE, kvpe].
    cache_all = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1]
    p = blockcyclic_positions(sp, CHUNK, SEQ_CACHE)

    nat = torch.empty(SEQ_CACHE, kvpe_dim, dtype=torch.float32)
    nat[p] = cache_all[target_slot, 0]
    dev_target = nat[:CHUNK]
    logger.info(f"Target slot {target_slot} cache vs kv_post_transform:")
    _pcc("target[nope]", g_post[:, :kv_lora], dev_target[:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("target[pe]", g_post[:, kv_lora:], dev_target[:, kv_lora:], THRESHOLDS.kv_pe)

    for u in range(num_users):
        if u == target_slot:
            continue
        other_max = cache_all[u, 0].abs().max().item()
        logger.info(f"Untouched slot {u} max|cache| = {other_max:.3e}")
        assert other_max < 1e-3, f"slot {u} was written despite cache_user_id={target_slot} (max {other_max:.3e})"

    logger.success(f"Multi-user slot routing passed (num_users={num_users}, target_slot={target_slot})")


@pytest.mark.parametrize("num_users", [2], ids=["U2"])
@pytest.mark.parametrize("layer_idx, gate_fallback_mode", [(2, None)], ids=["dense"])
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
@pytest.mark.timeout(1800)
def test_ds_prefill_block_chunked_multiuser(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    num_users,
    layer_idx,
    gate_fallback_mode,
    num_links,
    topology,
):
    run_chunked_block_multiuser(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        num_users,
        layer_idx,
        gate_fallback_mode,
        num_links,
        topology,
    )


def run_chunked_block_padded(
    variant, config, mesh_device, weight_cache_path, splits, layer_idx, gate_fallback_mode, num_links, topology
):
    """Variable/partial chunked prefill: a single prompt of sum(splits) real tokens fed in chunks of
    variable real length `splits` (e.g. [1024, 4096]), each run as a full CHUNK-wide tile padded with
    zeros. Exercises the rotated + partial MLA path (the second chunk starts mid-slab). Per the MLA
    tests the order is REORDER (block-cyclic gather) then PAD (zero rows whose global pos >= valid_end).
    """
    is_dense = layer_idx < variant.model_config.NUM_DENSE_LAYERS
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")

    profiler.clear()
    profiler.start("total_test_time")

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp = mesh_shape[sp_axis]
    tp = mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"
    tile = ttnn.TILE_SIZE

    chunk_local = CHUNK // sp  # 640
    total_real = sum(splits)
    for v in splits:
        assert 0 < v <= CHUNK and v % tile == 0, f"split {v} must be tile-aligned and <= {CHUNK}"

    # Slab-aligned cache big enough for the largest rotated write (kv_actual + CHUNK), >= 2 slabs.
    # A mid-slab chunk (kv_actual not a multiple of CHUNK) spills its behind-boundary chips' pad rows
    # into the next slab, so the physical cache must cover kv_actual+CHUNK even though only total_real
    # tokens are valid.
    max_window = CHUNK * 2
    ka = 0
    for v in splits:
        max_window = max(max_window, ka + CHUNK)
        ka += v
    seq_len_cache = ((max_window + CHUNK - 1) // CHUNK) * CHUNK

    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    kv_lora = config.kv_lora_rank
    config.max_seq_len = seq_len_cache

    logger.info(
        f"chunked-padded block: layer={layer_idx} ({'dense' if is_dense else 'moe'}) mesh={mesh_shape} "
        f"splits={splits} total_real={total_real} cache={seq_len_cache} chunk={CHUNK}"
    )

    # --- Golden trace (sliced to the real-token count). ---
    profiler.start("trace_loading")
    input_hidden = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx - 1, f"decoder_output_layer_{layer_idx - 1}", total_real
    )
    ref_out = _load_trace_tensor(trace_dir, "hidden_states", layer_idx, f"decoder_output_layer_{layer_idx}", total_real)
    ref_post_attn_norm = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx, f"post_attn_norm_layer_{layer_idx}", total_real
    )
    ref_post_mla_residual = _load_trace_tensor(
        trace_dir, "hidden_states", layer_idx, f"post_mla_residual_layer_{layer_idx}", total_real
    )
    g_compressed = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"compressed_kv_layer_{layer_idx}", total_real)
    g_nope = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_latent_normed_layer_{layer_idx}", total_real)
    g_rope = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_kpe_roped_layer_{layer_idx}", total_real)
    g_post = _load_trace_tensor(trace_dir, "kv_cache", layer_idx, f"kv_post_transform_layer_{layer_idx}", total_real)
    profiler.end("trace_loading")

    # --- Block from the prebuilt TTNN cache. ---
    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    init_checker(effective_cache_path)
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillBlock.check_cache_complete(
        effective_cache_path, layer_idx, is_dense=is_dense, experts_per_chip=experts_per_chip
    ), f"TTNN cache incomplete for layer {layer_idx} at {effective_cache_path}"

    block_kwargs = dict(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        layer_idx=layer_idx,
        seq_len=CHUNK,
        max_seq_len=seq_len_cache,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        weight_cache_path=effective_cache_path,
        slot_num=1,
        layer_num=1,
    )
    if gate_fallback_mode is not None:
        block_kwargs["gate_fallback_mode"] = gate_fallback_mode

    profiler.start("tt_block_creation")
    block = TtPrefillBlock(**block_kwargs)
    ttnn.synchronize_device(mesh_device)
    profiler.end("tt_block_creation")

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=seq_len_cache, chunk_size_global=CHUNK)
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=seq_len_cache,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=1,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[sp_axis] = -2
    hidden_shard_dims[tp_axis] = -1
    out_concat_dims = [None, None]
    out_concat_dims[sp_axis] = -2
    out_concat_dims[tp_axis] = -1

    out_accum = torch.zeros(total_real, emb_dim, dtype=torch.float32)
    hidden_accum = {
        "post_attn_norm": torch.zeros(total_real, emb_dim, dtype=torch.float32),
        "post_mla_residual": torch.zeros(total_real, emb_dim, dtype=torch.float32),
    }
    kv_accum = {
        "tt_kv": torch.zeros(total_real, kvpe_dim, dtype=torch.float32),
        "tt_kv_nope": torch.zeros(total_real, kv_lora, dtype=torch.float32),
        "tt_kv_rope": torch.zeros(total_real, kvpe_dim - kv_lora, dtype=torch.float32),
        "tt_kvpe": torch.zeros(total_real, kvpe_dim, dtype=torch.float32),
    }

    mesh_device.enable_program_cache()

    profiler.start("tt_forward")
    ka = 0
    for c, isl in enumerate(splits):
        kv_actual = ka
        valid_end = kv_actual + isl
        ka += isl

        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = [positions[ch][r] for ch in range(sp) for r in range(chunk_local)]  # global pos, len CHUNK
        gather_idx = torch.tensor([min(gp, total_real - 1) for gp in flat], dtype=torch.long)
        chunk_in = input_hidden[gather_idx].clone()  # 1. REORDER (block-cyclic gather)
        pad_mask = torch.tensor([gp >= valid_end for gp in flat])
        chunk_in[pad_mask] = 0.0  # 2. PAD (zero rows whose global position is beyond the valid range)

        tt_h = ttnn.from_torch(
            chunk_in.reshape(1, 1, CHUNK, emb_dim),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=hidden_shard_dims),
        )
        tt_out, kvi = block.forward(
            tt_h,
            indexed_rope,
            tt_kvpe_cache,
            cache_layer_idx=0,
            actual_start=kv_actual,
            actual_end=valid_end,
            cache_user_id=0,
            return_kv_intermediates=True,
        )

        # Scatter only the VALID rows (global pos < valid_end) back to natural order.
        valid_pairs = [(row, gp) for row, gp in enumerate(flat) if gp < valid_end]
        src = torch.tensor([row for row, _ in valid_pairs], dtype=torch.long)
        dst = torch.tensor([gp for _, gp in valid_pairs], dtype=torch.long)

        out_flat = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.float32)[0, 0]
        out_accum[dst] = out_flat[src]

        for name in kv_accum:
            kv_accum[name][dst] = _gather_kv(kvi[name], mesh_device)[src]
        for name in hidden_accum:
            full = ttnn.to_torch(
                kvi[name],
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape
                ),
            ).to(torch.float32)[0, 0]
            hidden_accum[name][dst] = full[src]

        ttnn.synchronize_device(mesh_device)
        logger.info(f"  chunk {c} done (kv_actual={kv_actual} isl={isl} valid_end={valid_end})")
    profiler.end("tt_forward")

    # --- PCC vs golden over the real tokens [:total_real] ---
    profiler.start("pcc_validation")
    logger.info("KV intermediates vs golden trace:")
    _pcc("compressed_kv[nope]", g_compressed[:, :kv_lora], kv_accum["tt_kv"][:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("compressed_kv[pe]", g_compressed[:, kv_lora:], kv_accum["tt_kv"][:, kv_lora:], THRESHOLDS.kv_pe)
    _pcc("kv_latent_normed", g_nope, kv_accum["tt_kv_nope"], THRESHOLDS.kv_nope)
    _pcc_pe("kv_kpe_roped", g_rope, kv_accum["tt_kv_rope"], THRESHOLDS.kv_pe)
    _pcc("kv_post_transform[nope]", g_post[:, :kv_lora], kv_accum["tt_kvpe"][:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("kv_post_transform[pe]", g_post[:, kv_lora:], kv_accum["tt_kvpe"][:, kv_lora:], THRESHOLDS.kv_pe)

    logger.info("Hidden intermediates vs golden trace:")
    _pcc("post_attn_norm", ref_post_attn_norm, hidden_accum["post_attn_norm"], THRESHOLDS.output)
    _pcc("post_mla_residual", ref_post_mla_residual, hidden_accum["post_mla_residual"], THRESHOLDS.output)
    _pcc("layer_output", ref_out, out_accum, THRESHOLDS.output)

    # --- Final: gather the device KV cache, un-rotate, compare the contiguous [:total_real] valid
    #     region to the trace's kv_post_transform (this is the "5k gathered from 2 chunks" check). ---
    cache_sr = ttnn.to_torch(
        tt_kvpe_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 1), mesh_shape=mesh_device.shape),
    ).to(torch.float32)[:, :1][
        0, 0
    ]  # [seq_len_cache, kvpe]
    p = blockcyclic_positions(sp, CHUNK, seq_len_cache)
    nat = torch.empty(seq_len_cache, kvpe_dim, dtype=torch.float32)
    nat[p] = cache_sr
    dev_cache = nat[:total_real]
    logger.info("Device KV cache (gathered across chunks) vs kv_post_transform:")
    _pcc("cache[nope]", g_post[:, :kv_lora], dev_cache[:, :kv_lora], THRESHOLDS.kv_nope)
    _pcc_pe("cache[pe]", g_post[:, kv_lora:], dev_cache[:, kv_lora:], THRESHOLDS.kv_pe)
    profiler.end("pcc_validation")

    profiler.end("total_test_time")
    logger.success(f"Chunked-padded block test passed (splits={splits}, total_real={total_real})")
    for key in profiler.times:
        logger.info(f"  {key}: {profiler.get(key) * 1000:.2f} ms")


@pytest.mark.parametrize("splits", [[1024, 4096], _PADDED_FULL_55K], ids=["1k+4k", "full55k"])
@pytest.mark.parametrize(
    "layer_idx, gate_fallback_mode",
    [(2, None), (3, GateComputeMode.DEVICE)],
    ids=["dense", "moe-gate_device"],
)
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
@pytest.mark.timeout(3600)
def test_ds_prefill_block_chunked_padded(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    splits,
    layer_idx,
    gate_fallback_mode,
    num_links,
    topology,
):
    run_chunked_block_padded(
        variant, config_only, mesh_device, weight_cache_path, splits, layer_idx, gate_fallback_mode, num_links, topology
    )


# ---------------------------------------------------------------------------
# Kimi K2.6 variants
# ---------------------------------------------------------------------------
# Same chunked-prefill machinery as the DeepSeek tests, with the kimi_k2_6 variant: the host gate
# (GateComputeMode.HOST_ALL — Kimi has a single expert group and is validated only with the host
# gate) and KimiK26Config fabric payload size. Kimi has a single dense layer (NUM_DENSE_LAYERS=1,
# layer 0); the block test reads layer L-1's decoder output as layer L's input, so we cannot drive
# the lone dense layer (would need layer -1) — only the first MoE layer (layer 1) is exercised.
# These skip until the Kimi golden trace lands (set PREFILL_TRACE_DIR; see tt/runners/adapters/).


@pytest.mark.parametrize("n_chunks", [1, 2, 5, 10, 11], ids=["chunks1", "chunks2", "chunks5", "chunks10", "chunks11"])
@pytest.mark.parametrize(
    "layer_idx, gate_fallback_mode",
    [(1, GateComputeMode.HOST_ALL)],
    ids=["moe-gate_host"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
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
@pytest.mark.timeout(1800)
def test_kimi_prefill_block_chunked(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    n_chunks,
    layer_idx,
    gate_fallback_mode,
    num_links,
    topology,
):
    run_chunked_block(
        variant,
        config_only,
        mesh_device,
        weight_cache_path,
        n_chunks,
        layer_idx,
        gate_fallback_mode,
        num_links,
        topology,
    )


@pytest.mark.parametrize("splits", [[1024, 4096], _PADDED_FULL_55K], ids=["1k+4k", "full55k"])
@pytest.mark.parametrize(
    "layer_idx, gate_fallback_mode",
    [(1, GateComputeMode.HOST_ALL)],
    ids=["moe-gate_host"],
)
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_1D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=KimiK26Config.FABRIC_PAYLOAD_SIZE),
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
@pytest.mark.timeout(3600)
def test_kimi_prefill_block_chunked_padded(
    variant,
    config_only,
    mesh_device,
    device_params,
    weight_cache_path,
    splits,
    layer_idx,
    gate_fallback_mode,
    num_links,
    topology,
):
    run_chunked_block_padded(
        variant, config_only, mesh_device, weight_cache_path, splits, layer_idx, gate_fallback_mode, num_links, topology
    )


# ---------------------------------------------------------------------------
# GLM DSA indexer-K teacher-forced check
# ---------------------------------------------------------------------------
# Feeds the golden decoder_output of layer_idx-1 as the block input (teacher-forced -> no cross-layer
# accumulation) and PCCs the device indexer-K cache + KVPE cache + block output against layer_idx's golden
# from the chunked_group_a_v1 indexer-kcache vLLM trace (PREFILL_TRACE_DIR). The PCC here (teacher-forced,
# expected ~1.0) isolates the per-layer op accuracy; contrast it with the chained transformer's per-layer
# PCC (which accumulates) to confirm the deep-layer sag is accumulation, not an indexer bug. The indexer_k
# golden is captured only for the DSA full-indexer layers (glm_5_1: all; glm_5_2: 0-2 + every 4th).


def run_chunked_block_glm_indexer(
    variant, config, mesh_device, weight_cache_path, n_chunks, layer_idx, num_links, topology
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    if not resolve_has_indexer(config):
        pytest.skip("indexer-K teacher-forced test is DSA-only (glm_5_1 / glm_5_2)")
    trace_dir = _resolve_trace_dir(variant)
    if not trace_dir.exists():
        pytest.skip(f"golden trace not found: {trace_dir}")
    idx_golden = trace_dir / "dsa" / f"indexer_k_layer_{layer_idx}"
    if not idx_golden.exists():
        pytest.skip(f"no indexer_k golden for layer {layer_idx} (not a captured full-indexer layer)")
    assert layer_idx >= 1, "teacher-forcing needs layer_idx-1's decoder output"

    sp_axis, tp_axis = 0, 1
    mesh_shape = list(mesh_device.shape)
    sp, tp = mesh_shape[sp_axis], mesh_shape[tp_axis]
    assert (sp, tp) == (8, 4), f"this test targets mesh-8x4, got {mesh_shape}"
    chunk_local = CHUNK // sp
    total_len = n_chunks * CHUNK
    assert total_len <= SEQ_CACHE, f"{n_chunks} chunks ({total_len}) exceed cache {SEQ_CACHE}"
    emb_dim = config.hidden_size
    kvpe_dim = config.qk_rope_head_dim + config.kv_lora_rank
    kv_lora = config.kv_lora_rank
    idx_dim = config.index_head_dim
    idx_rope = idx_dim // 2  # [rope | nope]; both compare directly for GLM (natively interleaved)
    config.max_seq_len = SEQ_CACHE
    is_dense = layer_idx < variant.model_config.NUM_DENSE_LAYERS

    logger.info(f"glm indexer block (teacher-forced): layer={layer_idx} n_chunks={n_chunks} total_len={total_len}")
    io = trace_dir / "decoder_io"
    input_hidden = read_sharded_rows(
        io / f"decoder_output_layer_{layer_idx - 1}", f"decoder_output_layer_{layer_idx - 1}", 0, total_len
    )
    ref_out = read_sharded_rows(
        io / f"decoder_output_layer_{layer_idx}", f"decoder_output_layer_{layer_idx}", 0, total_len
    )
    g_idx = read_sharded_rows(idx_golden, f"indexer_k_layer_{layer_idx}", 0, total_len)
    g_post = read_sharded_rows(
        trace_dir / "kv_cache" / f"layer_{layer_idx}", f"kv_post_transform_layer_{layer_idx}", 0, total_len
    )

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    init_checker(effective_cache_path)
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillBlock.check_cache_complete(
        effective_cache_path, layer_idx, is_dense=is_dense, experts_per_chip=experts_per_chip
    ), f"TTNN cache incomplete for layer {layer_idx} at {effective_cache_path}"

    block = TtPrefillBlock(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        layer_idx=layer_idx,
        seq_len=CHUNK,
        max_seq_len=SEQ_CACHE,
        dispatch_buffer_capacity_factor=8,
        num_links=num_links,
        topology=topology,
        sp_axis=sp_axis,
        tp_axis=tp_axis,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=effective_cache_path,
        slot_num=1,
        layer_num=1,
        routing_use_l1_small_for_semaphores=True,
    )
    ttnn.synchronize_device(mesh_device)

    rope_setup = RotarySetup(config, mesh_device, sp_axis=sp_axis, is_balanced=False)
    indexed_rope = rope_setup.get_rope_tensors_indexed(cache_seq_len_global=SEQ_CACHE, chunk_size_global=CHUNK)
    tt_kvpe_cache = init_kvpe_cache(
        kvpe_cache_head_dim=kvpe_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=1,
        num_users=1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    tt_index_kv_cache = init_kvpe_cache(
        kvpe_cache_head_dim=idx_dim,
        mesh_device=mesh_device,
        seq_len=SEQ_CACHE,
        mesh_shape=mesh_shape,
        sp_axis=sp_axis,
        num_kvpe_cache_layers=num_full_indexer_layers(config) or 1,
        num_users=1,
        dtype=ttnn.bfloat8_b,
    )

    hidden_shard_dims = [None, None]
    hidden_shard_dims[sp_axis] = -2
    hidden_shard_dims[tp_axis] = -1
    out_concat_dims = [None, None]
    out_concat_dims[sp_axis] = -2
    out_concat_dims[tp_axis] = -1
    out_accum = torch.zeros(total_len, emb_dim, dtype=torch.float32)

    mesh_device.enable_program_cache()
    for c in range(n_chunks):
        kv_actual = c * CHUNK
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        chunk_in = input_hidden[flat].reshape(1, 1, CHUNK, emb_dim)
        tt_h = ttnn.from_torch(
            chunk_in,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=hidden_shard_dims),
        )
        tt_out, _ = block.forward(
            tt_h,
            indexed_rope,
            tt_kvpe_cache,
            cache_layer_idx=0,
            actual_start=kv_actual,
            actual_end=kv_actual + CHUNK,
            cache_user_id=0,
            return_kv_intermediates=True,
            index_kv_cache=tt_index_kv_cache,
        )
        out_accum[flat] = ttnn.to_torch(
            tt_out,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=out_concat_dims, mesh_shape=mesh_device.shape),
        ).to(torch.float32)[0, 0]
        ttnn.synchronize_device(mesh_device)
        logger.info(f"  chunk {c} done (kv_actual={kv_actual})")

    p = blockcyclic_positions(sp, CHUNK, SEQ_CACHE)
    # Index cache is compact (GLM-5.2 reuse): this full layer wrote its full-indexer rank slot (== layer_idx
    # for glm_5_1). KVPE is per-layer and the block owns one slot (0).
    dev_idx = unrotate_cache_layer(
        gather_cache_tp0(tt_index_kv_cache, mesh_device)[full_indexer_rank(config, layer_idx)], p, total_len
    )
    dev_kvpe = unrotate_cache_layer(gather_cache_tp0(tt_kvpe_cache, mesh_device)[0], p, total_len)
    _, out_pcc = comp_pcc(ref_out, out_accum)
    idx_rope_pcc, idx_nope = cache_half_pccs(g_idx, dev_idx, idx_rope, pe_interleave=False)
    kv_nope, kv_pe = cache_half_pccs(g_post, dev_kvpe, kv_lora, pe_interleave=True)
    logger.info(f"[glm teacher-forced L{layer_idx}] block output PCC vs decoder_output: {out_pcc:.6f}")
    logger.info(f"[glm teacher-forced L{layer_idx}] indexer-K PCC: nope={idx_nope:.6f} rope={idx_rope_pcc:.6f}")
    logger.info(f"[glm teacher-forced L{layer_idx}] KVPE PCC: nope={kv_nope:.6f} pe(interleaved)={kv_pe:.6f}")
    # Teacher-forced single layer -> expect high PCC (op accuracy, no cross-layer accumulation).
    assert out_pcc >= 0.98, f"teacher-forced block output PCC {out_pcc:.6f} < 0.98"
    assert min(idx_nope, idx_rope_pcc) >= 0.98, f"teacher-forced indexer-K PCC {min(idx_nope, idx_rope_pcc):.6f} < 0.98"
    assert min(kv_nope, kv_pe) >= 0.98, f"teacher-forced KVPE PCC {min(kv_nope, kv_pe):.6f} < 0.98"
    logger.success(
        f"glm teacher-forced L{layer_idx}: output={out_pcc:.4f} "
        f"indexer-K min={min(idx_nope, idx_rope_pcc):.4f} KVPE min={min(kv_nope, kv_pe):.4f}"
    )


@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("layer_idx", [2, 6, 30, 62, 74], ids=lambda l: f"L{l}")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links, topology",
    [
        pytest.param(
            (8, 4),
            {
                "fabric_config": ttnn.FabricConfig.FABRIC_2D,
                "fabric_router_config": create_fabric_router_config(max_payload_size=GLM52Config.FABRIC_PAYLOAD_SIZE),
                "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
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
@pytest.mark.parametrize("variant", ["glm_5_2"], indirect=True, ids=["glm52"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA (indexer) is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_block_indexer_teacher_forced(
    variant, config_only, mesh_device, device_params, weight_cache_path, n_chunks, layer_idx, num_links, topology
):
    run_chunked_block_glm_indexer(
        variant, config_only, mesh_device, weight_cache_path, n_chunks, layer_idx, num_links, topology
    )
