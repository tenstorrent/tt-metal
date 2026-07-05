# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Perf-only end-to-end chunked-prefill timing for TtPrefillTransformer (DeepSeek V3).

Drives the full model with SYNTHETIC tokens in 5120-token (5*1024) chunks into a single-user KV
cache and reports device-synchronized wall-clock per chunk plus prefill throughput. Unlike
test_prefill_transformer_chunked.py there is no golden-trace / PCC comparison and no per-layer host
snapshot (return_intermediates=False) — the only external dependency is the prebuilt TTNN weight
cache. The first chunk pays JIT/program-cache compilation; steady-state (chunks 1..N) is reported
separately and drives the throughput number.

Requires the weight cache (set the variant's ttnn_cache_env, e.g. TT_DS_PREFILL_TTNN_CACHE).

Caveat: tokens are random, so MoE routing is ~uniform and lacks the in-column hotspots real corpora
produce; the dispatch/combine share of the number is a routing-averaged estimate, not a worst case.
"""

import gc
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole, profiler
from models.demos.deepseek_v3_d_p.reference.deepseek_v3_config import DeepSeekV3Config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import init_kvpe_cache

CHUNK = 5 * 1024  # 5120 tokens per chunk
SEQ_CACHE = 55 * 1024  # 56320 KV cache length (one user); bounds n_chunks * CHUNK


def run_prefill_perf(
    variant,
    config,
    mesh_device,
    weight_cache_path,
    num_layers,
    n_chunks,
    gate_fallback_mode,
    num_links,
    topology,
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")

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
    vocab = getattr(config, "vocab_size", None) or variant.model_config.VOCAB_SIZE

    logger.info(
        f"prefill perf: num_layers={num_layers} mesh={mesh_shape} n_chunks={n_chunks} "
        f"total_len={total_len} cache={SEQ_CACHE} chunk={CHUNK}"
    )

    effective_cache_path = weight_cache_path / f"{sp}x{tp}"
    experts_per_chip = variant.model_config.NUM_ROUTED_EXPERTS // (sp * tp)
    assert TtPrefillTransformer.check_cache_complete(
        effective_cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=variant.model_config.NUM_DENSE_LAYERS,
    ), f"TTNN cache incomplete for {num_layers} layers at {effective_cache_path}"

    profiler.clear()
    profiler.start("tt_transformer_creation")
    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=SEQ_CACHE,
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

    mesh_device.enable_program_cache()
    torch.manual_seed(0)

    # Per-chunk device-synchronized wall-clock. Chunk 0 pays program-cache compilation; chunks 1..N-1
    # are steady state (same shapes, only kv position advances via runtime args) and drive throughput.
    per_chunk_ms = []
    for c in range(n_chunks):
        kv_actual = c * CHUNK
        # Random valid token ids, SP-sharded on dim 0 as [sp, 1, chunk_local]. Values only feed the
        # embedding lookup + routing; they don't need to be a real sequence for a timing measurement.
        chunk_tok = torch.randint(0, vocab, (sp, 1, chunk_local), dtype=torch.int64)
        tt_tokens = ttnn.from_torch(
            chunk_tok,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_shape), dims=(0, None)),
        )

        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
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
        dt_ms = (time.perf_counter() - t0) * 1000.0
        per_chunk_ms.append(dt_ms)
        logger.info(f"  chunk {c} (kv_actual={kv_actual}): {dt_ms:.2f} ms")

    cold_ms = per_chunk_ms[0]
    warm = per_chunk_ms[1:] if len(per_chunk_ms) > 1 else per_chunk_ms
    warm_total_ms = sum(warm)
    warm_mean_ms = warm_total_ms / len(warm)
    warm_tokens = len(warm) * CHUNK
    warm_tps = warm_tokens / (warm_total_ms / 1000.0)
    full_total_ms = sum(per_chunk_ms)

    logger.success(
        f"prefill perf (num_layers={num_layers}, {n_chunks} chunks x {CHUNK}):\n"
        f"  creation:        {profiler.get('tt_transformer_creation') * 1000:.1f} ms\n"
        f"  chunk[0] (cold): {cold_ms:.1f} ms  (includes program-cache compile)\n"
        f"  chunk warm mean: {warm_mean_ms:.1f} ms  over {len(warm)} chunks\n"
        f"  forward total:   {full_total_ms:.1f} ms  ({total_len} tokens)\n"
        f"  throughput:      {warm_tps:,.0f} tok/s  (steady-state, cold chunk excluded)"
    )


@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [10, 61], ids=["L10", "L61"])
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
def test_ds_prefill_transformer_perf(
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
    run_prefill_perf(
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
