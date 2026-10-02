# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 chunked prefill, batch 4: one user per mesh column.

The chunked transformer of test_glm_prefill_transformer_chunked, with the 8x4 mesh's column axis
carrying four users instead of tensor parallelism (TtPrefillTransformer(batch_axis=1)):

  * attention is SP=8 / TP=1 per column -- replicated norms + MLA weights, per-column KVPE and index-K
    caches, SP-axis rings for the KVPE gather and the fused ring indexer;
  * the embedding, the MoE and the dense FFN stay tensor-parallel with their existing weights, fed by
    an all-to-all that interleaves every user's rows (tt/batch_axis.py) -- the MoE keeps today's four
    dispatch groups, one per column.

Data: the GLM-5.3 55k vLLM trace, replicated onto all four columns, so the four users are identical
and every column is checked against the same golden: per-layer decoder output per chunk, and after the
run the KVPE cache (every layer) and the index-K cache (every full-indexer layer). Floors are the
single-user chunked test's.

  test_glm_build_batch_axis_ttnn_cache          -- writes the replicated (``*_repl``) norm + MLA tensorbins
                                                   next to the existing GLM-5.3 cache; run once first
  test_glm_prefill_transformer_chunked_batch4  -- the prefill, untraced
"""

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_transformer_chunked import (
    CHUNK,
    GLM_L1_SMALL_SIZE,
    INDEXER_K_PCC_THRESHOLD,
    KV_CACHE_PCC_THRESHOLD,
    LAYER_PCC_THRESHOLD,
    SEQ_CACHE,
    _load_layer_rows,
    _load_metadata_token_ids,
    _ref_layer_slice,
    _resolve_trace_dir,
)
from models.demos.deepseek_v3_d_p.tt.mla.indexer import full_indexer_rank, normalized_hadamard_matrix
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.tt.tt_prefill_transformer import TtPrefillTransformer
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, unrotate_cache_layer
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import comp_pcc

SP_AXIS, BATCH_AXIS = 0, 1
# Dispatch capacity per chip = 8 chips x (4 users x 640 rows) x factor. The single-user test runs 8
# (40960 rows of 640-row chunks); 4 here keeps the same absolute headroom per expert as factor 8 at
# 4x the tokens would, at half the 4x-wider transient buffer (~1 GB/chip instead of ~2).
DISPATCH_BUFFER_CAPACITY_FACTOR = 4

_MESH_PARAMS = [
    pytest.param(
        (8, 4),
        torus_xy_device_params(fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE, l1_small_size=GLM_L1_SMALL_SIZE),
        2,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="torus-xy-8x4",
    ),
]


def _batch_cache_complete(cache_path: Path, num_layers: int, config, experts_per_chip: int) -> bool:
    return TtPrefillTransformer.check_cache_complete(
        cache_path,
        num_layers,
        experts_per_chip=experts_per_chip,
        first_k_dense=GLM53Config.NUM_DENSE_LAYERS,
        batch_axis=True,
        config=config,
    )


# ---------------------------------------------------------------------------
# Weight provisioning: the replicated norm + MLA tensorbins batch-axis attention loads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_layers", [10, 78], ids=["L10", "L78"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM requires Blackhole")
@pytest.mark.timeout(0)
def test_glm_build_batch_axis_ttnn_cache(
    variant, config_only, mesh_device, device_params, num_links, num_layers, weight_cache_path, model_path
):
    """Write ``layer_{i}.{attn_norm,ffn_norm}_repl`` and ``layer_{i}.mla_repl.*`` (+ indexer on full layers)
    into the existing GLM-5.3 8x4 cache directory. The embedding, dense FFN and MoE tensorbins there are
    reused unchanged, so nothing else is written. Layers already complete are skipped (resumable)."""
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    config = config_only
    sp, cols = list(mesh_device.shape)
    cache_path = weight_cache_path / f"{sp}x{cols}"
    assert cache_path.exists(), f"base GLM-5.3 cache missing at {cache_path}; build it with test_glm_build_ttnn_cache"
    # The shared cache is group-readable for everyone (rw-rw-r--); a 0007 umask would write files only
    # cache-writers can read.
    old_umask = os.umask(0o002)
    # The fast checker snapshots the directory listing; the per-layer skip checks below read that
    # snapshot (taken before anything is written), and the final check re-snapshots.
    init_checker(cache_path)
    try:
        weight_map = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
        for layer_idx in range(num_layers):
            if TtPrefillBlock.check_cache_complete(
                cache_path,
                layer_idx,
                layer_idx < GLM53Config.NUM_DENSE_LAYERS,
                GLM53Config.NUM_ROUTED_EXPERTS // (sp * cols),
                batch_axis=True,
                config=config,
            ):
                logger.info(f"[batch-axis cache] layer {layer_idx}: complete, skipping")
                continue
            prefix = f"model.layers.{layer_idx}."
            shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
            mla_weights = pretrained_mla_weights(
                config, layer=layer_idx, checkpoint_path=[str(Path(model_path) / s) for s in shards]
            )
            norms = load_hf_state_dict_filtered(
                str(model_path), [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."]
            )
            TtPrefillBlock.build_batch_axis_ttnn_cache(
                {
                    "mla_weights": mla_weights,
                    "attn_norm_weight": norms[f"{prefix}input_layernorm.weight"].to(torch.bfloat16),
                    "ffn_norm_weight": norms[f"{prefix}post_attention_layernorm.weight"].to(torch.bfloat16),
                },
                layer_idx,
                cache_path,
                mesh_device,
                config,
            )
    finally:
        os.umask(old_umask)
    experts_per_chip = GLM53Config.NUM_ROUTED_EXPERTS // (sp * cols)
    assert _batch_cache_complete(cache_path, num_layers, config, experts_per_chip), "batch-axis cache still incomplete"


# ---------------------------------------------------------------------------
# Chunked prefill, batch 4
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_chunks", [11], ids=["chunks11"])
@pytest.mark.parametrize("num_layers", [10, 78], ids=["L10", "L78"])
@pytest.mark.parametrize("check_layer_pcc", [True], ids=["layer_pcc"])
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links", _MESH_PARAMS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA ops (indexer / sparse SDPA) are Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_transformer_chunked_batch4(
    variant,
    config_only,
    mesh_device,
    device_params,
    num_links,
    weight_cache_path,
    num_layers,
    n_chunks,
    check_layer_pcc,
):
    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    topology = per_axis_topology(device_params["fabric_config"])
    sp, num_users = mesh_device.shape[SP_AXIS], mesh_device.shape[BATCH_AXIS]
    assert (sp, num_users) == (8, 4), f"this test targets mesh-8x4, got {list(mesh_device.shape)}"
    chunk_local = CHUNK // sp
    total_len = n_chunks * CHUNK
    seq_cache = SEQ_CACHE
    assert total_len <= seq_cache

    # Shared lru_cache'd config: copy before mutating (as the single-user driver does).
    config = copy.deepcopy(config_only)
    config.max_seq_len = seq_cache
    rope_scaling = getattr(config, "rope_scaling", None)
    if isinstance(rope_scaling, dict) and rope_scaling.get("factor", 1.0) == 1.0:
        rope_scaling["original_max_position_embeddings"] = seq_cache
    emb_dim, kv_lora = config.hidden_size, config.kv_lora_rank

    trace_dir = _resolve_trace_dir(variant)
    layout = variant.prefill_trace_layout
    token_ids = _load_metadata_token_ids(trace_dir, total_len, require_full=True) % config.vocab_size

    cache_path = weight_cache_path / f"{sp}x{num_users}"
    experts_per_chip = GLM53Config.NUM_ROUTED_EXPERTS // (sp * num_users)
    assert _batch_cache_complete(cache_path, num_layers, config, experts_per_chip), (
        f"batch-axis TTNN cache incomplete for {num_layers} layers at {cache_path}; "
        "run test_glm_build_batch_axis_ttnn_cache first"
    )
    logger.info(
        f"[batch4] GLM-5.3 chunked prefill: {num_users} users x {total_len} tokens ({n_chunks} chunks), "
        f"{num_layers} layers"
    )

    transformer = TtPrefillTransformer(
        mesh_device=mesh_device,
        config=config,
        model_cfg=variant.model_config,
        state_dict={},
        num_layers=num_layers,
        seq_len=CHUNK,
        max_seq_len=seq_cache,
        dispatch_buffer_capacity_factor=DISPATCH_BUFFER_CAPACITY_FACTOR,
        num_links=num_links,
        topology=topology,
        sp_axis=SP_AXIS,
        tp_axis=BATCH_AXIS,
        is_balanced=False,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        weight_cache_path=cache_path,
        is_chunked=True,
        slot_num=1,
        kv_only_last_layer=True,
        routing_use_l1_small_for_semaphores=True,
        batch_axis=BATCH_AXIS,
    )
    ttnn.synchronize_device(mesh_device)

    # Per-column caches: user c on column c, striped over that column's 8 chips.
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_layers,
        num_users=1,
        batch_axis=BATCH_AXIS,
    )
    num_index_slots = full_indexer_rank(config, num_layers)
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=config.index_head_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_index_slots,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        batch_axis=BATCH_AXIS,
    )

    # Per-layer decoder-output PCC, taken at each block's end through on_layer_hidden (the
    # return_intermediates dict would hold every layer of every user on host at once).
    layer_min = {}  # layer -> min PCC over chunks and users
    cross_user_max_diff = 0.0
    state = {}

    def on_layer_hidden(layer_idx, x):
        if not check_layer_pcc:
            return
        users = transformer._to_host(x).to(torch.float32)  # [U, CHUNK, emb], block-cyclic row order
        ref = _ref_layer_slice(trace_dir, layout, layer_idx, state["kv_actual"], state["kv_actual"] + CHUNK)
        nonlocal cross_user_max_diff
        for u in range(num_users):
            natural = torch.empty(CHUNK, emb_dim, dtype=torch.float32)
            natural[state["local_pos"]] = users[u]
            _, pcc = comp_pcc(ref, natural)
            layer_min[layer_idx] = min(layer_min.get(layer_idx, 1.0), pcc)
            if u == 0:
                logger.info(f"  chunk {state['chunk']} layer {layer_idx} user 0 PCC: {pcc:.6f}")
            else:
                cross_user_max_diff = max(cross_user_max_diff, (users[u] - users[0]).abs().max().item())

    mesh_device.enable_program_cache()
    for c in range(n_chunks):
        kv_actual = c * CHUNK
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        state.update(chunk=c, kv_actual=kv_actual, local_pos=flat - kv_actual)
        # Every chip of mesh row r gets row-block r of EVERY user, users in column order (the layout the
        # TP embedding consumes; the transformer de-interleaves right after it). The users are the same
        # trace, so the four blocks are identical.
        rows = token_ids[flat].reshape(sp, 1, chunk_local)
        chunk_tokens = rows.repeat(1, 1, num_users)  # [sp, 1, U * chunk_local] = [u0 | u1 | u2 | u3]
        tt_tokens = ttnn.from_torch(
            chunk_tokens,
            device=mesh_device,
            dtype=ttnn.uint32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(0, None)),
        )
        transformer.forward(
            tt_tokens,
            kvpe_cache,
            actual_isl=CHUNK,
            actual_start=kv_actual,
            actual_end=kv_actual + CHUNK,
            cache_user_id=0,
            index_kv_cache=index_cache,
            on_layer_hidden=on_layer_hidden,
        )
        ttnn.synchronize_device(mesh_device)
        ttnn.deallocate(tt_tokens)
        logger.info(f"[batch4] chunk {c} done (kv_actual={kv_actual})")

    failures = []
    if check_layer_pcc:
        # kv_only_last_layer leaves the last layer with no output; the KVPE PCC below covers it.
        assert len(layer_min) == num_layers - 1, f"compared {len(layer_min)} layers, expected {num_layers - 1}"
        for i in sorted(layer_min):
            logger.info(f"[batch4] layer {i} min PCC over chunks and users: {layer_min[i]:.6f}")
        worst = min(layer_min.values())
        logger.info(f"[batch4] per-layer min PCC {worst:.6f}; users vs user 0 max|diff| {cross_user_max_diff:.3e}")
        if worst < LAYER_PCC_THRESHOLD:
            failures.append(f"per-layer min PCC {worst:.6f} < {LAYER_PCC_THRESHOLD}")

    # KVPE cache, every layer, every user. Per chip [L, 1, T/sp, 576] -> host [U*L, 1, T, 576] (user u owns
    # slots [u*L, (u+1)*L)); bf16 on host, one slot converted at a time.
    composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape)
    p = blockcyclic_positions(sp, CHUNK, seq_cache)
    kv_host = ttnn.to_torch(kvpe_cache.storage, mesh_composer=composer)
    kv_min = 1.0
    for i in range(num_layers):
        g = _load_layer_rows(trace_dir, layout, "kv_cache", i, f"kv_post_transform_layer_{i}", 0, total_len)
        per_user = []
        for u in range(num_users):
            dev = unrotate_cache_layer(kv_host[u * num_layers + i, 0].float(), p, total_len)
            per_user.append(min(cache_half_pccs(g, dev, kv_lora, pe_interleave=True)))
        kv_min = min(kv_min, *per_user)
        logger.info(f"[batch4] KVPE layer {i} PCC per user: {[round(v, 6) for v in per_user]}")
    del kv_host
    logger.info(f"[batch4] KVPE min PCC over layers and users: {kv_min:.6f}")
    if kv_min < KV_CACHE_PCC_THRESHOLD:
        failures.append(f"KVPE min PCC {kv_min:.6f} < {KV_CACHE_PCC_THRESHOLD}")

    # Index-K cache, every full-indexer layer with a golden. Stored in the Hadamard basis (H symmetric and
    # orthonormal, so one more multiply returns the golden's basis).
    idx_host = ttnn.to_torch(index_cache, mesh_composer=composer).to(torch.float32)  # [U*slots, 1, T, D]
    hadamard = normalized_hadamard_matrix(config.index_head_dim).float()
    idx_layers = [i for i in range(num_layers) if (trace_dir / "dsa" / f"indexer_k_layer_{i}").exists()]
    idx_min = 1.0
    for i in idx_layers:
        g = _load_layer_rows(trace_dir, layout, "dsa", i, f"indexer_k_layer_{i}", 0, total_len)
        slot = full_indexer_rank(config, i)
        per_user = []
        for u in range(num_users):
            dev = unrotate_cache_layer(idx_host[u * num_index_slots + slot, 0], p, total_len) @ hadamard
            per_user.append(min(cache_half_pccs(g, dev, config.index_head_dim // 2, pe_interleave=False)))
        idx_min = min(idx_min, *per_user)
        logger.info(f"[batch4] index-K layer {i} PCC per user: {[round(v, 6) for v in per_user]}")
    logger.info(f"[batch4] index-K min PCC over {len(idx_layers)} layers and users: {idx_min:.6f}")
    if idx_layers and idx_min < INDEXER_K_PCC_THRESHOLD:
        failures.append(f"index-K min PCC {idx_min:.6f} < {INDEXER_K_PCC_THRESHOLD}")

    transformer.release_sub_device_managers()
    assert not failures, "; ".join(failures)
    logger.success(f"[batch4] GLM-5.3 L{num_layers} x {n_chunks} chunks: all {num_users} users pass")
