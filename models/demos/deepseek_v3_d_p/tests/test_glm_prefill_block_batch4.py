# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""GLM-5.3 batch-4 prefill block: one user per mesh column for attention, users interleaved for MoE.

Today a chunk of ONE user runs on the whole 8x4 mesh (SP=8 on rows, TP=4 on columns). Here four users
run at once, user c on column c:

  * Attention is SP=8 / TP=1 per column (``ttMLA(batch_axis=1)``): hidden and all 64 heads are local to
    each chip, weights are replicated, the KVPE and index-K caches are striped over the 8 chips of the
    column, and the KVPE gather + fused ring indexer are SP-axis rings (one per user). No attention
    collective crosses a column.
  * The MoE keeps today's 4 dispatch groups (one per column, 64 experts each) and is used UNCHANGED.
    An all-to-all over the column axis interleaves the four users: each chip trades its full-hidden
    [640, 6144] rows for a hidden quarter of all four users' rows [2560, 1536] -- exactly today's MoE
    input layout, with 4x the tokens. The inverse all-to-all hands each column its own user back.

Inputs are teacher-forced from the GLM-5.3 55k vLLM trace: the first ``n_chunks`` x 5120 tokens of
``decoder_output_layer_{L-1}`` are fed, identically, to all four columns, and every column's block
output, KVPE cache and index-K cache are PCC'd against layer L's golden. Because the four users are
identical, the four columns must also agree with each other.
"""

import json
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.deepseek_v3_d_p.reference.cpu_deepseek_v32 import pretrained_mla_weights
from models.demos.deepseek_v3_d_p.reference.glm_5_3_config import GLM53Config
from models.demos.deepseek_v3_d_p.tests.fabric_profiles import torus_xy_device_params
from models.demos.deepseek_v3_d_p.tests.test_prefill_block_chunked import _resolve_trace_dir
from models.demos.deepseek_v3_d_p.tt.mla.indexer import (
    full_indexer_rank,
    indexer_layer_is_reused,
    normalized_hadamard_matrix,
    num_full_indexer_layers,
)
from models.demos.deepseek_v3_d_p.tt.mla.mla import ttMLA
from models.demos.deepseek_v3_d_p.tt.mla.rope import RotarySetup
from models.demos.deepseek_v3_d_p.tt.mla.utils import blockcyclic_positions, rotated_chip_positions
from models.demos.deepseek_v3_d_p.tt.moe.tt_moe_gate_prefill import GateComputeMode
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.deepseek_v3_d_p.tt.tt_prefill_block import TtPrefillBlock
from models.demos.deepseek_v3_d_p.utils.chunk_config import PREFILL_CHUNK_TOKENS
from models.demos.deepseek_v3_d_p.utils.fast_cache_checker import init_checker
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat, init_kvpe_cache, init_mla_kv_cache
from models.demos.deepseek_v3_d_p.utils.test_utils import cache_half_pccs, read_sharded_rows, unrotate_cache_layer
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered
from tests.ttnn.utils_for_testing import comp_pcc

CHUNK = PREFILL_CHUNK_TOKENS  # 5120 tokens per user per chunk
SP_AXIS, BATCH_AXIS = 0, 1  # mesh rows carry SP, mesh columns carry users
# Same floors as the single-user teacher-forced test (test_glm_prefill_block_indexer_teacher_forced).
PCC_FLOOR = 0.98
# Routing semaphores use 512 B of L1_SMALL; the rest is the sparse-MLA gather's (as GLM_L1_SMALL_SIZE).
L1_SMALL_SIZE = 1216


def _log_shape(name: str, t: ttnn.Tensor) -> None:
    logger.info(f"[batch4] {name:<34} per chip {list(t.shape)}")


def _replicated_norm_weight(w: torch.Tensor, mesh_device) -> ttnn.Tensor:
    """[hidden] norm weight -> the ROW_MAJOR [1, 1, hidden/32, 32] layout ttnn.rms_norm reads, replicated:
    hidden is no longer split across the column axis, so every chip holds the whole weight."""
    return ttnn.from_torch(
        w.reshape(1, 1, -1, ttnn.TILE_SIZE).to(torch.bfloat16),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def interleave_users(x: ttnn.Tensor, num_links: int) -> ttnn.Tensor:
    """[1, 1, S/sp, H] (this column's user, full hidden) -> [1, 1, U*S/sp, H/U] (all U users, hidden
    quarter c on column c). Column c splits its hidden into U slices and sends slice h to column h, which
    concatenates the U received row blocks in source-column order -> rows are [u0 | u1 | u2 | u3]. That
    is today's TP-sharded MoE input, with U times the tokens."""
    return ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=2,  # grows: rows of all users
        out_dim=3,  # splits: hidden quarters
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=BATCH_AXIS,
    )


def deinterleave_users(x: ttnn.Tensor, num_links: int) -> ttnn.Tensor:
    """Inverse of interleave_users: [1, 1, U*S/sp, H/U] -> [1, 1, S/sp, H]. Column h splits its rows
    into the U user blocks and sends block u to column u, which concatenates the U hidden quarters."""
    return ttnn.experimental.all_to_all_async_generic(
        x,
        in_dim=3,  # grows: hidden quarters back to full hidden
        out_dim=2,  # splits: per-user row blocks
        num_links=num_links,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cluster_axis=BATCH_AXIS,
    )


def _per_user_cache(tt_cache: ttnn.Tensor, mesh_device) -> torch.Tensor:
    """Per-chip [slots, 1, T/sp, D] -> host [U*slots, 1, T, D]: SP stripes concatenated on dim 2 (still
    block-cyclic), the users of the U columns stacked on dim 0 (user u owns slots [u*slots, (u+1)*slots))."""
    return ttnn.to_torch(
        tt_cache,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape),
    ).to(torch.float32)


def _max_dispatched_rows_per_chip(moe, intermediates, mesh_device) -> int:
    """Largest per-chip routed-token sum. The dispatch kernel silently drops tokens past its buffer, so
    this is checked against the capacity instead of trusting the output."""
    counts = ttnn.to_torch(
        ttnn.unsqueeze_to_4D(intermediates.expert_token_counts),
        mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[1, 0])),
    ).squeeze(2)
    return int(counts.to(torch.int64).flatten().view(-1, moe.experts_per_chip).sum(dim=1).max().item())


@pytest.mark.parametrize("layer_idx", [6], ids=lambda l: f"L{l}")
@pytest.mark.parametrize("n_chunks", [1], ids=lambda n: f"chunks{n}")
@pytest.mark.parametrize(
    "mesh_device, device_params, num_links",
    [
        pytest.param(
            (8, 4),
            torus_xy_device_params(fabric_payload_size=GLM53Config.FABRIC_PAYLOAD_SIZE, l1_small_size=L1_SMALL_SIZE),
            2,
            marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
            id="torus-xy-8x4",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("variant", ["glm_5_3"], indirect=True, ids=["glm53"])
@pytest.mark.skipif(not is_blackhole(), reason="GLM DSA (indexer / sparse SDPA) is Blackhole-only")
@pytest.mark.timeout(0)
def test_glm_prefill_block_batch4(
    variant, config_only, mesh_device, device_params, num_links, model_path, weight_cache_path, n_chunks, layer_idx
):
    config = config_only
    topology = per_axis_topology(device_params["fabric_config"])
    sp, num_users = mesh_device.shape[SP_AXIS], mesh_device.shape[BATCH_AXIS]
    assert (sp, num_users) == (8, 4), f"this test targets mesh-8x4, got {list(mesh_device.shape)}"
    assert layer_idx >= variant.model_config.NUM_DENSE_LAYERS, "the batch-4 layout is about the MoE layers"
    assert not indexer_layer_is_reused(
        config, layer_idx
    ), f"layer {layer_idx} is a shared-indexer layer: an isolated block cannot supply the reused top-k"

    if weight_cache_path is None:
        pytest.skip(f"pretrained weights unavailable (set {variant.ttnn_cache_env} + {variant.env_var})")
    trace_dir = _resolve_trace_dir(variant)
    idx_golden = trace_dir / "dsa" / f"indexer_k_layer_{layer_idx}"
    if not idx_golden.exists():
        pytest.skip(f"no indexer_k golden for layer {layer_idx} in {trace_dir}")

    chunk_local = CHUNK // sp
    total_len = n_chunks * CHUNK
    seq_cache = total_len  # per-user KV capacity: exactly the prefix this test writes
    hidden = config.hidden_size
    kv_lora = config.kv_lora_rank
    idx_dim = config.index_head_dim
    config.max_seq_len = seq_cache
    logger.info(
        f"[batch4] GLM-5.3 layer {layer_idx}: {num_users} users x {total_len} tokens ({n_chunks} chunk(s)), "
        f"attention SP={sp}/TP=1 per column, MoE {num_users} dispatch groups over interleaved users"
    )

    # --- golden (teacher-forced: layer L-1's real output is layer L's input) ---
    io = trace_dir / "decoder_io"
    input_hidden = read_sharded_rows(
        io / f"decoder_output_layer_{layer_idx - 1}", f"decoder_output_layer_{layer_idx - 1}", 0, total_len
    )
    ref_out = read_sharded_rows(
        io / f"decoder_output_layer_{layer_idx}", f"decoder_output_layer_{layer_idx}", 0, total_len
    )
    g_idx = read_sharded_rows(idx_golden, f"indexer_k_layer_{layer_idx}", 0, total_len)
    g_kv = read_sharded_rows(
        trace_dir / "kv_cache" / f"layer_{layer_idx}", f"kv_post_transform_layer_{layer_idx}", 0, total_len
    )

    # --- attention weights: host checkpoint -> replicated device tensors (no TP-sharded cache reuse) ---
    prefix = f"model.layers.{layer_idx}."
    weight_map = json.load(open(Path(model_path) / "model.safetensors.index.json"))["weight_map"]
    shards = sorted({v for k, v in weight_map.items() if k.startswith(prefix)})
    mla_weights = pretrained_mla_weights(
        config, layer=layer_idx, checkpoint_path=[str(Path(model_path) / s) for s in shards]
    )
    norms = load_hf_state_dict_filtered(
        str(model_path), [f"{prefix}input_layernorm.", f"{prefix}post_attention_layernorm."]
    )
    attn_norm_w = _replicated_norm_weight(norms[f"{prefix}input_layernorm.weight"], mesh_device)
    ffn_norm_w = _replicated_norm_weight(norms[f"{prefix}post_attention_layernorm.weight"], mesh_device)

    mla = ttMLA(
        config,
        mla_weights,
        mesh_device,
        layer_idx=layer_idx,
        seq_len=seq_cache,
        sp_axis=SP_AXIS,
        tp_axis=BATCH_AXIS,
        topology=topology,
        weight_cache_path=None,  # replicated weights are built from the host checkpoint every run
        is_chunked=True,
        active_seq_len=CHUNK,
        slot_num=1,
        layer_num=1,
        has_indexer=True,
        sparse_mla_overlap_profile="off",
        batch_axis=BATCH_AXIS,
    )
    assert mla.tp_factor == 1 and not mla.tp_shard_kv

    # --- MoE: today's TtMoe, unchanged, sized for the interleaved token count ---
    effective_cache = weight_cache_path / f"{sp}x{num_users}"
    init_checker(effective_cache)
    experts_per_chip = GLM53Config.NUM_ROUTED_EXPERTS // (sp * num_users)
    assert TtPrefillBlock.check_cache_complete(
        effective_cache, layer_idx, is_dense=False, experts_per_chip=experts_per_chip
    ), f"TTNN MoE cache incomplete for layer {layer_idx} at {effective_cache}"
    moe = TtPrefillBlock._build_moe(
        mesh_device=mesh_device,
        model_cfg=GLM53Config,
        config=config,
        state_dict={},
        seq_len=num_users * CHUNK,  # seq_len_per_chip = U * 640 = 2560
        sp_axis=SP_AXIS,
        emb_dim=hidden,
        num_links=num_links,
        topology=topology,
        gate_fallback_mode=GateComputeMode.DEVICE_FP32,
        routed_expert_activations_dtype=ttnn.bfloat8_b,
        routed_expert_weights_dtype=ttnn.bfloat4_b,
        shared_expert_activations_dtype=ttnn.bfloat16,
        shared_expert_weights_dtype=ttnn.bfloat8_b,
        dispatch_buffer_capacity_factor=2,
        weight_cache_path=effective_cache,
        layer_idx=layer_idx,
        routing_use_l1_small_for_semaphores=True,
    )
    logger.info(
        f"[batch4] MoE seq_len_per_chip={moe.seq_len_per_chip} "
        f"dispatch capacity={moe.dispatch_module.max_dispatch_buffer_token_size} rows/chip"
    )

    # --- per-column caches: user c's rows striped over the 8 chips of column c ---
    kvpe_cache = init_mla_kv_cache(
        cache_format=MlaKvCacheFormat.BF16_RM,
        hf_config=config,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=1,
        num_users=1,
        batch_axis=BATCH_AXIS,
    )
    num_index_slots = num_full_indexer_layers(config) or 1
    index_cache = init_kvpe_cache(
        kvpe_cache_head_dim=idx_dim,
        mesh_device=mesh_device,
        seq_len=seq_cache,
        mesh_shape=list(mesh_device.shape),
        sp_axis=SP_AXIS,
        num_kvpe_cache_layers=num_index_slots,
        num_users=1,
        dtype=ttnn.bfloat8_b,
        batch_axis=BATCH_AXIS,
    )
    _log_shape("KVPE cache", kvpe_cache.storage)
    _log_shape("index-K cache", index_cache)

    rope = RotarySetup(config, mesh_device, sp_axis=SP_AXIS, is_balanced=False).get_rope_tensors_indexed(
        cache_seq_len_global=seq_cache, chunk_size_global=CHUNK
    )

    # Host [U, 1, CHUNK, H]: axis 0 (rows) splits the sequence, axis 1 (columns) selects the user.
    user_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, 0))
    user_composer = ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 0), mesh_shape=mesh_device.shape)
    out_accum = torch.zeros(num_users, total_len, hidden, dtype=torch.float32)

    mesh_device.enable_program_cache()
    for c in range(n_chunks):
        kv_actual = c * CHUNK
        positions = rotated_chip_positions(kv_actual, sp, chunk_local)
        flat = torch.tensor([positions[ch][r] for ch in range(sp) for r in range(chunk_local)], dtype=torch.long)
        chunk_in = input_hidden[flat].reshape(1, 1, CHUNK, hidden).expand(num_users, 1, CHUNK, hidden).contiguous()
        x = ttnn.from_torch(
            chunk_in,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=user_mapper,
        )
        log = _log_shape if c == 0 else (lambda *_: None)
        log("block input x", x)

        # --- attention: SP=8 / TP=1 per column ---
        h = ttnn.rms_norm(x, weight=attn_norm_w, epsilon=config.rms_norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        log("attn_norm (local)", h)
        attn = mla.forward(
            h,
            rope,
            kvpe_cache,
            cache_layer_idx=0,
            actual_start=kv_actual,
            actual_end=kv_actual + CHUNK,
            cache_user_id=0,
            index_kv_cache=index_cache,
        )
        ttnn.deallocate(h)
        log("MLA out", attn)
        x = ttnn.add(x, attn)
        ttnn.deallocate(attn)

        # --- MoE: interleave users -> unchanged TtMoe -> de-interleave ---
        h = ttnn.rms_norm(x, weight=ffn_norm_w, epsilon=config.rms_norm_eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        log("ffn_norm (local)", h)
        moe_in = interleave_users(h, num_links)
        ttnn.deallocate(h)
        log("interleaved MoE input (a2a)", moe_in)
        moe_out, inter = moe(ttnn.squeeze(moe_in, dim=0), return_intermediates=True)
        log("MoE out", moe_out)
        max_rows = _max_dispatched_rows_per_chip(moe, inter, mesh_device)
        capacity = moe.dispatch_module.max_dispatch_buffer_token_size
        logger.info(f"[batch4] chunk {c}: max dispatched rows per chip {max_rows} / capacity {capacity}")
        assert max_rows <= capacity, f"dispatch overflow: {max_rows} rows > capacity {capacity} (tokens dropped)"
        ffn = deinterleave_users(ttnn.unsqueeze(moe_out, dim=0), num_links)
        log("de-interleaved MoE out (a2a)", ffn)
        x = ttnn.add(x, ffn)
        ttnn.deallocate(ffn)
        log("block output", x)

        out_accum[:, flat] = ttnn.to_torch(x, mesh_composer=user_composer).to(torch.float32)[:, 0]
        ttnn.deallocate(x)
        ttnn.synchronize_device(mesh_device)
        logger.info(f"[batch4] chunk {c} done (kv_actual={kv_actual})")

    # --- validation: every column against the golden, and the columns against each other ---
    p = blockcyclic_positions(sp, CHUNK, seq_cache)
    kv_host = _per_user_cache(kvpe_cache.storage, mesh_device)  # [U, 1, T, 576]
    idx_host = _per_user_cache(index_cache, mesh_device)  # [U*slots, 1, T, 128]
    idx_slot = full_indexer_rank(config, layer_idx)
    # write_k stores keys in the orthonormal Hadamard basis (decode-compatible); H is symmetric and
    # orthonormal, so one more multiply by H returns the golden's plain basis.
    index_hadamard = normalized_hadamard_matrix(idx_dim).float()
    failures = []
    for u in range(num_users):
        _, out_pcc = comp_pcc(ref_out, out_accum[u])
        dev_kv = unrotate_cache_layer(kv_host[u, 0], p, total_len)
        dev_idx = unrotate_cache_layer(idx_host[u * num_index_slots + idx_slot, 0], p, total_len) @ index_hadamard
        kv_nope, kv_pe = cache_half_pccs(g_kv, dev_kv, kv_lora, pe_interleave=True)
        idx_rope, idx_nope = cache_half_pccs(g_idx, dev_idx, idx_dim // 2, pe_interleave=False)
        logger.info(
            f"[batch4] user {u} (column {u}): output {out_pcc:.6f} | KVPE nope {kv_nope:.6f} pe {kv_pe:.6f} | "
            f"index-K rope {idx_rope:.6f} nope {idx_nope:.6f}"
        )
        for name, val in (
            ("output", out_pcc),
            ("KVPE nope", kv_nope),
            ("KVPE pe", kv_pe),
            ("index-K rope", idx_rope),
            ("index-K nope", idx_nope),
        ):
            if val < PCC_FLOOR:
                failures.append(f"user {u} {name} PCC {val:.6f} < {PCC_FLOOR}")
        if u > 0:
            same = torch.equal(out_accum[0], out_accum[u])
            max_diff = (out_accum[0] - out_accum[u]).abs().max().item()
            logger.info(f"[batch4] user {u} vs user 0: bit-identical={same} max|diff|={max_diff:.3e}")
            _, cross = comp_pcc(out_accum[0], out_accum[u])
            if cross < 0.9999:
                failures.append(f"user {u} output diverges from user 0 (PCC {cross:.6f}) on identical inputs")
    assert not failures, "; ".join(failures)
    logger.success(f"[batch4] GLM-5.3 layer {layer_idx}: all {num_users} users pass (floor {PCC_FLOOR})")
