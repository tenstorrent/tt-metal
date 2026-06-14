# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bottom-up PCC for the Mistral-Small-4 MLA attention block (model-local, generic ttnn ops).

Step 1 (this file): the projection chains, PCC'd against reference goldens:
  q_b_out  = q_b_proj(q_a_layernorm(q_a_proj(x)))
  kv_a_out = kv_a_proj_with_mqa(x)
  kv_b_out = kv_b_proj(kv_a_layernorm(kv_a_out[..., :kv_lora_rank]))
Weights are bf16 (fp8-dequantized), replicated across the (1,8) mesh — correctness first;
sharding / flash-MLA / paged come later. Reference goldens from m4_text_reference.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import capture_golden, load_m4_text_reference


def _lin_w(hf_weight, mesh):
    # HF nn.Linear weight is [out, in]; ttnn.linear computes x @ W with W = [in, out].
    return ttnn.as_tensor(
        hf_weight.transpose(0, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _norm_w(hf_weight, mesh):
    return ttnn.as_tensor(
        hf_weight.reshape(1, -1).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_host(tt, mesh, batch):
    # replicated -> concat on dim 0 gives `mesh` identical copies; take the first `batch`
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()[:batch]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_mla_projections(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    eps = AutoConfig.from_pretrained(ckpt).text_config.rms_norm_eps
    kv_lora = AutoConfig.from_pretrained(ckpt).text_config.kv_lora_rank

    # reference + goldens
    model, cfg, _ = load_m4_text_reference(ckpt, n_layers=1)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    sa = model.model.layers[0].self_attn
    B = g["mla_in"].shape[0]

    # weights -> ttnn (replicated)
    w_qa = _lin_w(sa.q_a_proj.weight, mesh_device)
    w_qan = _norm_w(sa.q_a_layernorm.weight, mesh_device)
    w_qb = _lin_w(sa.q_b_proj.weight, mesh_device)
    w_kva = _lin_w(sa.kv_a_proj_with_mqa.weight, mesh_device)
    w_kvan = _norm_w(sa.kv_a_layernorm.weight, mesh_device)
    w_kvb = _lin_w(sa.kv_b_proj.weight, mesh_device)

    x = ttnn.from_torch(
        g["mla_in"].to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # q chain
    q_a = ttnn.linear(x, w_qa)
    q_a = ttnn.rms_norm(q_a, epsilon=eps, weight=w_qan)
    q_b = ttnn.linear(q_a, w_qb)

    # kv chain
    kv_a = ttnn.linear(x, w_kva)
    kv_pass = ttnn.slice(kv_a, [0, 0, 0], [B, kv_a.shape[1], kv_lora])
    kv_pass = ttnn.rms_norm(kv_pass, epsilon=eps, weight=w_kvan)
    kv_b = ttnn.linear(kv_pass, w_kvb)

    results = {
        "q_b_out": (_to_host(q_b, mesh_device, B), g["q_b_out"]),
        "kv_a_out": (_to_host(kv_a, mesh_device, B), g["kv_a_out"]),
        "kv_b_out": (_to_host(kv_b, mesh_device, B), g["kv_b_out"]),
    }
    all_pass = True
    for name, (tt, ref) in results.items():
        passing, msg = comp_pcc(ref, tt, pcc_required)
        logger.info(f"MLA projection {name}: {msg}")
        all_pass = all_pass and passing

    # --- SDPA + o_proj, fed the reference post-RoPE q/k/v (decoupled from RoPE assembly) ---
    # q/k/v goldens are [B, n_heads, S, head_dim]; ttnn SDPA expects [b, nqh/nkv, s, dh].
    def _bhsd(t):
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    head_dim = g["q_states"].shape[-1]
    qd, kd, vd = _bhsd(g["q_states"]), _bhsd(g["k_states"]), _bhsd(g["value_states"])
    attn = ttnn.transformer.scaled_dot_product_attention(qd, kd, vd, is_causal=True, scale=head_dim ** (-0.5))
    # [B, nh, S, vd] -> [B, S, nh*vd]
    attn = ttnn.transpose(attn, 1, 2)
    attn = ttnn.reshape(attn, (B, attn.shape[1], -1))
    o_in_tt = _to_host(attn, mesh_device, B)
    p_oin, m_oin = comp_pcc(g["o_proj_in"], o_in_tt, pcc_required)
    logger.info(f"MLA sdpa->o_proj_in: {m_oin}")

    w_o = _lin_w(sa.o_proj.weight, mesh_device)
    mla_out_tt = _to_host(ttnn.linear(attn, w_o), mesh_device, B)
    p_out, m_out = comp_pcc(g["mla_out"], mla_out_tt, pcc_required)
    logger.info(f"MLA o_proj->mla_out: {m_out}")
    all_pass = all_pass and p_oin and p_out

    # --- assembly: projections -> q_states / k_states / value (reshape, split, interleaved RoPE) ---
    cfg_t = AutoConfig.from_pretrained(ckpt).text_config
    H, qk, nope, rope_d, vd, kvl = (
        cfg_t.num_attention_heads,
        cfg_t.qk_head_dim,
        cfg_t.qk_nope_head_dim,
        cfg_t.qk_rope_head_dim,
        cfg_t.v_head_dim,
        cfg_t.kv_lora_rank,
    )
    S = q_b.shape[1]
    # de-interleave permutation matrix P (out[i]=x[2i] for i<d/2 else x[2(i-d/2)+1]) -> matmul-friendly
    perm = [2 * i for i in range(rope_d // 2)] + [2 * i + 1 for i in range(rope_d // 2)]
    P = torch.zeros(rope_d, rope_d)
    for i, p in enumerate(perm):
        P[p, i] = 1.0
    P_tt = ttnn.from_torch(
        P.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos = ttnn.reshape(_bhsd(g["rope_cos"]), (B, 1, S, rope_d))
    sin = ttnn.reshape(_bhsd(g["rope_sin"]), (B, 1, S, rope_d))

    def rotate_half(x):
        d = x.shape[-1]
        x1 = ttnn.slice(x, [0, 0, 0, 0], [x.shape[0], x.shape[1], x.shape[2], d // 2])
        x2 = ttnn.slice(x, [0, 0, 0, d // 2], [x.shape[0], x.shape[1], x.shape[2], d])
        return ttnn.concat([ttnn.neg(x2), x1], dim=-1)

    def apply_rope(x):  # x: [.., S, rope_d]; de-interleave then standard rope
        xd = ttnn.matmul(x, P_tt)
        return ttnn.add(ttnn.mul(xd, cos), ttnn.mul(rotate_half(xd), sin))

    # q: (B,S,H*qk) -> (B,H,S,qk) -> split
    qh = ttnn.transpose(ttnn.reshape(q_b, (B, S, H, qk)), 1, 2)
    q_nope = ttnn.slice(qh, [0, 0, 0, 0], [B, H, S, nope])
    q_rot = ttnn.slice(qh, [0, 0, 0, nope], [B, H, S, qk])
    q_states_tt = ttnn.concat([q_nope, apply_rope(q_rot)], dim=-1)

    # k/v: kv_b (B,S,H*(nope+vd)) -> (B,H,S,nope+vd) -> split; k_rot from kv_a tail (MQA, expand)
    kh = ttnn.transpose(ttnn.reshape(kv_b, (B, S, H, nope + vd)), 1, 2)
    k_nope = ttnn.slice(kh, [0, 0, 0, 0], [B, H, S, nope])
    value_tt2 = ttnn.slice(kh, [0, 0, 0, nope], [B, H, S, nope + vd])
    k_rot = ttnn.reshape(ttnn.slice(kv_a, [0, 0, kvl], [B, S, kvl + rope_d]), (B, 1, S, rope_d))
    k_rot = ttnn.repeat(apply_rope(k_rot), ttnn.Shape([1, H, 1, 1]))
    k_states_tt = ttnn.concat([k_nope, k_rot], dim=-1)

    for name, tt, ref in [
        ("q_states", q_states_tt, g["q_states"]),
        ("k_states", k_states_tt, g["k_states"]),
        ("value", value_tt2, g["value_states"]),
    ]:
        passing, msg = comp_pcc(ref, _to_host(tt, mesh_device, B), pcc_required)
        logger.info(f"MLA assembly {name}: {msg}")
        all_pass = all_pass and passing

    # --- end-to-end: my q/k/v -> sdpa -> o_proj -> mla_out ---
    e2e = ttnn.transformer.scaled_dot_product_attention(
        q_states_tt, k_states_tt, value_tt2, is_causal=True, scale=qk ** (-0.5)
    )
    e2e = ttnn.reshape(ttnn.transpose(e2e, 1, 2), (B, S, -1))
    e2e_out = _to_host(ttnn.linear(e2e, w_o), mesh_device, B)
    p_e2e, m_e2e = comp_pcc(g["mla_out"], e2e_out, pcc_required)
    logger.info(f"MLA end-to-end mla_in->mla_out: {m_e2e}")
    all_pass = all_pass and p_e2e

    assert all_pass, "MLA PCC below threshold"
