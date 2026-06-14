# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Bottom-up PCC for the Mistral-Small-4 MoE block (model-local, generic ttnn ops).

MoE = router (linear -> softmax -> top-4/128 -> normalize) + routed SwiGLU experts (stacked
gate_up/down, weighted top-4 sum) + a shared SwiGLU expert. n_group=1 so routing is plain
top-4-of-128 (no grouping, no custom gate op). This file PCC-gates the pieces vs reference
goldens (m4_text_reference): router_logits, shared_out (this step); experts_out + moe_out next.
"""
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral4.tests.m4_text_reference import capture_golden, load_m4_text_reference


def _lin_w(hf_weight, mesh):
    return ttnn.as_tensor(
        hf_weight.transpose(0, 1).contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _from(t, mesh):
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _to_host(tt, mesh, batch):
    return ttnn.to_torch(tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()[:batch]


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_moe_router_and_shared(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    model, cfg, _ = load_m4_text_reference(ckpt, n_layers=1)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    mlp = model.model.layers[0].mlp
    B = g["moe_in"].shape[0]

    x = _from(g["moe_in"], mesh_device)

    # router: logits = x @ gate.weight.T  (shape [B, S, 128])
    logits = ttnn.linear(x, _lin_w(mlp.gate.weight, mesh_device))
    p_r, m_r = comp_pcc(
        g["router_logits"].view(B, -1, cfg.n_routed_experts), _to_host(logits, mesh_device, B), pcc_required
    )
    logger.info(f"MoE router_logits: {m_r}")

    # shared expert: SwiGLU
    sh = mlp.shared_experts
    gate = ttnn.linear(x, _lin_w(sh.gate_proj.weight, mesh_device))
    up = ttnn.linear(x, _lin_w(sh.up_proj.weight, mesh_device))
    h = ttnn.mul(ttnn.silu(gate), up)
    shared = ttnn.linear(h, _lin_w(sh.down_proj.weight, mesh_device))
    p_s, m_s = comp_pcc(g["shared_out"].view(B, -1, cfg.hidden_size), _to_host(shared, mesh_device, B), pcc_required)
    logger.info(f"MoE shared_expert: {m_s}")

    # --- routed experts (dense, correctness-first) + combine ---
    # routing weights W[T, n_experts] from router logits (host; router already PCC'd above).
    # softmax -> top-4 -> normalize -> scatter to dense. ttnn-side routing comes in the module refactor.
    T = g["router_logits"].view(-1, cfg.n_routed_experts).shape[0]
    probs = g["router_logits"].view(-1, cfg.n_routed_experts).float().softmax(-1)
    tw, ti = probs.topk(cfg.num_experts_per_tok, dim=-1)
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)  # routed_scaling_factor == 1.0 (no-op)
    W = torch.zeros(T, cfg.n_routed_experts).scatter_(1, ti, tw)

    # on-device routing (no scatter): softmax -> kth-largest threshold mask -> normalize.
    k = cfg.num_experts_per_tok
    probs_tt = ttnn.softmax(logits, dim=-1)
    topk_vals = ttnn.topk(probs_tt, k, dim=-1)[0]  # descending; [B,T,k]
    kth = ttnn.slice(topk_vals, [0, 0, k - 1], [B, topk_vals.shape[1], k])  # [B,T,1]
    masked = ttnn.mul(probs_tt, ttnn.ge(probs_tt, kth))
    W_dev = ttnn.div(masked, ttnn.sum(masked, dim=-1, keepdim=True))
    p_w, m_w = comp_pcc(W.view(B, T, cfg.n_routed_experts), _to_host(W_dev, mesh_device, B), pcc_required)
    logger.info(f"MoE on-device routing W: {m_w}")

    W_tt = _from(W.view(B, T, cfg.n_routed_experts), mesh_device)

    interm = cfg.moe_intermediate_size
    gup, down = mlp.experts.gate_up_proj, mlp.experts.down_proj  # [E,2*interm,hid], [E,hid,interm]
    acc = None
    for e in range(cfg.n_routed_experts):
        gu = ttnn.linear(x, _lin_w(gup[e], mesh_device))  # [B,T,2*interm]
        gate_e = ttnn.slice(gu, [0, 0, 0], [B, T, interm])
        up_e = ttnn.slice(gu, [0, 0, interm], [B, T, 2 * interm])
        y = ttnn.linear(ttnn.mul(ttnn.silu(gate_e), up_e), _lin_w(down[e], mesh_device))  # [B,T,hid]
        w_e = ttnn.slice(W_tt, [0, 0, e], [B, T, e + 1])  # [B,T,1] broadcast
        contrib = ttnn.mul(y, w_e)
        acc = contrib if acc is None else ttnn.add(acc, contrib)

    p_e, m_e = comp_pcc(g["experts_out"].view(B, T, cfg.hidden_size), _to_host(acc, mesh_device, B), pcc_required)
    logger.info(f"MoE experts_out: {m_e}")

    moe_out = ttnn.add(acc, shared)
    p_m, m_m = comp_pcc(g["moe_out"].view(B, T, cfg.hidden_size), _to_host(moe_out, mesh_device, B), pcc_required)
    logger.info(f"MoE moe_in->moe_out: {m_m}")

    assert p_r and p_s and p_e and p_m and p_w, "MoE PCC below threshold"
