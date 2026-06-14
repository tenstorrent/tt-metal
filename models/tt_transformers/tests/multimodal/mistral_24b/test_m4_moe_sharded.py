# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Sharded-expert MoE PCC: shard 128 experts across the 8-device mesh (16/device) + all-reduce.

This is the memory-feasible path for the full 36-layer model (replicated dense experts cost
~6.4 GB/layer). Experts are sharded on the expert dim; routing weights W are sharded on the
expert dim too, so each device's local expert index aligns with its W columns. Each device sums
its 16 weighted experts; ttnn.all_reduce(Sum) combines across devices; the shared expert (replicated)
is added. Routing W is host-computed here (from the verified router logits) and uploaded sharded;
on-device routing + sparse dispatch are the perf follow-up. PCC vs the moe_out reference golden.
"""
import os

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.common.utility_functions import comp_pcc
from models.tt_transformers.tests.multimodal.mistral_24b.m4_text_reference import capture_golden, load_m4_text_reference


def _repl(t, mesh):
    return ttnn.as_tensor(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def _shard(t, mesh, dim):
    return ttnn.as_tensor(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
    )


def _lin(w, mesh):  # HF [out,in] -> [in,out] replicated
    return _repl(w.transpose(0, 1).contiguous(), mesh)


@pytest.mark.parametrize("mesh_device", [(1, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 30000000, "num_command_queues": 1}],
    indirect=True,
)
def test_m4_moe_sharded(mesh_device, reset_seeds):
    pcc_required = 0.99
    ckpt = os.environ["HF_MODEL"]
    cfg = AutoConfig.from_pretrained(ckpt).text_config
    model, _, _ = load_m4_text_reference(ckpt, n_layers=1)
    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    g = capture_golden(model, ids)
    mlp = model.model.layers[0].mlp
    B, S, H, I, E = (
        g["moe_in"].shape[0],
        g["moe_in"].shape[1],
        cfg.hidden_size,
        cfg.moe_intermediate_size,
        cfg.n_routed_experts,
    )
    n_dev = mesh_device.get_num_devices()
    per = E // n_dev  # experts per device

    # host routing weights W[T, E] (from verified router logits) -> shard on expert dim
    probs = g["router_logits"].view(-1, E).float().softmax(-1)
    tw, ti = probs.topk(cfg.num_experts_per_tok, -1)
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    W = torch.zeros(B * S, E).scatter_(1, ti, tw).view(B, S, E)
    W_sh = _shard(W, mesh_device, dim=-1)  # [B,S,per]/device

    # experts: pre-transpose for ttnn.linear, shard on expert dim
    gup_T = mlp.experts.gate_up_proj.detach().transpose(1, 2).contiguous()  # [E,H,2I]
    down_T = mlp.experts.down_proj.detach().transpose(1, 2).contiguous()  # [E,I,H]
    gup_sh = _shard(gup_T, mesh_device, dim=0)  # [per,H,2I]/device
    down_sh = _shard(down_T, mesh_device, dim=0)  # [per,I,H]/device

    x = _repl(g["moe_in"], mesh_device)  # [B,S,H]
    # BATCHED experts: do all `per` local experts in 2 batched matmuls (vs per sequential linears).
    xb = ttnn.repeat(x, ttnn.Shape([per, 1, 1]))  # [per,S,H]
    gu = ttnn.matmul(xb, gup_sh)  # [per,S,H]x[per,H,2I] -> [per,S,2I]
    h = ttnn.mul(ttnn.silu(ttnn.slice(gu, [0, 0, 0], [per, S, I])), ttnn.slice(gu, [0, 0, I], [per, S, 2 * I]))
    y = ttnn.matmul(h, down_sh)  # [per,S,I]x[per,I,H] -> [per,S,H]
    w = ttnn.permute(W_sh, (2, 1, 0))  # [B,S,per] -> [per,S,B]
    acc = ttnn.reshape(ttnn.sum(ttnn.mul(y, w), dim=0), (B, S, H))  # weighted sum over local experts
    experts_full = ttnn.all_reduce(acc)  # sum across the 8 devices -> replicated [B,S,H]

    sh = mlp.shared_experts
    shared = ttnn.linear(
        ttnn.mul(
            ttnn.silu(ttnn.linear(x, _lin(sh.gate_proj.weight, mesh_device))),
            ttnn.linear(x, _lin(sh.up_proj.weight, mesh_device)),
        ),
        _lin(sh.down_proj.weight, mesh_device),
    )
    moe_out = ttnn.add(experts_full, shared)
    out = ttnn.to_torch(moe_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)).float()[:B]
    passing, msg = comp_pcc(g["moe_out"].view(B, S, H), out, pcc_required)
    logger.info(f"Mistral-Small-4 sharded-MoE PCC: {msg}")
    assert passing, f"sharded MoE PCC below {pcc_required}: {msg}"
