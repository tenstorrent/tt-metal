# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for the MiniMax-M3 MoE block vs a hand-written torch reference.

M3 MoE = sigmoid router (+ correction-bias for selection) -> top-k -> gather unbiased -> normalize
-> x routed_scaling_factor; routed experts use the clamped swigluoai FFN (w1=gate, w3=up, w2=down);
plus an always-on shared expert added AFTER scaling (unscaled). Anchor: transformers minimax_m3_vl.

Self-authored torch ref + identical random weights, single card (non-fused, non-EP expert path —
the same composite FFN reused by EP=32 on the galaxy). Modest tile-aligned dims for speed.
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.mlp import MLP
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

H, INTER, SHARED_INTER, E, TOPK, SCALE, ALPHA, LIMIT = 512, 512, 512, 8, 2, 2.0, 1.702, 7.0


def _swiglu(gate, up):
    gate = gate.clamp(max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    return (up + 1.0) * (gate * torch.sigmoid(ALPHA * gate))


def _ffn(x, w1, w3, w2):
    """Expert/shared FFN: down(swiglu(gate(x), up(x))). w1=gate, w3=up, w2=down ([out,in])."""
    return _swiglu(x @ w1.t(), x @ w3.t()) @ w2.t()


def _torch_moe(x, gate_w, bias, experts, shared):
    """x: [T, H]. experts: list of (w1,w3,w2). shared: (gate,up,down)."""
    x = x.float()
    scores = torch.sigmoid(x @ gate_w.t())  # [T, E]
    choice = scores + bias  # selection only
    _, idx = torch.topk(choice, TOPK, dim=-1)  # [T, TOPK]
    w = torch.gather(scores, 1, idx)  # unbiased
    w = w / w.sum(-1, keepdim=True)
    w = w * SCALE  # routed_scaling_factor, after normalize

    routed = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            e = idx[t, j].item()
            routed[t] += w[t, j] * _ffn(x[t : t + 1], *experts[e]).squeeze(0)
    shared_out = _ffn(x, *shared)
    return routed + shared_out


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_moe_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """Full M3 MoE block (router + routed experts + shared + routed_scaling) vs torch ref, TP=1."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, seq_len, H) * 0.1

    gate_w = torch.randn(E, H) * 0.05
    bias = torch.randn(E) * 0.1
    experts = [
        (torch.randn(INTER, H) * 0.05, torch.randn(INTER, H) * 0.05, torch.randn(H, INTER) * 0.05) for _ in range(E)
    ]
    shared = (
        torch.randn(SHARED_INTER, H) * 0.05,
        torch.randn(SHARED_INTER, H) * 0.05,
        torch.randn(H, SHARED_INTER) * 0.05,
    )

    ref = _torch_moe(x.reshape(-1, H), gate_w, bias, experts, shared).reshape(1, 1, seq_len, H)

    state_dict = {
        "gate.weight": gate_w,
        "e_score_correction_bias": bias,
        "shared_experts.gate_proj.weight": shared[0],
        "shared_experts.up_proj.weight": shared[1],
        "shared_experts.down_proj.weight": shared[2],
    }
    for e, (w1, w3, w2) in enumerate(experts):
        state_dict[f"experts.{e}.w1.weight"] = w1
        state_dict[f"experts.{e}.w3.weight"] = w3
        state_dict[f"experts.{e}.w2.weight"] = w2

    hf_config = SimpleNamespace(
        hidden_size=H,
        intermediate_size=INTER,
        num_local_experts=E,
        num_experts_per_tok=TOPK,
        routed_scaling_factor=SCALE,
        swiglu_alpha=ALPHA,
        swiglu_limit=LIMIT,
    )

    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    moe = MLP(
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        mesh_config=mesh_config,
        expert_weight_dtype=ttnn.bfloat16,
    )

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out_tt = moe(x_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, seq_len, H)

    passing, pcc = comp_pcc(ref, out, 0.97)
    logger.info(f"moe block vs ref: {pcc}")
    assert passing, f"MoE PCC fail: {pcc}"
