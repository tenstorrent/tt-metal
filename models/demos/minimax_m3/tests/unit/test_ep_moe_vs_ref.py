# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M3 EP MoE MLP (router + always-on shared expert + expert-parallel routed experts) vs a
self-authored torch reference, at REAL M3 dims and the production 128-expert / 4-per-chip EP
dispatch.

Op-level MoE MLP check at the full 128-expert / 4-per-chip EP dispatch and real hidden size (the SP
whole-model test in test_model_sp_vs_ref.py runs a reduced expert count), so this uniquely exercises the
multi-expert-per-chip dispatch/combine buffers AND the shared-expert add + routed_scaling_factor.

Layout: each mesh ROW is fed an INDEPENDENT [1,1,S,H] token batch (row-sharded), replicated across the
TP cols (4) — exactly the [1,1,S,H] the MLP sees per device from the decoder layer. The MLP is blind to
how those rows relate (the model is SP-only: rows are sequence shards of one prompt), so independent
per-row inputs are just a convenient way to exercise the op. MLP does host top-k routing, EP
dispatch/combine (fused clamped-swigluoai kernel) across all 32 chips, then adds the shared expert.

Random weights. Needs TT_MESH_GRAPH_DESC_PATH=single_bh_galaxy ([8,4]). Anchor: transformers minimax_m3_vl.
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.mlp import MLP
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

# Real M3 MoE dims (from configs/MiniMax-M3/config.json).
HIDDEN, E, TOPK = 6144, 128, 4
INTER, SHARED_INTER = 3072, 3072  # routed intermediate_size / shared_intermediate_size
SCALE, ALPHA, LIMIT = 2.0, 1.702, 7.0  # routed_scaling_factor / swigluoai alpha / clamp limit


def _rand(*s):
    return torch.randn(*s) * 0.02


def _ffn(x, w1, w3, w2):
    """Clamped swigluoai FFN (w1=gate, w3=up, w2=down)."""
    g = (x @ w1.t()).clamp(max=LIMIT)
    u = (x @ w3.t()).clamp(min=-LIMIT, max=LIMIT)
    return ((u + 1.0) * (g * torch.sigmoid(ALPHA * g))) @ w2.t()


def _moe(x, w):
    """M3 MoE reference on [tokens, H]: sigmoid+bias top-k routing, normalized+scaled routed
    experts, plus the always-on shared expert (added unscaled)."""
    scores = torch.sigmoid(x @ w["gate"].t())
    _, idx = torch.topk(scores + w["bias"], TOPK, dim=-1)
    tw = torch.gather(scores, 1, idx)
    tw = (tw / tw.sum(-1, keepdim=True)) * SCALE
    routed = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(TOPK):
            routed[t] += tw[t, j] * _ffn(x[t : t + 1], *w["experts"][idx[t, j].item()]).squeeze(0)
    return routed + _ffn(x, *w["shared"])


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_ep_moe_vs_ref(mesh_device, device_params, seq_len, reset_seeds):
    """tt/mlp.py MLP (router + shared + 128-expert EP routed) on (8,4) EP=32 vs per-row torch ref."""
    rows, cols = mesh_device.shape
    assert (rows, cols) == (8, 4), "this test targets the (8,4) galaxy layout (EP=32, DP=8 rows)"
    torch.manual_seed(0)

    # Random MoE weights: gate [E,H], correction bias [E], E routed experts (w1/w3/w2), 1 shared expert.
    w = {
        "gate": torch.randn(E, HIDDEN) * 0.05,
        "bias": torch.randn(E) * 0.1,
        "experts": [(_rand(INTER, HIDDEN), _rand(INTER, HIDDEN), _rand(HIDDEN, INTER)) for _ in range(E)],
        "shared": (_rand(SHARED_INTER, HIDDEN), _rand(SHARED_INTER, HIDDEN), _rand(HIDDEN, SHARED_INTER)),
    }

    # One prompt per row (DP=8); post-layernorm hidden states feed MLP directly.
    x = torch.randn(rows, seq_len, HIDDEN, dtype=torch.bfloat16)

    # --- torch reference, per prompt ---
    ref = [_moe(x[r].float(), w) for r in range(rows)]  # each [S, H]

    # --- build the block_sparse_moe state dict MLP expects ---
    state = {
        "gate.weight": w["gate"],
        "e_score_correction_bias": w["bias"],
        "shared_experts.gate_proj.weight": w["shared"][0],
        "shared_experts.up_proj.weight": w["shared"][1],
        "shared_experts.down_proj.weight": w["shared"][2],
    }
    for e, (w1, w3, w2) in enumerate(w["experts"]):
        state[f"experts.{e}.w1.weight"] = w1
        state[f"experts.{e}.w3.weight"] = w3
        state[f"experts.{e}.w2.weight"] = w2

    # --- config from the real M3 config.json (128 experts / top-4, real dims) ---
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "..", "configs", "MiniMax-M3", "config.json")
    with open(cfg_path) as f:
        c = json.load(f)
    hf_config = SimpleNamespace(**c)
    assert hf_config.num_local_experts == E and hf_config.hidden_size == HIDDEN, "config.json is not stock M3"

    mesh_config = MeshConfig((rows, cols), tp=cols)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    mlp = MLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        expert_weight_dtype=ttnn.bfloat8_b,
        use_ep_moe=True,
        ep_seq_len_per_chip=seq_len,
    )

    # Shard: one prompt per row (dim 0), full emb replicated across TP cols (matches the decoder layer).
    tt_x = ttnn.from_torch(
        x.reshape(rows, 1, seq_len, HIDDEN),
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=mesh_device.shape),
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        dtype=ttnn.bfloat16,
    )
    out = mlp(tt_x)  # [1, 1, S, H] per device (full emb, replicated across cols)
    ttnn.synchronize_device(mesh_device)

    # Per-row read: device index = r*cols (col 0; all cols hold the same all-gathered full-emb output).
    dts = ttnn.get_device_tensors(out)
    for r in range(rows):
        row = ttnn.to_torch(dts[r * cols]).float().reshape(-1, HIDDEN)[:seq_len]
        passing, pcc = comp_pcc(ref[r], row, 0.95)
        logger.info(f"prompt{r}: pcc={pcc}")
        assert passing, f"prompt {r} EP MoE PCC fail: {pcc}"
    logger.info(f"EP MoE MLP (128 experts, shared): all {rows} prompts pass")
