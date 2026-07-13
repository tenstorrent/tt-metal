# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-2 PCC test for the MiniMax-M2 expert FFN (prefill) vs a torch reference.

Validates the expert math: per-expert SiLU SwiGLU (silu(w1 x) * (w3 x)) -> w2, no
bias, weighted-summed over the routed experts using the dense routing weights from
the router. Uses a REDUCED expert count (256 -> 32) so it fits/runs fast on a single
card, and bfloat8_b weights to isolate the compute logic from bfp4 quantization
(bfp4 accuracy is validated separately later). Runs at mesh (1,1)/TP=1/EP=1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m2.config import MeshConfig, ModeConfig
from models.demos.minimax_m2.tt.ccl import CCLManager
from models.demos.minimax_m2.tt.expert_configs import MiniMaxM2ExpertProgramConfig
from models.demos.minimax_m2.tt.experts import ExpertConfig, Experts
from models.demos.minimax_m2.utils.general_utils import get_default_num_links

from ..test_factory import minimax_config_dims, parametrize_mesh_with_fabric


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 8)], linear_fabric=True)
@pytest.mark.parametrize("seq_len", [128], ids=["s128"])
def test_experts_prefill_vs_hf(mesh_device, device_params, seq_len, reset_seeds):
    cfg = minimax_config_dims()
    H, I = cfg["hidden_size"], cfg["intermediate_size"]
    E, K = 32, 8  # reduced expert count (tile-aligned) for a single-card test

    scale = H**-0.5
    w1 = [torch.randn(I, H) * scale for _ in range(E)]  # gate
    w3 = [torch.randn(I, H) * scale for _ in range(E)]  # up
    w2 = [torch.randn(H, I) * (I**-0.5) for _ in range(E)]  # down
    h = torch.randn(seq_len, H)

    # Dense routing weights (top-k normalized sigmoid), as the router produces.
    logits = torch.randn(seq_len, E)
    rw = torch.sigmoid(logits)
    top_w, idx = torch.topk(rw, K, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    dense = torch.zeros(seq_len, E).scatter_(1, idx, top_w)

    # --- torch reference: weighted sum over experts of silu(w1 h) * (w3 h) -> w2 ---
    out_ref = torch.zeros(seq_len, H)
    for e in range(E):
        gated = torch.nn.functional.silu(h @ w1[e].t()) * (h @ w3[e].t())  # [seq, I]
        out_ref += dense[:, e : e + 1] * (gated @ w2[e].t())

    # --- TT experts ---
    expert_config = ExpertConfig(intermediate_size=I, num_experts=E, hidden_size=H, num_experts_per_tok=K)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=mesh_device.shape[1], ep=mesh_device.shape[0]))
    # Linear topology: this Galaxy is a plain MESH (no torus). Harmless at TP=1.
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    state = {}
    for e in range(E):
        state[f"{e}.w1.weight"] = w1[e]
        state[f"{e}.w3.weight"] = w3[e]
        state[f"{e}.w2.weight"] = w2[e]

    # Default program config: the experts config auto-snaps core grids to divide Nt
    # at TP=1 (default grids assume TP-sharded dims).
    experts = Experts(
        mesh_device=mesh_device,
        config=expert_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=MiniMaxM2ExpertProgramConfig(),
        weight_dtype=ttnn.bfloat8_b,  # isolate logic from bfp4 quantization
    )

    repl = ttnn.ReplicateTensorToMesh(mesh_device)
    h_tt = ttnn.from_torch(
        h.reshape(1, 1, seq_len, H), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl
    )
    dense_tt = ttnn.from_torch(
        dense, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=repl
    )

    out_tt = experts(h_tt, topk_expert_weights=dense_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(seq_len, H)

    passing, pcc = comp_pcc(out_ref, out, 0.97)
    logger.info(f"experts prefill vs torch ref (E={E}, K={K}): {pcc}")
    assert passing, f"experts PCC fail: {pcc}"
