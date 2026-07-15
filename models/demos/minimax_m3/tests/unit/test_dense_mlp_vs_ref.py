# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tier-1 PCC test for MiniMax-M3 dense MLP (layers 0-2) vs a hand-written torch reference.

The dense MLP is a plain clamped-swigluoai SwiGLU FFN at dense_intermediate_size:
    down((up_clamped + 1) * (gate_clamped * sigmoid(alpha * gate_clamped)))
Anchor: transformers minimax_m3_vl MLP. Builds the real DenseMLP class with random weights and
compares to the torch reference. Torch-only (no HF/checkpoint) — single card (TP=1, no CCL).
"""

from types import SimpleNamespace

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.dense_mlp import DenseMLP
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric


def _torch_dense_mlp(x, gate_w, up_w, down_w, alpha, limit):
    """Dense clamped-swigluoai FFN reference (fp32). Weights are [out, in] (HF layout)."""
    x = x.float()
    gate = x @ gate_w.float().t()
    up = x @ up_w.float().t()
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    act = (up + 1.0) * (gate * torch.sigmoid(alpha * gate))
    return act @ down_w.float().t()


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "seq_len, hidden, inter",
    [
        (128, 6144, 12288),  # M3 dense layer: hidden_size, dense_intermediate_size
    ],
    ids=["s128"],
)
def test_dense_mlp_vs_ref(mesh_device, device_params, seq_len, hidden, inter, reset_seeds):
    """DenseMLP vs torch reference, random weights. (1,1)=TP=1; (8,4)=TP=4 (gate/up col-parallel +
    down-proj all-reduce). Output is full-hidden post-allreduce -> device[0] holds it. (8,4) needs
    TT_MESH_GRAPH_DESC_PATH=single_bh_galaxy."""
    alpha, limit = 1.702, 7.0
    x = torch.randn(1, 1, seq_len, hidden) * 0.1
    # HF Linear layout [out, in]; small scale so post-clamp distribution isn't degenerate.
    gate_w = torch.randn(inter, hidden) * 0.05
    up_w = torch.randn(inter, hidden) * 0.05
    down_w = torch.randn(hidden, inter) * 0.05

    ref = _torch_dense_mlp(x, gate_w, up_w, down_w, alpha, limit)

    hf_config = SimpleNamespace(hidden_size=hidden, swiglu_limit=limit, swiglu_alpha=alpha)
    state_dict = {"gate_proj.weight": gate_w, "up_proj.weight": up_w, "down_proj.weight": down_w}
    mesh_config = MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
    # TP>1 needs CCL for the down-proj all-reduce; at TP=1 it's unused.
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    mlp = DenseMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        weight_dtype=ttnn.bfloat16,
    )

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    out_tt = mlp(x_tt)
    out = ttnn.to_torch(ttnn.get_device_tensors(out_tt)[0]).reshape(1, 1, seq_len, hidden)

    passing, pcc = comp_pcc(ref, out, 0.99)
    logger.info(f"dense_mlp seq={seq_len} hidden={hidden} inter={inter}: {pcc}")
    assert passing, f"PCC fail (seq={seq_len}): {pcc}"
