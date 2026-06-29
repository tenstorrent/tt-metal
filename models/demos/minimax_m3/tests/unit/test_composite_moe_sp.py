# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Composite EP-MoE at the REAL config (SP=8 × TP=4 × EP=32) vs a from-scratch M3 golden, REAL layer-3 weights.

Validates CompositeEPMoE (self-owned dispatch/combine; bypasses the DeepSeek all-to-all that is wrong for
M3's skewed real-gate routing) end-to-end with SP-sharded input: x [1, S_local, H] per row, routing passed
in (as the model does), output SP-sharded [1, S_local, H]. Reassemble across rows and compare to the
from-scratch M3 router+clamped-experts golden. (NOT a random-weight test — uses real checkpoint weights.)
"""

import os

import pytest
import torch
from loguru import logger
from safetensors import safe_open

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig, ModeConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.experts_throughput.composite_moe import CompositeEPMoE
from models.demos.minimax_m3.utils.general_utils import get_default_num_links

from ..test_factory import parametrize_mesh_with_fabric

EMB, HID, E, K = 6144, 3072, 128, 4
CKPT = os.environ.get("M3_CKPT", "/data/vmelnykov/MiniMax-M3-ref")
P = "language_model.model.layers.3.block_sparse_moe."


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("s_total", [1024], ids=["s1024"])
def test_composite_moe_sp(mesh_device, device_params, s_total, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp_axis = 0
    s_local = s_total // rows
    shard = f"{CKPT}/model-00003-of-00059.safetensors"

    with safe_open(shard, framework="pt") as f:
        gW = f.get_tensor(P + "gate.weight").float()
        gB = f.get_tensor(P + "e_score_correction_bias").float()
        routed_w = [
            {
                "gate_proj": f.get_tensor(P + f"experts.{e}.w1.weight").float(),
                "up_proj": f.get_tensor(P + f"experts.{e}.w3.weight").float(),
                "down_proj": f.get_tensor(P + f"experts.{e}.w2.weight").float(),
            }
            for e in range(E)
        ]

    torch.manual_seed(0)
    x = torch.randn(s_total, EMB)
    x = x / x.norm(dim=-1, keepdim=True) * (EMB**0.5)

    # M3-correct routing (host)
    scores = torch.sigmoid(x @ gW.T)
    idx = torch.topk(scores + gB, K, dim=-1).indices  # [N,K]
    wsel = scores.gather(1, idx)
    wsel = wsel / wsel.sum(-1, keepdim=True)  # unbiased, normalized

    # from-scratch M3 golden
    vec = torch.zeros(E, s_total)
    for j in range(K):
        vec[idx[:, j], torch.arange(s_total)] = wsel[:, j]
    gold = torch.zeros(s_total, EMB)
    for g in range(E):
        m = vec[g] > 0
        if not m.any():
            continue
        xt = x[m]
        ga = (xt @ routed_w[g]["gate_proj"].T).clamp(max=7.0)
        up = (xt @ routed_w[g]["up_proj"].T).clamp(-7.0, 7.0)
        act = (up + 1.0) * (ga * torch.sigmoid(1.702 * ga))
        gold[m] += vec[g][m].unsqueeze(-1) * (act @ routed_w[g]["down_proj"].T)

    mesh_config = MeshConfig((rows, cols), decode=ModeConfig(tp=cols, ep=rows))
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    moe = CompositeEPMoE(mesh_device, ccl, mesh_config, routed_w, EMB, HID, E, weights_dtype=ttnn.bfloat8_b)

    # SP-shard the input + routing across rows (seq dim 1), replicate across cols.
    def sp_shard(t, dt, layout):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dt,
            layout=layout,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=(1, None)),
        )

    x_tt = sp_shard(x.reshape(1, s_total, EMB).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)
    idx_tt = sp_shard(idx.reshape(1, s_total, K).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    wts_tt = sp_shard(wsel.reshape(1, s_total, K).to(torch.bfloat16), ttnn.bfloat16, ttnn.TILE_LAYOUT)

    out = moe(x_tt, idx_tt, wts_tt)  # [1, s_local, H] SP seq-shard
    # reassemble across rows: device (r,0) holds rows [r*s_local:(r+1)*s_local]
    dts = ttnn.get_device_tensors(out)
    full = torch.cat([ttnn.to_torch(dts[r * cols]).float().reshape(s_local, EMB) for r in range(rows)], dim=0)

    passing, pcc = comp_pcc(gold, full, 0.99)
    logger.info(f"composite EP-MoE SP=8xTP=4xEP=32 (real layer-3 weights) vs from-scratch M3 golden: pcc={pcc}")
    assert passing, f"composite MoE PCC fail: {pcc}"
