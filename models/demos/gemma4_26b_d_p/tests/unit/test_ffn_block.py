# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Gemma-4 FFN sub-layer (dense MLP + router + EP MoE + norms + layer_scalar), real weights, vs torch."""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.tt_ccl import per_axis_topology
from models.demos.gemma4_26b_d_p.bringup.registry import record_result
from models.demos.gemma4_26b_d_p.reference.blocks import DecoderLayer
from models.demos.gemma4_26b_d_p.reference.config import Gemma4TextConfig
from models.demos.gemma4_26b_d_p.reference.weights import CheckpointReader
from models.demos.gemma4_26b_d_p.tests.mesh import MESH_PARAMS, mesh_id, sp_tp
from models.demos.gemma4_26b_d_p.tt.ffn import TtFFN


def ref_ffn(layer: DecoderLayer, r):
    B, S, H = r.shape
    m1 = layer.post_feedforward_layernorm_1(layer.mlp(layer.pre_feedforward_layernorm(r)))
    routing, idx = layer.router(r)
    xe = layer.pre_feedforward_layernorm_2(r).reshape(B * S, H)
    m2 = layer.post_feedforward_layernorm_2(layer.experts(xe, routing.reshape(B * S, -1), idx.reshape(B * S, -1)).view(B, S, H))
    return (r + layer.post_feedforward_layernorm(m1 + m2)) * layer.layer_scalar, m1, m2


@MESH_PARAMS
@pytest.mark.parametrize("layer_idx", [0, 5])
@pytest.mark.parametrize("tokens", [4096])
def test_ffn_block(mesh_device, device_params, layer_idx, tokens):
    cfg = Gemma4TextConfig.from_json()
    r = CheckpointReader()
    sd = r.layer_state(layer_idx)
    sp, tp, _, _ = sp_tp(mesh_device)
    S_local = tokens // sp

    torch.manual_seed(0)
    ref = DecoderLayer(cfg, layer_idx).float()
    ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    x = torch.randn(1, tokens, cfg.hidden_size).bfloat16().float()
    with torch.no_grad():
        ref_out, ref_m1, ref_m2 = ref_ffn(ref, x)

    sp_topo, _ = per_axis_topology(device_params["fabric_config"])
    ffn = TtFFN(mesh_device, cfg, sd, seq_len_per_chip=S_local, sp_topology=sp_topo, layer_idx=layer_idx)
    x_map = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=(2, None))
    tt_x = ttnn.from_torch(x[None], device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=x_map)
    out = ffn(tt_x)
    dts = ttnn.get_device_tensors(out)
    got = torch.cat([ttnn.to_torch(dts[s * tp]).float()[0, 0] for s in range(sp)], 0)[None]
    ok, pcc = comp_pcc(ref_out, got, 0.99)
    delta_ok, delta_pcc = comp_pcc(ref_out - x * ref.layer_scalar, got - x * ref.layer_scalar)
    logger.info(f"FFN L{layer_idx} mesh={mesh_id(mesh_device)}: out PCC {pcc}, ffn-delta PCC {delta_pcc}")
    record_result("layer:ffn", mesh_id(mesh_device), float(delta_pcc), bool(ok and delta_pcc > 0.98), f"L{layer_idx} {tokens} tokens; PCC of (out - r*scalar)")
    assert ok, pcc
    assert delta_pcc > 0.98, delta_pcc
