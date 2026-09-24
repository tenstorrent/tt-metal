# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TtHyperConnection / TtHyperHead vs the reference DeepseekV4HyperConnection / HyperHead (random weights, seed 42).

Gate (plan M2): post and collapsed PCC >= 0.999; the full mix site (post * y + comb^T @ streams) PCC >= 0.999 per
stream; comb max|delta| <= 2e-3 = one bf16 ulp at 0.25 (comb sits near 0.25 everywhere so PCC is uninformative, and
the reference itself casts comb to bf16 before mixing, so an fp32 error under one ulp is invisible downstream;
MEASURED 2026-09-24 on one BH chip: 1.5e-3, from fp32 eltwise rounding over the 40-step division chain).
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size
from models.demos.deepseek_v3_d_p.tt.v4 import mhc_math as M
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import TtHyperConnection, TtHyperHead

_SEED = 42
_PCC = 0.999
_COMB_ATOL = 2e-3

# (mesh shape, device_params, sp_axis, tp_axis). The sub-galaxy rungs carry NO requires_mesh_topology mark: on a
# Blackhole galaxy host that mark only admits whole-galaxy shapes, while the bring-up ladder runs them by shrinking
# TT_VISIBLE_DEVICES (+ TT_MESH_GRAPH_DESC_PATH for the pod's [2,1] / [2,4] shapes); the mesh_device fixture still
# skips a shape the visible devices cannot form.
_MESH_CONFIGS = [
    pytest.param((1, 1), {}, 0, 1, id="single-1x1"),
    pytest.param(
        (2, 1),
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
        1,
        0,  # 2 chips: TP over the 2-long axis, SP = 1
        id="tp2-2x1",
    ),
    pytest.param(
        (2, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        id="fabric2d-mesh-2x4",
    ),
    pytest.param(
        (8, 4),
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D,
            "fabric_router_config": create_fabric_router_config(max_payload_size=get_max_payload_size()),
            "reliability_mode": ttnn.FabricReliabilityMode.RELAXED_INIT,
        },
        0,
        1,
        marks=pytest.mark.requires_mesh_topology(mesh_shape=(8, 4), topology="mesh-8x4"),
        id="fabric2d-mesh-8x4",
    ),
]


def _cfg(hidden):
    cfg = deepseek_v4_flash_hf_config(num_hidden_layers=4)
    cfg.hidden_size = hidden
    return cfg


def _mapper(mesh_device, sp_axis, tp_axis):
    if not hasattr(mesh_device, "shape"):
        return None
    dims = [None, None]
    if mesh_device.shape[sp_axis] > 1:
        dims[sp_axis] = 2
    if mesh_device.shape[tp_axis] > 1:
        dims[tp_axis] = 3
    if dims == [None, None]:
        return ttnn.ReplicateTensorToMesh(mesh_device)
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims)


def _composer(mesh_device, sp_axis, tp_axis):
    dims = [0, 0]  # a dim of 0 with a mesh extent of 1 concatenates nothing useful; ConcatMesh2d needs distinct dims
    dims[sp_axis] = 2
    dims[tp_axis] = 3
    return ttnn.ConcatMesh2dToTensor(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=tuple(dims))


def _to_device(mesh_device, x, sp_axis, tp_axis, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        x,
        device=mesh_device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=_mapper(mesh_device, sp_axis, tp_axis),
    )


def _to_host(mesh_device, t, sp_axis, tp_axis, *, replicated_cols=False):
    if not hasattr(mesh_device, "shape"):
        return ttnn.to_torch(t)
    if replicated_cols:  # a [1,1,S_l,32] row: SP-sharded rows, identical on every TP chip -> take TP index 0
        full = ttnn.to_torch(t, mesh_composer=_composer(mesh_device, sp_axis, tp_axis))
        return full[..., : t.shape[-1]]
    return ttnn.to_torch(t, mesh_composer=_composer(mesh_device, sp_axis, tp_axis))


def _check(name, ref, got, floor=_PCC):
    ok, pcc = comp_pcc(ref, got, floor)
    logger.info(f"{name}: PCC {pcc}")
    assert ok, f"{name}: PCC {pcc} < {floor}"


@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("hidden,seq", [(4096, 512)])
def test_hyper_connection_site(mesh_device, device_params, sp_axis, tp_axis, hidden, seq):
    cfg = _cfg(hidden)
    torch.manual_seed(_SEED)
    ref = DeepseekV4HyperConnection(cfg).eval()
    with torch.no_grad():
        ref.fn.normal_(0.0, 0.02)
        ref.base.normal_(0.0, 0.3)
        ref.scale.copy_(torch.tensor([1.1, 0.9, 1.3]))
    sp = mesh_device.shape[sp_axis] if hasattr(mesh_device, "shape") else 1
    assert seq % (32 * sp) == 0
    h = (torch.randn(1, seq, M.HC, hidden) * 2.0).to(torch.bfloat16)
    y = torch.randn(1, seq, hidden).to(torch.bfloat16)
    post_ref, comb_ref, collapsed_ref = ref(h)
    mixed_ref = post_ref.to(h.dtype).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(
        comb_ref.to(h.dtype).transpose(-1, -2), h
    )

    tt = TtHyperConnection.from_reference(mesh_device, ref, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    streams = [_to_device(mesh_device, h[:, :, i, :].unsqueeze(0), sp_axis, tp_axis) for i in range(M.HC)]
    y_tt = _to_device(mesh_device, y.unsqueeze(0), sp_axis, tp_axis)

    post_row, comb_row, collapsed = tt(streams)
    out = tt.mix(streams, y_tt, post_row, comb_row)

    post_h = _to_host(mesh_device, post_row, sp_axis, tp_axis, replicated_cols=True)[0, 0]  # [S, 32]
    comb_h = _to_host(mesh_device, comb_row, sp_axis, tp_axis, replicated_cols=True)[0, 0]
    _check("post", post_ref[0], post_h[:, M.POST0 : M.POST0 + M.HC])
    comb_4x4 = comb_h[:, M.COMB0 : M.COMB0 + 16].reshape(seq, M.HC, M.HC)
    dcomb = (comb_4x4 - comb_ref[0]).abs().max().item()
    logger.info(f"comb max|delta| {dcomb:.3e}")
    assert dcomb <= _COMB_ATOL, f"comb max|delta| {dcomb} > {_COMB_ATOL}"
    _check("collapsed", collapsed_ref[0].float(), _to_host(mesh_device, collapsed, sp_axis, tp_axis)[0, 0].float())
    for k in range(M.HC):
        _check(
            f"mixed stream {k}",
            mixed_ref[0, :, k, :].float(),
            _to_host(mesh_device, out[k], sp_axis, tp_axis)[0, 0].float(),
        )


@pytest.mark.parametrize(
    "mesh_device, device_params, sp_axis, tp_axis", _MESH_CONFIGS, indirect=["mesh_device", "device_params"]
)
def test_hyper_head(mesh_device, device_params, sp_axis, tp_axis):
    cfg = _cfg(4096)
    torch.manual_seed(_SEED)
    head = DeepseekV4HyperHead(cfg).eval()
    with torch.no_grad():
        head.hc_fn.normal_(0.0, 0.02)
        head.hc_base.normal_(0.0, 0.3)
        head.hc_scale.fill_(0.8)
    seq = 256
    h = torch.randn(1, seq, M.HC, 4096).to(torch.bfloat16)
    out_ref = head(h)
    tt = TtHyperHead.from_reference(mesh_device, head, cfg, sp_axis=sp_axis, tp_axis=tp_axis)
    streams = [_to_device(mesh_device, h[:, :, i, :].unsqueeze(0), sp_axis, tp_axis) for i in range(M.HC)]
    out = tt(streams)
    _check("hyper_head", out_ref[0].float(), _to_host(mesh_device, out, sp_axis, tp_axis)[0, 0].float())
