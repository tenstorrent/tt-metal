# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""The small-row expert path (GEMMA4_SMALL_ROW_EXPERTS=1: custom_ops/expert_gather_small + the GeGLU op + W @ out +
one all-reduce) against the token-dispatch path on the same (4,1) column with the real 26B expert weights: the
layer-level expert output of a 128- and a 256-token call must match the dispatch path's to bf16 rounding
(PCC >= 0.9999), both are reported against HF, and a 1024-token call must keep the dispatch path."""


import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import TestFactory, mesh_tp_and_axis, parametrize_mesh_with_fabric
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts


def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _build(mesh_device, state_dict, hf_config):
    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    ep_axis = 1 - tp_axis
    ep = mesh_device.shape[ep_axis] if mesh_device.shape[ep_axis] > 1 else 1
    mesh_config = MeshConfig(
        mesh_device.shape, decode=ModeConfig(tp=tp, ep=ep), prefill=ModeConfig(tp=tp, ep=ep), tp_axis=tp_axis
    )
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None
    return Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )


@pytest.mark.skipif(not is_blackhole(), reason="token-dispatch prefill runs the Blackhole-only routed-expert op")
@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1)])
@pytest.mark.parametrize("seq_len", [128, 256, 1024], ids=lambda s: f"seq{s}")
def test_experts_small_rows_vs_dispatch(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_FIT_CHUNK", "1")
    hf_text_config = TestFactory.create_hf_text_config()
    hf_experts = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0).experts
    state_dict = {"gate_up_proj": hf_experts.gate_up_proj.data.clone(), "down_proj": hf_experts.down_proj.data.clone()}
    hf_config = TestFactory.create_hf_config()
    E, K, H = hf_config.num_experts, hf_config.top_k_experts, hf_config.hidden_size
    x = torch.randn(1, 1, seq_len, H, dtype=torch.bfloat16)
    routing = torch.zeros(1, 1, seq_len, E)
    hf_idx, hf_w = torch.zeros(seq_len, K, dtype=torch.int64), torch.zeros(seq_len, K)
    for s in range(seq_len):
        sel = torch.randperm(E)[:K]
        w = torch.rand(K) + 0.1
        w = w / w.sum()
        hf_idx[s], hf_w[s] = sel, w
        routing[0, 0, s, sel] = w
    routing = routing.to(torch.bfloat16)
    with torch.no_grad():
        ref = hf_experts(x.reshape(-1, H).float(), hf_idx, hf_w)
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
    r_tt = ttnn.from_torch(routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
    outs = {}
    for name, flag in (("dispatch", "0"), ("small_rows", "1")):
        monkeypatch.setenv("GEMMA4_SMALL_ROW_EXPERTS", flag)
        experts = _build(mesh_device, state_dict, hf_config)
        plans = experts.dispatch_experts._plans
        o = experts(x_tt, r_tt)
        outs[name] = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(-1, H).float()[:seq_len]
        # the plan the call used: the smallest chunk that holds the call (the full-chunk plan always exists)
        plan = min((q for q in plans.values() if q.chunk_tokens >= seq_len), key=lambda q: q.chunk_tokens)
        took_small = plan.gather is not None
        assert took_small == (flag == "1" and seq_len <= 256), (name, seq_len, took_small)
        print(
            f"[seq{seq_len} {name}] chunk {plan.chunk_tokens} small_path={took_small} pcc vs HF {_pcc(ref, outs[name]):.5f}"
        )
        del experts
    d = (outs["dispatch"] - outs["small_rows"]).abs()
    p = _pcc(outs["dispatch"], outs["small_rows"])
    print(
        f"[seq{seq_len}] small_rows vs dispatch: pcc {p:.6f} max|d| {d.max():.4f} mean|d| {d.mean():.5f} (|out| max {outs['dispatch'].abs().max():.3f})"
    )
    assert p > 0.9999, p
    assert d.max() < 0.05 * outs["dispatch"].abs().max() + 0.05, d.max()
