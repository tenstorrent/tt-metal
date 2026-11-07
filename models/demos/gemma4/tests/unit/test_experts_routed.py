# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Routed (token-grouped) expert prefill vs the all-experts sparse_matmul prefill and the HF reference.

    pytest models/demos/gemma4/tests/unit/test_experts_routed.py -k 1x1

One MoE layer on a single card with reduced experts and random weights. The routed path builds its dispatch and
combine indices on device from the dense routing; the old path is selected with GEMMA4_MOE_ROUTED_PREFILL=0.
"""

import pytest
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import (
    TestFactory,
    compare_tensors,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts
from models.demos.gemma4.tt.experts.routed_prefill import routed_prefill_enabled

PCC = 0.999
NUM_EXPERTS = 8
TOP_K = 4


def _host(tt_tensor, hidden_size, seq_len):
    return ttnn.to_torch(tt_tensor).reshape(-1, hidden_size).float()[:seq_len]


def _build(mesh_device):
    hf_text_config = TestFactory.create_hf_text_config(num_experts=NUM_EXPERTS, top_k=TOP_K)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts
    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = NUM_EXPERTS
    hf_config.top_k_experts = TOP_K
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=1, ep=1), prefill=ModeConfig(tp=1, ep=1))
    experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=None,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat16,
    )
    return hf_config, hf_experts, experts


def _inputs(hf_config, seq_len, skewed):
    g = torch.Generator().manual_seed(seq_len)
    hidden = torch.randn(1, 1, seq_len, hf_config.hidden_size, generator=g, dtype=torch.bfloat16)
    routing = torch.zeros(1, 1, seq_len, NUM_EXPERTS, dtype=torch.float32)
    idx = torch.zeros(seq_len, TOP_K, dtype=torch.int64)
    wts = torch.zeros(seq_len, TOP_K, dtype=torch.float32)
    for s in range(seq_len):
        if skewed:  # most tokens pile onto experts 0..TOP_K-1: one expert gets (nearly) every token
            e = torch.randperm(TOP_K, generator=g) if s % 8 else torch.randperm(NUM_EXPERTS, generator=g)[:TOP_K]
        else:
            e = torch.randperm(NUM_EXPERTS, generator=g)[:TOP_K]
        w = torch.rand(TOP_K, generator=g) + 0.1
        w = w / w.sum()
        idx[s], wts[s] = e, w
        routing[0, 0, s, e] = w
    return hidden, routing.to(torch.bfloat16), idx, wts


def test_routed_prefill_enabled_flag(monkeypatch):
    monkeypatch.delenv("GEMMA4_MOE_ROUTED_PREFILL", raising=False)
    monkeypatch.delenv("GEMMA4_MOE_ROUTED_MIN_TOKENS", raising=False)
    assert routed_prefill_enabled(1024)
    assert not routed_prefill_enabled(32)  # decode frame keeps sparse_matmul
    assert not routed_prefill_enabled(1024, override=False)
    assert routed_prefill_enabled(32, override=True)
    monkeypatch.setenv("GEMMA4_MOE_ROUTED_PREFILL", "0")
    assert not routed_prefill_enabled(1024)


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("skewed", [False, True], ids=["uniform", "skewed"])
@pytest.mark.parametrize("seq_len", [128, 1024], ids=["prefill_128", "prefill_1024"])
def test_experts_routed_prefill(seq_len, skewed, mesh_device, reset_seeds, monkeypatch):
    """Routed path (exact capacity) vs the old all-experts path and the HF reference, PCC >= 0.999."""
    hf_config, hf_experts, experts = _build(mesh_device)
    hidden, routing, idx, wts = _inputs(hf_config, seq_len, skewed)
    with torch.no_grad():
        ref = hf_experts(hidden[0, 0].float(), idx, wts)

    def run(routed):
        monkeypatch.setenv("GEMMA4_MOE_ROUTED_PREFILL", "1" if routed else "0")
        monkeypatch.delenv("GEMMA4_MOE_ROUTED_CAPACITY", raising=False)
        x = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        r = ttnn.from_torch(routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        return _host(experts(x, r), hf_config.hidden_size, seq_len)

    old = run(routed=False)
    new = run(routed=True)
    ok, msg = compare_tensors(old, ref, pcc_threshold=PCC)
    assert ok, f"old path vs HF: {msg}"
    ok, msg = compare_tensors(new, ref, pcc_threshold=PCC)
    assert ok, f"routed path vs HF: {msg}"
    ok, msg = compare_tensors(new, old, pcc_threshold=PCC)
    assert ok, f"routed path vs old path: {msg}"


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize("seq_len", [128, 1024], ids=["prefill_128", "prefill_1024"])
def test_experts_routed_prefill_full_capacity(seq_len, mesh_device, reset_seeds, monkeypatch):
    """GEMMA4_MOE_ROUTED_CAPACITY=full: no host readback (traceable form), still exact."""
    hf_config, hf_experts, experts = _build(mesh_device)
    hidden, routing, idx, wts = _inputs(hf_config, seq_len, skewed=True)
    with torch.no_grad():
        ref = hf_experts(hidden[0, 0].float(), idx, wts)
    monkeypatch.setenv("GEMMA4_MOE_ROUTED_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_ROUTED_CAPACITY", "full")
    x = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    r = ttnn.from_torch(routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    out = _host(experts(x, r), hf_config.hidden_size, seq_len)
    ok, msg = compare_tensors(out, ref, pcc_threshold=PCC)
    assert ok, f"routed path (full capacity) vs HF: {msg}"
