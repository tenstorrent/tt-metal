# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local token-sorted routed-expert prefill (GEMMA4_MOE_LOCAL_ROUTED_PREFILL=1, tt/experts/local_routed_prefill.py)
vs the HF reference experts on the whole-expert TP lines, plus the per-chip routed golden (the index algebra) on
skewed routing and the chunked call.

    pytest models/demos/gemma4/tests/unit/test_experts_local_routed.py -k 1x4 -s
"""

import os

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.gemma4.tests.unit.test_experts import run_experts
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts
from models.demos.gemma4.tt.experts.local_routed_index import local_routed_golden

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    mesh_tp_and_axis,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)

_BH_ONLY = pytest.mark.skipif(not is_blackhole(), reason="the GeGLU routed-expert kernel runs on Blackhole only")


@skip_if_not_moe
@_BH_ONLY
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4), (1, 8)])
@pytest.mark.parametrize("seq_len", [128, 1024, 4096 + 128], ids=["prefill_128", "prefill_1024", "prefill_4224"])
def test_experts_local_routed(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    """Dense routing matrix in (the experts top-k it on device); uniform random routing; vs the HF experts."""
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS", "128")
    run_experts(1, seq_len, mesh_device, request)


@skip_if_not_moe
@_BH_ONLY
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_experts_local_routed_chunked(mesh_device, reset_seeds, request, monkeypatch):
    """4224 rows through 2048-row chunks (2048 + 2048 + 128): the chunk loop and a non-power-of-two tail."""
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS", "128")
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_CHUNK", "2048")
    run_experts(1, 4096 + 128, mesh_device, request)


def _build(mesh_device):
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    tp, tp_axis = mesh_tp_and_axis(mesh_device)
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts
    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    mesh_config = MeshConfig(
        mesh_device.shape, decode=ModeConfig(tp=tp, ep=1), prefill=ModeConfig(tp=tp, ep=1), tp_axis=tp_axis
    )
    ccl_manager = CCLManager(mesh_device, num_links=1)
    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    return tt_experts, hf_experts, hf_config, state_dict


@skip_if_not_moe
@_BH_ONLY
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_experts_local_routed_skewed(mesh_device, reset_seeds, request, monkeypatch):
    """Router-style (values, indices) in; 90% of the tokens route to chip 1's first 8 experts (regions up to
    0.9 * S rows, empty experts elsewhere); vs the HF experts and, per chip, vs the routed golden of the index
    algebra (chip 1 sees the hot regions, chip 0 only the random tail)."""
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS", "128")
    seq_len = 512
    tt_experts, hf_experts, hf_config, state_dict = _build(mesh_device)
    E, K, H = hf_config.num_experts, hf_config.top_k_experts, hf_config.hidden_size
    E_loc = tt_experts.num_local_experts
    assert tt_experts.local_routed is not None

    g = torch.Generator().manual_seed(1)
    indices = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(seq_len)])
    indices[: int(0.9 * seq_len)] = torch.arange(K) + E_loc  # chip 1's first K experts
    weights = torch.rand(seq_len, K, generator=g) + 0.1
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
    x = torch.randn(1, 1, seq_len, H, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_experts(x.reshape(-1, H).float(), indices, weights.float())

    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    values_tt = ttnn.from_torch(
        weights.reshape(1, 1, seq_len, K),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=replicate,
    )
    idx_tt = ttnn.from_torch(
        indices.reshape(1, 1, seq_len, K).to(torch.int32),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.uint16,
        mesh_mapper=replicate,
    )
    # per-chip partials (before the TP all-reduce) vs the routed golden of the index algebra
    partial = tt_experts.local_routed.forward(x_tt, (values_tt, idx_tt))
    gate_up = state_dict["gate_up_proj"]  # [E, 2I, H], gate rows first
    down = state_dict["down_proj"]  # [E, H, I]
    I = gate_up.shape[1] // 2
    chips = ttnn.get_device_tensors(partial)
    for chip, t in enumerate(chips):
        base = chip * E_loc
        gate = [gate_up[base + e, :I, :].T for e in range(E_loc)]
        up = [gate_up[base + e, I:, :].T for e in range(E_loc)]
        dn = [down[base + e].T for e in range(E_loc)]
        golden = local_routed_golden(x.reshape(-1, H), weights, indices, base, E_loc, gate, up, dn)
        got = ttnn.to_torch(t).reshape(-1, H).float()[:seq_len]
        passing, pcc = comp_pcc(golden, got, 0.99)
        assert passing, f"chip {chip} partial vs routed golden: {pcc}"
    # the full path (TP all-reduce included) vs the HF experts
    tt_output = tt_experts(x_tt, (values_tt, idx_tt))
    tt_out_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]).reshape(-1, H).float()[:seq_len]
    passing, pcc_msg = compare_tensors(tt_out_torch, ref_output, pcc_threshold=0.99)
    assert passing, f"local routed experts (skewed) PCC too low: {pcc_msg}"


@skip_if_not_moe
@_BH_ONLY
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 4)])
def test_experts_local_routed_threshold(mesh_device, reset_seeds, request, monkeypatch):
    """Calls below GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS (and the decode frame) keep the sparse_matmul path."""
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_LOCAL_ROUTED_MIN_TOKENS", "256")
    tt_experts, _, hf_config, _ = _build(mesh_device)
    H = hf_config.hidden_size
    short = ttnn.from_torch(
        torch.zeros(1, 1, 128, H, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert not tt_experts._use_local_routed(short)
    assert not tt_experts.wants_sparse_routing(short)
    long = ttnn.from_torch(
        torch.zeros(1, 1, 256, H, dtype=torch.bfloat16),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    assert tt_experts._use_local_routed(long)
    assert tt_experts.wants_sparse_routing(long)
    run_experts(1, 128, mesh_device, request)  # the sparse_matmul path, unchanged
    assert os.environ.get("GEMMA4_MOE_LOCAL_ROUTED_PREFILL") == "1"
