# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""EP experts under lane-sharded batches on a (2,4) block (tp=4 over axis 1, ep=2 = lanes over axis 0):
lane-parallel rows, single-owner prefill for every owner lane, decode-shaped rows, and the served
interleaving (owner prefill -> decode -> other owner prefill -> decode), each against the HF experts."""
import torch

import ttnn
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)

ROWS = 64  # rows per lane for prefill-shaped calls


def _build(mesh_device):
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts
    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    tp, ep = mesh_device.shape[1], mesh_device.shape[0]
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp, ep=ep), prefill=ModeConfig(tp=tp, ep=ep))
    mesh_config.lane_sharded = True  # lanes = the ep axis
    ccl_manager = CCLManager(mesh_device, num_links=1)
    experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    return hf_config, hf_experts, mesh_config, experts


def _inputs(hf_config, lanes, rows, seed):
    g = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, 1, lanes * rows, hf_config.hidden_size, generator=g, dtype=torch.bfloat16)
    routing = torch.zeros(1, 1, lanes * rows, hf_config.num_experts, dtype=torch.float32)
    idx = torch.zeros(lanes * rows, hf_config.top_k_experts, dtype=torch.int64)
    wts = torch.zeros(lanes * rows, hf_config.top_k_experts, dtype=torch.float32)
    for s in range(lanes * rows):
        e = torch.randperm(hf_config.num_experts, generator=g)[: hf_config.top_k_experts]
        w = torch.rand(hf_config.top_k_experts, generator=g) + 0.1
        w = w / w.sum()
        idx[s], wts[s] = e, w
        routing[0, 0, s, e] = w
    return hidden, routing.to(torch.bfloat16), idx, wts


def _ref(hf_experts, hidden, idx, wts, lane, rows):
    sl = slice(lane * rows, (lane + 1) * rows)
    with torch.no_grad():
        return hf_experts(hidden[0, 0, sl].float(), idx[sl], wts[sl])


def _run(experts, mesh_device, mesh_config, hidden, routing):
    mapper = mesh_config.lane_shard_mapper(mesh_device, tensor_dim=2)
    x = ttnn.from_torch(hidden, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)
    r = ttnn.from_torch(routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)
    out = experts(x, r)
    per_lane = []
    tp = mesh_device.shape[1]
    for lane in range(mesh_device.shape[0]):
        per_lane.append(ttnn.to_torch(ttnn.get_device_tensors(out)[lane * tp]).reshape(-1, hidden.shape[-1]).float())
    return per_lane


def _check(got, ref, request, what):
    passing, msg = compare_tensors(got[: ref.shape[0]], ref, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"{what}: {msg}"


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4)])
def test_lane_parallel_prefill(mesh_device, reset_seeds, request):
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = mesh_device.shape[0]
    hidden, routing, idx, wts = _inputs(hf_config, lanes, ROWS, seed=1)
    mesh_config._g4_single_lane_prefill_owner = None
    per_lane = _run(experts, mesh_device, mesh_config, hidden, routing)
    for lane in range(lanes):
        _check(per_lane[lane], _ref(hf_experts, hidden, idx, wts, lane, ROWS), request, f"lane {lane} rows")


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4)])
def test_single_owner_prefill_every_lane(mesh_device, reset_seeds, request):
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = mesh_device.shape[0]
    hidden, routing, idx, wts = _inputs(hf_config, lanes, ROWS, seed=2)
    for owner in range(lanes):
        mesh_config._g4_single_lane_prefill_owner = owner
        per_lane = _run(experts, mesh_device, mesh_config, hidden, routing)
        ref = _ref(hf_experts, hidden, idx, wts, owner, ROWS)
        # every lane holds the owner's result after the fast path's all-reduce
        for lane in range(lanes):
            _check(per_lane[lane], ref, request, f"owner {owner} seen from lane {lane}")
    mesh_config._g4_single_lane_prefill_owner = None


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4)])
def test_decode_rows_after_owner_prefill(mesh_device, reset_seeds, request):
    """Served order: owner-0 prefill, decode rows (1 per lane), owner-1 prefill, decode rows again."""
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = mesh_device.shape[0]
    hidden_p, routing_p, idx_p, wts_p = _inputs(hf_config, lanes, ROWS, seed=3)
    hidden_d, routing_d, idx_d, wts_d = _inputs(hf_config, lanes, 1, seed=4)
    for owner in range(lanes):
        mesh_config._g4_single_lane_prefill_owner = owner
        per_lane = _run(experts, mesh_device, mesh_config, hidden_p, routing_p)
        _check(
            per_lane[owner], _ref(hf_experts, hidden_p, idx_p, wts_p, owner, ROWS), request, f"owner {owner} prefill"
        )
        mesh_config._g4_single_lane_prefill_owner = None  # decode: every lane's rows are real
        per_lane = _run(experts, mesh_device, mesh_config, hidden_d, routing_d)
        for lane in range(lanes):
            _check(
                per_lane[lane],
                _ref(hf_experts, hidden_d, idx_d, wts_d, lane, 1),
                request,
                f"decode row lane {lane} after owner {owner}",
            )
