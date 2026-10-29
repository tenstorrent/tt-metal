# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul.

    pytest -k "1x1"        # single card (reduced experts)
    pytest -k "1x8"        # T3K with TP-sharded experts + CCL
    pytest -k "decode"     # decode only
    pytest -k "prefill"    # prefill only
"""

import os
import time

import pytest
import torch

import ttnn
from models.common.utility_functions import comp_pcc, is_blackhole
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tests.test_factory import (
    TestFactory,
    lanes_fabric_config,
    mesh_tp_and_axis,
    parametrize_mesh_with_fabric,
    skip_if_config_only_checkpoint,
)
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts
from models.demos.gemma4.tt.experts.local_routed_index import local_routed_golden

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    lanes_fabric_config,
    mesh_tp_and_axis,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 2), (1, 4), (2, 4), (8, 4)])
@parametrize_batch_seq()
def test_experts(batch_size, seq_len, mesh_device, reset_seeds, request):
    run_experts(batch_size, seq_len, mesh_device, request)


# the direct dispatch/combine ops run along mesh axis 0 only: single-row meshes take test_experts_dispatch_ring; the
# 4-chip line runs them as the (4,1) column mesh (GEMMA4_TP_AXIS=0: TP over the rows = the dispatch group)
@skip_if_not_moe
@pytest.mark.skipif(
    not is_blackhole(), reason="token-dispatch prefill runs the Blackhole-only unified_routed_expert_ffn"
)
@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize("seq_len", [128, 1024, 4096 + 128], ids=["prefill_128", "prefill_1024", "prefill_4224"])
def test_experts_dispatch(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    """Token-dispatch prefill (DeepSeek dispatch/combine + fused GeGLU experts) vs the HF reference; 4224 tokens
    exercise a partial second dispatch chunk."""
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    run_experts(1, seq_len, mesh_device, request)


# GEMMA4_MOE_DISPATCH_FIT_CHUNK=1: a call shorter than the 4096-token chunk runs on a chunk of its own length (128
# tokens on 128 rows instead of 4096 padded rows; 2065 on a 2176-row chunk with the sentinel-marked padding; 4224 as
# 4096 + a 128 remainder). ``sparse``: the router's (values, indices) pair instead of the dense routing matrix
# (GEMMA4_MOE_DISPATCH_SPARSE_ROUTING=1 in the model: no dense scatter, no second top-k).
@skip_if_not_moe
@pytest.mark.skipif(
    not is_blackhole(), reason="token-dispatch prefill runs the Blackhole-only unified_routed_expert_ffn"
)
@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1), (2, 4), (8, 4)])
@pytest.mark.parametrize(
    "seq_len, sparse",
    [(128, False), (128, True), (1024, True), (2065, True), (4096 + 128, True)],
    ids=["fit_dense_128", "fit_sparse_128", "fit_sparse_1024", "fit_sparse_2065", "fit_sparse_4224"],
)
def test_experts_dispatch_fit(seq_len, sparse, mesh_device, reset_seeds, request, monkeypatch):
    """Token-dispatch prefill on a chunk fitted to the call (and the sparse routing input) vs the HF reference."""
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_FIT_CHUNK", "1")
    run_experts(1, seq_len, mesh_device, request, sparse_routing=sparse)


def _skip_unless_ring_payload_fits(hidden):
    """dispatch_fabric2d sends a whole bf16 token page plus 64 B per fabric packet and refuses more than the fabric
    max payload (4352 B by default). Opening the mesh with a raised router config (6528 B for the 26B's 2816-wide
    hidden) hangs the loop-wired QB2 inside the op (tt-metal CI at ca39639f7c1, bh_quietbox_2: DispatchFabric running,
    the board left with eth firmware not ready), so a model whose token exceeds the default is skipped here; the
    planner refuses the ring for it at runtime (ring_transport_reason) and test_dispatch_ring_payload_host.py pins
    the numbers. The ring transport is unit-only: no served configuration selects it."""
    from models.demos.gemma4.tt.experts.dispatch_prefill import (
        FABRIC_DEFAULT_PAYLOAD_BYTES,
        ring_payload_required_bytes,
    )

    need = ring_payload_required_bytes(hidden)
    if need > FABRIC_DEFAULT_PAYLOAD_BYTES:
        pytest.skip(
            f"26B token payload exceeds the fabric default ({need} B > {FABRIC_DEFAULT_PAYLOAD_BYTES} B); upstream "
            "dispatch_fabric2d cannot carry it on this build"
        )


# single-row meshes (the 4-chip and 8-chip entries) dispatch along their columns over the hop-forwarding
# dispatch_fabric2d / combine_fabric2d pair, which needs the axis wrap-wired (FABRIC_1D_RING; skipped on the line)
@skip_if_not_moe
@pytest.mark.skipif(
    not is_blackhole(), reason="token-dispatch prefill runs the Blackhole-only unified_routed_expert_ffn"
)
@parametrize_mesh_with_fabric(
    mesh_shapes=[(1, 4), (1, 8)], device_params_extra={"fabric_config": lanes_fabric_config()}
)
@pytest.mark.parametrize("seq_len", [128, 1024, 4096 + 128], ids=["prefill_128", "prefill_1024", "prefill_4224"])
def test_experts_dispatch_ring(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    """Token-dispatch prefill on the ring transport (single-row mesh) vs the HF reference."""
    if lanes_fabric_config() != ttnn.FabricConfig.FABRIC_1D_RING:
        pytest.skip("the ring transport (dispatch_fabric2d / combine_fabric2d) needs FABRIC_1D_RING")
    _skip_unless_ring_payload_fits(int(TestFactory.create_hf_text_config().hidden_size))
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_RING", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    run_experts(1, seq_len, mesh_device, request)


# the decode MoE kernel dispatches on the lane fabric: FABRIC_1D_RING on the Galaxy, a FABRIC_1D line on the (2,4)
# block (GEMMA4_TEST_FABRIC_1D=1; the wrapper then picks the Linear topology from the fabric)
@skip_if_not_moe
@pytest.mark.skipif(
    not is_blackhole(), reason="the decode MoE kernel's sharded CCL address generation is Blackhole-only"
)
@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)], device_params_extra={"fabric_config": lanes_fabric_config()}
)
@pytest.mark.parametrize("seq_len", [64, 128, 256], ids=["frame_64", "frame_128", "frame_256"])
def test_experts_decode_kernel(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    """Decode-frame routed experts on the decode MoE kernel (dispatch within the column, moe_compute, bfp4 experts)
    vs the HF reference; the frame is the call size so each mesh row takes seq_len / rows tokens."""
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", str(seq_len))
    # the kernel quantizes the expert weights to bfloat4_b: 0.98 vs the bf16 HF reference (the module's own
    # golden test reports 0.991); the bf8 sparse path reaches 0.9993
    run_experts(1, seq_len, mesh_device, request, pcc_threshold=0.97)


def run_experts(batch_size, seq_len, mesh_device, request, pcc_threshold=None, sparse_routing=False):
    """Test MoE experts against HF reference.

    1x1: Uses reduced experts (8) and bfloat16 weights for fast single-card test.
    1x8: Uses real model dimensions with TP-sharded bfloat8_b weights.
    ``sparse_routing``: hand the experts the router's sparse ``(values, indices)`` pair (the dense matrix's top-k,
    bf16 / uint16 TILE) instead of the dense routing matrix (the token-dispatch prefill takes both).
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    # (8,4): the Galaxy one-instance orientation (TP on the size-8 axis 0, fractured dense weights,
    # EP over the 4 columns); every other 2-D mesh: TP over axis 1, EP over axis 0 -- or, with
    # GEMMA4_TP_AXIS=0, TP over axis 0 and EP over axis 1 (the 4-chip line as a (4,1) column mesh).
    galaxy = is_mesh and tuple(mesh_device.shape) == (8, 4)
    tp, tp_axis = mesh_tp_and_axis(mesh_device)

    if tp > 1:
        # Multi-device: real model dims, TP-sharded, HF reference layer for weights
        hf_text_config = TestFactory.create_hf_text_config()
        hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
        hf_experts = hf_layer.experts
        state_dict = {
            "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
            "down_proj": hf_experts.down_proj.data.clone(),
        }
        hf_config = TestFactory.create_hf_config()
        weight_dtype = ttnn.bfloat8_b
    else:
        # Single device: reduced experts for speed
        num_experts_override = 8
        top_k_override = 4
        hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts_override, top_k=top_k_override)
        hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
        hf_experts = hf_layer.experts
        state_dict = {
            "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
            "down_proj": hf_experts.down_proj.data.clone(),
        }
        hf_config = TestFactory.create_hf_config()
        hf_config.num_experts = num_experts_override
        hf_config.top_k_experts = top_k_override
        weight_dtype = ttnn.bfloat16

    num_experts = hf_config.num_experts
    top_k = hf_config.top_k_experts
    hidden_size = hf_config.hidden_size

    expert_config = Gemma4ExpertConfig(hf_config)
    # 2-D meshes: TP over axis 1, expert parallelism over axis 0 (one local-expert slice per row)
    if galaxy:
        ep = mesh_device.shape[1]
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=tp, ep=ep),
            prefill=ModeConfig(tp=tp, ep=ep),
            tp_axis=0,
            weight_fracture=True,
        )
    else:
        ep_axis = 1 - tp_axis
        ep = mesh_device.shape[ep_axis] if hasattr(mesh_device, "shape") and mesh_device.shape[ep_axis] > 1 else 1
        mesh_config = MeshConfig(
            mesh_device.shape, decode=ModeConfig(tp=tp, ep=ep), prefill=ModeConfig(tp=tp, ep=ep), tp_axis=tp_axis
        )
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=expert_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=weight_dtype,
    )

    # Create input + dense routing
    x = torch.randn(1, 1, seq_len, hidden_size, dtype=torch.bfloat16)
    hf_indices = torch.zeros(seq_len, top_k, dtype=torch.int64)
    hf_weights = torch.zeros(seq_len, top_k, dtype=torch.float32)
    routing = torch.zeros(1, 1, seq_len, num_experts, dtype=torch.float32)
    for s in range(seq_len):
        experts_selected = torch.randperm(num_experts)[:top_k]
        weights = torch.rand(top_k) + 0.1
        weights = weights / weights.sum()
        hf_indices[s] = experts_selected
        hf_weights[s] = weights
        routing[0, 0, s, experts_selected] = weights
    routing = routing.to(torch.bfloat16)

    # HF reference
    x_flat = x.reshape(-1, hidden_size).float()
    with torch.no_grad():
        ref_output = hf_experts(x_flat, hf_indices, hf_weights)

    # TT forward
    replicate = ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate)
    if sparse_routing:
        # the router's sparse form: the dense matrix's top-k values and expert ids ([1,1,S,K] bf16 / uint16)
        values, indices = routing.float().topk(top_k, dim=-1)
        routing_tt = (
            ttnn.from_torch(
                values.to(torch.bfloat16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                mesh_mapper=replicate,
            ),
            ttnn.from_torch(
                indices.to(torch.int16),
                device=mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.uint16,
                mesh_mapper=replicate,
            ),
        )
    else:
        routing_tt = ttnn.from_torch(
            routing, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=replicate
        )
    tt_output = tt_experts(x_tt, routing_tt)

    tt_out_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .reshape(-1, hidden_size)
        .float()[:seq_len]
    )

    passing, pcc_msg = compare_tensors(
        tt_out_torch, ref_output, pcc_threshold=pcc_threshold or get_pcc_threshold(request)
    )
    assert passing, f"Experts (tp={tp}) PCC too low: {pcc_msg}"


# ===== merged from tests/unit/test_experts_lanes.py: EP experts under lane-sharded batches on a (2,4) block (tp=4 over axis 1, ep=2 = lanes over axis 0):
_LANES_BH_ONLY = pytest.mark.skipif(
    not is_blackhole(), reason="the lane engine (lane-sharded EP batches) ships on Blackhole only"
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
    if tuple(mesh_device.shape) == (8, 4):
        # the Galaxy one-instance orientation: TP on the size-8 axis 0, fractured dense weights, lanes/EP over
        # the 4 columns (4 whole experts per chip)
        tp, ep = mesh_device.shape[0], mesh_device.shape[1]
        mesh_config = MeshConfig(
            mesh_device.shape,
            decode=ModeConfig(tp=tp, ep=ep),
            prefill=ModeConfig(tp=tp, ep=ep),
            tp_axis=0,
            weight_fracture=True,
        )
    else:
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
    return _read_lanes(out, mesh_device, hidden.shape[-1])


def _check(got, ref, request, what):
    passing, msg = compare_tensors(got[: ref.shape[0]], ref, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"{what}: {msg}"


@_LANES_BH_ONLY
@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
def test_lane_parallel_prefill(mesh_device, reset_seeds, request):
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = _num_lanes(mesh_device)
    hidden, routing, idx, wts = _inputs(hf_config, lanes, ROWS, seed=1)
    mesh_config._g4_single_lane_prefill_owner = None
    per_lane = _run(experts, mesh_device, mesh_config, hidden, routing)
    for lane in range(lanes):
        _check(per_lane[lane], _ref(hf_experts, hidden, idx, wts, lane, ROWS), request, f"lane {lane} rows")


@_LANES_BH_ONLY
@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
def test_single_owner_prefill_every_lane(mesh_device, reset_seeds, request):
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = _num_lanes(mesh_device)
    hidden, routing, idx, wts = _inputs(hf_config, lanes, ROWS, seed=2)
    for owner in range(lanes):
        mesh_config._g4_single_lane_prefill_owner = owner
        per_lane = _run(experts, mesh_device, mesh_config, hidden, routing)
        ref = _ref(hf_experts, hidden, idx, wts, owner, ROWS)
        # every lane holds the owner's result after the fast path's all-reduce
        for lane in range(lanes):
            _check(per_lane[lane], ref, request, f"owner {owner} seen from lane {lane}")
    mesh_config._g4_single_lane_prefill_owner = None


@_LANES_BH_ONLY
@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
def test_decode_rows_after_owner_prefill(mesh_device, reset_seeds, request):
    """Served order: owner-0 prefill, decode rows (1 per lane), owner-1 prefill, decode rows again."""
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = _num_lanes(mesh_device)
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


DECODE_ROWS = 32  # one tile of decode rows per lane (the served B=32 per lane)


def _lane_device_indices(mesh_device):
    """Row-major index of each lane's first chip: lanes on axis 1 for the (8,4) Galaxy orientation, axis 0 otherwise."""
    rows, cols = (int(x) for x in mesh_device.shape)
    if (rows, cols) == (8, 4):
        return list(range(cols))
    return [lane * cols for lane in range(rows)]


def _num_lanes(mesh_device):
    return len(_lane_device_indices(mesh_device))


def _read_lanes(out, mesh_device, hidden_dim):
    shards = ttnn.get_device_tensors(out)
    return [ttnn.to_torch(shards[i]).reshape(-1, hidden_dim).float() for i in _lane_device_indices(mesh_device)]


@_LANES_BH_ONLY
@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)], device_params_extra={"trace_region_size": 64_000_000})
def test_traced_decode_rows_around_eager_prefill(mesh_device, reset_seeds, request):
    """Served order with a captured decode trace: capture the decode-row graph once (as warmup does),
    replay it, run an eager owner prefill, replay with reloaded rows, run an eager lane-parallel prefill,
    replay again. Every replay and every prefill is checked against the HF experts."""
    hf_config, hf_experts, mesh_config, experts = _build(mesh_device)
    lanes = _num_lanes(mesh_device)
    hidden_dim = hf_config.hidden_size
    mapper = mesh_config.lane_shard_mapper(mesh_device, tensor_dim=2)

    def _dev(t):
        return ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    def _host(t):
        return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=mapper)

    hidden_d, routing_d, idx_d, wts_d = _inputs(hf_config, lanes, DECODE_ROWS, seed=5)
    x_in, r_in = _dev(hidden_d), _dev(routing_d)
    mesh_config._g4_single_lane_prefill_owner = None

    def _decode_graph():
        # the model hands the experts an intermediate activation, never the bound trace input itself
        return experts(ttnn.clone(x_in), ttnn.clone(r_in))

    _decode_graph().deallocate(True)  # compile
    trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    out_t = _decode_graph()
    ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(mesh_device)

    def _replay(hidden, routing, idx, wts, what):
        ttnn.copy_host_to_device_tensor(_host(hidden), x_in)
        ttnn.copy_host_to_device_tensor(_host(routing), r_in)
        ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=True)
        got = _read_lanes(out_t, mesh_device, hidden_dim)
        for lane in range(lanes):
            _check(got[lane], _ref(hf_experts, hidden, idx, wts, lane, DECODE_ROWS), request, f"{what} lane {lane}")

    _replay(hidden_d, routing_d, idx_d, wts_d, "replay 0")

    hidden_p, routing_p, idx_p, wts_p = _inputs(hf_config, lanes, ROWS, seed=6)
    for owner in range(lanes):
        mesh_config._g4_single_lane_prefill_owner = owner
        per_lane = _run(experts, mesh_device, mesh_config, hidden_p, routing_p)
        _check(
            per_lane[owner], _ref(hf_experts, hidden_p, idx_p, wts_p, owner, ROWS), request, f"owner {owner} prefill"
        )
        mesh_config._g4_single_lane_prefill_owner = None
        hidden_n, routing_n, idx_n, wts_n = _inputs(hf_config, lanes, DECODE_ROWS, seed=10 + owner)
        _replay(hidden_n, routing_n, idx_n, wts_n, f"replay after owner {owner} prefill")

    per_lane = _run(experts, mesh_device, mesh_config, hidden_p, routing_p)
    for lane in range(lanes):
        _check(
            per_lane[lane],
            _ref(hf_experts, hidden_p, idx_p, wts_p, lane, ROWS),
            request,
            f"lane-parallel prefill lane {lane}",
        )
    _replay(hidden_d, routing_d, idx_d, wts_d, "replay after lane-parallel prefill")
    ttnn.release_trace(mesh_device, trace_id)


def _build_engine(mesh_device):
    """The lane engine's own orientation on any (R, C) mesh: TP over the rows (axis 0), the C columns are the lanes and
    the EP axis, fractured dense weights, the decode MoE kernel on (frame 128 = one 32-row tile per lane over 2 rows on
    the (2,4) block; the Galaxy's default frame on the (8,4)). This is the layout the served Galaxy entry and the (2,4)
    block emulation run, as opposed to ``_build``'s lanes-over-rows (2,4) case."""
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_experts = hf_layer.experts
    state_dict = {
        "gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "down_proj": hf_experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    mesh_config.lane_sharded = True
    experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=1),
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    assert experts.moe_decode is not None, "the engine layout runs the decode MoE kernel"
    return hf_config, hf_experts, mesh_config, experts


@skip_if_not_moe
@pytest.mark.skipif(not is_blackhole(), reason="the decode MoE kernel is Blackhole-only")
@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)
def test_single_owner_prefill_every_lane_engine_layout(mesh_device, reset_seeds, request, monkeypatch):
    """A 128-token single-user prefill owned by each lane in turn through the decode MoE kernel, in the engine's own
    orientation (TP over the rows, lanes over the columns). On the (2,4) block the full model's eager prefill passed for
    lane 0 and hung on lane 1 (2026-10-24 devq 0593: fetch-queue timeout after lane 1's call was enqueued); this is the
    one-layer version of that sequence."""
    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", "1")
    if tuple(mesh_device.shape) == (2, 4):
        monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", "128")
    rows = 128  # one user's 128-token prompt on the owner lane
    hf_config, hf_experts, mesh_config, experts = _build_engine(mesh_device)
    lanes = mesh_config.lanes
    hidden, routing, idx, wts = _inputs(hf_config, lanes, rows, seed=11)
    for owner in range(lanes):
        mesh_config._g4_single_lane_prefill_owner = owner
        per_lane = _run(experts, mesh_device, mesh_config, hidden, routing)
        ref = _ref(hf_experts, hidden, idx, wts, owner, rows)
        for lane in range(lanes):
            # the decode MoE kernel's bar (bfloat4_b experts; test_experts_decode_kernel passes at ~0.98)
            passing, msg = compare_tensors(per_lane[lane][:rows], ref, pcc_threshold=0.97)
            assert passing, f"engine layout owner {owner} seen from lane {lane}: {msg}"
    mesh_config._g4_single_lane_prefill_owner = None


# ===== merged from tests/unit/test_experts_local_routed.py: Local token-sorted routed-expert prefill (GEMMA4_MOE_LOCAL_ROUTED_PREFILL=1, tt/experts/local_routed_prefill.py)
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


def _build_experts_local_routed(mesh_device):
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
    tt_experts, hf_experts, hf_config, state_dict = _build_experts_local_routed(mesh_device)
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
    tt_experts, _, hf_config, _ = _build_experts_local_routed(mesh_device)
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


# ===== merged from tests/unit/test_experts_small_rows.py: The small-row expert path (GEMMA4_SMALL_ROW_EXPERTS=1: custom_ops/expert_gather_small + the GeGLU op + W @ out +
def _pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def _build_experts_small_rows(mesh_device, state_dict, hf_config):
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


@skip_if_not_moe
@pytest.mark.skipif(not is_blackhole(), reason="token-dispatch prefill runs the Blackhole-only routed-expert op")
@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1)])
@pytest.mark.parametrize("seq_len", [128, 256, 1024], ids=lambda s: f"seq{s}")
def test_experts_small_rows_vs_dispatch(seq_len, mesh_device, reset_seeds, request, monkeypatch):
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_FIT_CHUNK", "1")
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "256" if seq_len == 256 else "128")
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
        experts = _build_experts_small_rows(mesh_device, state_dict, hf_config)
        plans = experts.dispatch_experts._plans
        o = experts(x_tt, r_tt)
        outs[name] = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).reshape(-1, H).float()[:seq_len]
        # the plan the call used: the smallest chunk that holds the call (the full-chunk plan always exists)
        plan = min((q for q in plans.values() if q.chunk_tokens >= seq_len), key=lambda q: q.chunk_tokens)
        took_small = plan.gather is not None
        assert took_small == (flag == "1" and seq_len <= 256), (name, seq_len, took_small)
        if took_small:
            assert plan.chunk_tokens == seq_len, (plan.chunk_tokens, seq_len)  # the path's own tile-aligned plan
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


@skip_if_not_moe
@pytest.mark.skipif(not is_blackhole(), reason="the small-row path is a Blackhole (4,1) feature")
@parametrize_mesh_with_fabric(mesh_shapes=[(4, 1)])
@pytest.mark.parametrize("tokens", [32, 64, 128], ids=lambda t: f"tok{t}")  # the experts take tile-aligned rows
def test_experts_small_rows_eager_timing(tokens, mesh_device, reset_seeds, request, monkeypatch):
    """Eager per-call wall of one layer's experts at a few tokens, dispatch path vs the small-row path (the served
    cached-prefix continuations run eagerly per user; the traced replay cannot see this). Timing only: prints one
    RESULT-style line per path; the small path is forced for every size (min bound 1). devq 0465 (4,1): dispatch /
    small-row ms per call = 2.17 / 0.90 at 32 tokens (the dispatch number is the masked-dense path below the
    dispatch minimum), 2.08 / 1.18 at 64, 3.27 / 1.45 at 128."""
    import time

    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_MIN_TOKENS", "128")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_FIT_CHUNK", "1")
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MIN_TOKENS", "1")
    monkeypatch.setenv("GEMMA4_SMALL_ROW_MAX_TOKENS", "128")
    hf_text_config = TestFactory.create_hf_text_config()
    hf_experts = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0).experts
    state_dict = {"gate_up_proj": hf_experts.gate_up_proj.data.clone(), "down_proj": hf_experts.down_proj.data.clone()}
    hf_config = TestFactory.create_hf_config()
    E, K, H = hf_config.num_experts, hf_config.top_k_experts, hf_config.hidden_size
    x = torch.randn(1, 1, tokens, H, dtype=torch.bfloat16)
    routing = torch.zeros(1, 1, tokens, E)
    for s_ in range(tokens):
        sel = torch.randperm(E)[:K]
        w = torch.rand(K) + 0.1
        routing[0, 0, s_, sel] = w / w.sum()
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    x_tt = ttnn.from_torch(x, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep)
    r_tt = ttnn.from_torch(
        routing.to(torch.bfloat16), device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=rep
    )
    iters = 10
    for name, flag in (("dispatch", "0"), ("small_rows", "1")):
        monkeypatch.setenv("GEMMA4_SMALL_ROW_EXPERTS", flag)
        experts = _build_experts_small_rows(mesh_device, state_dict, hf_config)
        for _ in range(2):  # compile + program cache
            ttnn.deallocate(experts(x_tt, r_tt))
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(iters):
            ttnn.deallocate(experts(x_tt, r_tt))
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / iters * 1e3
        plan = min(
            (q for q in experts.dispatch_experts._plans.values() if q.chunk_tokens >= tokens),
            key=lambda q: q.chunk_tokens,
        )
        print(
            f"[eager tok{tokens} {name}] {ms:.2f} ms per call (chunk {plan.chunk_tokens}, small_path={plan.gather is not None})"
        )
        del experts


# ===== merged from tests/unit/test_dispatch_padding.py: Padding-aware token-dispatch prefill (GEMMA4_DISPATCH_PADDING): a chunk with fewer real rows than
@pytest.mark.skipif(os.environ.get("HF_MODEL", "google/gemma-4-26B-A4B-it").find("26B") < 0, reason="MoE only")
@parametrize_mesh_with_fabric(
    mesh_shapes=[(8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)
@pytest.mark.parametrize("n_tokens", [128, 2065], ids=["n128", "n2065"])
def test_dispatch_padding_matches_full_chunk(mesh_device, n_tokens, reset_seeds, request, monkeypatch):
    skip_if_config_only_checkpoint()
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1")
    monkeypatch.setenv("GEMMA4_MOE_DISPATCH_PREFILL", "1")
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tests.test_factory import TestFactory
    from models.demos.gemma4.tt.ccl import CCLManager
    from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

    # layer-0 routed experts on the real checkpoint, the Galaxy one-instance orientation (as test_experts.py)
    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    state_dict = {
        "gate_up_proj": hf_layer.experts.gate_up_proj.data.clone(),
        "down_proj": hf_layer.experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    tt_experts = Gemma4Experts(
        mesh_device=mesh_device,
        config=Gemma4ExpertConfig(hf_config),
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=1),
        mesh_config=mesh_config,
        program_config=None,
        weight_dtype=ttnn.bfloat8_b,
    )
    experts = tt_experts.dispatch_experts
    assert experts is not None, "GEMMA4_MOE_DISPATCH_PREFILL did not build the dispatch experts"
    H, E = experts.hidden, experts.num_experts
    torch.manual_seed(0)
    hidden = torch.randn(1, 1, n_tokens, H, dtype=torch.bfloat16) * 0.5
    routing = torch.zeros(1, 1, n_tokens, E, dtype=torch.bfloat16)
    top = torch.rand(n_tokens, E).topk(experts.top_k, dim=-1)
    routing[0, 0].scatter_(1, top.indices, torch.softmax(top.values, dim=-1).to(torch.bfloat16))

    def run(padding):
        experts.PADDING_ENABLED = padding
        h = ttnn.from_torch(
            hidden,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        r = ttnn.from_torch(
            routing,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        experts.forward(h, r)  # warm (program cache)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        out = experts.forward(h, r)
        ttnn.synchronize_device(mesh_device)
        dt = time.perf_counter() - t0
        host = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0:1].float()
        return host[0, 0, :n_tokens], dt

    full, t_full = run(False)
    padded, t_pad = run(True)
    assert torch.isfinite(padded).all()
    pcc = torch.corrcoef(torch.stack([full.flatten(), padded.flatten()]))[0, 1].item()
    print(
        f"[n_tokens={n_tokens}] pcc(full, padded)={pcc:.6f} max|diff|={(full - padded).abs().max():.4f} "
        f"time full={t_full*1e3:.1f} ms padded={t_pad*1e3:.1f} ms"
    )
    assert pcc > 0.999, f"padded dispatch diverges from the full chunk: pcc={pcc}"
    if n_tokens <= 128:
        assert t_pad < 0.8 * t_full, f"short chunk did not get cheaper: {t_pad*1e3:.1f} vs {t_full*1e3:.1f} ms"
