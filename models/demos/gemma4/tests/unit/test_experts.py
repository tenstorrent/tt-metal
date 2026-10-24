# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 routed experts — on-device via sparse_matmul.

    pytest -k "1x1"        # single card (reduced experts)
    pytest -k "1x8"        # T3K with TP-sharded experts + CCL
    pytest -k "decode"     # decode only
    pytest -k "prefill"    # prefill only
"""

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.ccl import CCLManager
from models.demos.gemma4.tt.experts import Gemma4ExpertConfig, Gemma4Experts

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
