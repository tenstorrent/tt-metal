# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 MoE block — fully on device.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K with TP-sharded experts + CCL
"""


import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.demos.gemma4.tt.moe import MoEBlock

from ...tests.test_factory import (
    PREFILL_BUCKETS,
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    lanes_fabric_config,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
    skip_if_not_moe,
)


@skip_if_not_moe
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (8, 4)])
@parametrize_batch_seq(configs=[(1, L) for L in PREFILL_BUCKETS])
def test_moe(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test MoE end-to-end on device against HF reference.

    Uses HF routing for the reference, TT router+experts for the test.
    Relaxed PCC: router may select different experts due to bf16 precision.
    """
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    num_experts = 8
    top_k = 4
    hf_text_config = TestFactory.create_hf_text_config(num_experts=num_experts, top_k=top_k)
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_router = hf_layer.router
    hf_experts = hf_layer.experts

    # Boost router proj σ so softmax produces peaked distributions. The
    # factory's default N(0, 0.02) gives expert_scores std ≈ 0.019, which
    # makes the post-softmax distribution near-uniform across experts; bf16
    # and fp32 then disagree on topk rankings purely due to mantissa
    # precision (a test artifact, not a routing bug). Real trained Gemma4
    # routing is peaked, so we mirror that here.
    with torch.no_grad():
        hf_router.proj.weight.normal_(0, 1.0)

    state_dict = {
        "router.scale": hf_router.scale.data.clone(),
        "router.proj.weight": hf_router.proj.weight.data.clone(),
        "router.per_expert_scale": hf_router.per_expert_scale.data.clone(),
        "experts.gate_up_proj": hf_experts.gate_up_proj.data.clone(),
        "experts.down_proj": hf_experts.down_proj.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    hf_config.num_experts = num_experts
    hf_config.top_k_experts = top_k

    # (8,4): the Galaxy one-instance orientation (TP on the size-8 axis 0, fractured dense weights).
    galaxy = hasattr(mesh_device, "shape") and tuple(mesh_device.shape) == (8, 4)
    tp = (mesh_device.shape[0] if galaxy else mesh_device.shape[1]) if hasattr(mesh_device, "shape") else 1
    if galaxy:
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=0, weight_fracture=True)
    else:
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    moe = MoEBlock(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # HF reference
    x_flat = x_torch.reshape(-1, hf_config.hidden_size).float()
    with torch.no_grad():
        _, ref_weights, ref_indices = hf_router(x_flat)
        ref_output = hf_experts(x_flat, ref_indices, ref_weights)

    # TT forward
    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = moe(x_tt, x_tt)
    tt_output_torch = (
        (ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output))
        .reshape(-1, hf_config.hidden_size)
        .float()[:seq_len]
    )

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"MoE (tp={tp}) PCC too low: {pcc_msg}"


@pytest.mark.skipif(not is_blackhole(), reason="the decode MoE kernel is Blackhole-only")
@parametrize_mesh_with_fabric(
    mesh_shapes=[(2, 4), (8, 4)],
    device_params_extra={"fabric_config": lanes_fabric_config(), "l1_small_size": 65536},
)
@pytest.mark.parametrize(
    "lanes",
    [
        pytest.param(
            False,
            id="tp",
            marks=pytest.mark.skip(reason="decode MoE kernel on the non-lane 8x4 frame hangs (not a served layout)"),
        ),
        pytest.param(True, id="lanes"),
    ],
)
@pytest.mark.parametrize("fused", [False, True], ids=["ttnn_router", "fused_router"])
@pytest.mark.parametrize("rows", [8, 128], ids=["rows8", "rows128"])
def test_moe_sparse_routing_matches_dense(mesh_device, lanes, fused, rows, reset_seeds, request, monkeypatch):
    """Decode rows through the decode MoE kernel: the router's sparse (values, indices) must give the same
    expert sum as the dense routing matrix the kernel used to re-top-k, on the TP layout and through the lane
    wrapper (which gathers the indices across lanes). ``fused_router`` = GEMMA4_MOE_FUSED=1: the packed router
    tail (custom_ops/router_topk.op_packed) + custom_ops/moe_frame_prep feed the kernel. The (2,4) block runs the
    lane engine's shape at frame 128 (TP=2 over the rows, 4 lanes; 64 tokens per row-chip)."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    monkeypatch.setenv("GEMMA4_MOE_DECODE_KERNEL", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_FRACTURE", "1")
    monkeypatch.setenv("GEMMA4_GALAXY_LANES", "1" if lanes else "0")
    monkeypatch.setenv("GEMMA4_MOE_FUSED", "0")
    if tuple(mesh_device.shape) == (2, 4):
        if rows == 128:
            pytest.skip("4 lanes x 128 rows exceed the block's frame (the 8x4 case covers the prefill-sized rows)")
        monkeypatch.setenv("GEMMA4_MOE_DECODE_FRAME", "128")  # 4 lanes x one row tile over 2 mesh rows
    hf_text_config = TestFactory.create_hf_text_config()  # real expert geometry: the kernel config is the 26B's
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    with torch.no_grad():
        hf_layer.router.proj.weight.normal_(0, 1.0)
    state_dict = {
        "router.scale": hf_layer.router.scale.data.clone(),
        "router.proj.weight": hf_layer.router.proj.weight.data.clone(),
        "router.per_expert_scale": hf_layer.router.per_expert_scale.data.clone(),
        "experts.gate_up_proj": hf_layer.experts.gate_up_proj.data.clone(),
        "experts.down_proj": hf_layer.experts.down_proj.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    hf_config.num_hidden_layers = 1  # the decode MoE kernel module builds once every declared layer registered
    tp, ep = mesh_device.shape[0], mesh_device.shape[1]
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=tp, ep=ep),
        prefill=ModeConfig(tp=tp, ep=ep),
        tp_axis=0,
        weight_fracture=True,
    )
    mesh_config.lane_sharded = lanes  # the lane wrapper keys on the config, not the env flag
    moe = MoEBlock(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        ccl_manager=CCLManager(mesh_device, num_links=1),
        mesh_config=mesh_config,
        dtype=ttnn.bfloat8_b,
    )
    # rows: 8 = decode-like, 128 = a short single-user prefill that the kernel also takes (per lane when lanes are on)
    x_torch = torch.randn(1, 1, rows, hf_config.hidden_size, dtype=torch.bfloat16)
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run(sparse):
        monkeypatch.setenv("GEMMA4_SPARSE_ROUTING", "1" if sparse else "0")
        monkeypatch.setenv("GEMMA4_MOE_FUSED", "1" if (sparse and fused) else "0")
        assert moe.experts.wants_sparse_routing(x_tt) == sparse
        out = moe(x_tt, x_tt)
        # The block's input is the caller's (the decoder layer frees expert_input after the MoE call): a lane gather
        # that consumed it freed this very tensor on the (2,4) at frame 128 and the second pass segfaulted in the
        # router's rms_norm.
        assert x_tt.is_allocated(), "the MoE block freed its input tensor"
        return ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(-1, hf_config.hidden_size).float()[:rows]

    # A device hang here is bounded by TT_METAL_OPERATION_TIMEOUT_SECONDS (also the per-device teardown wait);
    # run with a short value (10 s). Never skip the mesh close: it leaves the fabric up for the next process.
    dense = run(False)
    print("[sparse-test] dense run done", flush=True)
    sparse = run(True)
    print("[sparse-test] sparse run done", flush=True)
    assert torch.isfinite(sparse).all()
    pcc = torch.corrcoef(torch.stack([dense.flatten(), sparse.flatten()]))[0, 1].item()
    print(f"[lanes={lanes} fused={fused}] pcc(dense, sparse)={pcc:.6f} max|diff|={(dense - sparse).abs().max():.5f}")
    assert pcc > 0.999, f"sparse routing diverges from the dense path: pcc={pcc}"
