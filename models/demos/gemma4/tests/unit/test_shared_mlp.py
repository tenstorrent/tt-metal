# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Gemma4 SharedMLP — uses HF Gemma4TextMLP as reference.

    pytest -k "1x1"   # single card
    pytest -k "1x8"   # T3K with CCL all-reduce
"""

from types import SimpleNamespace

import torch

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP, resolve_shared_mlp_intermediate_size

from ...tests.test_factory import (
    TestFactory,
    compare_tensors,
    get_pcc_threshold,
    parametrize_batch_seq,
    parametrize_mesh_with_fabric,
)


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (8, 4)])
@parametrize_batch_seq()
def test_shared_mlp(batch_size, seq_len, mesh_device, reset_seeds, request):
    """Test SharedMLP against HF Gemma4TextMLP (GeGLU)."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    # (8,4): the Galaxy one-instance orientation (TP on the size-8 axis 0, fractured dense weights).
    galaxy = hasattr(mesh_device, "shape") and tuple(mesh_device.shape) == (8, 4)
    tp = (mesh_device.shape[0] if galaxy else mesh_device.shape[1]) if hasattr(mesh_device, "shape") else 1
    if galaxy:
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp), tp_axis=0, weight_fracture=True)
    else:
        mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=tp))
    ccl_manager = CCLManager(mesh_device, num_links=1) if tp > 1 else None

    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=ccl_manager,
        dtype=ttnn.bfloat16,
    )

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).float()).unsqueeze(0).to(torch.bfloat16)

    is_mesh = hasattr(mesh_device, "shape") and mesh_device.get_num_devices() > 1
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_output = tt_mlp(x_tt)
    tt_output_torch = ttnn.to_torch(ttnn.get_device_tensors(tt_output)[0]) if is_mesh else ttnn.to_torch(tt_output)

    passing, pcc_msg = compare_tensors(tt_output_torch, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"SharedMLP (tp={tp}) PCC too low: {pcc_msg}"


# ===== merged from tests/unit/test_shared_mlp_fracture.py: SharedMLP under 2D weight fracture vs HF GeGLU: one weight copy with the inter dim sharded over all 32
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@parametrize_batch_seq()
def test_shared_mlp_fracture(batch_size, seq_len, mesh_device, reset_seeds, request):
    """One weight copy over 32 chips; output PCC vs HF Gemma4TextMLP."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    assert mesh_config.fracture_ways == rows * cols

    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=None,  # fractured all-reduce uses the sync per-axis path
        dtype=ttnn.bfloat16,
    )
    # One weight copy across the mesh: per-chip GeGLU shard is inter/ways.
    inter = tt_mlp.intermediate_size
    assert tt_mlp._inter_per_device * mesh_config.fracture_ways >= inter
    assert tt_mlp._inter_per_device < inter // rows  # strictly smaller than plain TP=rows

    x_torch = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).float()).unsqueeze(0).to(torch.bfloat16)

    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_output = tt_mlp(x_tt)

    # Output must be identical (replicated) on every chip; check the two mesh
    # corners plus PCC vs HF on chip 0.
    device_tensors = ttnn.get_device_tensors(tt_output)
    out0 = ttnn.to_torch(device_tensors[0])
    out_last = ttnn.to_torch(device_tensors[-1])
    assert torch.equal(out0, out_last), "fractured MLP output not replicated across the mesh"

    passing, pcc_msg = compare_tensors(out0, ref_output, pcc_threshold=get_pcc_threshold(request))
    assert passing, f"SharedMLP fractured ({rows}x{cols}) PCC too low: {pcc_msg}"


# ===== merged from tests/unit/test_shared_mlp_lanes.py: SharedMLP with lane-sharded activations: each lane holds different rows, the forward gathers them,
@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
def test_shared_mlp_lanes(mesh_device, reset_seeds, request):
    from models.demos.gemma4.config import MeshConfig, ModeConfig

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp

    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }

    hf_config = TestFactory.create_hf_config()
    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(
        mesh_device.shape,
        decode=ModeConfig(tp=rows),
        tp_axis=0,
        weight_fracture=True,
    )
    mesh_config.lane_sharded = True
    assert mesh_config.lanes == cols

    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=None,  # lane choreography uses the sync per-axis path
        dtype=ttnn.bfloat16,
    )

    rows_per_lane = 32  # one tile of rows per lane
    total_rows = rows_per_lane * cols
    x_torch = torch.randn(1, 1, total_rows, hf_config.hidden_size, dtype=torch.bfloat16)

    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).squeeze(0).float()).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)

    # Rows sharded across lanes (axis 1), replicated within a column (axis 0).
    x_tt = ttnn.from_torch(
        x_torch,
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 2)),
    )
    tt_output = tt_mlp(x_tt)

    device_tensors = ttnn.get_device_tensors(tt_output)
    assert len(device_tensors) == rows * cols
    threshold = get_pcc_threshold(request)

    # Device index is row-major over (rows, cols): chip (r, c) = r*cols + c.
    for c in range(cols):
        lane_ref = ref_output[0, :, c * rows_per_lane : (c + 1) * rows_per_lane, :]
        out_top = ttnn.to_torch(device_tensors[c])  # (0, c)
        out_bot = ttnn.to_torch(device_tensors[(rows - 1) * cols + c])  # (rows-1, c)
        assert torch.equal(out_top, out_bot), f"lane {c} output differs within its column"
        passing, pcc_msg = compare_tensors(out_top.squeeze(0).float(), lane_ref.float(), pcc_threshold=threshold)
        assert passing, f"lane {c} PCC too low: {pcc_msg}"


@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
def test_shared_mlp_lanes_single_owner(mesh_device, reset_seeds, request):
    """Single-user prefill fast path: only the owner lane's rows are computed (1x instead of lanes x rows) and
    every lane ends up holding the owner's result after the lane + tp all-reduces."""
    from models.demos.gemma4.config import MeshConfig, ModeConfig
    from models.demos.gemma4.tt.ccl import CCLManager

    hf_text_config = TestFactory.create_hf_text_config()
    hf_layer = TestFactory.create_hf_reference_layer(hf_text_config, layer_idx=0)
    hf_mlp = hf_layer.mlp
    state_dict = {
        "gate_proj.weight": hf_mlp.gate_proj.weight.data.clone(),
        "up_proj.weight": hf_mlp.up_proj.weight.data.clone(),
        "down_proj.weight": hf_mlp.down_proj.weight.data.clone(),
    }
    hf_config = TestFactory.create_hf_config()
    rows, cols = tuple(mesh_device.shape)
    mesh_config = MeshConfig(mesh_device.shape, decode=ModeConfig(tp=rows), tp_axis=0, weight_fracture=True)
    mesh_config.lane_sharded = True
    tt_mlp = SharedMLP(
        mesh_device=mesh_device,
        hf_config=hf_config,
        state_dict=state_dict,
        mesh_config=mesh_config,
        ccl_manager=CCLManager(mesh_device, num_links=1),
        dtype=ttnn.bfloat16,
    )
    rows_per_lane = 64  # above the decode-frame threshold of the fast path
    total_rows = rows_per_lane * cols
    x_torch = torch.randn(1, 1, total_rows, hf_config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        ref_output = hf_mlp(x_torch.squeeze(0).squeeze(0).float()).unsqueeze(0).unsqueeze(0).to(torch.bfloat16)
    threshold = get_pcc_threshold(request)
    for owner in range(cols):
        mesh_config._g4_single_lane_prefill_owner = owner
        x_tt = ttnn.from_torch(
            x_torch,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_device.shape, dims=(None, 2)),
        )
        tt_output = tt_mlp(x_tt)
        device_tensors = ttnn.get_device_tensors(tt_output)
        owner_ref = ref_output[0, :, owner * rows_per_lane : (owner + 1) * rows_per_lane, :]
        for c in range(cols):
            out = ttnn.to_torch(device_tensors[c])  # chip (0, c)
            passing, pcc_msg = compare_tensors(out.squeeze(0).float(), owner_ref.float(), pcc_threshold=threshold)
            assert passing, f"owner {owner} seen from lane {c}: PCC too low: {pcc_msg}"
    mesh_config._g4_single_lane_prefill_owner = None


# ===== merged from tests/unit/test_shared_mlp_intermediate.py
def test_resolve_intermediate_from_weights():
    state = {"gate_proj.weight": torch.zeros(12288, 1536)}
    cfg = SimpleNamespace(
        intermediate_size=6144, use_double_wide_mlp=True, num_kv_shared_layers=20, num_hidden_layers=35
    )
    assert resolve_shared_mlp_intermediate_size(cfg, state, layer_idx=15) == 12288


def test_resolve_double_wide_from_layer_idx():
    cfg = SimpleNamespace(
        intermediate_size=6144, use_double_wide_mlp=True, num_kv_shared_layers=20, num_hidden_layers=35
    )
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=14) == 6144
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=15) == 12288


def test_resolve_no_double_wide():
    cfg = SimpleNamespace(
        intermediate_size=10240, use_double_wide_mlp=False, num_kv_shared_layers=18, num_hidden_layers=42
    )
    assert resolve_shared_mlp_intermediate_size(cfg, None, layer_idx=24) == 10240
