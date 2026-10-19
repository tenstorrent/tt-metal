# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""SharedMLP with lane-sharded activations: each lane holds different rows, the forward gathers them,
runs the fractured GeGLU and reduce-scatters back, so every chip must end with exactly its lane's rows."""

import torch

import ttnn
from models.demos.gemma4.tt.shared_mlp import SharedMLP

from ...tests.test_factory import TestFactory, compare_tensors, get_pcc_threshold, parametrize_mesh_with_fabric


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
