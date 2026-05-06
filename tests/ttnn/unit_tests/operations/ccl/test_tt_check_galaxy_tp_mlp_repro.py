# SPDX-FileCopyrightText: 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn


def _num_devices() -> int:
    for name in ("get_num_devices", "GetNumAvailableDevices"):
        fn = getattr(ttnn, name, None)
        if fn is not None:
            return int(fn())
    return 1


def _ttnn_mlp_forward(x, w1, w3, w2):
    w1_out = ttnn.linear(x, w1, dtype=ttnn.bfloat16)
    w3_out = ttnn.linear(x, w3, dtype=ttnn.bfloat16)
    hidden = ttnn.mul(ttnn.silu(w1_out), w3_out, dtype=ttnn.bfloat16)
    partial_output = ttnn.linear(hidden, w2, dtype=ttnn.bfloat16)
    return ttnn.all_reduce(partial_output, cluster_axis=1, topology=ttnn.Topology.Linear)


@pytest.mark.parametrize("rows", [1024])
def test_tt_check_tensor_parallel_prefill_mlp_repro(rows):
    device_count = _num_devices()
    if device_count < 2:
        pytest.skip("requires a multi-device mesh")

    activation_width = 1024
    per_device_intermediate_width = 1024
    global_intermediate_width = per_device_intermediate_width * device_count

    torch.manual_seed(0)
    x = (torch.randn((1, 1, rows, activation_width), dtype=torch.float32) * 0.5).to(torch.bfloat16)
    w1 = torch.randn((activation_width, global_intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w3 = torch.randn((activation_width, global_intermediate_width), dtype=torch.float32) / math.sqrt(activation_width)
    w2 = torch.randn((global_intermediate_width, activation_width), dtype=torch.float32) / math.sqrt(
        global_intermediate_width
    )

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, device_count), trace_region_size=0)
    try:
        tt_x = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        tt_w1 = ttnn.from_torch(
            w1,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )
        tt_w3 = ttnn.from_torch(
            w3,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1),
        )
        tt_w2 = ttnn.from_torch(
            w2,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0),
        )

        output = _ttnn_mlp_forward(tt_x, tt_w1, tt_w3, tt_w2)
        shards = [ttnn.to_torch(device_tensor) for device_tensor in ttnn.get_device_tensors(output)]
        assert len(shards) == device_count
        assert all(torch.equal(shards[0], shard) for shard in shards[1:])
    finally:
        ttnn.close_mesh_device(device)
