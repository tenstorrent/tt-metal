# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Runnable examples for the CCL operations. The body of each test is injected verbatim
# into the corresponding op's rendered docstring by docs/source/ttnn/_ext/doc_modifier.py
# (keyed through tests/ttnn/docs_examples/examples_mapping.py), so keep the bodies clean
# and illustrative.
#
# The tests take the root conftest `mesh_device` fixture (with `device_params` for fabric).
# Using the fixture means the device is only opened when pytest runs the test on hardware;
# the docs build merely imports this module to extract the bodies, so it never touches a
# device. The mesh_device fixture also skips cleanly when the requested mesh is larger than
# the available devices.
#
# NOTE: `device_params` must be a module-level constant (not an inline dict in the
# decorator). doc_modifier's body extractor mis-parses a decorator line that contains a
# top-level `:` (as an inline `{"fabric_config": ...}` dict would), which corrupts the
# rendered example. The decorators themselves are not rendered into the docs.

import pytest
import torch
import ttnn

FABRIC_1D = [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}]
FABRIC_2D = [{"fabric_config": ttnn.FabricConfig.FABRIC_2D}]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_gather(mesh_device):
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Gather each device's shard along dim 0: output[dim] = input[dim] * num_devices.
    output = ttnn.all_gather(tt_input, dim=0)
    print(output.shape)  # [2, 1, 32, 256]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_broadcast(mesh_device):
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Returns a list of num_devices tensors, each identical and of the input shape.
    outputs = ttnn.all_broadcast(tt_input)
    print(len(outputs), outputs[0].shape)  # 2 [1, 1, 32, 256]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_broadcast(mesh_device):
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Copy the data on the sender coordinate to every device along the cluster axis.
    output = ttnn.broadcast(tt_input, ttnn.MeshCoordinate(0, 0), cluster_axis=1)
    print(output.shape)  # [1, 1, 32, 256]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_all_reduce(mesh_device):
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Sum-reduce across devices; result is replicated and shape is preserved.
    output = ttnn.all_reduce(tt_input, cluster_axis=1)
    print(output.shape)  # [1, 1, 32, 256]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_reduce_scatter(mesh_device):
    # Each device holds a full-width slice; the scatter dim must be divisible by num_devices.
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Sum-reduce across devices then scatter along dim 3: output[dim] = input[dim] / num_devices.
    output = ttnn.reduce_scatter(tt_input, dim=3, cluster_axis=1)
    print(output.shape)  # [1, 1, 32, 128]


@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_mesh_partition(mesh_device):
    # mesh_partition is a per-device slice and does not use the fabric (no device_params needed).
    # The partition dim must be evenly divisible by the cluster-axis device count.
    torch_input = torch.randn([1, 4, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Keep this device's 1/num_devices slice along dim 1: output[dim] = input[dim] / num_devices.
    output = ttnn.mesh_partition(tt_input, dim=1, cluster_axis=1)
    print(output.shape)  # [1, 2, 32, 256]


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(1, 2)], indirect=True)
def test_point_to_point(mesh_device):
    torch_input = torch.randn([1, 1, 32, 256], dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Send the shard on the sender coordinate to the receiver coordinate (same row/column).
    sender_coord = ttnn.MeshCoordinate(0, 0)
    receiver_coord = ttnn.MeshCoordinate(0, 1)
    output = ttnn.point_to_point(tt_input, sender_coord, receiver_coord, topology=ttnn.Topology.Linear)
    print(output.shape)  # same spec as the input


@pytest.mark.parametrize("device_params", FABRIC_1D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_reduce_to_root(mesh_device):
    # SDPA tree-reduction state tensors (values / running-sum / running-max), sharded per device.
    l = ttnn.from_torch(
        torch.randn((8, 128), dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    s = ttnn.from_torch(
        torch.randn((8, 32), dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    m = ttnn.from_torch(
        torch.randn((8, 32), dtype=torch.bfloat16),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )

    # Reduce the three states along the line to the root coordinate; outputs match the input specs.
    root_coord = ttnn.MeshCoordinate(1, 0)
    out_l, out_s, out_m = ttnn.reduce_to_root(l, s, m, root_coord, scale_fp32=1.0, topology=ttnn.Topology.Linear)
    print(out_l.shape, out_s.shape, out_m.shape)  # (8, 128) (8, 32) (8, 32)


@pytest.mark.parametrize("device_params", FABRIC_2D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_all_to_all_dispatch(mesh_device):
    devices = 8
    experts = 8  # must be divisible by the number of devices
    select_experts_k = 2
    hidden_size = 128
    batch, seq = 8, 2

    # Tokens: [batch, 1, seq, hidden], bfloat16, row-major, sharded along the cluster axis.
    tokens = ttnn.from_torch(
        torch.randn([batch, 1, seq, hidden_size], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(None, 0)),
    )
    # Expert indices: [batch, 1, seq, k], uint16, sharded like the tokens.
    expert_indices = ttnn.from_torch(
        torch.randint(0, experts, [batch, 1, seq, select_experts_k], dtype=torch.int16),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(None, 0)),
    )
    # Expert-to-device mapping: [1, 1, experts, devices], uint16, fully replicated across the mesh.
    mapping_torch = torch.zeros([1, 1, experts, devices], dtype=torch.int16)
    for e in range(experts):
        mapping_torch[0, 0, e, e % devices] = 1
    expert_mapping = ttnn.from_torch(
        mapping_torch,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(None, None)),
    )

    output_tokens, output_metadata = ttnn.all_to_all_dispatch(
        tokens, expert_indices, expert_mapping, cluster_axis=1, num_links=1
    )
    print(output_tokens.shape, output_metadata.shape)


@pytest.mark.parametrize("device_params", FABRIC_2D, indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_all_to_all_combine(mesh_device):
    devices = 8
    experts = 8  # must be divisible by the number of devices
    select_experts_k = 2
    hidden_size = 128
    batch, seq = 8, 2

    # Expert contributions to combine: [experts, batch, seq, hidden], bfloat16, row-major,
    # sharded on dim 0 so each device holds its experts // devices slice.
    contributions = ttnn.from_torch(
        torch.randn([experts, batch, seq, hidden_size], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    # Expert metadata (indices per token), replicated per device: [devices, batch, seq, k],
    # uint16, sharded on dim 0 so each device receives a full [1, batch, seq, k] copy.
    expert_metadata = ttnn.from_torch(
        torch.randint(0, experts, [devices, batch, seq, select_experts_k], dtype=torch.int16),
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    # Expert-to-device mapping: [1, 1, experts, devices], uint16, fully replicated.
    mapping_torch = torch.zeros([1, 1, experts, devices], dtype=torch.int16)
    for e in range(experts):
        mapping_torch[0, 0, e, e % devices] = 1
    expert_mapping = ttnn.from_torch(
        mapping_torch,
        dtype=ttnn.uint16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 4), dims=(None, None)),
    )

    # cluster_axis is required for all_to_all_combine.
    output = ttnn.all_to_all_combine(contributions, expert_metadata, expert_mapping, cluster_axis=1, num_links=1)
    print(output.shape)
