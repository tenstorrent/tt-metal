# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Runnable examples for the CCL operations. The body of each test is injected verbatim
# into the corresponding op's rendered docstring by docs/source/ttnn/_ext/doc_modifier.py
# (keyed through tests/ttnn/docs_examples/examples_mapping.py), so keep the bodies clean
# and illustrative. The skipif decorators are NOT rendered into the docs; they only let
# the docs_examples CI job skip gracefully on hosts without enough devices / fabric.

import pytest
import torch
import ttnn


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.all_gather requires at least 2 devices")
def test_all_gather():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.all_broadcast requires at least 2 devices")
def test_all_broadcast():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.broadcast requires at least 2 devices")
def test_broadcast():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.all_reduce requires at least 2 devices")
def test_all_reduce():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.reduce_scatter requires at least 2 devices")
def test_reduce_scatter():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.mesh_partition requires a cluster axis of size > 1")
def test_mesh_partition():
    # mesh_partition is a per-device slice and does not use the fabric.
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)


@pytest.mark.skipif(ttnn.get_num_devices() < 2, reason="ttnn.point_to_point requires at least 2 devices")
def test_point_to_point():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 4, reason="ttnn.reduce_to_root uses a fixed 4-device line topology")
def test_reduce_to_root():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 2))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ttnn.all_to_all_dispatch example uses an 8-device mesh")
def test_all_to_all_dispatch():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.mark.skipif(ttnn.get_num_devices() < 8, reason="ttnn.all_to_all_combine example uses an 8-device mesh")
def test_all_to_all_combine():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
    mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))

    devices = 8
    experts = 8  # must be divisible by the number of devices
    select_experts_k = 2
    hidden_size = 128
    batch, seq = 8, 2

    # Expert contributions to combine: [experts_per_device, batch, seq, hidden], bfloat16, row-major.
    contributions = ttnn.from_torch(
        torch.randn([experts // devices, batch, seq, hidden_size], dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    # Expert metadata (indices per token): [batch, seq, 1, k], uint16.
    expert_metadata = ttnn.from_torch(
        torch.randint(0, experts, [batch, seq, 1, select_experts_k], dtype=torch.int16),
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

    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
