# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
import torch
import ttnn
from conftest import requires_hybrid_allocator


def assert_tensor_linked_to_its_allocation(graph):
    """Checks that a captured device tensor is linked to the buffer node its allocation was recorded under."""
    # Allocation events connect to the buffer node they were recorded under.
    allocated_buffers = {
        connection for node in graph if node["node_type"] == "buffer_allocate" for connection in node["connections"]
    }
    assert allocated_buffers, "the tensor's allocation was not captured"

    # A buffer node connects to the tensors it holds.
    device_tensors = {
        node["counter"]
        for node in graph
        if node["node_type"] == "tensor" and node["params"].get("storage_type") == "DEVICE"
    }
    assert device_tensors, "the tensor was not captured"
    linked = [
        node
        for node in graph
        if node["counter"] in allocated_buffers and device_tensors.intersection(node["connections"])
    ]
    assert linked, "the tensor is not linked to the buffer node of its allocation"


@requires_hybrid_allocator
def test_graph_capture_links_per_core_tensor_to_its_allocation(device):
    """A per-core allocated tensor has no backing buffer. Graph capture must still link it to the buffer node its
    allocation was recorded under."""
    core = ttnn.CoreCoord(0, 0)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, 64], ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    mem_config.experimental_set_per_core_allocation(True)
    data = torch.zeros(1, 64, dtype=torch.uint8)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tensor = ttnn.from_torch(
        data, dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mem_config
    )
    # Passing the tensor to an operation records it as a tensor node.
    ttnn.from_device(tensor)
    graph = ttnn.graph.end_graph_capture()

    assert tensor.buffer_unique_id() is not None

    assert_tensor_linked_to_its_allocation(graph)


@requires_hybrid_allocator
def test_graph_capture_reports_each_device_of_per_core_tensor(mesh_device):
    """Each device places a per-core allocation itself, so graph capture must report every device's tensor under its
    own device ID, and give the tensor no single address."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires at least 2 devices")
    core = ttnn.CoreCoord(0, 0)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, 64], ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    mem_config.experimental_set_per_core_allocation(True)
    data = torch.zeros(1, 64, dtype=torch.uint8)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tensor = ttnn.from_torch(
        data,
        dtype=ttnn.uint8,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=mem_config,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    ttnn.from_device(tensor)
    graph = ttnn.graph.end_graph_capture()

    tensor_nodes = [
        node
        for node in graph
        if node["node_type"] == "tensor"
        and node["params"].get("storage_type") == "DEVICE"
        and "device_tensors" in node["params"]
    ]
    assert tensor_nodes, "the tensor was not captured"
    for node in tensor_nodes:
        device_tensors = json.loads(node["params"]["device_tensors"])
        assert len(device_tensors) == mesh_device.get_num_devices()
        assert len({dt["mesh_device_id"] for dt in device_tensors}) == len(device_tensors)
        assert all(dt["mesh_device_id"] == dt["device_id"] for dt in device_tensors)
        assert "address" not in node["params"]


@requires_hybrid_allocator
def test_graph_capture_links_single_device_tensor_to_its_allocation(mesh_device):
    """A tensor allocated on one device of a mesh holds its allocation there only. Graph capture must link it to that
    allocation even when the device isn't the mesh's first."""
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Requires at least 2 devices")
    core = ttnn.CoreCoord(0, 0)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]), [1, 64], ttnn.ShardOrientation.ROW_MAJOR
    )
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)
    mem_config.experimental_set_per_core_allocation(True)
    host_tensor = ttnn.from_torch(torch.zeros(1, 64, dtype=torch.uint8), dtype=ttnn.uint8, layout=ttnn.ROW_MAJOR_LAYOUT)

    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    tensor = ttnn._ttnn.tensor.experimental_to_single_device(
        host_tensor, mesh_device, ttnn.MeshCoordinate(0, 1), mem_config
    )
    ttnn.from_device(tensor)
    graph = ttnn.graph.end_graph_capture()

    assert tensor.buffer_unique_id() is not None
    assert_tensor_linked_to_its_allocation(graph)
