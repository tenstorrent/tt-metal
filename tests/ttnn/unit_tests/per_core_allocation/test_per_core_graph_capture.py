# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from conftest import requires_hybrid_allocator


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
