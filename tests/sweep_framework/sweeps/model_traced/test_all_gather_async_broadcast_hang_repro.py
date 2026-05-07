# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Faithful reproduction of sweep vector 1f53dcabb0819464 (all_gather_async)
that hangs intermittently in CI lead models sweep.

This test mirrors the exact execution path in
all_gather_async_model_traced.py::run() for the model_traced suite,
including replicate_with_topology, device_context, and op kwargs.
"""

import os
import pytest
import torch
import ttnn

from tests.sweep_framework.sweep_utils.ccl_common import device_context
from tests.sweep_framework.sweep_utils.mesh_tensor_utils import replicate_with_topology

MESH_SHAPE = (4, 8)
INPUT_SHAPE = (1, 1, 4, 16384)
DIM = 2
CLUSTER_AXIS = 1
NUM_LINKS = 4
TOPOLOGY = ttnn.Topology.Linear
LAYOUT = ttnn.Layout.ROW_MAJOR
INPUT_DTYPE = ttnn.DataType.BFLOAT16
USE_BROADCAST = True
FABRIC_CONFIG = ttnn.FabricConfig.FABRIC_2D

INPUT_MEMORY_CONFIG = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)
OUTPUT_MEMORY_CONFIG = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.L1)

INPUT_A_TENSOR_PLACEMENT = {
    "distribution_shape": "[4, 8]",
    "mesh_device_shape": "[4, 8]",
    "placement": "['PlacementShard(1)', 'PlacementShard(-1)']",
}


@pytest.fixture(scope="function")
def mesh_device():
    prev_timeout = os.environ.get("TT_METAL_OPERATION_TIMEOUT_SECONDS")
    os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = "30"
    try:
        with device_context(MESH_SHAPE, FABRIC_CONFIG) as (device, device_err):
            assert device_err is None, f"Device setup error: {device_err}"
            assert tuple(device.shape) == MESH_SHAPE, f"Expected {MESH_SHAPE}, got {device.shape}"
            yield device
    finally:
        if prev_timeout is not None:
            os.environ["TT_METAL_OPERATION_TIMEOUT_SECONDS"] = prev_timeout
        else:
            os.environ.pop("TT_METAL_OPERATION_TIMEOUT_SECONDS", None)


def run_all_gather_async(mesh_device):
    global_shape = list(INPUT_SHAPE)
    torch_global = torch.rand(global_shape).bfloat16()

    cluster_size = MESH_SHAPE[CLUSTER_AXIS]
    repeats = [1] * len(global_shape)
    repeats[DIM] = cluster_size
    torch_reference = torch_global.repeat(*repeats)
    torch_input = torch_global

    tt_input = replicate_with_topology(
        torch_input,
        mesh_device,
        INPUT_DTYPE,
        LAYOUT,
        INPUT_MEMORY_CONFIG,
        INPUT_A_TENSOR_PLACEMENT,
    )

    compute_grid_size = mesh_device.compute_with_storage_grid_size()
    ccl_sub_device_crs = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(compute_grid_size.x - 1, compute_grid_size.y - 1))}
    )
    worker_sub_device_id = ttnn.SubDeviceId(0)
    sub_device_stall_group = [worker_sub_device_id]

    mesh_device.set_sub_device_stall_group(sub_device_stall_group)

    ccl_semaphore_handles = [ttnn.create_global_semaphore(mesh_device, ccl_sub_device_crs, 0) for _ in range(2)]

    op_kwargs = {
        "dim": DIM,
        "multi_device_global_semaphore": ccl_semaphore_handles,
        "num_links": NUM_LINKS,
        "topology": TOPOLOGY,
        "cluster_axis": CLUSTER_AXIS,
        "use_broadcast": USE_BROADCAST,
    }

    tt_out_tensor = ttnn.experimental.all_gather_async(tt_input, **op_kwargs)
    ttnn.synchronize_device(mesh_device, sub_device_ids=sub_device_stall_group)

    mesh_device.reset_sub_device_stall_group()

    device_tensors = ttnn.get_device_tensors(tt_out_tensor)
    tt_output_tensor = ttnn.to_torch(device_tensors[0])

    tt_output_tensor = tt_output_tensor[tuple(slice(0, s) for s in torch_reference.shape)]

    assert tt_output_tensor.shape == torch_reference.shape, (
        f"Shape mismatch: got {tt_output_tensor.shape}, expected {torch_reference.shape}"
    )

    if not torch.allclose(tt_output_tensor, torch_reference, atol=0.1, rtol=0.1):
        pcc = torch.corrcoef(torch.stack([tt_output_tensor.flatten().float(), torch_reference.flatten().float()]))[0, 1]
        assert pcc > 0.99, f"PCC too low: {pcc}"


@pytest.mark.parametrize("iteration", range(10))
def test_all_gather_async_broadcast_hang(mesh_device, iteration):
    """Run the exact sweep config 10 times to catch intermittent hangs."""
    run_all_gather_async(mesh_device)
