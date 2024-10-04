# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
import os
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor, ListMeshToTensor


@pytest.mark.parametrize("shape", [(1, 1, 512, 512)])
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_multi_device_events(t3k_mesh_device, shape):
    if t3k_mesh_device.get_num_devices() <= 1:
        pytest.skip("This test requires multiple devices")

    # Enable Program Cache and Async Mode
    t3k_mesh_device.enable_async(True)
    for device_id in t3k_mesh_device.get_device_ids():
        t3k_mesh_device.get_device(device_id).enable_program_cache()

    # Preallocate activation tensors.
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, t3k_mesh_device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, t3k_mesh_device)

    # Send workload/ops on CQ 0 and Data on CQ 1. Use events for synchronization
    workload_cq = 0
    data_movement_cq = 1

    # Op chain to run on device, using the workload_cq
    def run_op_chain(input_0, input_1, workload_cq):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    for i in range(10):
        # Create events to synchronize data movement with workload execution.
        write_event = ttnn.create_event(t3k_mesh_device)
        workload_event = ttnn.create_event(t3k_mesh_device)
        # Create torch inputs, for validation
        torch_input_tensor_0 = torch.rand(
            (t3k_mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (t3k_mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        # Compute torch golden for validation
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        )
        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(t3k_mesh_device, dim=0)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors, using data-movement CQ
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev, cq_id=data_movement_cq)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev, cq_id=data_movement_cq)
        # Wait for write to be completed before issuing workload
        ttnn.record_event(data_movement_cq, write_event)
        ttnn.wait_for_event(workload_cq, write_event)
        logger.info("Execute Workload")
        # Execute workload
        ttnn_output = run_op_chain(input_0_dev, input_1_dev, workload_cq)
        # Wait for workload to be completed before issuing read
        ttnn.record_event(workload_cq, workload_event)
        ttnn.wait_for_event(data_movement_cq, workload_event)
        logger.info("Read Back Workload Outputs")
        # Read device outputs and validate
        ttnn_torch_output_tensor = ttnn.to_torch(
            ttnn_output,
            mesh_composer=ConcatMeshToTensor(t3k_mesh_device, dim=0),
            device=t3k_mesh_device,
            cq_id=data_movement_cq,
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.96)

    t3k_mesh_device.enable_async(False)
