# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
import os
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor

NUM_TRACE_LOOPS = int(os.getenv("NUM_TRACE_LOOPS", 15))


@pytest.mark.parametrize(
    "shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 32, 32), (1, 1, 256, 256), (1, 3, 512, 512), (1, 3, 128, 128)]
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("enable_multi_cq", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 60000, "num_command_queues": 2}], indirect=True)
def test_multi_device_single_trace(mesh_device, shape, enable_multi_cq):
    if mesh_device.get_num_devices() < 32:
        pytest.skip("Test is only valid on Galaxy")
    # Trace requires program cache to be enabled
    mesh_device.enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)

    # Op chains to be traced
    def run_op_chain(input_0, input_1):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    if enable_multi_cq:
        trace_cq = 0
        data_movement_cq = 1

        def event_sync(device, record_cq, wait_cq):
            event = ttnn.record_event(device, record_cq)
            ttnn.wait_for_event(wait_cq, event)

    else:
        trace_cq = 0
        data_movement_cq = 0

        def event_sync(device, record_cq, wait_cq):
            pass

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev)

    # Capture Trace
    logger.info("Capture Trace")

    tid = ttnn.begin_trace_capture(mesh_device, cq_id=trace_cq)
    output_tensor = run_op_chain(input_0_dev, input_1_dev)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=trace_cq)
    logger.info("Done Trace Capture")

    for i in range(NUM_TRACE_LOOPS):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(
            (mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        )
        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev, cq_id=data_movement_cq)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev, cq_id=data_movement_cq)
        event_sync(mesh_device, data_movement_cq, trace_cq)
        logger.info("Execute Trace")
        # Execute trace
        ttnn.execute_trace(mesh_device, tid, cq_id=trace_cq, blocking=False)
        event_sync(mesh_device, trace_cq, data_movement_cq)
        # Perform host All-Gather
        logger.info("Read Back Trace Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(
            output_tensor,
            mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
            device=mesh_device,
            cq_id=data_movement_cq,
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.96)

    # Release trace buffer once workload is complete
    ttnn.release_trace(mesh_device, tid)


@pytest.mark.parametrize(
    "shape",
    [(1, 1, 256, 256), (1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 32, 32)],
)
@pytest.mark.parametrize("mesh_device", [pytest.param((8, 4), id="8x4_grid")], indirect=True)
@pytest.mark.parametrize("enable_multi_cq", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000, "num_command_queues": 2}], indirect=True)
def test_multi_device_multi_trace(mesh_device, shape, enable_multi_cq):
    torch.manual_seed(0)
    if mesh_device.get_num_devices() < 32:
        pytest.skip("Test is only valid on Galaxy")

    # Trace requires program cache to be enabled
    mesh_device.enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)
    weight_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, mesh_device)

    # Op chains to be traced
    def run_op_chain(input_0, input_1, weight):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1))) @ ttnn.silu(
            weight
        )

    def run_op_chain_1(input_0, input_1, weight):
        return ttnn.tanh(ttnn.mul(ttnn.sub(input_0, input_1), weight)) @ ttnn.softmax(weight, dim=1)

    def run_op_chain_2(input_0, input_1, weight):
        return ttnn.neg(ttnn.mul(input_0, input_1)) @ ttnn.gelu(weight)

    if enable_multi_cq:
        trace_cq = 0
        data_movement_cq = 1

        def event_sync(device, record_cq, wait_cq):
            event = ttnn.record_event(device, record_cq)
            ttnn.wait_for_event(wait_cq, event)

    else:
        trace_cq = 0
        data_movement_cq = 0

        def event_sync(device, record_cq, wait_cq):
            pass

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev, weight_dev)
    run_op_chain_1(input_0_dev, input_1_dev, weight_dev)
    run_op_chain_2(input_0_dev, input_1_dev, weight_dev)

    # Capture Trace 0
    logger.info("Capture Trace 0")
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=trace_cq)
    output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(mesh_device, tid, cq_id=trace_cq)

    # Capture Trace 1
    logger.info("Capture Trace 1")
    tid_1 = ttnn.begin_trace_capture(mesh_device, cq_id=trace_cq)
    output_tensor_1 = run_op_chain_1(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(mesh_device, tid_1, cq_id=trace_cq)

    # Capture Trace 1
    logger.info("Capture Trace 2")
    tid_2 = ttnn.begin_trace_capture(mesh_device, cq_id=trace_cq)
    output_tensor_2 = run_op_chain_2(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(mesh_device, tid_2, cq_id=trace_cq)

    # Execute and verify trace against pytorch
    torch_silu = torch.nn.SiLU()
    torch_softmax = torch.nn.Softmax(dim=1)
    # Decrease loop count for larger shapes, since they time out on CI
    num_trace_loops = NUM_TRACE_LOOPS
    if shape == (1, 1, 512, 512):
        num_trace_loops = 5

    for i in range(num_trace_loops):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(
            (mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_input_tensor_1 = torch.rand(
            (mesh_device.get_num_devices(), shape[1], shape[2], shape[3]), dtype=torch.bfloat16
        )
        torch_weight = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        ) @ torch_silu(torch_weight)

        torch_output_golden_1 = torch.tanh(
            torch.mul(torch.sub(torch_input_tensor_0, torch_input_tensor_1), torch_weight)
        ) @ torch_softmax(torch_weight)

        torch_output_golden_2 = torch.neg(
            torch.mul(torch_input_tensor_0, torch_input_tensor_1)
        ) @ torch.nn.functional.gelu(torch_weight)

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(
            torch_input_tensor_0, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)
        )
        ttnn_input_tensor_1 = ttnn.from_torch(
            torch_input_tensor_1, layout=ttnn.TILE_LAYOUT, mesh_mapper=ShardTensorToMesh(mesh_device, dim=0)
        )
        ttnn_weight = ttnn.from_torch(
            torch_weight, layout=ttnn.TILE_LAYOUT, mesh_mapper=ReplicateTensorToMesh(mesh_device)
        )

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev, cq_id=data_movement_cq)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev, cq_id=data_movement_cq)
        ttnn.copy_host_to_device_tensor(ttnn_weight, weight_dev, cq_id=data_movement_cq)
        event_sync(mesh_device, data_movement_cq, trace_cq)
        # Execute trace
        logger.info("Execute Trace 0")
        ttnn.execute_trace(mesh_device, tid, cq_id=trace_cq, blocking=False)
        logger.info("Execute Trace 1")
        ttnn.execute_trace(mesh_device, tid_1, cq_id=trace_cq, blocking=False)
        logger.info("Execute Trace 2")
        ttnn.execute_trace(mesh_device, tid_2, cq_id=trace_cq, blocking=False)
        event_sync(mesh_device, trace_cq, data_movement_cq)

        # Perform host All-Gather
        logger.info("Read Back Trace 0 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(
            output_tensor,
            mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
            device=mesh_device,
            cq_id=data_movement_cq,
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.96)

        logger.info("Read Back Trace 1 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(
            output_tensor_1,
            mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
            device=mesh_device,
            cq_id=data_movement_cq,
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden_1, pcc=0.96)

        logger.info("Read Back Trace 2 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(
            output_tensor_2,
            mesh_composer=ConcatMeshToTensor(mesh_device, dim=0),
            device=mesh_device,
            cq_id=data_movement_cq,
        )
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden_2, pcc=0.96)

    # Release trace buffer once workload is complete
    ttnn.release_trace(mesh_device, tid)
    ttnn.release_trace(mesh_device, tid_1)
    ttnn.release_trace(mesh_device, tid_2)
