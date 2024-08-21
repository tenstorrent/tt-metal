# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import typing
import pytest
import ttnn
import tempfile
from loguru import logger
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("shape", [[1, 3, 1024, 1024], (1, 1, 512, 512), (1, 3, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("enable_async", [True, False])
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_single_device_single_trace(device, shape, enable_async, blocking):
    device.enable_async(enable_async)
    device.enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chain to be traced
    def run_op_chain(input_0, input_1):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1)))

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev)

    # Capture Trace
    logger.info("Capture Trace")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensor_1 = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        )

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(torch_input_tensor_0, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor_1 = ttnn.from_torch(torch_input_tensor_1, layout=ttnn.TILE_LAYOUT)

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)

        if blocking:
            ttnn.synchronize_device(device)
        logger.info("Execute Trace")
        # Execute trace
        ttnn.execute_trace(device, tid, cq_id=0, blocking=blocking)
        # Readback data
        logger.info("Read Back Trace Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)

    ttnn.release_trace(device, tid)
    device.enable_async(False)


@pytest.mark.parametrize("shape", [(1, 1, 512, 512), (1, 1, 32, 32), (1, 3, 512, 512), (1, 3, 32, 32)])
@pytest.mark.parametrize("enable_async", [True, False])
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device_params", [{"trace_region_size": 266240}], indirect=True)
def test_single_device_multi_trace(device, shape, enable_async, blocking):
    device.enable_async(enable_async)
    device.enable_program_cache()

    # Preallocate activation tensors. These will be used when capturing and executing the trace
    input_0_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    input_1_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
    weight_dev = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

    # Op chains to be traced
    def run_op_chain(input_0, input_1, weight):
        return ttnn.neg(ttnn.add(ttnn.mul(input_1, ttnn.neg(ttnn.gelu(input_0))), ttnn.relu(input_1))) @ ttnn.silu(
            weight
        )

    def run_op_chain_1(input_0, input_1, weight):
        return ttnn.gelu(ttnn.add(ttnn.tanh(ttnn.mul(ttnn.sub(input_0, input_1), weight)), input_1))

    # Compile program binaries
    run_op_chain(input_0_dev, input_1_dev, weight_dev)
    run_op_chain_1(input_0_dev, input_1_dev, weight_dev)

    # Capture Trace 0
    logger.info("Capture Trace 0")
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor = run_op_chain(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    # Capture Trace 1
    logger.info("Capture Trace 1")
    tid_1 = ttnn.begin_trace_capture(device, cq_id=0)
    output_tensor_1 = run_op_chain_1(input_0_dev, input_1_dev, weight_dev)
    ttnn.end_trace_capture(device, tid_1, cq_id=0)

    # Execute and verify trace against pytorch
    torch_silu = torch.nn.SiLU()
    for i in range(50):
        # Create torch inputs
        torch_input_tensor_0 = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensor_1 = torch.rand(shape, dtype=torch.bfloat16)
        torch_weight = torch.rand(shape, dtype=torch.bfloat16)
        # Compute PT Golden
        torch_output_golden = torch.neg(
            torch.add(
                torch.mul(torch_input_tensor_1, torch.neg(torch.nn.functional.gelu(torch_input_tensor_0))),
                torch.relu(torch_input_tensor_1),
            )
        ) @ torch_silu(torch_weight)

        torch_output_golden_1 = torch.nn.functional.gelu(
            torch.add(
                torch.tanh(torch.mul(torch.sub(torch_input_tensor_0, torch_input_tensor_1), torch_weight)),
                torch_input_tensor_1,
            )
        )

        # Convert torch tensors to TTNN Multi-Device Host Tensors
        ttnn_input_tensor_0 = ttnn.from_torch(torch_input_tensor_0, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor_1 = ttnn.from_torch(torch_input_tensor_1, layout=ttnn.TILE_LAYOUT)
        ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT)

        # Copy TTNN host tensors into preallocated Mult-Device tensors
        logger.info("Send Inputs to Device")
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_0, input_0_dev)
        ttnn.copy_host_to_device_tensor(ttnn_input_tensor_1, input_1_dev)
        ttnn.copy_host_to_device_tensor(ttnn_weight, weight_dev)

        if blocking:
            ttnn.synchronize_device(device)
        logger.info("Execute Trace 0")
        # Execute trace
        ttnn.execute_trace(device, tid, cq_id=0, blocking=blocking)
        logger.info("Execute Trace 1")
        ttnn.execute_trace(device, tid_1, cq_id=0, blocking=blocking)

        logger.info("Read Back Trace 0 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden, pcc=0.99)
        logger.info("Read Back Trace 1 Outputs")
        ttnn_torch_output_tensor = ttnn.to_torch(output_tensor_1)
        assert_with_pcc(ttnn_torch_output_tensor, torch_output_golden_1, pcc=0.99)

    # Release trace buffer once workload is complete
    ttnn.release_trace(device, tid)
    ttnn.release_trace(device, tid_1)

    device.enable_async(False)
