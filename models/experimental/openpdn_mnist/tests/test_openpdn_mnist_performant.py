# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
from models.utility_functions import run_for_wormhole_b0
from models.experimental.openpdn_mnist.tests.openpdn_mnist_e2e_performant import OpenPDNMnistTrace2CQ
from models.experimental.openpdn_mnist.tests.openpdn_mnist_performant import (
    run_openpdn_mnist_inference,
    run_openpdn_mnist_trace_inference,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_run_openpdn_mnist_inference(device, use_program_cache, model_location_generator):
    run_openpdn_mnist_inference(device, model_location_generator)


@run_for_wormhole_b0()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 1843200}], indirect=True)
def test_run_openpdn_mnist_trace_inference(
    device,
    use_program_cache,
    model_location_generator,
):
    run_openpdn_mnist_trace_inference(
        device,
        model_location_generator,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 3686400, "num_command_queues": 2}], indirect=True
)
def test_run_openpdn_mnist_trace_2cqs_inference(
    device,
    use_program_cache,
    model_location_generator,
):
    openpdn_mnist_trace_2cq = OpenPDNMnistTrace2CQ()

    openpdn_mnist_trace_2cq.initialize_openpdn_mnist_trace_2cqs_inference(
        device,
        model_location_generator,
    )

    input_shape = (1, 5, 78, 78)
    batch_size = input_shape[0]
    torch_input_tensor = torch.randn(input_shape, dtype=torch.float32)
    n, c, h, w = torch_input_tensor.shape
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    torch_input_tensor = torch_input_tensor.reshape(1, 1, h * w * n, c)
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
    inference_iter_count = 10
    inference_time_iter = []
    for iter in range(0, inference_iter_count):
        output = openpdn_mnist_trace_2cq.execute_openpdn_mnist_trace_2cqs_inference(tt_inputs_host)
    openpdn_mnist_trace_2cq.release_openpdn_mnist_trace_2cqs_inference()
