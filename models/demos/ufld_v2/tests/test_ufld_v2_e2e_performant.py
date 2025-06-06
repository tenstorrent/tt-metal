# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.ufld_v2.reference.ufld_v2_model import TuSimple34
from models.demos.ufld_v2.tests.ufld_v2_performant_runner import UFLDPerformantRunner
from models.utility_functions import run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 79104, "trace_region_size": 23887872, "num_command_queues": 2}], indirect=True
)
@pytest.mark.parametrize(
    "batch_size, act_dtype, weight_dtype",
    ((1, ttnn.bfloat8_b, ttnn.bfloat8_b),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 800),
    ],
)
@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
def test_ufldv2_e2e_performant(
    device,
    use_program_cache,
    batch_size,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    torch_input_tensor = torch.load("/home/ubuntu/venkatesh_latest/tt-metal/models/demos/ufld_v2/dumps/real_input")  #
    # torch_input_tensor = torch.randn((1,3,320,800), dtype=torch.float32)
    print("real input is", torch_input_tensor.shape)
    torch_model = TuSimple34(input_height=320, input_width=800)
    torch_model.eval()
    weights_path = "models/demos/ufld_v2/tusimple_res34.pth"
    state_dict = torch.load(weights_path)
    new_state_dict = {}
    for key, value in state_dict["model"].items():
        new_key = key.replace("model.", "res_model.")
        new_state_dict[new_key] = value
    torch_model.load_state_dict(new_state_dict)
    torch_out1, torch_out2 = torch_model(torch_input_tensor)
    performant_runner = UFLDPerformantRunner(
        device,
        batch_size,
        act_dtype,
        weight_dtype,
        resolution=resolution,
        model_location_generator=None,
        torch_input_tensor=torch_input_tensor,
    )
    print("before capture")
    print(assert_with_pcc(performant_runner.runner_infra.torch_output_tensor_1, torch_out1, 0.0))
    performant_runner._capture_ufldv2_trace_2cqs()
    print("after capture")
    print(assert_with_pcc(performant_runner.runner_infra.torch_output_tensor_1, torch_out1, 0.0))
    # input_shape = (1, *resolution, 3)

    n, c, h, w = torch_input_tensor.shape
    torch_input_tensor = torch_input_tensor.permute(0, 2, 3, 1)
    # torch_input_tensor = F.pad(torch_input_tensor, (0, 13))
    torch_input_tensor = torch_input_tensor.reshape(
        1,
        1,
        (torch_input_tensor.shape[0] * torch_input_tensor.shape[1] * torch_input_tensor.shape[2]),
        torch_input_tensor.shape[3],
    )
    tt_inputs_host = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # tt_inputs_host = ttnn.pad(tt_inputs_host, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)
    tt_inputs_host = tt_inputs_host.to(device)
    inference_times = []
    for _ in range(10):
        t0 = time.time()
        _ = performant_runner._execute_ufldv2_trace_2cqs_inference(None)
        print(assert_with_pcc(ttnn.to_torch(_).squeeze(0).squeeze(0), torch_out1, 0.0))
        print(assert_with_pcc(performant_runner.runner_infra.torch_output_tensor_1, torch_out1, 0.0))
        print(
            assert_with_pcc(
                performant_runner.runner_infra.torch_output_tensor_1, ttnn.to_torch(_).squeeze(0).squeeze(0), 0.0
            )
        )
        t1 = time.time()
        inference_times.append(t1 - t0)

    performant_runner.release()

    inference_time_avg = round(sum(inference_times) / len(inference_times), 6)
    logger.info(
        f"ttnn_ufld_v2_batch_size: {batch_size}, resolution: {resolution}. One inference iteration time (sec): {inference_time_avg}, FPS: {round(batch_size/inference_time_avg)}"
    )
