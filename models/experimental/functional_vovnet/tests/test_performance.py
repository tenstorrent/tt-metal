# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import pytest
import timm
from loguru import logger
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_vovnet.tt import ttnn_functional_vovnet
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_vovnet.tests.demo_utils import *


def get_expected_times(functional_vovnet):
    return {
        ttnn_functional_vovnet: (30, 45),
    }[functional_vovnet]


@pytest.mark.models_performance_bare_metal
@pytest.mark.models_performance_virtual_machine
@pytest.mark.parametrize(
    "model_name",
    ["hf_hub:timm/ese_vovnet19b_dw.ra_in1k"],
)
@pytest.mark.parametrize(
    "batch_size",
    [1],
)
@pytest.mark.parametrize(
    "functional_vovnet",
    [ttnn_functional_vovnet],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_performance_vovnet(device, model_name, batch_size, functional_vovnet):
    model = timm.create_model("hf_hub:timm/ese_vovnet19b_dw.ra_in1k", pretrained=True)
    model = model.eval()
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        convert_to_ttnn=lambda *_: False,
    )
    input_path = "models/experimental/functional_vovnet/demo/dataset/"
    data_loader = get_data_loader(input_path, batch_size, 2)
    torch_pixel_values, labels = get_batch(data_loader, model)
    ttnn_input = ttnn.from_torch(torch_pixel_values.permute(0, 2, 3, 1), dtype=ttnn.bfloat16)

    durations = []
    for _ in range(2):
        start = time.time()

        ttnn_output = ttnn_functional_vovnet.vovnet(
            device=device,
            x=ttnn_input,
            torch_model=model.state_dict(),
            parameters=parameters,
            model=model,
            batch_size=1,
            layer_per_block=3,
            residual=False,
            depthwise=True,
            debug=False,
            bias=False,
        )
        end = time.time()
        durations.append(end - start)

    inference_and_compile_time, *inference_times = durations
    average_inference_time = sum(inference_times) / len(inference_times)
    expected_compile_time, expected_inference_time = get_expected_times(functional_vovnet)

    prep_perf_report(
        model_name,
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=average_inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - average_inference_time}")
    logger.info(f"Inference time: {average_inference_time}")
    logger.info(f"Inference times: {inference_times}")
    logger.info(f"Sample(s) per second: {1 / average_inference_time * batch_size}")
