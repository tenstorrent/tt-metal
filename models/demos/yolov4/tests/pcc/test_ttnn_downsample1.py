# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import load_torch_model
from models.demos.yolov4.tt.downsample1 import Down1
from models.demos.yolov4.tt.model_preprocessing import create_ds1_model_parameters
from models.utility_functions import skip_for_grayskull
from tests.ttnn.utils_for_testing import assert_with_pcc


@skip_for_grayskull()
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_down1(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    torch_model = load_torch_model(model_location_generator, "down1")
    torch_input = torch.randn((1, 3, *resolution), dtype=torch.float)
    ref = torch_model(torch_input)

    parameters = create_ds1_model_parameters(torch_model, torch_input, resolution, device)

    ttnn_model = Down1(device, parameters, parameters.conv_args)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)

    result_ttnn = ttnn_model(ttnn_input)
    if resolution == (320, 320):
        start_time = time.time()
        for x in range(100):
            result_ttnn = ttnn_model(ttnn_input)
        logger.info(f"Time taken: {time.time() - start_time}")

    result = ttnn.to_torch(result_ttnn)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.99)
