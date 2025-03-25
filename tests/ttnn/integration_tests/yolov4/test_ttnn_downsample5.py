# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time
from models.utility_functions import skip_for_grayskull
from models.demos.yolov4.reference.downsample5 import DownSample5
from models.demos.yolov4.ttnn.downsample5 import Down5
from models.demos.yolov4.ttnn.model_preprocessing import create_ds5_model_parameters
from loguru import logger
import os


@skip_for_grayskull()
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_down5(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)
    model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

    if resolution[0] == 320:
        torch_input = torch.randn((1, 512, 20, 20), dtype=torch.bfloat16)
    else:
        torch_input = torch.randn((1, 512, 40, 40), dtype=torch.bfloat16)

    torch_input = torch_input.float()
    torch_model = DownSample5()

    torch_dict = torch.load(weights_pth)
    ds_state_dict = {k: v for k, v in torch_dict.items() if (k.startswith("down5."))}
    new_state_dict = dict(zip(torch_model.state_dict().keys(), ds_state_dict.values()))
    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()
    ref = torch_model(torch_input)

    parameters = create_ds5_model_parameters(torch_model, torch_input, resolution, device)

    torch_input = torch_input.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16)
    ttnn_model = Down5(device, parameters, parameters.conv_args)
    result_ttnn = ttnn_model(ttnn_input)

    start_time = time.time()
    for x in range(2):
        result_ttnn = ttnn_model(ttnn_input)
    logger.info(f"Time taken: {time.time() - start_time}")

    result = ttnn.to_torch(result_ttnn)
    result = result.permute(0, 3, 1, 2)
    result = result.reshape(ref.shape)
    assert_with_pcc(result, ref, 0.90)  # PCC 0.9 - The PCC will improve once #3612 is resolved.
