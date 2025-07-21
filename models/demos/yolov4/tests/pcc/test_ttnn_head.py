# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.yolov4.common import load_torch_model
from models.demos.yolov4.tt.head import TtHead
from models.demos.yolov4.tt.model_preprocessing import create_head_model_parameters
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
def test_head(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    torch_model = load_torch_model(model_location_generator, module="head")

    if resolution == (320, 320):
        torch_input_tensor1 = torch.randn(1, 40, 40, 128, dtype=torch.float)
        torch_input_tensor2 = torch.randn(1, 10, 10, 512, dtype=torch.float)
        torch_input_tensor3 = torch.randn(1, 20, 20, 256, dtype=torch.float)

        ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
        ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 1600, 128))
        ttnn_input_tensor1 = ttnn.to_layout(ttnn_input_tensor1, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device=device)

        ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn.bfloat16)
        ttnn_input_tensor2 = ttnn.reshape(ttnn_input_tensor2, (1, 1, 100, 512))
        ttnn_input_tensor2 = ttnn.to_layout(ttnn_input_tensor2, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor2 = ttnn.to_device(ttnn_input_tensor2, device=device)

        ttnn_input_tensor3 = ttnn.from_torch(torch_input_tensor3, dtype=ttnn.bfloat16)
        ttnn_input_tensor3 = ttnn.reshape(ttnn_input_tensor3, (1, 1, 400, 256))
        ttnn_input_tensor3 = ttnn.to_layout(ttnn_input_tensor3, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor3 = ttnn.to_device(ttnn_input_tensor3, device=device)
    elif resolution == (640, 640):
        torch_input_tensor1 = torch.randn(1, 80, 80, 128, dtype=torch.float)
        torch_input_tensor2 = torch.randn(1, 20, 20, 512, dtype=torch.float)
        torch_input_tensor3 = torch.randn(1, 40, 40, 256, dtype=torch.float)

        ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
        ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 6400, 128))
        ttnn_input_tensor1 = ttnn.to_layout(ttnn_input_tensor1, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device=device)

        ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn.bfloat16)
        ttnn_input_tensor2 = ttnn.reshape(ttnn_input_tensor2, (1, 1, 400, 512))
        ttnn_input_tensor2 = ttnn.to_layout(ttnn_input_tensor2, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor2 = ttnn.to_device(ttnn_input_tensor2, device=device)

        ttnn_input_tensor3 = ttnn.from_torch(torch_input_tensor3, dtype=ttnn.bfloat16)
        ttnn_input_tensor3 = ttnn.reshape(ttnn_input_tensor3, (1, 1, 1600, 256))
        ttnn_input_tensor3 = ttnn.to_layout(ttnn_input_tensor3, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor3 = ttnn.to_device(ttnn_input_tensor3, device=device)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    ttnn_input_tensor = [ttnn_input_tensor1, ttnn_input_tensor2, ttnn_input_tensor3]
    torch_input_tensor1 = torch_input_tensor1.permute(0, 3, 1, 2)
    torch_input_tensor2 = torch_input_tensor2.permute(0, 3, 1, 2)
    torch_input_tensor3 = torch_input_tensor3.permute(0, 3, 1, 2)
    torch_input_tensor = [torch_input_tensor1, torch_input_tensor2, torch_input_tensor3]

    ref1, ref2, ref3 = torch_model(torch_input_tensor[0], torch_input_tensor[1], torch_input_tensor[2])

    parameters = create_head_model_parameters(torch_model, torch_input_tensor, resolution, device)

    ttnn_model = TtHead(device, parameters, parameters.conv_args)

    result_ttnn = ttnn_model(ttnn_input_tensor)
    start_time = time.time()
    for x in range(1):
        result_ttnn = ttnn_model(ttnn_input_tensor)
    logger.info(f"Time taken: {time.time() - start_time}")

    result_1 = ttnn.to_torch(result_ttnn[0])
    result_2 = ttnn.to_torch(result_ttnn[1])
    result_3 = ttnn.to_torch(result_ttnn[2])

    num_channels = ref1.shape[1]  # 255
    num_channels_padded = num_channels + 1

    result_1 = result_1.reshape(1, ref1.shape[2], ref1.shape[3], num_channels_padded)
    result_1 = result_1.permute(0, 3, 1, 2)

    result_2 = result_2.reshape(1, ref2.shape[2], ref2.shape[3], num_channels_padded)
    result_2 = result_2.permute(0, 3, 1, 2)

    result_3 = result_3.reshape(1, ref3.shape[2], ref3.shape[3], num_channels_padded)
    result_3 = result_3.permute(0, 3, 1, 2)

    # Output is sliced because ttnn.conv returns 256 channels instead of 255.
    result_1 = result_1[:, :num_channels, :, :]
    result_2 = result_2[:, :num_channels, :, :]
    result_3 = result_3[:, :num_channels, :, :]

    assert_with_pcc(result_1, ref1, 0.99)
    assert_with_pcc(result_2, ref2, 0.99)
    assert_with_pcc(result_3, ref3, 0.99)
