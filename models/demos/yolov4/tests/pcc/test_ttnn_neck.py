# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.yolov4.common import load_torch_model
from models.demos.yolov4.tt.model_preprocessing import create_neck_model_parameters
from models.demos.yolov4.tt.neck import TtNeck
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
def test_neck(device, reset_seeds, model_location_generator, resolution):
    torch.manual_seed(0)

    torch_model = load_torch_model(model_location_generator, module="neck")

    if resolution == (320, 320):
        torch_input_tensor1 = torch.randn(1, 10, 10, 1024, dtype=torch.float)
        torch_input_tensor2 = torch.randn(1, 20, 20, 512, dtype=torch.float)
        torch_input_tensor3 = torch.randn(1, 40, 40, 256, dtype=torch.float)
        ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
        ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 100, 1024))
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
        ttnn_input_tensor = [ttnn_input_tensor1, ttnn_input_tensor2, ttnn_input_tensor3]
    elif resolution == (640, 640):
        torch_input_tensor1 = torch.randn(1, 20, 20, 1024, dtype=torch.float)
        torch_input_tensor2 = torch.randn(1, 40, 40, 512, dtype=torch.float)
        torch_input_tensor3 = torch.randn(1, 80, 80, 256, dtype=torch.float)
        ttnn_input_tensor1 = ttnn.from_torch(torch_input_tensor1, dtype=ttnn.bfloat16)
        ttnn_input_tensor1 = ttnn.reshape(ttnn_input_tensor1, (1, 1, 400, 1024))
        ttnn_input_tensor1 = ttnn.to_layout(ttnn_input_tensor1, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor1 = ttnn.to_device(ttnn_input_tensor1, device=device)
        ttnn_input_tensor2 = ttnn.from_torch(torch_input_tensor2, dtype=ttnn.bfloat16)
        ttnn_input_tensor2 = ttnn.reshape(ttnn_input_tensor2, (1, 1, 1600, 512))
        ttnn_input_tensor2 = ttnn.to_layout(ttnn_input_tensor2, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor2 = ttnn.to_device(ttnn_input_tensor2, device=device)
        ttnn_input_tensor3 = ttnn.from_torch(torch_input_tensor3, dtype=ttnn.bfloat16)
        ttnn_input_tensor3 = ttnn.reshape(ttnn_input_tensor3, (1, 1, 6400, 256))
        ttnn_input_tensor3 = ttnn.to_layout(ttnn_input_tensor3, layout=ttnn.TILE_LAYOUT)
        ttnn_input_tensor3 = ttnn.to_device(ttnn_input_tensor3, device=device)
        ttnn_input_tensor = [ttnn_input_tensor1, ttnn_input_tensor2, ttnn_input_tensor3]
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    torch_input_tensor1 = torch_input_tensor1.permute(0, 3, 1, 2)
    torch_input_tensor2 = torch_input_tensor2.permute(0, 3, 1, 2)
    torch_input_tensor3 = torch_input_tensor3.permute(0, 3, 1, 2)
    torch_input_tensor = [torch_input_tensor1, torch_input_tensor2, torch_input_tensor3]

    ref1, ref2, ref3 = torch_model(torch_input_tensor[0], torch_input_tensor[1], torch_input_tensor[2])

    parameters = create_neck_model_parameters(torch_model, torch_input_tensor, resolution, device)

    ttnn_model = TtNeck(device, parameters, parameters.conv_args)

    result_ttnn = ttnn_model(ttnn_input_tensor)

    result_1 = ttnn.to_torch(result_ttnn[0])
    result_2 = ttnn.to_torch(result_ttnn[1])
    result_3 = ttnn.to_torch(result_ttnn[2])
    ref1 = ref1.permute(0, 2, 3, 1)
    ref2 = ref2.permute(0, 2, 3, 1)
    ref3 = ref3.permute(0, 2, 3, 1)
    result1 = result_1.reshape(ref1.shape)
    result2 = result_2.reshape(ref2.shape)
    result3 = result_3.reshape(ref3.shape)

    assert_with_pcc(result1, ref1, 0.99)
    assert_with_pcc(result2, ref2, 0.985)
    assert_with_pcc(result3, ref3, 0.96)
