# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import sys
from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters_detect
from models.experimental.yolov6l.tt.ttnn_detect import TtDetect
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc

sys.path.append("models/experimental/yolov6l/reference/")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_detect(device, reset_seeds):
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    stride = int(model.stride.max())

    model = model.detect
    # print(model)

    torch_input_0 = torch.randn(1, 128, 80, 60)
    torch_input_1 = torch.randn(1, 256, 40, 30)
    torch_input_2 = torch.randn(1, 512, 20, 15)

    parameters = create_yolov6l_model_parameters_detect(model, [torch_input_0, torch_input_1, torch_input_2], device)
    # print(parameters)

    ttnn_model = TtDetect(device, parameters, parameters.model_args)

    input_tensor_0 = torch.permute(torch_input_0, (0, 2, 3, 1))
    ttnn_input_0 = ttnn.from_torch(
        input_tensor_0,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_1 = torch.permute(torch_input_1, (0, 2, 3, 1))
    ttnn_input_1 = ttnn.from_torch(
        input_tensor_1,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    input_tensor_2 = torch.permute(torch_input_2, (0, 2, 3, 1))
    ttnn_input_2 = ttnn.from_torch(
        input_tensor_2,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    output = ttnn_model([ttnn_input_0, ttnn_input_1, ttnn_input_2])

    torch_output = model([torch_input_0, torch_input_1, torch_input_2])

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output, output, pcc=1.0)  # 0.9999822426296433
