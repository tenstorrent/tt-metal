# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import os
from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters
from models.experimental.yolov6l.tt.ttnn_yolov6l import TtYolov6l

# from models.experimental.yolov6l.reference.yolov6l import Model, BottleRep
import sys
from tests.ttnn.utils_for_testing import assert_with_pcc

sys.path.append("models/experimental/yolov6l/reference/")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l(device, reset_seeds):
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    if not os.path.exists(weights):
        os.system("bash models/experimental/yolov6l/weights_download.sh")

    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    stride = int(model.stride.max())
    # print(model)

    torch_input = torch.randn(1, 3, 640, 480)

    parameters = create_yolov6l_model_parameters(model, torch_input, device)
    # print(parameters)

    ttnn_model = TtYolov6l(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output = ttnn_model(ttnn_input)
    print("output: ", output.shape)

    torch_output = model(torch_input)
    print("torch_output: ", torch_output[0].shape)

    output = ttnn.to_torch(output)
    assert_with_pcc(torch_output[0], output, pcc=0.999)  # PCC: 0.9994452308868368
