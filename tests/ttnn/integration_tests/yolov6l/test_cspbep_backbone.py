# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import sys
from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters
from models.experimental.yolov6l.tt.ttnn_cspbep_backbone import TtCSPBepBackbone
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import comp_pcc

sys.path.append("models/experimental/yolov6l/reference/")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_cspbep_backbone(device, reset_seeds):
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    stride = int(model.stride.max())

    model = model.backbone

    torch_input = torch.randn(1, 3, 640, 480)

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtCSPBepBackbone(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output = ttnn_model(ttnn_input)
    # print("output: ", output[0].shape, output[1].shape, output[2].shape, output[3].shape)

    torch_output = model(torch_input)
    # print("torch_output: ", len(torch_output), torch_output[0].shape, torch_output[1].shape, torch_output[2].shape, torch_output[3].shape)

    # output = ttnn.to_torch(output)
    # output = output.permute(0, 3, 1, 2)
    # assert_with_pcc(torch_output, output, pcc=1.0)

    output_0 = ttnn.to_torch(output[0])
    output_0 = output_0.permute(0, 3, 1, 2)
    pcc_passed, pcc_message = comp_pcc(torch_output[0], output_0, pcc=0.99)  # 0.9973235311581851
    print("output_0: ", pcc_passed, pcc_message)

    output_1 = ttnn.to_torch(output[1])
    output_1 = output_1.permute(0, 3, 1, 2)
    pcc_passed, pcc_message = comp_pcc(torch_output[1], output_1, pcc=0.99)  # 0.9948524164851411
    print("output_1: ", pcc_passed, pcc_message)

    output_2 = ttnn.to_torch(output[2])
    output_2 = output_2.permute(0, 3, 1, 2)
    pcc_passed, pcc_message = comp_pcc(torch_output[2], output_2, pcc=0.99)  # 0.9962125864082163
    print("output_2: ", pcc_passed, pcc_message)

    output_3 = ttnn.to_torch(output[3])
    output_3 = output_3.permute(0, 3, 1, 2)
    pcc_passed, pcc_message = comp_pcc(torch_output[3], output_3, pcc=0.99)  # 0.9959534335651179
    print("output_3: ", pcc_passed, pcc_message)
