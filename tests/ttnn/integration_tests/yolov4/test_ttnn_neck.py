# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.neck import TtNeck
from models.experimental.yolov4.reference.neck import Neck
from tests.ttnn.utils_for_testing import assert_with_pcc
import pytest
import time
from loguru import logger
import os


@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_neck(device, reset_seeds, model_location_generator):
    model_path = model_location_generator("models", model_subdir="Yolo")

    if model_path == "models":
        if not os.path.exists("tests/ttnn/integration_tests/yolov4/yolov4.pth"):  # check if yolov4.th is availble
            os.system(
                "tests/ttnn/integration_tests/yolov4/yolov4_weights_download.sh"
            )  # execute the yolov4_weights_download.sh file

        weights_pth = "tests/ttnn/integration_tests/yolov4/yolov4.pth"
    else:
        weights_pth = str(model_path / "yolov4.pth")

    ttnn_model = TtNeck(weights_pth)

    torch_input_tensor1 = torch.randn(1, 10, 10, 1024, dtype=torch.bfloat16)
    torch_input_tensor2 = torch.randn(1, 20, 20, 512, dtype=torch.bfloat16)
    torch_input_tensor3 = torch.randn(1, 40, 40, 256, dtype=torch.bfloat16)
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
    torch_input_tensor1 = torch_input_tensor1.permute(0, 3, 1, 2).float()
    torch_input_tensor2 = torch_input_tensor2.permute(0, 3, 1, 2).float()
    torch_input_tensor3 = torch_input_tensor3.permute(0, 3, 1, 2).float()
    torch_input_tensor = [torch_input_tensor1, torch_input_tensor2, torch_input_tensor3]
    torch_model = Neck()

    new_state_dict = {}
    ds_state_dict = {k: v for k, v in ttnn_model.torch_model.items() if (k.startswith("neek."))}

    keys = [name for name, parameter in torch_model.state_dict().items()]
    values = [parameter for name, parameter in ds_state_dict.items()]
    for i in range(len(keys)):
        new_state_dict[keys[i]] = values[i]

    torch_model.load_state_dict(new_state_dict)
    torch_model.eval()

    result_ttnn = ttnn_model(device, ttnn_input_tensor)
    start_time = time.time()
    for x in range(2):
        result_ttnn = ttnn_model(device, ttnn_input_tensor)
    logger.info(f"Time taken: {time.time() - start_time}")

    result_1 = ttnn.to_torch(result_ttnn[0])
    result_2 = ttnn.to_torch(result_ttnn[1])
    result_3 = ttnn.to_torch(result_ttnn[2])
    ref1, ref2, ref3 = torch_model(torch_input_tensor[0], torch_input_tensor[1], torch_input_tensor[2])
    ref1 = ref1.permute(0, 2, 3, 1)
    ref2 = ref2.permute(0, 2, 3, 1)
    ref3 = ref3.permute(0, 2, 3, 1)
    result1 = result_1.reshape(ref1.shape)
    result2 = result_2.reshape(ref2.shape)
    result3 = result_3.reshape(ref3.shape)
    assert_with_pcc(result1, ref1, 0.94)  # PCC = 0.94
    assert_with_pcc(result2, ref2, 0.99)  # PCC = 0.99
    assert_with_pcc(result3, ref3, 0.96)  # PCC = 0.96
