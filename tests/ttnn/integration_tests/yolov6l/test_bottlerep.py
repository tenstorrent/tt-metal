# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import sys
from models.experimental.yolov6l.reference.yolov6l_utils import fuse_model
from models.experimental.yolov6l.reference.yolov6l import BottleRep
from models.experimental.yolov6l.tt.model_preprocessing import create_yolov6l_model_parameters
from models.experimental.yolov6l.tt.ttnn_bottlerep import TtBottleRep
from tests.ttnn.utils_for_testing import assert_with_pcc

sys.path.append("models/experimental/yolov6l/reference/")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_yolov6l_bottlerep(device, reset_seeds):
    weights = "tests/ttnn/integration_tests/yolov6l/yolov6l.pt"
    ckpt = torch.load(weights, map_location=torch.device("cpu"), weights_only=False)
    model = ckpt["ema" if ckpt.get("ema") else "model"].float()
    model = fuse_model(model).eval()
    stride = int(model.stride.max())

    model = model.backbone.ERBlock_2[1].m.conv1

    torch_input = torch.randn(1, 64, 160, 120)

    parameters = create_yolov6l_model_parameters(model, torch_input, device)

    ttnn_model = TtBottleRep(device, parameters, parameters.model_args)

    input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    output = ttnn_model(ttnn_input)

    torch_output = model(torch_input)

    output = ttnn.to_torch(output)
    output = output.permute(0, 3, 1, 2)
    assert_with_pcc(torch_output, output, pcc=0.99)
