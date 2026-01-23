# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.BEVFormerV2.reference.resnet import ResNet
from models.experimental.BEVFormerV2.tt.ttnn_backbone import TtResNet50
from models.experimental.BEVFormerV2.common import load_torch_model
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import custom_preprocessor_resnet


def create_backbone_parameters(model: ResNet, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model,
        run_model=lambda model: model(input_tensor),
        device=None,
    )
    assert parameters is not None
    for key in parameters.conv_args.keys():
        if hasattr(model, key):
            parameters.conv_args[key].module = getattr(model, key)
    return parameters


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bevformer_backbone(
    device,
    reset_seeds,
    model_location_generator,
):
    # Create torch model
    torch_model = ResNet(
        depth=50,
        in_channels=3,
        out_indices=(1, 2, 3),
        style="caffe",
    )
    torch_model = load_torch_model(
        torch_model=torch_model, layer="img_backbone", model_location_generator=model_location_generator
    )

    torch_input = torch.randn((6, 3, 384, 640), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    torch_outputs = torch_model(torch_input)

    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16)

    parameter = create_backbone_parameters(torch_model, torch_input, device=device)

    ttnn_model = TtResNet50(
        parameter.conv_args,
        parameter,
        device,
        out_indices=(1, 2, 3),
    )

    ttnn_outputs = ttnn_model(ttnn_input_tensor, batch_size=6, input_height=384, input_width=640)

    for idx, (torch_output, ttnn_output) in enumerate(zip(torch_outputs, ttnn_outputs)):
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape(
            torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
        ).to(torch.float32)
        ttnn_output = ttnn_output.permute(0, 3, 1, 2)

        pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.96)
        logger.info(f"Level {idx}: {pcc_message}")
