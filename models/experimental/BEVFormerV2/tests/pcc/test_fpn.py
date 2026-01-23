# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger
import ttnn
from models.experimental.BEVFormerV2.reference.resnet import ResNet
from models.experimental.BEVFormerV2.reference.fpn import FPN
from models.experimental.BEVFormerV2.tt.ttnn_backbone import TtResNet50
from models.experimental.BEVFormerV2.tt.ttnn_fpn import TtFPN
from models.experimental.BEVFormerV2.common import load_torch_model
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
)
from models.experimental.BEVFormerV2.tests.pcc.custom_preprocessors import (
    custom_preprocessor_resnet,
    custom_preprocessor_fpn,
)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_bevformer_fpn(
    device,
    reset_seeds,
    model_location_generator,
):
    torch_backbone = ResNet(
        depth=50,
        in_channels=3,
        out_indices=(1, 2, 3),
        style="caffe",
    )
    torch_backbone = load_torch_model(
        torch_model=torch_backbone, layer="img_backbone", model_location_generator=model_location_generator
    )

    torch_fpn = FPN(
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    )
    torch_fpn = load_torch_model(
        torch_model=torch_fpn, layer="img_neck", model_location_generator=model_location_generator
    )

    torch_input = torch.randn((6, 3, 384, 640), dtype=torch.bfloat16)
    torch_input = torch_input.float()

    backbone_outputs = torch_backbone(torch_input)
    torch_outputs = torch_fpn(backbone_outputs)

    ttnn_input_tensor = torch.permute(torch_input, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        (ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2]),
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, device=device, dtype=ttnn.bfloat16)

    backbone_params = preprocess_model_parameters(
        initialize_model=lambda: torch_backbone,
        custom_preprocessor=custom_preprocessor_resnet,
        device=device,
    )
    backbone_params.conv_args = infer_ttnn_module_args(
        model=torch_backbone,
        run_model=lambda model: model(torch_input),
        device=None,
    )
    for key in backbone_params.conv_args.keys():
        if hasattr(torch_backbone, key):
            backbone_params.conv_args[key].module = getattr(torch_backbone, key)

    fpn_params = preprocess_model_parameters(
        initialize_model=lambda: torch_fpn,
        custom_preprocessor=custom_preprocessor_fpn,
        device=device,
    )
    fpn_params.conv_args = infer_ttnn_module_args(
        model=torch_fpn,
        run_model=lambda model: model(backbone_outputs),
        device=None,
    )

    ttnn_backbone = TtResNet50(
        backbone_params.conv_args,
        backbone_params,
        device,
        out_indices=(1, 2, 3),
    )

    ttnn_fpn = TtFPN(
        fpn_params.conv_args,
        fpn_params,
        device,
    )

    backbone_ttnn_outputs = ttnn_backbone(ttnn_input_tensor, batch_size=6, input_height=384, input_width=640)
    ttnn_outputs = ttnn_fpn(backbone_ttnn_outputs, batch_size=6)

    for idx, (torch_output, ttnn_output) in enumerate(zip(torch_outputs, ttnn_outputs)):
        ttnn_output = ttnn.to_torch(ttnn_output)
        ttnn_output = ttnn_output.reshape(
            torch_output.shape[0], torch_output.shape[2], torch_output.shape[3], torch_output.shape[1]
        ).to(torch.float32)
        ttnn_output = ttnn_output.permute(0, 3, 1, 2)

        pcc_passed, pcc_message = assert_with_pcc(ttnn_output, torch_output, 0.95)
        logger.info(f"FPN Level {idx}: {pcc_message}")
