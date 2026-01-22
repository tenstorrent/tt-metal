# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
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
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor_backbone(model, name):
    parameters = {}
    if isinstance(model, ResNet):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.norm1)
        parameters["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            for block_idx, block in enumerate(layer):
                prefix = f"layer{layer_idx}_{block_idx}"
                parameters[prefix] = {}

                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    norm = getattr(block, f"norm{conv_name[-1]}")
                    w, b = fold_batch_norm2d_into_conv2d(conv, norm)
                    parameters[prefix][conv_name] = {
                        "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                        "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                    }

                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        norm = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, norm)
                        parameters[prefix]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }
    return parameters


def custom_preprocessor_fpn(model, name):
    parameters = {}
    if isinstance(model, FPN):
        parameters["lateral_convs"] = []
        for lateral_conv in model.lateral_convs:
            conv = lateral_conv.conv
            lateral_params = {
                "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                if conv.bias is not None
                else None,
            }
            parameters["lateral_convs"].append(lateral_params)

        parameters["fpn_convs"] = []
        for fpn_conv in model.fpn_convs:
            conv = fpn_conv.conv
            fpn_params = {
                "weight": ttnn.from_torch(conv.weight, dtype=ttnn.float32),
                "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32)
                if conv.bias is not None
                else None,
            }
            parameters["fpn_convs"].append(fpn_params)
    return parameters


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
        custom_preprocessor=custom_preprocessor_backbone,
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
