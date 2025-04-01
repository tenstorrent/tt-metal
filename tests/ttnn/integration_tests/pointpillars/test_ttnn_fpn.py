# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN
from models.experimental.functional_pointpillars.reference.fpn import FPN
from models.experimental.functional_pointpillars.tt.ttnn_fpn import TtFPN


def fold_batch_norm2d_into_conv2d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")

    weight = conv.weight
    bias = conv.bias
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    return weight, bias


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, FPN):
            parameters["lateral_convs"] = {}
            for index, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][index] = {}
                parameters["lateral_convs"][index]["ConvModule"] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                parameters["lateral_convs"][index]["ConvModule"]["weight"] = ttnn.from_torch(conv_weight)
                parameters["lateral_convs"][index]["ConvModule"]["bias"] = ttnn.from_torch(
                    conv_bias.reshape(1, 1, 1, -1),
                )
            parameters["fpn_convs"] = {}
            for index, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][index] = {}
                parameters["fpn_convs"][index]["ConvModule"] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                parameters["fpn_convs"][index]["ConvModule"]["weight"] = ttnn.from_torch(conv_weight)
                parameters["fpn_convs"][index]["ConvModule"]["bias"] = ttnn.from_torch(
                    conv_bias.reshape(1, 1, 1, -1),
                )

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize(
    "use_pretrained_weight",
    [
        True,
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@run_for_wormhole_b0()
def test_ttnn_fpn(device, use_pretrained_weight, reset_seeds):
    reference_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )
    if use_pretrained_weight == True:
        state_dict = torch.load(
            "/home/ubuntu/pointpillars_mmdetect/mmdetection3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        )["state_dict"]
        reference_model.load_state_dict(state_dict)
    reference_model.eval()
    reference_model = reference_model.pts_neck

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    inputs = [torch.randn(1, 64, 200, 200), torch.randn(1, 128, 100, 100), torch.randn(1, 256, 50, 50)]

    ttnn_inputs = inputs[:]
    for i in range(len(ttnn_inputs)):
        input_0 = ttnn_inputs[i].permute(0, 2, 3, 1)
        ttnn_inputs[i] = ttnn.from_torch(input_0, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(inputs=inputs)

    ttnn_model = TtFPN(
        in_channels=[64, 128, 256],
        start_level=0,
        num_outs=3,
        out_channels=256,
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(inputs=ttnn_inputs)

    for i in range(len(ttnn_output)):
        torch_output = ttnn.to_torch(ttnn_output[i])
        torch_output = torch_output.reshape(
            reference_output[i].shape[0],
            reference_output[i].shape[2],
            reference_output[i].shape[3],
            reference_output[i].shape[1],
        )
        passing, pcc = assert_with_pcc(reference_output[i], torch_output.permute(0, 3, 1, 2), 0.99)
        logger.info(f"Passing: {passing}, PCC: {pcc}")
