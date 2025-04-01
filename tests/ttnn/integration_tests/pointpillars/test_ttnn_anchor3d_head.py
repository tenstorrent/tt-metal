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
from models.experimental.functional_pointpillars.reference.anchor3d_head import Anchor3DHead
from models.experimental.functional_pointpillars.tt.ttnn_anchor3d_head import TtAnchor3DHead


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
        if isinstance(model, Anchor3DHead):
            parameters["conv_cls"] = {}
            parameters["conv_cls"]["weight"] = ttnn.from_torch(model.conv_cls.weight)
            parameters["conv_cls"]["bias"] = ttnn.from_torch(
                model.conv_cls.bias.reshape(1, 1, 1, -1),
            )

            parameters["conv_reg"] = {}
            parameters["conv_reg"]["weight"] = ttnn.from_torch(model.conv_reg.weight)
            parameters["conv_reg"]["bias"] = ttnn.from_torch(
                model.conv_reg.bias.reshape(1, 1, 1, -1),
            )

            parameters["conv_dir_cls"] = {}
            parameters["conv_dir_cls"]["weight"] = ttnn.from_torch(model.conv_dir_cls.weight)
            parameters["conv_dir_cls"]["bias"] = ttnn.from_torch(
                model.conv_dir_cls.bias.reshape(1, 1, 1, -1),
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
def test_ttnn_anchor3d_head(device, use_pretrained_weight, reset_seeds):
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
    reference_model = reference_model.pts_bbox_head

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    x = [torch.randn(1, 256, 200, 200), torch.randn(1, 256, 100, 100), torch.randn(1, 256, 50, 50)]

    ttnn_x = x[:]
    for i in range(len(ttnn_x)):
        input_0 = ttnn_x[i].permute(0, 2, 3, 1)
        ttnn_x[i] = ttnn.from_torch(input_0, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    reference_output = reference_model(x=x)

    ttnn_model = TtAnchor3DHead(
        num_classes=10,
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        anchor_generator={
            "type": "AlignedAnchor3DRangeGenerator",
            "ranges": [[-50, -50, -1.8, 50, 50, -1.8]],
            "scales": [1, 2, 4],
            "sizes": [[2.5981, 0.866, 1.0], [1.7321, 0.5774, 1.0], [1.0, 1.0, 1.0], [0.4, 0.4, 1]],
            "custom_values": [0, 0],
            "rotations": [0, 1.57],
            "reshape_out": True,
        },
        assign_per_class=False,
        diff_rad_by_sin=True,
        dir_offset=-0.7854,
        bbox_coder={"type": "DeltaXYZWLHRBBoxCoder", "code_size": 9},
        test_cfg={
            "pts": {
                "use_rotate_nms": True,
                "nms_across_levels": False,
                "nms_pre": 1000,
                "nms_thr": 0.2,
                "score_thr": 0.05,
                "min_bbox_size": 0,
                "max_num": 500,
            }
        },
        parameters=parameters,
        device=device,
    )

    ttnn_output = ttnn_model(x=ttnn_x)

    for i in range(len(ttnn_output)):
        for j in range(len(ttnn_output[i])):
            torch_output = ttnn.to_torch(ttnn_output[i][j])
            torch_output = torch_output.reshape(
                reference_output[i][j].shape[0],
                reference_output[i][j].shape[2],
                reference_output[i][j].shape[3],
                reference_output[i][j].shape[1],
            )
            passing, pcc = assert_with_pcc(reference_output[i][j], torch_output.permute(0, 3, 1, 2), 0.99)
            logger.info(f"Passing: {passing}, PCC: {pcc}")
