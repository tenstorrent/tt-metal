# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import run_for_wormhole_b0
from ttnn.model_preprocessing import preprocess_model_parameters, preprocess_linear_weight, preprocess_linear_bias
from models.experimental.functional_pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN
from models.experimental.functional_pointpillars.reference.second import SECOND
from models.experimental.functional_pointpillars.tt.ttnn_mvx_faster_rcnn import TtMVXFasterRCNN
from models.experimental.functional_pointpillars.tt.ttnn_point_pillars_utils import TtLiDARInstance3DBoxes


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
        if isinstance(model, MVXFasterRCNN):
            pts_voxel_encoder = {}  # HardVFE
            pts_middle_encoder = {}  # PointPillarsScatter
            pts_backbone = {}  # SECOND
            pts_neck = {}  # FPN
            pts_bbox_head = {}  # Anchor3DHead

            # Anchor3DHead
            pts_bbox_head["conv_cls"] = {}
            pts_bbox_head["conv_cls"]["weight"] = ttnn.from_torch(model.pts_bbox_head.conv_cls.weight)
            pts_bbox_head["conv_cls"]["bias"] = ttnn.from_torch(
                model.pts_bbox_head.conv_cls.bias.reshape(1, 1, 1, -1),
            )

            pts_bbox_head["conv_reg"] = {}
            pts_bbox_head["conv_reg"]["weight"] = ttnn.from_torch(model.pts_bbox_head.conv_reg.weight)
            pts_bbox_head["conv_reg"]["bias"] = ttnn.from_torch(
                model.pts_bbox_head.conv_reg.bias.reshape(1, 1, 1, -1),
            )

            pts_bbox_head["conv_dir_cls"] = {}
            pts_bbox_head["conv_dir_cls"]["weight"] = ttnn.from_torch(model.pts_bbox_head.conv_dir_cls.weight)
            pts_bbox_head["conv_dir_cls"]["bias"] = ttnn.from_torch(
                model.pts_bbox_head.conv_dir_cls.bias.reshape(1, 1, 1, -1),
            )

            # FPN
            pts_neck["lateral_convs"] = {}
            for index, child in enumerate(model.pts_neck.lateral_convs):
                pts_neck["lateral_convs"][index] = {}
                pts_neck["lateral_convs"][index]["ConvModule"] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                pts_neck["lateral_convs"][index]["ConvModule"]["weight"] = ttnn.from_torch(conv_weight)
                pts_neck["lateral_convs"][index]["ConvModule"]["bias"] = ttnn.from_torch(
                    conv_bias.reshape(1, 1, 1, -1),
                )
            pts_neck["fpn_convs"] = {}
            for index, child in enumerate(model.pts_neck.fpn_convs):
                pts_neck["fpn_convs"][index] = {}
                pts_neck["fpn_convs"][index]["ConvModule"] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child.conv, child.bn)
                pts_neck["fpn_convs"][index]["ConvModule"]["weight"] = ttnn.from_torch(conv_weight)
                pts_neck["fpn_convs"][index]["ConvModule"]["bias"] = ttnn.from_torch(
                    conv_bias.reshape(1, 1, 1, -1),
                )

            # SECOND
            pts_backbone["blocks"] = {}
            for index, child in enumerate(model.pts_backbone.blocks):
                pts_backbone["blocks"][index] = {}
                for i in range(0, len(child), 3):
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child[i], child[i + 1])
                    pts_backbone["blocks"][index][i] = {}
                    pts_backbone["blocks"][index][i]["weight"] = ttnn.from_torch(conv_weight)
                    pts_backbone["blocks"][index][i]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )

            # HardVFE
            pts_voxel_encoder = {}
            pts_voxel_encoder["vfe_layers"] = {}
            for index, child in enumerate(model.pts_voxel_encoder.vfe_layers):
                pts_voxel_encoder["vfe_layers"][index] = {}
                # As we are using torch batch_norm the norm weights are torch
                pts_voxel_encoder["vfe_layers"][index]["norm"] = {}
                pts_voxel_encoder["vfe_layers"][index]["norm"] = child.norm
                # pts_voxel_encoder["vfe_layers"][index]["norm"]["bias"] = child.norm.weight

                pts_voxel_encoder["vfe_layers"][index]["linear"] = {}
                pts_voxel_encoder["vfe_layers"][index]["linear"]["weight"] = preprocess_linear_weight(
                    child.linear.weight, dtype=ttnn.bfloat16
                )
                pts_voxel_encoder["vfe_layers"][index]["linear"]["weight"] = ttnn.to_device(
                    pts_voxel_encoder["vfe_layers"][index]["linear"]["weight"], device=device
                )
                pts_voxel_encoder["vfe_layers"][index]["linear"]["bias"] = None

            parameters["pts_voxel_encoder"] = pts_voxel_encoder
            parameters["pts_middle_encoder"] = pts_middle_encoder
            parameters["pts_backbone"] = pts_backbone
            parameters["pts_neck"] = pts_neck
            parameters["pts_bbox_head"] = pts_bbox_head

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
def test_ttnn_mvx_faster_rcnn(device, use_pretrained_weight, reset_seeds):
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

    parameters = preprocess_model_parameters(
        initialize_model=lambda: reference_model, custom_preprocessor=create_custom_preprocessor(device)
    )

    batch_inputs_dict = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/functional_pointpillars/reference/batch_inputs_dict.pt"
    )
    # print("batch_inputs_dict",batch_inputs_dict["voxels"]['coors'].shape,batch_inputs_dict["voxels"]['coors'].dtype)

    batch_data_samples_modified = torch.load(
        "/home/ubuntu/punith/tt-metal/models/experimental/functional_pointpillars/reference/batch_inputs_metas_motdified.pt"
    )  # modified
    # print("batch_data_samples_modified", batch_data_samples_modified)

    reference_output = reference_model(
        batch_inputs_dict=batch_inputs_dict, batch_data_samples=batch_data_samples_modified
    )

    ttnn_batch_inputs_dict = batch_inputs_dict.copy()
    ttnn_batch_inputs_dict["points"][0] = ttnn.from_torch(
        ttnn_batch_inputs_dict["points"][0], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["num_points"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["num_points"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["voxel_centers"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["voxel_centers"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["voxels"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["voxels"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device
    )
    ttnn_batch_inputs_dict["voxels"]["coors"] = ttnn.from_torch(
        ttnn_batch_inputs_dict["voxels"]["coors"], layout=ttnn.TILE_LAYOUT, dtype=ttnn.uint32, device=device
    )
    ttnn_batch_data_samples_modified = batch_data_samples_modified.copy()
    ttnn_batch_data_samples_modified[0]["box_type_3d"] = TtLiDARInstance3DBoxes

    ttnn_model = TtMVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
        parameters=parameters,
        device=device,
    )
    ttnn_output = ttnn_model(
        batch_inputs_dict=ttnn_batch_inputs_dict, batch_data_samples=ttnn_batch_data_samples_modified
    )

    reference_output_final = reference_model.pts_bbox_head.predict(reference_output, batch_data_samples_modified)
    for i in range(len(ttnn_output)):
        for j in range(len(ttnn_output[i])):
            ttnn_output[i][j] = ttnn.to_torch(ttnn_output[i][j])
            ttnn_output[i][j] = ttnn_output[i][j].permute(0, 3, 1, 2)
            ttnn_output[i][j] = ttnn_output[i][j].to(dtype=torch.float)

    ttnn_output_final = ttnn_model.pts_bbox_head.predict(ttnn_output, ttnn_batch_data_samples_modified)

    print("reference_output_final", reference_output_final)
    print("ttnn_output_final", ttnn_output_final)

    # for i in range(len(ttnn_output)):
    #     for j in range(len(ttnn_output[i])):
    #         output_temp=ttnn_output[i][j]
    #         output_temp=ttnn.to_torch(output_temp)
    #         output_temp=output_temp.permute(0,3,1,2)
    #         passing, pcc = assert_with_pcc(reference_output[i][j], output_temp,0.97)
    #         logger.info(f"Passing: {passing}, PCC: {pcc}")
