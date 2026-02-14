# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


from models.experimental.pointpillars.reference.fpn import FPN
from models.experimental.pointpillars.reference.second import SECOND
from models.experimental.pointpillars.reference.hard_vfe import HardVFE
from models.experimental.pointpillars.reference.anchor3d_head import Anchor3DHead
from models.experimental.pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN

from ttnn.model_preprocessing import preprocess_linear_weight


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

        if isinstance(model, HardVFE):
            parameters["vfe_layers"] = {}
            for index, child in enumerate(model.vfe_layers):
                parameters["vfe_layers"][index] = {}
                # As we are using torch batch_norm1d the norm weights are torch
                parameters["vfe_layers"][index]["norm"] = {}
                parameters["vfe_layers"][index]["norm"] = child.norm

                parameters["vfe_layers"][index]["linear"] = {}
                parameters["vfe_layers"][index]["linear"]["weight"] = preprocess_linear_weight(
                    child.linear.weight, dtype=ttnn.bfloat16
                )
                parameters["vfe_layers"][index]["linear"]["weight"] = ttnn.to_device(
                    parameters["vfe_layers"][index]["linear"]["weight"], device=device
                )
                parameters["vfe_layers"][index]["linear"]["bias"] = None

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

        if isinstance(model, SECOND):
            parameters["blocks"] = {}
            for index, child in enumerate(model.blocks):
                parameters["blocks"][index] = {}
                for i in range(0, len(child), 3):
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(child[i], child[i + 1])
                    parameters["blocks"][index][i] = {}
                    parameters["blocks"][index][i]["weight"] = ttnn.from_torch(conv_weight)
                    parameters["blocks"][index][i]["bias"] = ttnn.from_torch(
                        conv_bias.reshape(1, 1, 1, -1),
                    )
        return parameters

    return custom_preprocessor
