# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import numpy as np
from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.tt.ttnn_petr_head import ttnn_PETRHead, pos2posemb3d
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    ParameterDict,
    ParameterList,
    infer_ttnn_module_args,
)
from torch.nn import Conv2d, Linear
from torch import nn
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes

# from models.experimental.functional_petr.tt.ttnn_petr_head import ttnn_pos2posemb3d
# from models.experimental.functional_petr.reference.petr_head import pos2posemb3d

# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# def test_pos2posemb3d(device):
#     input=torch.randn(900,3,dtype=torch.bfloat16)
#     output=pos2posemb3d(input)
#     print(output.shape)
#     ttnn_input=ttnn.from_torch(input,dtype=ttnn.bfloat16,layout=ttnn.TILE_LAYOUT,device=device)
#     ttnn_output=ttnn_pos2posemb3d(ttnn_input,device=device)


def move_to_device(object, device):
    if isinstance(object, ParameterDict):
        for name, value in list(object.items()):
            if name in ["input_proj", "adapt_pos3d", "position_encoder"]:
                continue
            object[name] = move_to_device(value, device)
        return object
    elif isinstance(object, ParameterList):
        for index, element in enumerate(object):
            object[index] = move_to_device(element, device)
        return object
    elif isinstance(object, ttnn.Tensor):
        return ttnn.to_device(object, device)
    else:
        return object


def create_custom_preprocessor(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PETRHead):
            parameters["input_proj"] = {}
            parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.bfloat16)
            parameters["input_proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.bfloat16,
            )

            parameters["cls_branches"] = {}
            for index, child in enumerate(model.cls_branches):
                parameters["cls_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["cls_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.bfloat8_b
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.bfloat8_b
                        )

            parameters["adapt_pos3d"] = {}
            for index, child in enumerate(model.adapt_pos3d):
                parameters["adapt_pos3d"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["position_encoder"] = {}
            for index, child in enumerate(model.position_encoder):
                parameters["position_encoder"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.bfloat16)
                    parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.bfloat16,
                    )

            parameters["query_embedding"] = {}
            for index, child in enumerate(model.query_embedding):
                parameters["query_embedding"][index] = {}
                if isinstance(child, Linear):
                    parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                        child.weight, dtype=ttnn.bfloat8_b
                    )
                    parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                        child.bias, dtype=ttnn.bfloat8_b
                    )
            parameters["reference_points"] = {}
            parameters["reference_points"]["weight"] = ttnn.from_torch(model.reference_points.weight, device=device)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_head_without_saved_input(device, reset_seeds):
    mlvl_feats = [torch.randn(1, 6, 256, 20, 50), torch.randn(1, 6, 256, 10, 25)]

    img_metas_dict = dict()
    img_metas_dict["lidar2cam"] = np.random.randn(6, 4, 4)
    img_metas_dict["img_shape"] = [(900, 600)] * 6
    img_metas_dict["pad_shape"] = (320, 800)
    img_metas_dict["box_type_3d"] = LiDARInstance3DBoxes
    img_metas_dict["cam2img"] = [
        np.random.randn(4, 4),
        np.random.randn(4, 4),
        np.random.randn(4, 4),
        np.random.randn(4, 4),
        np.random.randn(4, 4),
        np.random.randn(4, 4),
    ]
    img_metas_dict["input_shape"] = [320, 800]
    img_metas_dict["lidar2img"] = [
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
        torch.randn(4, 4),
    ]
    img_metas = [img_metas_dict]

    torch_model = PETRHead(
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    parameters = move_to_device(parameters, device)
    output = torch_model(mlvl_feats, img_metas)

    query_embedding_input = torch_model.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)

    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    # transformer module preprocess
    child = torch_model.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, 20, 50),
            torch.zeros((1, 6, 20, 50), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, 20, 50),
        ),
        device=None,
    )
    assert x is not None
    for key in x.keys():
        x[key].module = getattr(child, key)
    parameters["transformer"] = x

    ttnn_model = ttnn_PETRHead(
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        parameters=parameters,
        device=device,
        query_embedding_input=query_embedding_input,
    )

    for i in range(len(mlvl_feats)):
        mlvl_feats[i] = ttnn.from_torch(mlvl_feats[i], layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn_model(mlvl_feats, img_metas, device=device)
    ttnn_output["all_cls_scores"] = ttnn.to_torch(ttnn_output["all_cls_scores"])
    ttnn_output["all_bbox_preds"] = ttnn.to_torch(ttnn_output["all_bbox_preds"])

    assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)  # Pcc > 0.99


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_head(device, reset_seeds):
    mlvl_feats = torch.load("models/experimental/functional_petr/reference/golden_mlvl_feats_petr_head.pt")
    img_metas = torch.load("models/experimental/functional_petr/reference/golden_img_metas_petr_head.pt")

    torch_model = PETRHead(
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        normedlinear=False,
    )

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model, custom_preprocessor=create_custom_preprocessor(None), device=None
    )
    parameters = move_to_device(parameters, device)
    output = torch_model(mlvl_feats, img_metas)

    query_embedding_input = torch_model.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)

    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    # transformer module preprocess
    child = torch_model.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, 20, 50),
            torch.zeros((1, 6, 20, 50), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, 20, 50),
        ),
        device=None,
    )
    assert x is not None
    for key in x.keys():
        x[key].module = getattr(child, key)
    parameters["transformer"] = x

    ttnn_model = ttnn_PETRHead(
        num_classes=10,
        in_channels=256,
        num_query=900,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        parameters=parameters,
        device=device,
        query_embedding_input=query_embedding_input,
    )

    for i in range(len(mlvl_feats)):
        mlvl_feats[i] = ttnn.from_torch(mlvl_feats[i], layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_output = ttnn_model(mlvl_feats, img_metas, device=device)
    ttnn_output["all_cls_scores"] = ttnn.to_torch(ttnn_output["all_cls_scores"])
    ttnn_output["all_bbox_preds"] = ttnn.to_torch(ttnn_output["all_bbox_preds"])

    assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)  # Pcc > 0.99
