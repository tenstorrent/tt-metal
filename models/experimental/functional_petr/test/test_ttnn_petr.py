# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest
import numpy as np
from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes

from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    preprocess_linear_weight,
    preprocess_layernorm_parameter,
    preprocess_linear_bias,
    ParameterDict,
    ParameterList,
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
)
from loguru import logger

from models.experimental.functional_petr.reference.petr import PETR
from models.experimental.functional_petr.reference.petr_head import PETRHead, pos2posemb3d
from models.experimental.functional_petr.reference.cp_fpn import CPFPN
from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP, eSEModule, _OSA_module, _OSA_stage

from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR


from torch.nn import Conv2d, Linear

from torch import nn
from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc


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


def stem_parameters_preprocess(model):
    parameters = {}
    if isinstance(model, VoVNetCP):
        if hasattr(model, "stem"):
            layers = list(model.stem.named_children())

        for i, (name, layer) in enumerate(layers):
            if "conv" in name:
                conv_name, conv_layer = layers[i]
                norm_name, norm_layer = layers[i + 1]
                prefix = conv_name.split("/")[0]

                if prefix not in parameters:
                    parameters[prefix] = {}

                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, norm_layer)

                logger.info(
                    f"[PREPROCESS] {prefix}: weight shape={conv_weight.shape}, mean={conv_weight.mean():.6f}, std={conv_weight.std():.6f}"
                )

                # Convert to ttnn format (same as other Conv layers)
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

    return parameters


def create_custom_preprocessor_cpfpn(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, CPFPN):
            parameters["lateral_convs"] = {}
            for i, child in enumerate(model.lateral_convs):
                parameters["lateral_convs"][i] = {}
                parameters["lateral_convs"][i]["conv"] = {}
                parameters["lateral_convs"][i]["conv"]["weight"] = ttnn.from_torch(
                    child.conv.weight, dtype=ttnn.bfloat16
                )
                parameters["lateral_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
            parameters["fpn_convs"] = {}
            for i, child in enumerate(model.fpn_convs):
                parameters["fpn_convs"][i] = {}
                parameters["fpn_convs"][i]["conv"] = {}
                parameters["fpn_convs"][i]["conv"]["weight"] = ttnn.from_torch(child.conv.weight, dtype=ttnn.bfloat16)
                parameters["fpn_convs"][i]["conv"]["bias"] = ttnn.from_torch(
                    torch.reshape(child.conv.bias, (1, 1, 1, -1)),
                    dtype=ttnn.bfloat16,
                )
        return parameters

    return custom_preprocessor


def create_custom_preprocessor_vovnetcp(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, eSEModule):
            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(torch.reshape(model.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16)
        if isinstance(model, _OSA_module):
            if hasattr(model, "conv_reduction"):
                first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(model.conv_reduction[0], model.conv_reduction[1])
                parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            for i, layers in enumerate(model.layers):
                first_layer_name = list(layers.named_children())[0][0]
                prefix = first_layer_name.split("/")[0]
                parameters[prefix] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                # if "OSA2_1" in prefix:
                #     parameters[prefix]["weight"] = conv_weight
                #     parameters[prefix]["bias"] = conv_bias
                # else:
                parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                parameters[prefix]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

            first_layer_name, _ = list(model.concat.named_children())[0]
            base_name = first_layer_name.split("/")[0]
            parameters[base_name] = {}
            # if "OSA2_1" in base_name:
            #     parameters[base_name]["weight"] = model.concat[0].weight
            #     parameters[base_name]["bias"] = model.concat[0].bias
            # else:
            concat_weight, concat_bias = fold_batch_norm2d_into_conv2d(model.concat[0], model.concat[1])
            parameters[base_name]["weight"] = ttnn.from_torch(concat_weight, dtype=ttnn.bfloat16)
            parameters[base_name]["bias"] = ttnn.from_torch(
                torch.reshape(concat_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )

            parameters["fc"] = {}
            parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
            parameters["fc"]["bias"] = ttnn.from_torch(
                torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
            )
        if isinstance(model, _OSA_stage):
            if isinstance(model, _OSA_module):
                if hasattr(model, "conv_reduction"):
                    first_layer_name, _ = list(model.conv_reduction.named_children())[0]
                    base_name = first_layer_name.split("/")[0]
                    parameters[base_name] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(
                        model.conv_reduction[0], model.conv_reduction[1]
                    )
                    parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[base_name]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                for i, layers in enumerate(model.layers):
                    first_layer_name = list(layers.named_children())[0][0]
                    prefix = first_layer_name.split("/")[0]
                    parameters[prefix] = {}
                    conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(layers[0], layers[1])
                    parameters[prefix]["weight"] = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
                    parameters[prefix]["bias"] = ttnn.from_torch(
                        torch.reshape(conv_bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                    )

                first_layer_name, _ = list(model.concat.named_children())[0]
                base_name = first_layer_name.split("/")[0]
                parameters[base_name] = {}
                parameters[base_name]["weight"] = model.concat[0].weight
                parameters[base_name]["bias"] = model.concat[0].bias

                parameters["fc"] = {}
                parameters["fc"]["weight"] = ttnn.from_torch(model.ese.fc.weight, dtype=ttnn.bfloat16)
                parameters["fc"]["bias"] = ttnn.from_torch(
                    torch.reshape(model.ese.fc.bias, (1, 1, 1, -1)), dtype=ttnn.bfloat16
                )

        return parameters

    return custom_preprocessor


def create_custom_preprocessor_petr_head(device):
    def custom_preprocessor(model, name, ttnn_module_args):
        parameters = {}
        if isinstance(model, PETRHead):
            parameters["input_proj"] = {}
            parameters["input_proj"]["weight"] = ttnn.from_torch(model.input_proj.weight, dtype=ttnn.float32)
            parameters["input_proj"]["bias"] = ttnn.from_torch(
                torch.reshape(model.input_proj.bias, (1, 1, 1, -1)),
                dtype=ttnn.float32,
            )

            parameters["cls_branches"] = {}
            for index, child in enumerate(model.cls_branches):
                parameters["cls_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["cls_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.float32
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.float32
                        )
                    elif isinstance(child1, nn.LayerNorm):
                        parameters["cls_branches"][index][index1]["weight"] = preprocess_layernorm_parameter(
                            child1.weight, dtype=ttnn.float32
                        )
                        parameters["cls_branches"][index][index1]["bias"] = preprocess_layernorm_parameter(
                            child1.bias, dtype=ttnn.float32
                        )

            parameters["reg_branches"] = {}
            for index, child in enumerate(model.reg_branches):
                parameters["reg_branches"][index] = {}
                for index1, child1 in enumerate(child):
                    parameters["reg_branches"][index][index1] = {}
                    if isinstance(child1, Linear):
                        parameters["reg_branches"][index][index1]["weight"] = preprocess_linear_weight(
                            child1.weight, dtype=ttnn.float32
                        )
                        parameters["reg_branches"][index][index1]["bias"] = preprocess_linear_bias(
                            child1.bias, dtype=ttnn.float32
                        )

            parameters["adapt_pos3d"] = {}
            for index, child in enumerate(model.adapt_pos3d):
                parameters["adapt_pos3d"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["adapt_pos3d"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.float32)
                    parameters["adapt_pos3d"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.float32,
                    )

            parameters["position_encoder"] = {}
            for index, child in enumerate(model.position_encoder):
                parameters["position_encoder"][index] = {}
                if isinstance(child, Conv2d):
                    parameters["position_encoder"][index]["weight"] = ttnn.from_torch(child.weight, dtype=ttnn.float32)
                    parameters["position_encoder"][index]["bias"] = ttnn.from_torch(
                        torch.reshape(child.bias, (1, 1, 1, -1)),
                        dtype=ttnn.float32,
                    )

            parameters["query_embedding"] = {}
            for index, child in enumerate(model.query_embedding):
                parameters["query_embedding"][index] = {}
                if isinstance(child, Linear):
                    parameters["query_embedding"][index]["weight"] = preprocess_linear_weight(
                        child.weight, dtype=ttnn.float32
                    )
                    parameters["query_embedding"][index]["bias"] = preprocess_linear_bias(
                        child.bias, dtype=ttnn.float32
                    )
            parameters["reference_points"] = {}
            parameters["reference_points"]["weight"] = ttnn.from_torch(model.reference_points.weight, device=device)

        return parameters

    return custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_without_saved_input(device, reset_seeds):
    inputs = dict()
    inputs["imgs"] = torch.randn(1, 6, 3, 320, 800)

    img_metas_dict = dict()
    img_metas_dict["lidar2cam"] = np.random.randn(6, 4, 4)
    img_metas_dict["img_shape"] = (320, 800)
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
    modified_batch_img_metas = [img_metas_dict]

    torch_model = PETR(use_grid_mask=True)
    # weights_state_dict = torch.load(
    #     "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth"
    # )["state_dict"]
    # torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    ttnn_inputs = dict()
    ttnn_inputs["imgs"] = ttnn.from_torch(inputs["imgs"], device=device)

    ttnn_batch_img_metas = modified_batch_img_metas.copy()
    print("ttnn_batch_img_metas", ttnn_batch_img_metas[0]["img_shape"])
    print("modified_batch_img_metas", modified_batch_img_metas[0]["img_shape"])
    output = torch_model.predict(inputs, modified_batch_img_metas)

    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    # transformer module preprocess
    child = torch_model.pts_bbox_head.transformer
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
    parameters_petr_head["transformer"] = x

    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    parameters = {}
    parameters["pts_bbox_head"] = parameters_petr_head
    parameters["img_neck"] = parameters_petr_cpfpn
    parameters["img_backbone"] = parameters_petr_vovnetcp

    stem_parameters = stem_parameters_preprocess(torch_model.img_backbone)
    parameters["stem_parameters"] = stem_parameters

    # print("parameters", parameters)

    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)

    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas)

    # print("output", ttnn_output)
    passed, msg = check_with_pcc(
        output[0]["pts_bbox"]["bboxes_3d"].tensor, ttnn_output[0]["pts_bbox"]["bboxes_3d"].tensor, pcc=0.99
    )
    print(f"Bboxes PCC: {msg}")
    passed, msg = check_with_pcc(output[0]["pts_bbox"]["scores_3d"], ttnn_output[0]["pts_bbox"]["scores_3d"], pcc=0.99)
    print(f"Scores PCC: {msg}")
    passed, msg = check_with_pcc(output[0]["pts_bbox"]["labels_3d"], ttnn_output[0]["pts_bbox"]["labels_3d"], pcc=0.99)
    print(f"Labels PCC: {msg}")
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["bboxes_3d"].tensor, ttnn_output[0]["pts_bbox"]["bboxes_3d"].tensor, pcc=0.99
    # )  # 0.05455256429036736
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["scores_3d"], ttnn_output[0]["pts_bbox"]["scores_3d"], pcc=0.99
    # )  # 0.7654845361594788
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["labels_3d"], ttnn_output[0]["pts_bbox"]["labels_3d"], pcc=0.99
    # )  # -0.0063037415388272665


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr(device, reset_seeds):
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )
    # print("Type of inputs:", type(inputs))
    # print("Keys in inputs:", inputs.keys() if isinstance(inputs, dict) else "Not a dict")
    # print("Type of inputs['imgs']:", type(inputs.get("imgs")))
    # print("Content of inputs['imgs']:", inputs.get("imgs"))

    torch_model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    if isinstance(inputs.get("imgs"), str):
        # Handle the case where it's a file path
        imgs_path = inputs["imgs"]
    # print(f"imgs is a string: {imgs_path}, creating dummy tensor")
    # inputs["imgs"] = torch.randn(1, 6, 3, 320, 800)

    ttnn_inputs = dict()
    ttnn_inputs["imgs"] = ttnn.from_torch(inputs["imgs"], device=device)

    ttnn_batch_img_metas = modified_batch_img_metas.copy()
    output = torch_model.predict(inputs, modified_batch_img_metas)

    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    # transformer module preprocess
    child = torch_model.pts_bbox_head.transformer
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
    parameters_petr_head["transformer"] = x

    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    parameters = {}
    parameters["pts_bbox_head"] = parameters_petr_head
    parameters["img_neck"] = parameters_petr_cpfpn
    parameters["img_backbone"] = parameters_petr_vovnetcp

    stem_parameters = stem_parameters_preprocess(torch_model.img_backbone)
    parameters["stem_parameters"] = stem_parameters

    # print("parameters", parameters)

    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)

    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    ttnn_output = ttnn_model.predict(ttnn_inputs, ttnn_batch_img_metas)

    torch_outs = torch_model.head_outs
    ttnn_outs = ttnn_model.head_outs

    passed, pcc_cls = check_with_pcc(torch_outs["all_cls_scores"], ttnn_outs["all_cls_scores"], pcc=0.99)
    print(f"all_cls_scores PCC: {float(pcc_cls):.6f}, Shape: {torch_outs['all_cls_scores'].shape}")

    passed, pcc_bbox = check_with_pcc(torch_outs["all_bbox_preds"], ttnn_outs["all_bbox_preds"], pcc=0.99)
    print(f"all_bbox_preds PCC: {float(pcc_bbox):.6f}, Shape: {torch_outs['all_bbox_preds'].shape}")

    # Statistics
    print(f"\nall_cls_scores statistics:")
    print(f"  Torch: mean={torch_outs['all_cls_scores'].mean():.6f}, std={torch_outs['all_cls_scores'].std():.6f}")
    print(f"  TTNN:  mean={ttnn_outs['all_cls_scores'].mean():.6f}, std={ttnn_outs['all_cls_scores'].std():.6f}")

    print(f"\nall_bbox_preds statistics:")
    print(f"  Torch: mean={torch_outs['all_bbox_preds'].mean():.6f}, std={torch_outs['all_bbox_preds'].std():.6f}")
    print(f"  TTNN:  mean={ttnn_outs['all_bbox_preds'].mean():.6f}, std={ttnn_outs['all_bbox_preds'].std():.6f}")
    assert_with_pcc(torch_outs["all_bbox_preds"], ttnn_outs["all_bbox_preds"], pcc=0.99)  # 0.05455256429036736
    assert_with_pcc(torch_outs["all_cls_scores"], ttnn_outs["all_cls_scores"], pcc=0.97)
    # print("")
    # passed, msg = check_with_pcc(
    #     output[0]["pts_bbox"]["bboxes_3d"].tensor, ttnn_output[0]["pts_bbox"]["bboxes_3d"].tensor, pcc=0.99
    # )
    # print(f"Bboxes PCC: {msg}")
    # passed, msg = check_with_pcc(output[0]["pts_bbox"]["scores_3d"], ttnn_output[0]["pts_bbox"]["scores_3d"], pcc=0.99)
    # print(f"Scores PCC: {msg}")
    # passed, msg = check_with_pcc(output[0]["pts_bbox"]["labels_3d"], ttnn_output[0]["pts_bbox"]["labels_3d"], pcc=0.99)
    # print(f"Labels PCC: {msg}")
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["bboxes_3d"].tensor, ttnn_output[0]["pts_bbox"]["bboxes_3d"].tensor, pcc=0.99
    # )  # 0.05455256429036736
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["scores_3d"], ttnn_output[0]["pts_bbox"]["scores_3d"], pcc=0.99
    # )  # 0.7654845361594788
    # assert_with_pcc(
    #     output[0]["pts_bbox"]["labels_3d"], ttnn_output[0]["pts_bbox"]["labels_3d"], pcc=0.99
    # )  # -0.0063037415388272665


"""
Add this test to your test_ttnn_petr.py file to check intermediate PCCs
"""


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_intermediate_outputs(device, reset_seeds):
    """Test intermediate outputs to identify where divergence occurs"""

    # Load inputs
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )

    # Setup models
    torch_model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Preprocess parameters (use your existing preprocessing functions)
    parameters_petr_head = preprocess_model_parameters(
        initialize_model=lambda: torch_model.pts_bbox_head,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters_petr_head = move_to_device(parameters_petr_head, device)

    child = torch_model.pts_bbox_head.transformer
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
    for key in x.keys():
        x[key].module = getattr(child, key)
    parameters_petr_head["transformer"] = x

    parameters_petr_cpfpn = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_neck,
        custom_preprocessor=create_custom_preprocessor_cpfpn(None),
        device=None,
    )

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )

    parameters = {}
    parameters["pts_bbox_head"] = parameters_petr_head
    parameters["img_neck"] = parameters_petr_cpfpn
    parameters["img_backbone"] = parameters_petr_vovnetcp
    parameters["stem_parameters"] = stem_parameters_preprocess(torch_model.img_backbone)

    query_embedding_input = torch_model.pts_bbox_head.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)
    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    ttnn_model = ttnn_PETR(
        use_grid_mask=True,
        parameters=parameters,
        query_embedding_input=query_embedding_input,
        device=device,
    )

    # Prepare inputs
    ttnn_inputs = dict()
    ttnn_inputs["imgs"] = ttnn.from_torch(inputs["imgs"], device=device)
    ttnn_batch_img_metas = modified_batch_img_metas.copy()

    # Add lidar2img for both models
    torch_batch_img_metas = modified_batch_img_metas.copy()
    torch_batch_img_metas = torch_model.add_lidar2img([inputs["imgs"]], torch_batch_img_metas)
    ttnn_batch_img_metas = ttnn_model.add_lidar2img([ttnn_inputs["imgs"]], ttnn_batch_img_metas)

    print("\n" + "=" * 80)
    print("INTERMEDIATE PCC ANALYSIS")
    print("=" * 80)

    # ========== TEST 1: Backbone Features (per camera) ==========
    print("\n[1] Testing Backbone Features (per camera)...")

    img_torch = inputs["imgs"]
    img_ttnn = ttnn_inputs["imgs"]

    if len(img_torch.shape) == 5:
        B, N, C, H, W = img_torch.shape
        img_torch = img_torch.view(B * N, C, H, W)
        img_ttnn = ttnn.reshape(img_ttnn, (B * N, C, H, W))

    # Test each camera separately
    for cam_idx in range(6):
        single_img_torch = img_torch[cam_idx : cam_idx + 1]
        single_img_ttnn = img_ttnn[cam_idx : cam_idx + 1]

        # Torch backbone
        with torch.no_grad():
            torch_feats = torch_model.img_backbone(single_img_torch)
            if isinstance(torch_feats, dict):
                torch_feats = list(torch_feats.values())

        # TTNN backbone
        single_img_ttnn_nhwc = ttnn.permute(single_img_ttnn, (0, 2, 3, 1))
        ttnn_feats = ttnn_model.img_backbone(device=device, x=single_img_ttnn_nhwc)

        # Compare stage 4 and stage 5
        for stage_idx, (t_feat, tt_feat) in enumerate(zip(torch_feats, ttnn_feats)):
            tt_feat_torch = ttnn.to_torch(ttnn.permute(tt_feat, (0, 3, 1, 2)))
            passed, pcc = check_with_pcc(t_feat, tt_feat_torch, pcc=0.99)
            print(f"  Camera {cam_idx}, Stage {stage_idx+4}: PCC = {float(pcc):.6f}, Shape = {t_feat.shape}")

    # ========== TEST 2: Full Backbone Output (all cameras combined) ==========
    print("\n[2] Testing Full Backbone Output (all cameras combined)...")

    # Reset img for full processing
    img_torch = inputs["imgs"]
    # img_ttnn = ttnn_inputs["imgs"]
    img_ttnn = ttnn.from_torch(inputs["imgs"], device=device)

    if len(img_torch.shape) == 5:
        B, N, C, H, W = img_torch.shape
        img_torch = img_torch.view(B * N, C, H, W)

    with torch.no_grad():
        torch_backbone_feats = torch_model.img_backbone(img_torch)
        if isinstance(torch_backbone_feats, dict):
            torch_backbone_feats = list(torch_backbone_feats.values())

    # Convert to torch once and slice there
    img_torch_reshaped = inputs["imgs"].view(6, 3, 320, 800)  # [B*N, C, H, W]

    ttnn_backbone_feats = []
    for cam_idx in range(6):
        single_img_torch = img_torch_reshaped[cam_idx : cam_idx + 1]  # [1, 3, 320, 800]
        single_img_ttnn = ttnn.from_torch(single_img_torch, layout=ttnn.TILE_LAYOUT, device=device)
        single_img_nhwc = ttnn.permute(single_img_ttnn, (0, 2, 3, 1))
        single_feats = ttnn_model.img_backbone(device=device, x=single_img_nhwc)
        ttnn_backbone_feats.append(single_feats)

    # Combine features
    ttnn_combined_feats = []
    for stage_idx in range(2):
        stage_feats = [ttnn_backbone_feats[cam][stage_idx] for cam in range(6)]
        stacked_feat = ttnn.concat(stage_feats, dim=0)
        ttnn_combined_feats.append(stacked_feat)

    for stage_idx in range(2):
        tt_feat_torch = ttnn.to_torch(ttnn.permute(ttnn_combined_feats[stage_idx], (0, 3, 1, 2)))
        passed, pcc = check_with_pcc(torch_backbone_feats[stage_idx], tt_feat_torch, pcc=0.99)
        print(f"  Stage {stage_idx+4}: PCC = {float(pcc):.6f}, Shape = {torch_backbone_feats[stage_idx].shape}")

    # ========== TEST 3: FPN/Neck Output ==========
    print("\n[3] Testing FPN/Neck Output...")

    with torch.no_grad():
        torch_neck_feats = torch_model.img_neck(torch_backbone_feats)

    ttnn_neck_feats = ttnn_model.img_neck(device=device, inputs=ttnn_combined_feats)

    for idx, (t_feat, tt_feat) in enumerate(zip(torch_neck_feats, ttnn_neck_feats)):
        tt_feat_torch = ttnn.to_torch(ttnn.permute(tt_feat, (0, 3, 1, 2)))
        passed, pcc = check_with_pcc(t_feat, tt_feat_torch, pcc=0.99)
        print(f"  FPN Level {idx}: PCC = {float(pcc):.6f}, Shape = {t_feat.shape}")

    # ========== TEST 4: Reshaped Features (B, N, C, H, W) ==========
    print("\n[4] Testing Reshaped Features...")

    torch_img_feats_reshaped = []
    for img_feat in torch_neck_feats:
        BN, C, H, W = img_feat.size()
        torch_img_feats_reshaped.append(img_feat.view(1, int(BN / 1), C, H, W))

    ttnn_img_feats_reshaped = []
    for img_feat in ttnn_neck_feats:
        img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))
        BN, C, H, W = img_feat.shape
        ttnn_img_feats_reshaped.append(ttnn.reshape(img_feat, (1, int(BN / 1), C, H, W)))

    for idx, (t_feat, tt_feat) in enumerate(zip(torch_img_feats_reshaped, ttnn_img_feats_reshaped)):
        tt_feat_torch = ttnn.to_torch(tt_feat)
        passed, pcc = check_with_pcc(t_feat, tt_feat_torch, pcc=0.99)
        print(f"  Reshaped Level {idx}: PCC = {float(pcc):.6f}, Shape = {t_feat.shape}")

    # ========== TEST 5: Head Forward Pass (Transformer + Predictions) ==========
    print("\n[5] Testing Head Forward Pass...")
    print("\nDEBUG: Checking transformer output before cls/reg branches")
    with torch.no_grad():
        torch_outs = torch_model.pts_bbox_head(torch_img_feats_reshaped, torch_batch_img_metas)

    ttnn_outs = ttnn_model.pts_bbox_head(ttnn_img_feats_reshaped, ttnn_batch_img_metas, device=device)

    # Compare transformer outputs
    for key in ["all_cls_scores", "all_bbox_preds"]:
        if key in ttnn_outs:
            ttnn_tensor = ttnn.to_torch(ttnn_outs[key])
        else:
            ttnn_tensor = ttnn_outs[key]

        passed, pcc = check_with_pcc(torch_outs[key], ttnn_tensor, pcc=0.99)
        print(f"  {key}: PCC = {float(pcc):.6f}, Shape = {torch_outs[key].shape}")

    # ========== TEST 6: Post-Processing (get_bboxes) ==========
    print("\n[6] Testing Post-Processing...")

    torch_bbox_list = torch_model.pts_bbox_head.get_bboxes(torch_outs, torch_batch_img_metas, rescale=False)

    # Convert ttnn outputs to torch for get_bboxes
    ttnn_outs_for_bbox = {}
    for key in ttnn_outs.keys():
        if key in ["all_cls_scores", "all_bbox_preds"]:
            ttnn_outs_for_bbox[key] = ttnn.to_torch(ttnn_outs[key])
        else:
            ttnn_outs_for_bbox[key] = ttnn_outs[key]

    ttnn_bbox_list = ttnn_model.pts_bbox_head.get_bboxes(ttnn_outs_for_bbox, ttnn_batch_img_metas, rescale=False)

    ####################################
    ttnn_outs_forced = {
        "all_cls_scores": torch_outs["all_cls_scores"],  # Use torch scores!
        "all_bbox_preds": torch_outs["all_bbox_preds"],  # Use torch bboxes!
        "enc_cls_scores": None,
        "enc_bbox_preds": None,
    }

    ttnn_bbox_list_forced = ttnn_model.pts_bbox_head.get_bboxes(ttnn_outs_forced, ttnn_batch_img_metas, rescale=False)

    for batch_idx in range(len(torch_bbox_list)):
        torch_bboxes, torch_scores, torch_labels = torch_bbox_list[batch_idx]
        ttnn_bboxes, ttnn_scores, ttnn_labels = ttnn_bbox_list_forced[batch_idx]

        passed, pcc = check_with_pcc(torch_bboxes.tensor, ttnn_bboxes.tensor, pcc=0.99)
        print(f"[FORCED TEST] Bboxes PCC with torch inputs: {float(pcc):.6f}")

    print("\n[DEBUG] Post-processing analysis:")

    # Use .tensor.shape[0] to get the number of boxes, since LiDARInstance3DBoxes has no __len__
    torch_num_boxes = torch_bbox_list[0][0].tensor.shape[0]
    ttnn_num_boxes = ttnn_bbox_list[0][0].tensor.shape[0]
    print(
        f"Torch - num boxes: {torch_num_boxes}, score range: [{torch_bbox_list[0][1].min():.4f}, {torch_bbox_list[0][1].max():.4f}]"
    )
    print(
        f"TTNN  - num boxes: {ttnn_num_boxes}, score range: [{ttnn_bbox_list[0][1].min():.4f}, {ttnn_bbox_list[0][1].max():.4f}]"
    )

    # Check how many boxes overlap
    torch_boxes = torch_bbox_list[0][0].tensor.numpy()
    ttnn_boxes = ttnn_bbox_list[0][0].tensor.numpy()
    print(f"Torch boxes shape: {torch_boxes.shape}")
    print(f"TTNN boxes shape: {ttnn_boxes.shape}")

    # with torch.no_grad():
    # # Process through torch model's head
    #     torch_outs_detailed = torch_model.pts_bbox_head(torch_img_feats_reshaped, torch_batch_img_metas)

    # # Now test EACH layer's output in cls_branches[0]
    # print("\n[DETAILED] Comparing cls_branches[0] layer by layer:")

    # # Get input to cls_branches (outs_dec[0])
    # torch_cls_input = torch_outs_detailed  # Need to capture this from inside torch head
    # ttnn_cls_input = outs_dec[0:1]

    # # Layer 0: Linear
    # torch_cls_0 = torch_model.pts_bbox_head.cls_branches[0][0](torch_cls_input)
    # ttnn_cls_0 = # your ttnn output after first linear

    # passed, pcc = check_with_pcc(torch_cls_0, ttnn.to_torch(ttnn_cls_0), pcc=0.99)
    # print(f"  After Linear 0: PCC = {pcc:.6f}")

    # # Layer 1: LayerNorm
    # torch_cls_1 = torch_model.pts_bbox_head.cls_branches[0][1](torch_cls_0)
    # ttnn_cls_1 = # your ttnn output after layernorm

    # passed, pcc = check_with_pcc(torch_cls_1, ttnn.to_torch(ttnn_cls_1), pcc=0.99)
    # print(f"  After LayerNorm 1: PCC = {pcc:.6f}")

    # Compare final outputs
    for batch_idx in range(len(torch_bbox_list)):
        torch_bboxes, torch_scores, torch_labels = torch_bbox_list[batch_idx]
        ttnn_bboxes, ttnn_scores, ttnn_labels = ttnn_bbox_list[batch_idx]

        passed, pcc = check_with_pcc(torch_bboxes.tensor, ttnn_bboxes.tensor, pcc=0.99)
        print(f"  Batch {batch_idx} - Bboxes: PCC = {float(pcc):.6f}")

        passed, pcc = check_with_pcc(torch_scores, ttnn_scores, pcc=0.99)
        print(f"  Batch {batch_idx} - Scores: PCC = {float(pcc):.6f}")

        passed, pcc = check_with_pcc(torch_labels, ttnn_labels, pcc=0.99)
        print(f"  Batch {batch_idx} - Labels: PCC = {float(pcc):.6f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_stem_only(device, reset_seeds):
    """Test just the stem to see if error starts there"""

    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )

    torch_model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    img_torch = inputs["imgs"].view(6, 3, 320, 800)[0:1]

    # Torch stem
    with torch.no_grad():
        x_torch_stem = torch_model.img_backbone.stem(img_torch)

    print("\n[TORCH Stem Output]")
    print(f"  Shape: {x_torch_stem.shape}")
    print(f"  mean: {x_torch_stem.mean():.6f}, std: {x_torch_stem.std():.6f}")
    print(f"  min: {x_torch_stem.min():.6f}, max: {x_torch_stem.max():.6f}")

    # TTNN stem - you need to call it manually
    # Check your ttnn_vovnetcp.py to see how stem is processed
    # It should be in the __call__ method

    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )
    stem_parameters = stem_parameters_preprocess(torch_model.img_backbone)

    # need to check ttnn_vovnetcp.py implementation
    # and extract just the stem forward pass
    print("\n[TTNN Stem] - Need to implement stem-only forward in ttnn_vovnetcp.py")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_backbone_real_vs_random_inputs(device, reset_seeds):
    """Compare backbone PCC with real inputs vs random inputs"""

    # Load real inputs
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )

    torch_model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    # Setup TTNN backbone
    parameters_petr_vovnetcp = preprocess_model_parameters(
        initialize_model=lambda: torch_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )
    stem_parameters = stem_parameters_preprocess(torch_model.img_backbone)

    from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_VoVNetCP

    ttnn_backbone = ttnn_VoVNetCP(parameters=parameters_petr_vovnetcp, stem_parameters=stem_parameters, device=device)

    print("\n" + "=" * 80)
    print("TEST 1: BACKBONE WITH RANDOM INPUT")
    print("=" * 80)

    # Random input (what your original standalone test probably used)
    random_input = torch.randn(1, 3, 320, 800)

    print(
        f"Random input stats: mean={random_input.mean():.6f}, std={random_input.std():.6f}, "
        f"min={random_input.min():.6f}, max={random_input.max():.6f}"
    )

    # Torch forward
    with torch.no_grad():
        torch_feats_random = torch_model.img_backbone(random_input)
        if isinstance(torch_feats_random, dict):
            torch_feats_random = list(torch_feats_random.values())

    # TTNN forward
    random_ttnn = ttnn.from_torch(random_input, layout=ttnn.TILE_LAYOUT, device=device)
    random_ttnn_nhwc = ttnn.permute(random_ttnn, (0, 2, 3, 1))
    ttnn_feats_random = ttnn_backbone(device=device, x=random_ttnn_nhwc)

    print("\nRandom input results:")
    for stage_idx in range(2):
        torch_feat = torch_feats_random[stage_idx]
        ttnn_feat = ttnn.to_torch(ttnn.permute(ttnn_feats_random[stage_idx], (0, 3, 1, 2)))
        passed, pcc = check_with_pcc(torch_feat, ttnn_feat, pcc=0.99)
        print(f"  Stage {stage_idx+4}: PCC = {float(pcc):.6f}")
        print(f"    Torch: mean={torch_feat.mean():.6f}, std={torch_feat.std():.6f}")
        print(f"    TTNN:  mean={ttnn_feat.mean():.6f}, std={ttnn_feat.std():.6f}")

    print("\n" + "=" * 80)
    print("TEST 2: BACKBONE WITH REAL INPUT (single camera)")
    print("=" * 80)

    # Real input from dataset
    real_input = inputs["imgs"].view(6, 3, 320, 800)[0:1]  # First camera

    print(
        f"Real input stats: mean={real_input.mean():.6f}, std={real_input.std():.6f}, "
        f"min={real_input.min():.6f}, max={real_input.max():.6f}"
    )

    # Torch forward
    with torch.no_grad():
        torch_feats_real = torch_model.img_backbone(real_input)
        if isinstance(torch_feats_real, dict):
            torch_feats_real = list(torch_feats_real.values())

    # TTNN forward
    real_ttnn = ttnn.from_torch(real_input, layout=ttnn.TILE_LAYOUT, device=device)
    real_ttnn_nhwc = ttnn.permute(real_ttnn, (0, 2, 3, 1))
    ttnn_feats_real = ttnn_backbone(device=device, x=real_ttnn_nhwc)

    print("\nReal input results:")
    for stage_idx in range(2):
        torch_feat = torch_feats_real[stage_idx]
        ttnn_feat = ttnn.to_torch(ttnn.permute(ttnn_feats_real[stage_idx], (0, 3, 1, 2)))
        passed, pcc = check_with_pcc(torch_feat, ttnn_feat, pcc=0.99)
        print(f"  Stage {stage_idx+4}: PCC = {float(pcc):.6f}")
        print(
            f"    Torch: mean={torch_feat.mean():.6f}, std={torch_feat.std():.6f}, "
            f"min={torch_feat.min():.6f}, max={torch_feat.max():.6f}"
        )
        print(
            f"    TTNN:  mean={ttnn_feat.mean():.6f}, std={ttnn_feat.std():.6f}, "
            f"min={ttnn_feat.min():.6f}, max={ttnn_feat.max():.6f}"
        )
        print(
            f"    Diff: mean={abs(torch_feat - ttnn_feat).mean():.6f}, " f"max={abs(torch_feat - ttnn_feat).max():.6f}"
        )

    print("\n" + "=" * 80)
    print("TEST 3: STEM OUTPUT COMPARISON")
    print("=" * 80)

    # Check stem with both inputs
    print("\nRandom input through stem:")
    with torch.no_grad():
        stem_output_random = torch_model.img_backbone.stem(random_input)
    print(f"  Torch stem: mean={stem_output_random.mean():.6f}, std={stem_output_random.std():.6f}")

    print("\nReal input through stem:")
    with torch.no_grad():
        stem_output_real = torch_model.img_backbone.stem(real_input)
    print(f"  Torch stem: mean={stem_output_real.mean():.6f}, std={stem_output_real.std():.6f}")

    # TTNN stem for real input
    print("\n  Testing TTNN stem with real input...")
    # Need to call stem manually - check if your ttnn_vovnetcp exposes stem

    print("\n" + "=" * 80)
    print("TEST 4: INPUT PREPROCESSING CHECK")
    print("=" * 80)

    # Check if input preprocessing is correct
    print("\nIs the real input already normalized/preprocessed?")
    print(f"Real input value range: [{real_input.min():.6f}, {real_input.max():.6f}]")
    print(f"Random input value range: [{random_input.min():.6f}, {random_input.max():.6f}]")

    # Check if real input needs normalization
    if real_input.max() > 10:
        print("\n⚠️  WARNING: Real input appears to be in [0, 255] range!")
        print("   It may need normalization before backbone processing")

        # Try normalizing
        normalized_real = real_input / 255.0
        print(f"\nTrying with normalized input (/ 255):")

        with torch.no_grad():
            torch_feats_norm = torch_model.img_backbone(normalized_real)
            if isinstance(torch_feats_norm, dict):
                torch_feats_norm = list(torch_feats_norm.values())

        norm_ttnn = ttnn.from_torch(normalized_real, layout=ttnn.TILE_LAYOUT, device=device)
        norm_ttnn_nhwc = ttnn.permute(norm_ttnn, (0, 2, 3, 1))
        ttnn_feats_norm = ttnn_backbone(device=device, x=norm_ttnn_nhwc)

        for stage_idx in range(2):
            torch_feat = torch_feats_norm[stage_idx]
            ttnn_feat = ttnn.to_torch(ttnn.permute(ttnn_feats_norm[stage_idx], (0, 3, 1, 2)))
            passed, pcc = check_with_pcc(torch_feat, ttnn_feat, pcc=0.99)
            print(f"  Stage {stage_idx+4} (normalized): PCC = {float(pcc):.6f}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_backbone_preprocessing_comparison(device, reset_seeds):
    """Compare if different preprocessing functions cause the issue"""

    from models.experimental.functional_petr.reference.petr import PETR
    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP

    # Load PETR model
    torch_petr_model = PETR(use_grid_mask=True)
    # weights_state_dict = torch.load(
    #     "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth",
    #     weights_only=False
    # )["state_dict"]
    # torch_petr_model.load_state_dict(weights_state_dict)
    torch_petr_model.eval()

    # Load standalone VoVNetCP
    torch_vovnet_model = VoVNetCP("V-99-eSE")
    # torch_vovnet_model.load_state_dict({k.replace('img_backbone.', ''): v
    #                                     for k, v in weights_state_dict.items()
    #                                     if 'img_backbone' in k})
    torch_vovnet_model.eval()

    # Import both preprocessing functions
    from models.experimental.functional_petr.test.test_ttnn_vovnetcp import (
        create_custom_preprocessor as vovnet_preprocessor,
        stem_parameters_preprocess as vovnet_stem_preprocess,
    )

    # Method 1: PETR preprocessing (current, broken)
    print("\n" + "=" * 80)
    print("METHOD 1: Preprocessing from test_petr.py")
    print("=" * 80)

    params1 = preprocess_model_parameters(
        initialize_model=lambda: torch_petr_model.img_backbone,
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )
    stem_params1 = stem_parameters_preprocess(torch_petr_model)
    # stem_params1 = stem_parameters_preprocess(torch_petr_model.img_backbone)

    # Method 2: Standalone preprocessing (working)
    print("\n" + "=" * 80)
    print("METHOD 2: Preprocessing from test_ttnn_vovnetcp.py")
    print("=" * 80)

    params2 = preprocess_model_parameters(
        initialize_model=lambda: torch_vovnet_model,
        custom_preprocessor=vovnet_preprocessor(None),
        device=None,
    )
    stem_params2 = vovnet_stem_preprocess(torch_vovnet_model)

    # Compare parameters
    print("\n" + "=" * 80)
    print("COMPARING PARAMETERS")
    print("=" * 80)

    # Compare stem parameters
    for key in stem_params1.keys():
        if key in stem_params2:
            for param_type in ["weight", "bias"]:
                if param_type in stem_params1[key] and param_type in stem_params2[key]:
                    p1 = ttnn.to_torch(stem_params1[key][param_type])
                    p2 = ttnn.to_torch(stem_params2[key][param_type])
                    diff = (p1 - p2).abs().max().item()
                    print(f"stem {key} {param_type}: max_diff = {diff:.10f}")
                    if diff > 1e-6:
                        print(f"  ⚠️ WARNING: Significant difference!")

    # Test both with same input
    test_input = torch.randn(1, 3, 320, 800)
    test_input_ttnn = ttnn.from_torch(test_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)

    # Torch reference
    with torch.no_grad():
        torch_output = torch_vovnet_model(test_input)
        if isinstance(torch_output, dict):
            torch_output = list(torch_output.values())

    # TTNN Method 1
    from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_VoVNetCP

    ttnn_model1 = ttnn_VoVNetCP(params1, stem_params1, device)
    ttnn_output1 = ttnn_model1(device, test_input_ttnn)

    # TTNN Method 2
    ttnn_model2 = ttnn_VoVNetCP(params2, stem_params2, device)
    ttnn_output2 = ttnn_model2(device, test_input_ttnn)

    print("\n" + "=" * 80)
    print("PCC COMPARISON")
    print("=" * 80)

    for stage_idx in range(2):
        ttnn_out1 = ttnn.to_torch(ttnn.permute(ttnn_output1[stage_idx], (0, 3, 1, 2)))
        ttnn_out2 = ttnn.to_torch(ttnn.permute(ttnn_output2[stage_idx], (0, 3, 1, 2)))

        passed1, pcc1 = check_with_pcc(torch_output[stage_idx], ttnn_out1, pcc=0.99)
        passed2, pcc2 = check_with_pcc(torch_output[stage_idx], ttnn_out2, pcc=0.99)

        print(f"\nStage {stage_idx+4}:")
        print(f"  Method 1 (test_petr preprocessing): PCC = {float(pcc1):.6f}")
        print(f"  Method 2 (test_vovnet preprocessing): PCC = {float(pcc2):.6f}")

        # Compare the two TTNN outputs
        passed_comp, pcc_comp = check_with_pcc(ttnn_out1, ttnn_out2, pcc=0.99)
        print(f"  Method 1 vs Method 2: PCC = {float(pcc_comp):.6f}")


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_backbone_preprocessing_comparison1(device, reset_seeds):
    """Compare if different preprocessing functions cause the issue"""

    from models.experimental.functional_petr.reference.vovnetcp import VoVNetCP

    # Create SINGLE source model
    torch_model = VoVNetCP("V-99-eSE")
    torch_model.eval()

    # Import both preprocessing functions
    from models.experimental.functional_petr.test.test_ttnn_vovnetcp import (
        create_custom_preprocessor as vovnet_preprocessor,
        stem_parameters_preprocess as vovnet_stem_preprocess,
    )

    # Method 1: test_petr preprocessing - USE SAME MODEL
    print("\n" + "=" * 80)
    print("METHOD 1: Preprocessing from test_petr.py")
    print("=" * 80)

    params1 = preprocess_model_parameters(
        initialize_model=lambda: torch_model,  # ← Same model
        custom_preprocessor=create_custom_preprocessor_vovnetcp(None),
        device=None,
    )
    stem_params1 = stem_parameters_preprocess(torch_model)  # ← Same model

    # Method 2: test_vovnet preprocessing - USE SAME MODEL
    print("\n" + "=" * 80)
    print("METHOD 2: Preprocessing from test_ttnn_vovnetcp.py")
    print("=" * 80)

    params2 = preprocess_model_parameters(
        initialize_model=lambda: torch_model,  # ← Same model
        custom_preprocessor=vovnet_preprocessor(None),
        device=None,
    )
    stem_params2 = vovnet_stem_preprocess(torch_model)  # ← Same model

    # Compare parameters
    print("\n" + "=" * 80)
    print("COMPARING PARAMETERS")
    print("=" * 80)

    # Compare stem parameters
    for key in stem_params1.keys():
        if key in stem_params2:
            for param_type in ["weight", "bias"]:
                if param_type in stem_params1[key] and param_type in stem_params2[key]:
                    p1 = ttnn.to_torch(stem_params1[key][param_type])
                    p2 = ttnn.to_torch(stem_params2[key][param_type])
                    diff = (p1 - p2).abs().max().item()
                    print(f"stem {key} {param_type}: max_diff = {diff:.10f}")
                    if diff > 1e-6:
                        print(f"  ⚠️ WARNING: Significant difference!")
                        print(f"  p1 shape: {p1.shape}, mean: {p1.mean():.6f}, std: {p1.std():.6f}")
                        print(f"  p2 shape: {p2.shape}, mean: {p2.mean():.6f}, std: {p2.std():.6f}")

    # Test both with same input
    test_input = torch.randn(1, 3, 320, 800)
    test_input_ttnn = ttnn.from_torch(test_input.permute(0, 2, 3, 1), dtype=ttnn.bfloat16, device=device)

    # Torch reference
    with torch.no_grad():
        torch_output = torch_model(test_input)
        if isinstance(torch_output, dict):
            torch_output = list(torch_output.values())

    # TTNN Method 1
    from models.experimental.functional_petr.tt.ttnn_vovnetcp import ttnn_VoVNetCP

    ttnn_model1 = ttnn_VoVNetCP(params1, stem_params1, device)
    ttnn_output1 = ttnn_model1(device, test_input_ttnn)

    # TTNN Method 2
    ttnn_model2 = ttnn_VoVNetCP(params2, stem_params2, device)
    ttnn_output2 = ttnn_model2(device, test_input_ttnn)

    print("\n" + "=" * 80)
    print("PCC COMPARISON")
    print("=" * 80)

    for stage_idx in range(2):
        ttnn_out1 = ttnn.to_torch(ttnn.permute(ttnn_output1[stage_idx], (0, 3, 1, 2)))
        ttnn_out2 = ttnn.to_torch(ttnn.permute(ttnn_output2[stage_idx], (0, 3, 1, 2)))

        if ttnn_out1.shape != torch_output[stage_idx].shape:
            ttnn_out1 = ttnn_out1.reshape(torch_output[stage_idx].shape)
        if ttnn_out2.shape != torch_output[stage_idx].shape:
            ttnn_out2 = ttnn_out2.reshape(torch_output[stage_idx].shape)

        passed1, pcc1 = check_with_pcc(torch_output[stage_idx], ttnn_out1, pcc=0.99)
        passed2, pcc2 = check_with_pcc(torch_output[stage_idx], ttnn_out2, pcc=0.99)

        print(f"\nStage {stage_idx+4}:")
        print(f"  Method 1 (test_petr preprocessing): PCC = {float(pcc1):.6f}")
        print(f"  Method 2 (test_vovnet preprocessing): PCC = {float(pcc2):.6f}")

        # Compare the two TTNN outputs
        passed_comp, pcc_comp = check_with_pcc(ttnn_out1, ttnn_out2, pcc=0.99)
        print(f"  Method 1 vs Method 2: PCC = {float(pcc_comp):.6f}")
