# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.efficientdetd0.reference.modules import Classifier
from models.experimental.efficientdetd0.tt.classifier import Classifier as TTClassifier
from models.experimental.efficientdetd0.tt.custom_preprocessor import (
    create_custom_mesh_preprocessor,
    infer_torch_module_args,
)
from ttnn.dot_access import make_dot_access_dict
from ttnn.model_preprocessing import ModuleArgs


# @pytest.mark.parametrize(
#     "input_dim, hidden_dims,
#     [
#         (56, [256]),
#     ],
# )
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_classifier(
    device,
):
    compound_coef = 0
    num_classes = 80
    fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
    box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]
    pyramid_levels = [1, 5, 5, 5, 5, 5, 5, 5, 6]
    # pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
    aspect_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    num_scales = len([2**0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])
    num_anchors = len(aspect_ratios) * num_scales
    torch_model = Classifier(
        in_channels=fpn_num_filters[compound_coef],
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_layers=box_class_repeats[compound_coef],
        pyramid_levels=pyramid_levels[compound_coef],
    )
    # load_torch_model_state(torch_model, weight_key_prefix)

    features = (
        # torch.randn([1, 64, 64, 64]),
        torch.randn([1, 64, 32, 32]),
        # torch.randn([1, 64, 16, 16]),
        # torch.randn([1, 64, 8, 8]),
        # torch.randn([1, 64, 4, 4]),
        # torch.randn([1, 64, 32, 32]),
    )
    torch_out = torch_model(features)
    # device = None
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    # module_args = infer_ttnn_module_args(
    #     model=torch_model, run_model=lambda torch_model: torch_model(features), device=None
    # )
    # import pdb; pdb.set_trace()
    # parameters = None
    module_args = infer_torch_module_args(model=torch_model, input=features, layer_type=torch.nn.Conv2d)

    module_args_pruned = {}
    i = 0
    p_i = 0
    while i < len(module_args):
        i += 1
        module_args_pruned[p_i] = module_args[i]
        p_i += 1
        i += 2
        module_args_pruned[p_i] = module_args[i]
        p_i += 1
        i += 1

    conv_args = {}
    conv_args["conv_list"] = {}
    conv_args["header_list"] = {}

    i = 0
    for p_level in range(pyramid_levels[compound_coef]):
        conv_args["conv_list"][p_level] = {}
        conv_args["header_list"][p_level] = {}
        for layer in range(box_class_repeats[compound_coef]):
            conv_args["conv_list"][p_level][layer] = {}
            conv_args["conv_list"][p_level][layer]["depthwise_conv"] = module_args_pruned[i]
            conv_args["conv_list"][p_level][layer]["pointwise_conv"] = module_args_pruned[i + 1]
            i += 2
        conv_args["header_list"][p_level]["depthwise_conv"] = module_args_pruned[i]
        conv_args["header_list"][p_level]["pointwise_conv"] = module_args_pruned[i + 1]
        i += 2

    conv_args = make_dot_access_dict(conv_args, ignore_types=(ModuleArgs,))

    ttnn_model = TTClassifier(
        device,
        parameters,
        conv_args,
        num_anchors=num_anchors,
        num_classes=num_classes,
        num_layers=box_class_repeats[compound_coef],
        pyramid_levels=pyramid_levels[compound_coef],
    )
    ttnn_features = [
        ttnn.from_torch(
            x.permute(0, 2, 3, 1),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for x in features
    ]
    ttnn_out = ttnn_model(ttnn_features)
    ttnn_out = ttnn.to_torch(ttnn_out)
    # ttnn_out = ttnn_out.permute(0, 3, 1, 2)

    passing, pcc_message = comp_pcc(torch_out, ttnn_out, 0.999)
    logger.info(f"Output PCC: {pcc_message}")
    logger.info(comp_allclose(torch_out, ttnn_out))

    if passing:
        logger.info("Classifier Test Passed!")
    else:
        logger.warning("Classifier Test Failed!")

    assert passing, f"PCC value is lower than 0.999. Check implementation! {pcc_message}"


# test_classifier()
