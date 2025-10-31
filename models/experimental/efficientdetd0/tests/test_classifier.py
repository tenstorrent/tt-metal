# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest

from loguru import logger
from ttnn.model_preprocessing import preprocess_model_parameters, infer_ttnn_module_args
from models.common.utility_functions import comp_pcc, comp_allclose

from models.experimental.efficientdetd0.reference.modules import Classifier
from models.experimental.efficientdetd0.tt.classifier import Classifier as TTClassifier
from models.experimental.efficientdetd0.tt.custom_preprocessor import create_custom_mesh_preprocessor
from models.experimental.efficientdetd0.common import infer_ttnn_module_args


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
    pyramid_levels = [5, 5, 5, 5, 5, 5, 5, 5, 6]
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
        torch.randn([1, 64, 64, 64]),
        torch.randn([1, 64, 32, 32]),
        torch.randn([1, 64, 16, 16]),
        torch.randn([1, 64, 8, 8]),
        torch.randn([1, 64, 4, 4]),
    )
    torch_out = torch_model(features)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_mesh_preprocessor(None),
        device=device,
    )
    parameters.conv_args = infer_ttnn_module_args(
        model=torch_model, run_model=lambda torch_model: torch_model(features), device=None
    )
    ttnn_model = TTClassifier(
        device,
        parameters,
        parameters.conv_args,
        num_anchors,
        num_classes,
        box_class_repeats[compound_coef],
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
