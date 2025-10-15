# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import pytest
from loguru import logger
import ttnn
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)

from models.experimental.functional_petr.reference.petr import PETR
from models.experimental.functional_petr.reference.petr_head import pos2posemb3d
from models.experimental.functional_petr.tt.ttnn_petr import ttnn_PETR
from models.experimental.functional_petr.tt.common import (
    create_custom_preprocessor_petr_head,
    create_custom_preprocessor_cpfpn,
    create_custom_preprocessor_vovnetcp,
    stem_parameters_preprocess,
)
from models.experimental.functional_petr.tt.common import move_to_device
from tests.ttnn.utils_for_testing import check_with_pcc, assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr(device, reset_seeds):
    inputs = torch.load(
        "models/experimental/functional_petr/resources/golden_input_inputs_sample1.pt", weights_only=False
    )
    modified_batch_img_metas = torch.load(
        "models/experimental/functional_petr/resources/modified_input_batch_img_metas_sample1.pt", weights_only=False
    )

    torch_model = PETR(use_grid_mask=True)
    weights_state_dict = torch.load(
        "models/experimental/functional_petr/resources/petr_vovnet_gridmask_p4_800x320-e2191752.pth", weights_only=False
    )["state_dict"]
    torch_model.load_state_dict(weights_state_dict)
    torch_model.eval()

    if isinstance(inputs.get("imgs"), str):
        # Handle the case where it's a file path
        imgs_path = inputs["imgs"]

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

    passed, pcc_cls = check_with_pcc(torch_outs["all_cls_scores"], ttnn_outs["all_cls_scores"], pcc=0.97)
    logger.info(f"PETR all_cls_scores PCC: {float(pcc_cls):.6f}")

    passed, pcc_bbox = check_with_pcc(torch_outs["all_bbox_preds"], ttnn_outs["all_bbox_preds"], pcc=0.97)
    logger.info(f"PETR all_bbox_preds PCC: {float(pcc_bbox):.6f}")

    assert_with_pcc(torch_outs["all_cls_scores"], ttnn_outs["all_cls_scores"], pcc=0.97)
    assert_with_pcc(torch_outs["all_bbox_preds"], ttnn_outs["all_bbox_preds"], pcc=0.97)
