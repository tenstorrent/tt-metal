# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
from loguru import logger
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)
from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.tt.ttnn_petr_head import ttnn_PETRHead
from models.experimental.functional_petr.reference.petr_head import pos2posemb3d
from models.experimental.functional_petr.tt.common import create_custom_preprocessor_petr_head, move_to_device
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_head(device, reset_seeds):
    mlvl_feats = torch.load(
        "models/experimental/functional_petr/resources/golden_mlvl_feats_petr_head.pt", weights_only=False
    )
    img_metas = torch.load(
        "models/experimental/functional_petr/resources/golden_img_metas_petr_head.pt", weights_only=False
    )
    for meta in img_metas:
        if "img_shape" in meta and isinstance(meta["img_shape"], tuple):
            meta["img_shape"] = [meta["img_shape"]] * 6
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
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
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

    passed, msg = check_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    passed1, msg1 = check_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)

    logger.info(f"petr_head_cls_scores test passed: " f"PCC={msg}")
    logger.info(f"petr_head_bbox_preds test passed: " f"PCC={msg1}")
    assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)  # Pcc > 0.99
