# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import numpy as np
from models.experimental.functional_petr.reference.petr_head import PETRHead
from models.experimental.functional_petr.tt.ttnn_petr_head import ttnn_PETRHead, pos2posemb3d
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc
from loguru import logger
from models.experimental.functional_petr.reference.utils import LiDARInstance3DBoxes
from models.experimental.functional_petr.tt.common import create_custom_preprocessor_petr_head, move_to_device


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_head_without_saved_input(device, reset_seeds):
    mlvl_feats = [torch.randn(1, 6, 256, 20, 50), torch.randn(1, 6, 256, 10, 25)]

    img_metas_dict = dict()
    img_metas_dict["lidar2cam"] = np.random.randn(6, 4, 4)
    img_metas_dict["img_shape"] = [(320, 800)] * 6
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

    logger.info(f"petr_head_without_saved_input_cls_scores test passed: " f"PCC={msg}")
    logger.info(f"petr_head_without_saved_input_bbox_preds test passed: " f"PCC={msg1}")
    torch_bbox = output["all_bbox_preds"]
    ttnn_bbox = ttnn_output["all_bbox_preds"]

    logger.info(f"Overall bbox PCC: {check_with_pcc(torch_bbox, ttnn_bbox, pcc=0.99)}")

    for lvl in range(torch_bbox.shape[0]):
        passed, msg = check_with_pcc(torch_bbox[lvl], ttnn_bbox[lvl], pcc=0.99)
        logger.info(f"Layer {lvl} bbox PCC: {msg}")

    # Compare each coordinate dimension
    for dim in range(torch_bbox.shape[-1]):
        passed, msg = check_with_pcc(torch_bbox[..., dim], ttnn_bbox[..., dim], pcc=0.99)
        logger.info(f"Dimension {dim} bbox PCC: {msg}")

    # Check for NaN or Inf
    logger.info(f"Torch bbox - NaN: {torch.isnan(torch_bbox).any()}, Inf: {torch.isinf(torch_bbox).any()}")
    logger.info(f"TTNN bbox - NaN: {torch.isnan(ttnn_bbox).any()}, Inf: {torch.isinf(ttnn_bbox).any()}")

    # Statistical comparison
    logger.info(
        f"Torch bbox stats - mean: {torch_bbox.mean()}, std: {torch_bbox.std()}, min: {torch_bbox.min()}, max: {torch_bbox.max()}"
    )
    logger.info(
        f"TTNN bbox stats - mean: {ttnn_bbox.mean()}, std: {ttnn_bbox.std()}, min: {ttnn_bbox.min()}, max: {ttnn_bbox.max()}"
    )
    assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)  # Pcc > 0.99


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
