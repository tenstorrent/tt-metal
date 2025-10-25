# # SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# # SPDX-License-Identifier: Apache-2.0

# import ttnn
# import torch
# import pytest
# from loguru import logger
# from ttnn.model_preprocessing import (
#     preprocess_model_parameters,
#     infer_ttnn_module_args,
# )
# from models.experimental.petr.reference.petr_head import PETRHead
# from models.experimental.petr.tt.ttnn_petr_head import ttnn_PETRHead
# from models.experimental.petr.reference.petr_head import pos2posemb3d
# from models.experimental.petr.tt.common import create_custom_preprocessor_petr_head, move_to_device
# from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc


# @pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
# def test_petr_head(device, reset_seeds):
#     mlvl_feats = torch.load(
#         "models/experimental/functional_petr/resources/golden_mlvl_feats_petr_head.pt", weights_only=False
#     )
#     img_metas = torch.load(
#         "models/experimental/functional_petr/resources/golden_img_metas_petr_head.pt", weights_only=False
#     )
#     for meta in img_metas:
#         if "img_shape" in meta and isinstance(meta["img_shape"], tuple):
#             meta["img_shape"] = [meta["img_shape"]] * 6
#     torch_model = PETRHead(
#         num_classes=10,
#         in_channels=256,
#         num_query=900,
#         LID=True,
#         with_position=True,
#         with_multiview=True,
#         position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#         normedlinear=False,
#     )

#     parameters = preprocess_model_parameters(
#         initialize_model=lambda: torch_model,
#         custom_preprocessor=create_custom_preprocessor_petr_head(None),
#         device=None,
#     )
#     parameters = move_to_device(parameters, device)
#     output = torch_model(mlvl_feats, img_metas)

#     query_embedding_input = torch_model.reference_points.weight
#     query_embedding_input = pos2posemb3d(query_embedding_input)

#     query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

#     # transformer module preprocess
#     child = torch_model.transformer
#     x = infer_ttnn_module_args(
#         model=child,
#         run_model=lambda model: model(
#             torch.randn(1, 6, 256, 20, 50),
#             torch.zeros((1, 6, 20, 50), dtype=torch.bool),
#             torch.rand(900, 256),
#             torch.rand(1, 6, 256, 20, 50),
#         ),
#         device=None,
#     )
#     assert x is not None
#     for key in x.keys():
#         x[key].module = getattr(child, key)
#     parameters["transformer"] = x

#     ttnn_model = ttnn_PETRHead(
#         num_classes=10,
#         in_channels=256,
#         num_query=900,
#         LID=True,
#         with_position=True,
#         with_multiview=True,
#         position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#         parameters=parameters,
#         device=device,
#         query_embedding_input=query_embedding_input,
#     )

#     for i in range(len(mlvl_feats)):
#         mlvl_feats[i] = ttnn.from_torch(mlvl_feats[i], layout=ttnn.TILE_LAYOUT, device=device)
#     ttnn_output = ttnn_model(mlvl_feats, img_metas, device=device)
#     ttnn_output["all_cls_scores"] = ttnn.to_torch(ttnn_output["all_cls_scores"])
#     ttnn_output["all_bbox_preds"] = ttnn.to_torch(ttnn_output["all_bbox_preds"])

#     passed, msg = check_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
#     passed1, msg1 = check_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)

#     logger.info(f"petr_head_cls_scores test passed: " f"PCC={msg}")
#     logger.info(f"petr_head_bbox_preds test passed: " f"PCC={msg1}")
#     assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
#     assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)  # Pcc > 0.99
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import pytest
import numpy as np
from loguru import logger
from ttnn.model_preprocessing import (
    preprocess_model_parameters,
    infer_ttnn_module_args,
)
from models.experimental.petr.reference.petr_head import PETRHead
from models.experimental.petr.tt.ttnn_petr_head import ttnn_PETRHead
from models.experimental.petr.reference.petr_head import pos2posemb3d
from models.experimental.petr.tt.common import create_custom_preprocessor_petr_head, move_to_device
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc


def generate_realistic_mlvl_feats_and_img_metas():
    """
    multi-level features and image metadata.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    B, num_cams, C = 1, 6, 256

    feature_configs = [
        {"H": 16, "W": 40, "scale": "1/20", "name": "P3"},
    ]

    mlvl_feats = []
    for config in feature_configs:
        H, W = config["H"], config["W"]

        feat = torch.randn(B, num_cams, C, H, W, dtype=torch.float32) * 0.7

        for cam_idx in range(num_cams):
            smooth_h = max(H // 2, 1)
            smooth_w = max(W // 2, 1)
            smooth_component = torch.randn(B, C, smooth_h, smooth_w)
            smooth_component = torch.nn.functional.interpolate(
                smooth_component, size=(H, W), mode="bilinear", align_corners=False
            )
            feat[:, cam_idx] += smooth_component * 0.3

        channel_scaling = torch.rand(C) * 0.5 + 0.75  # Scale between 0.75 and 1.25
        feat = feat * channel_scaling.view(1, 1, C, 1, 1)

        feat = torch.clamp(feat, -3.0, 3.0)

        mlvl_feats.append(feat)

    H_img, W_img = 320, 800
    scale_x = 800 / 1600
    scale_y = 320 / 900

    img_metas = []

    lidar2cam = []
    cam2img = []
    lidar2img = []

    cam_configs = [
        {"pos": [1.70, 0.0, 1.54], "yaw": 0.0},  # FRONT
        {"pos": [1.52, 0.49, 1.51], "yaw": 55.0},  # FRONT_RIGHT
        {"pos": [1.52, -0.49, 1.51], "yaw": -55.0},  # FRONT_LEFT
        {"pos": [-1.00, 0.0, 1.54], "yaw": 180.0},  # BACK
        {"pos": [-1.03, -0.48, 1.51], "yaw": -110.0},  # BACK_LEFT
        {"pos": [-1.03, 0.48, 1.51], "yaw": 110.0},  # BACK_RIGHT
    ]

    base_fx = 1266.0 * scale_x
    base_fy = 1266.0 * scale_y
    base_cx = 800.0 * scale_x
    base_cy = 450.0 * scale_y

    for cam_idx, config in enumerate(cam_configs):
        # Camera intrinsics with per-camera variation
        fx = base_fx + np.random.randn() * 10
        fy = base_fy + np.random.randn() * 10
        cx = base_cx + np.random.randn() * 5
        cy = base_cy + np.random.randn() * 5

        K = np.array([[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        cam2img.append(K)

        # Lidar to camera transformation
        transform = np.eye(4, dtype=np.float32)

        yaw_rad = np.radians(config["yaw"])
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)

        transform[0, 0] = cos_yaw
        transform[0, 1] = -sin_yaw
        transform[1, 0] = sin_yaw
        transform[1, 1] = cos_yaw

        transform[0, 3] = config["pos"][0]
        transform[1, 3] = config["pos"][1]
        transform[2, 3] = config["pos"][2]

        # Add calibration noise
        transform[:3, :3] += np.random.randn(3, 3) * 0.01
        transform[:3, 3] += np.random.randn(3) * 0.05

        lidar2cam.append(transform)

        # Lidar to image transformation
        lidar2img.append(torch.from_numpy(K @ transform).float())

    lidar2cam = np.stack(lidar2cam)

    # Create metadata dictionary
    meta = {
        "lidar2cam": lidar2cam,
        "cam2img": cam2img,
        "lidar2img": lidar2img,
        "img_shape": (H_img, W_img),
        "ori_shape": (H_img, W_img),
        "pad_shape": (H_img, W_img),
        "input_shape": (H_img, W_img),
        "scale_factor": (scale_x, scale_y, scale_x, scale_y),
        "flip": False,
        "flip_direction": None,
        "sample_idx": "generated_sample_0",
        "scene_token": "generated_scene",
        "timestamp": 1533151603547836,
        "filename": f"CAM_{cam_idx}.jpg",
        "pc_range": np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
        "box_type_3d": type("LiDARInstance3DBoxes", (), {}),
    }
    img_metas.append(meta)

    return mlvl_feats, img_metas


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_petr_head(device, reset_seeds):
    """Test PETR head with dynamically generated realistic inputs (memory-optimized)"""

    # Generate realistic inputs dynamically with memory constraints
    logger.info("Generating memory-optimized PETR head test data...")
    logger.info(f"L1 small size: 24576 B per core")
    mlvl_feats, img_metas = generate_realistic_mlvl_feats_and_img_metas()

    # Ensure img_shape is in the correct format
    for meta in img_metas:
        if "img_shape" in meta and isinstance(meta["img_shape"], list):
            pass  # Already correct
        elif "img_shape" in meta and isinstance(meta["img_shape"], tuple):
            meta["img_shape"] = [meta["img_shape"]] * 6

    # Create PyTorch model
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

    # Preprocess parameters
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        custom_preprocessor=create_custom_preprocessor_petr_head(None),
        device=None,
    )
    parameters = move_to_device(parameters, device)

    # Get PyTorch output
    logger.info("Running PyTorch model...")
    output = torch_model(mlvl_feats, img_metas)
    logger.info(
        f"PyTorch output - cls_scores: {output['all_cls_scores'].shape}, "
        f"bbox_preds: {output['all_bbox_preds'].shape}"
    )

    # Prepare query embedding input
    query_embedding_input = torch_model.reference_points.weight
    query_embedding_input = pos2posemb3d(query_embedding_input)
    query_embedding_input = ttnn.from_torch(query_embedding_input, layout=ttnn.TILE_LAYOUT, device=device)

    # Transformer module preprocessing
    # Use the actual feature dimensions from our generated data
    feat_h, feat_w = mlvl_feats[0].shape[-2:]
    logger.info(f"Using feature dimensions for transformer: H={feat_h}, W={feat_w}")

    child = torch_model.transformer
    x = infer_ttnn_module_args(
        model=child,
        run_model=lambda model: model(
            torch.randn(1, 6, 256, feat_h, feat_w),
            torch.zeros((1, 6, feat_h, feat_w), dtype=torch.bool),
            torch.rand(900, 256),
            torch.rand(1, 6, 256, feat_h, feat_w),
        ),
        device=None,
    )
    assert x is not None
    for key in x.keys():
        x[key].module = getattr(child, key)
    parameters["transformer"] = x

    # Create TTNN model
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

    # Convert features to TTNN format
    logger.info("Converting features to TTNN format...")
    for i in range(len(mlvl_feats)):
        logger.info(f"  Converting level {i}: {mlvl_feats[i].shape}")
        mlvl_feats[i] = ttnn.from_torch(mlvl_feats[i], layout=ttnn.TILE_LAYOUT, device=device)

    # Run TTNN model
    logger.info("Running TTNN model...")
    ttnn_output = ttnn_model(mlvl_feats, img_metas, device=device)

    # Convert outputs back to torch
    ttnn_output["all_cls_scores"] = ttnn.to_torch(ttnn_output["all_cls_scores"])
    ttnn_output["all_bbox_preds"] = ttnn.to_torch(ttnn_output["all_bbox_preds"])

    # Verify outputs
    passed, msg = check_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    passed1, msg1 = check_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)

    logger.info(f"petr_head_cls_scores test: PCC={msg}")
    logger.info(f"petr_head_bbox_preds test: PCC={msg1}")

    assert_with_pcc(output["all_cls_scores"], ttnn_output["all_cls_scores"], pcc=0.99)
    assert_with_pcc(output["all_bbox_preds"], ttnn_output["all_bbox_preds"], pcc=0.99)
