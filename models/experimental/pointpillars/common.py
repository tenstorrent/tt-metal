# # SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# # SPDX-License-Identifier: Apache-2.0

import os

import torch
from models.experimental.pointpillars.reference.mvx_faster_rcnn import MVXFasterRCNN


POINTPILLARS_L1_SMALL_SIZE = 32768


def load_torch_model(model_location_generator=None):
    if model_location_generator is None or "TT_GH_CI_INFRA" not in os.environ:
        ckpt = torch.load(
            "models/experimental/pointpillars/inputs_weights/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        )["state_dict"]

    else:
        weights_path = (
            model_location_generator("<path_in_civ2", model_subdir="", download_if_ci_v2=True)
            / "hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth"
        )
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        state_dict = model.state_dict() if hasattr(model, "state_dict") else ckpt["model"]
    elif isinstance(ckpt, torch.nn.Module):
        state_dict = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unexpected checkpoint format: {type(ckpt)}")

    torch_model = MVXFasterRCNN(
        pts_voxel_encoder=True,
        pts_middle_encoder=True,
        pts_backbone=True,
        pts_neck=True,
        pts_bbox_head=True,
        train_cfg=None,
    )
    torch_model.load_state_dict(state_dict, strict=False)
    torch_model.eval()

    return torch_model
