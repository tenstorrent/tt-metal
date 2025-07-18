# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn

from models.experimental.vadv2.reference import nms_free_coder
from models.experimental.vadv2.tt import tt_nms_free_coder

from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 4 * 8192}], indirect=True)
def test_vadv2_mapnmsfreecoder(
    device,
    reset_seeds,
):
    map_bbox_coder = {
        "type": "MapNMSFreeCoder",
        "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
        "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        "max_num": 50,
        "voxel_size": [0.15, 0.15, 4],
        "num_classes": 3,
    }

    torch_map_bbox_coder = nms_free_coder.MapNMSFreeCoder(
        map_bbox_coder["pc_range"],
        voxel_size=map_bbox_coder["voxel_size"],
        post_center_range=map_bbox_coder["post_center_range"],
        max_num=map_bbox_coder["max_num"],
        num_classes=map_bbox_coder["num_classes"],
    )

    tt_map_bbox_coder = tt_nms_free_coder.TtMapNMSFreeCoder(
        map_bbox_coder["pc_range"],
        voxel_size=map_bbox_coder["voxel_size"],
        post_center_range=map_bbox_coder["post_center_range"],
        max_num=map_bbox_coder["max_num"],
        num_classes=map_bbox_coder["num_classes"],
    )

    cls_scores = torch.rand(100, 3)
    bbox_preds = torch.rand(100, 4)
    pts_preds = torch.rand(100, 20, 2)

    output = torch_map_bbox_coder.decode_single(cls_scores, bbox_preds, pts_preds)

    tt_cls_scores = ttnn.from_torch(cls_scores, dtype=ttnn.float32, device=device)
    tt_bbox_preds = ttnn.from_torch(bbox_preds, dtype=ttnn.float32, device=device)
    tt_pts_preds = ttnn.from_torch(pts_preds, dtype=ttnn.float32, device=device)

    logger.info("Testing MapNMSFreeCoder")

    tt_output = tt_map_bbox_coder.decode_single(tt_cls_scores, tt_bbox_preds, tt_pts_preds)

    assert torch.allclose(output["map_bboxes"], tt_output["map_bboxes"])
    assert torch.allclose(output["map_scores"], tt_output["map_scores"])
    assert torch.allclose(output["map_labels"], tt_output["map_labels"])
    assert torch.allclose(output["map_pts"], tt_output["map_pts"])

    assert_with_pcc(map_bbox_coder, tt_nms_free_coder, device)
