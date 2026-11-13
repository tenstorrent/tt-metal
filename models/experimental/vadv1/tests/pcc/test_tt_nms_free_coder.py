# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The detection and map coders' top-k, decoding and range test on device, against the reference
decoding of the same picks."""

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.vadv1.reference.utils import denormalize_2d_bbox, denormalize_2d_pts, denormalize_bbox
from models.experimental.vadv1.tt.tt_nms_free_coder import TtVADDetectionCoder, TtVADMapCoder
from tests.ttnn.utils_for_testing import assert_with_pcc

PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
POST_CENTER_RANGE = [-20, -35, -10.0, 20, 35, 10.0]
MAP_POST_CENTER_RANGE = [-20, -35, -20, -35, 20, 35, 20, 35]
NUM_LAYERS = 3


def _bf16(tensor):
    """The values the device holds, so both sides decode the same numbers."""
    return tensor.bfloat16().float()


def _to_device(tensor, device):
    return ttnn.from_torch(tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def _in_range(values, low, high):
    return ((values >= torch.tensor(low)) & (values <= torch.tensor(high))).all(-1)


def _check_scores(logits, scores, max_num):
    """The device's scores are the top max_num, whatever order equal logits took."""
    expected = logits.reshape(-1).sigmoid().topk(max_num).values
    actual = scores[0].sort(descending=True).values
    assert torch.allclose(actual, expected, atol=1e-3), "device scores are not the top-k"


def _check_picks(logits, scores, labels, query_index):
    """Each pick's score is the sigmoid of its own (query, label) logit."""
    picked = logits[query_index, labels[0].long()].sigmoid()
    assert torch.allclose(scores[0], picked, atol=1e-3), "a pick's label or query does not match its score"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv1_detection_coder(device, reset_seeds):
    num_query, num_classes, max_num, fut_mode, fut_ts = 300, 10, 100, 6, 6
    cls_scores = _bf16(torch.randn(NUM_LAYERS, 1, num_query, num_classes) * 3)
    bbox_preds = torch.randn(NUM_LAYERS, 1, num_query, 10) * 0.5
    # Centres in metres, some outside post_center_range so the validity mask varies.
    bbox_preds[..., 0] = torch.empty(NUM_LAYERS, 1, num_query).uniform_(-25, 25)
    bbox_preds[..., 1] = torch.empty(NUM_LAYERS, 1, num_query).uniform_(-40, 40)
    bbox_preds[..., 4] = torch.empty(NUM_LAYERS, 1, num_query).uniform_(-12, 12)
    bbox_preds = _bf16(bbox_preds)
    traj_preds = _bf16(torch.randn(NUM_LAYERS, 1, num_query, fut_mode, fut_ts * 2))

    coder = TtVADDetectionCoder(PC_RANGE, POST_CENTER_RANGE, max_num=max_num, num_classes=num_classes)
    outputs = coder(*(_to_device(t, device) for t in (cls_scores, bbox_preds, traj_preds)))
    scores, labels, query_index, boxes, trajs, valid = (ttnn.to_torch(t).float() for t in outputs)

    _check_scores(cls_scores[-1, 0], scores, max_num)
    query_index = query_index[0].long()
    _check_picks(cls_scores[-1, 0], scores, labels, query_index)

    expected_boxes = denormalize_bbox(bbox_preds[-1, 0][query_index], PC_RANGE)
    expected_valid = _in_range(expected_boxes[..., :3], POST_CENTER_RANGE[:3], POST_CENTER_RANGE[3:])
    expected_boxes[:, 2] = expected_boxes[:, 2] - expected_boxes[:, 5] * 0.5
    expected_trajs = traj_preds[-1, 0][query_index].reshape(max_num, -1)

    assert torch.equal(valid[0].bool(), expected_valid), "validity differs from the reference's range test"
    logger.info(f"valid detections: {int(expected_valid.sum())} of {max_num}")
    _, message = assert_with_pcc(expected_boxes, boxes[0], 0.999)
    logger.info(f"boxes: {message}")
    assert torch.allclose(boxes[0][:, :6], expected_boxes[:, :6], rtol=1e-2, atol=1e-2)
    assert torch.equal(trajs[0], expected_trajs), "trajectories differ from the gathered rows"


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
def test_vadv1_map_coder(device, reset_seeds):
    num_query, num_classes, max_num, num_pts = 100, 3, 50, 20
    cls_scores = _bf16(torch.randn(NUM_LAYERS, 1, num_query, num_classes) * 3)
    # Normalized (cx, cy, w, h); some centres past the patch so the range test drops them.
    bbox_preds = torch.rand(NUM_LAYERS, 1, num_query, 4)
    bbox_preds[..., :2] = bbox_preds[..., :2] * 1.6 - 0.3
    bbox_preds = _bf16(bbox_preds)
    pts_preds = _bf16(torch.rand(NUM_LAYERS, 1, num_query, num_pts, 2))

    coder = TtVADMapCoder(PC_RANGE, MAP_POST_CENTER_RANGE, max_num=max_num, num_classes=num_classes)
    outputs = coder(*(_to_device(t, device) for t in (cls_scores, bbox_preds, pts_preds)))
    scores, labels, query_index, boxes, pts, valid = (ttnn.to_torch(t).float() for t in outputs)

    _check_scores(cls_scores[-1, 0], scores, max_num)
    query_index = query_index[0].long()
    _check_picks(cls_scores[-1, 0], scores, labels, query_index)
    expected_boxes = denormalize_2d_bbox(bbox_preds[-1, 0][query_index].clone(), PC_RANGE)
    expected_valid = _in_range(expected_boxes, MAP_POST_CENTER_RANGE[:4], MAP_POST_CENTER_RANGE[4:])
    expected_pts = denormalize_2d_pts(pts_preds[-1, 0][query_index], PC_RANGE).reshape(max_num, -1)

    assert torch.equal(valid[0].bool(), expected_valid), "validity differs from the reference's range test"
    logger.info(f"valid map elements: {int(expected_valid.sum())} of {max_num}")
    assert torch.allclose(boxes[0], expected_boxes, atol=0.1), "map boxes differ from the reference decoding"
    assert torch.allclose(pts[0], expected_pts, atol=0.1), "map points differ from the reference decoding"
    _, message = assert_with_pcc(expected_pts, pts[0], 0.999)
    logger.info(f"map points: {message}")
