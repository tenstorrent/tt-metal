# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
NMS-free box coder in PyTorch.

This module turns the detection head's last-layer class logits and box predictions into
detections: the top ``max_num`` (query, class) pairs by sigmoid score, their predictions
decoded to ``(cx, cy, cz, w, l, h, yaw, vx, vy)`` and filtered to ``post_center_range``. It
is the reference the TTNN coder in ``tt/tt_nms_free_coder.py`` is checked against.

BEVFormer is trained with one-to-one matching, so the top scores need no NMS. cz stays at
the box's gravity center: upstream ``BEVFormerHead.get_bboxes`` moves it to the bottom face
(``cz - h / 2``) when it builds the LiDAR boxes, after the coder.

Adapted from UniAD's ``NMSFreeCoder`` in ``models/experimental/uniad/reference/nms_free_coder.py``
and BEVFormer's coder and box utilities:
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/coders/nms_free_coder.py
https://github.com/fundamentalvision/BEVFormer/blob/master/projects/mmdet3d_plugin/core/bbox/util.py
"""

import torch

from models.experimental.bevformer.config.decoder_config import (
    CODE_COS,
    CODE_H,
    CODE_SIN,
    CODE_VELOCITY,
    CODE_WL,
    CODE_XY,
    CODE_Z,
)
from models.experimental.bevformer.config.head_config import (
    BOX_CENTER,
    MAX_NUM,
    NUM_CLASSES,
    PC_RANGE,
    POST_CENTER_RANGE,
)


def denormalize_bbox(normalized_bboxes):
    """Box predictions (``config/decoder_config.py``'s code, centers in metres) to
    ``(cx, cy, cz, w, l, h, yaw, vx, vy)`` boxes."""
    return torch.cat(
        [
            normalized_bboxes[..., CODE_XY],
            normalized_bboxes[..., CODE_Z],
            normalized_bboxes[..., CODE_WL].exp(),
            normalized_bboxes[..., CODE_H].exp(),
            torch.atan2(normalized_bboxes[..., CODE_SIN], normalized_bboxes[..., CODE_COS]),
            normalized_bboxes[..., CODE_VELOCITY],
        ],
        dim=-1,
    )


def filter_boxes(scores, labels, boxes, post_center_range, score_threshold=None):
    """One sample's top-k boxes whose centers lie in ``post_center_range`` and, when
    ``score_threshold`` is set, whose score passes it. While no score passes, the threshold
    drops by 10% at a time; once below 0.01, every score passes."""
    post_center_range = torch.tensor(post_center_range)
    centers = boxes[..., BOX_CENTER]
    mask = (centers >= post_center_range[:3]).all(1)
    mask &= (centers <= post_center_range[3:]).all(1)
    if score_threshold is not None:
        thresh_mask = scores > score_threshold
        tmp_score = score_threshold
        while thresh_mask.sum() == 0:
            tmp_score *= 0.9
            if tmp_score < 0.01:
                thresh_mask = scores > -1
                break
            thresh_mask = scores >= tmp_score
        mask &= thresh_mask
    return {"bboxes": boxes[mask], "scores": scores[mask], "labels": labels[mask]}


class NMSFreeCoder:
    """
    BEVFormer's NMS-free box coder: top-k by score, decode, range filter.

    Args:
        pc_range, voxel_size: Unused, accepted so upstream's ``bbox_coder`` config applies.
            Upstream stores them unused too: its ``denormalize_bbox`` ignores ``pc_range``, as
            the head already emits metres.
        post_center_range (tuple[float]): ``(x_min, y_min, z_min, x_max, y_max, z_max)`` box
            centers are kept in.
        max_num (int): Boxes kept per sample, by score.
        score_threshold (float, optional): Minimum score; BEVFormer sets none.
        num_classes (int): Class logits per query.
    """

    def __init__(
        self,
        pc_range=PC_RANGE,
        voxel_size=None,
        post_center_range=POST_CENTER_RANGE,
        max_num=MAX_NUM,
        score_threshold=None,
        num_classes=NUM_CLASSES,
    ):
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def topk(self, cls_scores, bbox_preds):
        """One sample's ``(num_query, num_classes)`` logits and ``(num_query, code_size)`` box predictions
        to the top ``max_num`` scores, sorted, with their labels, query indexes and decoded boxes."""
        scores, indexes = cls_scores.sigmoid().view(-1).topk(self.max_num)
        query_index = indexes // self.num_classes
        return scores, indexes % self.num_classes, query_index, denormalize_bbox(bbox_preds[query_index])

    def decode(self, all_cls_scores, all_bbox_preds):
        """The head's ``(L, bs, num_query, *)`` outputs to one dict of boxes, scores and labels per
        sample, from the last decoder layer."""
        detections = []
        for cls_scores, bbox_preds in zip(all_cls_scores[-1], all_bbox_preds[-1]):
            scores, labels, _, boxes = self.topk(cls_scores, bbox_preds)
            detections.append(filter_boxes(scores, labels, boxes, self.post_center_range, self.score_threshold))
        return detections
