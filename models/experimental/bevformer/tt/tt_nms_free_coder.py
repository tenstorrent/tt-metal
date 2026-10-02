# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's NMS-free box coder, from UniAD's ``TtNMSFreeCoder``
(``models/experimental/uniad/tt/ttnn_nms_free_coder.py``).

The top-k selection and the box decoding run on device. Only the final range and score
filter (``reference.nms_free_coder.filter_boxes``) runs on host, on the ``max_num`` selected
boxes: it keeps a data-dependent number of them, so it ends the pipeline.
"""

import ttnn
from models.experimental.bevformer.config.decoder_config import (
    CODE_COS,
    CODE_H,
    CODE_SIN,
    CODE_VELOCITY,
    CODE_WL,
    CODE_XY,
    CODE_Z,
)
from models.experimental.bevformer.config.head_config import MAX_NUM, NUM_CLASSES, PC_RANGE, POST_CENTER_RANGE
from models.experimental.bevformer.reference.nms_free_coder import filter_boxes
from models.experimental.bevformer.tt.tt_common import SCORE_DTYPE


def denormalize_bbox(normalized_bboxes):
    """Box predictions (``config/decoder_config.py``'s code, centers in metres) to
    ``(cx, cy, cz, w, l, h, yaw, vx, vy)`` boxes."""
    return ttnn.concat(
        [
            normalized_bboxes[..., CODE_XY],
            normalized_bboxes[..., CODE_Z],
            ttnn.exp(normalized_bboxes[..., CODE_WL]),
            ttnn.exp(normalized_bboxes[..., CODE_H]),
            ttnn.atan2(normalized_bboxes[..., CODE_SIN], normalized_bboxes[..., CODE_COS]),
            normalized_bboxes[..., CODE_VELOCITY],
        ],
        dim=-1,
    )


class TtNMSFreeCoder:
    """BEVFormer's NMS-free box coder with top-k and decoding on device; the arguments are
    ``reference.nms_free_coder.NMSFreeCoder``'s."""

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
        """``(bs, num_query, num_classes)`` ``SCORE_DTYPE`` logits and ``(bs, num_query, code_size)``
        box predictions to each sample's top ``max_num`` scores, sorted, as device tensors: scores
        ``(bs, max_num)`` float32, labels ``(bs, max_num)`` uint32, query indexes ``(bs, max_num, 1)``
        uint32 and boxes ``(bs, max_num, 9)`` in the box predictions' dtype."""
        if cls_scores.dtype != SCORE_DTYPE:
            raise ValueError(f"cls_scores must be {SCORE_DTYPE}, got {cls_scores.dtype}")
        bs, num_query, num_classes = cls_scores.shape
        assert num_classes == self.num_classes, f"{num_classes} class logits, coder has {self.num_classes}"
        # floor_div below emits float32, exact for indexes up to 2**24.
        assert (
            num_query * num_classes <= 2**24
        ), f"{num_query * num_classes} scores, float32 indexes are exact to 2**24"
        scores = ttnn.reshape(ttnn.sigmoid(cls_scores), (bs, 1, 1, num_query * num_classes))
        scores, indexes = ttnn.topk(scores, k=self.max_num, dim=-1)
        labels = ttnn.remainder(indexes, num_classes)
        # gather takes uint32 indexes.
        query_index = ttnn.typecast(ttnn.floor_div(indexes, num_classes), ttnn.uint32)
        query_index = ttnn.reshape(query_index, (bs, self.max_num, 1))
        code_index = ttnn.repeat(query_index, (1, 1, bbox_preds.shape[-1]))
        boxes = denormalize_bbox(ttnn.gather(bbox_preds, 1, code_index))
        return (
            ttnn.reshape(scores, (bs, self.max_num)),
            ttnn.reshape(labels, (bs, self.max_num)),
            query_index,
            boxes,
        )

    def decode(self, all_cls_scores, all_bbox_preds):
        """The head's ``(L, bs, num_query, *)`` device outputs to one dict of host boxes, scores
        and labels per sample, from the last decoder layer."""
        scores, labels, _, boxes = self.topk(all_cls_scores[-1], all_bbox_preds[-1])
        scores, labels, boxes = (ttnn.to_torch(t) for t in (scores, labels, boxes))
        return [
            filter_boxes(*sample, self.post_center_range, self.score_threshold)
            for sample in zip(scores.float(), labels.long(), boxes.float())
        ]
