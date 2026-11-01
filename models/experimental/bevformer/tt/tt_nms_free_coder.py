# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""TTNN port of BEVFormer's NMS-free box coder, from UniAD's ``TtNMSFreeCoder``
(``models/experimental/uniad/tt/ttnn_nms_free_coder.py``).

The top-k selection, the box decoding, the ``post_center_range`` and ``score_threshold`` tests
and upstream ``BEVFormerHead.get_bboxes``' move of z to the box's bottom face run on device, on
fixed ``(bs, max_num)`` tensors, with the tests as a validity mask. Only dropping the invalid boxes
runs on host (:meth:`TTNMSFreeCoder.compact`): it leaves a data-dependent number
of boxes, and ttnn and trace shapes are fixed. With BEVFormer's configs, which set no score
threshold, it drops none: the head scales the centers into ``pc_range``, inside
``post_center_range``.
"""

import ttnn
from models.experimental.bevformer.model_config import (
    BOX_CENTER,
    BOX_H,
    BOX_Z,
    CODER_BISECTION_STEPS,
    CODE_COS,
    CODE_H,
    CODE_SIN,
    CODE_VELOCITY,
    CODE_WL,
    CODE_XY,
    CODE_Z,
    MAX_NUM,
    NUM_CLASSES,
    POST_CENTER_RANGE,
    SCORE_DTYPE,
)


def denormalize_bbox(normalized_bboxes):
    """Box predictions (``model_config.py``'s box code, centers in metres) to
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


class TTNMSFreeCoder:
    """BEVFormer's NMS-free box coder with top-k and decoding on device; the arguments after
    ``device`` are the ones of ``reference.nms_free_coder.NMSFreeCoder`` that decoding reads."""

    def __init__(
        self,
        device,
        post_center_range=POST_CENTER_RANGE,
        max_num=MAX_NUM,
        score_threshold=None,
        num_classes=NUM_CLASSES,
    ):
        self.post_center_range = post_center_range
        # Upstream tests the threshold for truth, so 0.0 applies none.
        self.score_threshold = score_threshold or None
        # Upstream's fallback when no score passes the threshold: it lowers the threshold by 10%
        # at a time and stops at the first one some score reaches, or lets every score pass once
        # it drops below 0.01. The thresholds depend on score_threshold only, so they are
        # computed here, by upstream's own float arithmetic, and the 0.0 that ends them stands
        # for "every score passes", as sigmoid scores are never negative.
        self._fallback = []
        if self.score_threshold is not None:
            threshold = self.score_threshold
            while True:
                threshold *= 0.9
                if threshold < 0.01:
                    break
                self._fallback.append(threshold)
        self._fallback.append(0.0)
        # Uploaded here, so a trace may capture the coder's first call.
        self._fallback_tensor = ttnn.to_device(
            ttnn.Tensor(self._fallback, [1, len(self._fallback)], ttnn.float32, ttnn.TILE_LAYOUT), device
        )
        self.max_num = max_num
        self.num_classes = num_classes
        # The bisection's counts reach num_query * num_classes, which bfloat16 accumulation would
        # round; float32 holds them exactly.
        self._count_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

    def _kth_logit(self, logits):
        """The exact k-th largest of ``logits`` ``(bs, 1, 1, n)``, ``(bs, 1, 1, 1)`` float32. A
        bfloat16 top-k gives it rounded, the pivot, as rounding is monotonic; within the pivot's
        bracket a bisection keeps the largest value with at least k logits at or above it."""
        k = self.max_num
        pivot, _ = ttnn.topk(ttnn.typecast(logits, ttnn.bfloat16), k=k, dim=-1)
        pivot = ttnn.typecast(pivot[..., k - 1 : k], SCORE_DTYPE)
        # Wider than the pivot's bfloat16 rounding step on either side; the floor covers a pivot of 0.
        half_width = ttnn.add(ttnn.multiply(ttnn.abs(pivot), 2.0**-6), 1e-30)
        low, high = ttnn.subtract(pivot, half_width), ttnn.add(pivot, half_width)
        for _ in range(CODER_BISECTION_STEPS):
            middle = ttnn.multiply(ttnn.add(low, high), 0.5)
            count = ttnn.sum(ttnn.ge(logits, middle), dim=-1, keepdim=True, compute_kernel_config=self._count_config)
            enough = ttnn.ge(count, float(k))
            low = ttnn.where(enough, middle, low)
            high = ttnn.where(enough, high, middle)
        return low

    def topk(self, cls_scores, bbox_preds):
        """``(bs, num_query, num_classes)`` ``SCORE_DTYPE`` logits and ``(bs, num_query, code_size)``
        box predictions to each sample's top ``max_num`` scores, sorted, as device tensors: scores
        ``(bs, max_num)`` float32, labels ``(bs, max_num)`` uint32, query indexes ``(bs, max_num)``
        uint32 and boxes ``(bs, max_num, 9)`` in the box predictions' dtype.

        ttnn.topk ranks float32 on a single core, far slower than bfloat16, whose large k takes
        a multi-core route on Blackhole; bfloat16 alone ties many scores. So the logits, which
        rank as their sigmoid scores do, are ranked in three steps:

        1. The exact k-th logit, ``_kth_logit``.
        2. A bfloat16 top-k of the logits minus it: larger logits stay positive, smaller ones
           negative and equal ones zero, so it selects exactly the top-k set; it ties only
           logits equal in float32.
        3. A float32 top-k of those k logits, which sorts them and returns their indexes.
        """
        if cls_scores.dtype != SCORE_DTYPE:
            raise ValueError(f"cls_scores must be {SCORE_DTYPE}, got {cls_scores.dtype}")
        bs, num_query, num_classes = cls_scores.shape
        if num_classes != self.num_classes:
            raise ValueError(f"{num_classes} class logits, the coder has {self.num_classes} classes")
        if num_query * num_classes < self.max_num:
            raise ValueError(f"{num_query * num_classes} scores for a top-{self.max_num}")
        logits = ttnn.reshape(cls_scores, (bs, 1, 1, num_query * num_classes))
        shifted = ttnn.typecast(ttnn.subtract(logits, self._kth_logit(logits)), ttnn.bfloat16)
        _, selected = ttnn.topk(shifted, k=self.max_num, dim=-1)
        # A float32 top-k takes uint32 indexes.
        selected = ttnn.typecast(selected, ttnn.uint32)
        top_logits, indexes = ttnn.topk(
            ttnn.gather(logits, -1, selected), k=self.max_num, dim=-1, indices_tensor=selected
        )
        scores = ttnn.sigmoid(top_logits)
        labels = ttnn.remainder(indexes, num_classes)
        # Integer division, exact for any index; gather takes uint32 indexes.
        query_index = ttnn.div(ttnn.typecast(indexes, ttnn.int32), num_classes, rounding_mode="floor")
        query_index = ttnn.reshape(ttnn.typecast(query_index, ttnn.uint32), (bs, self.max_num, 1))
        code_index = ttnn.repeat(query_index, (1, 1, bbox_preds.shape[-1]))
        boxes = denormalize_bbox(ttnn.gather(bbox_preds, 1, code_index))
        return tuple(ttnn.reshape(t, (bs, self.max_num)) for t in (scores, labels, query_index)) + (boxes,)

    def valid(self, scores, boxes):
        """``(bs, max_num)`` validity of :meth:`topk`'s scores and boxes, 1.0 or 0.0, on device: the
        box center inside ``post_center_range`` and, when set, the score above ``score_threshold``
        or, when no score of the sample is, at or above the fallback threshold."""
        valid = None
        for axis in range(3):
            center = boxes[..., BOX_CENTER.start + axis : BOX_CENTER.start + axis + 1]
            in_range = ttnn.logical_and(
                ttnn.ge(center, self.post_center_range[axis]), ttnn.le(center, self.post_center_range[3 + axis])
            )
            valid = in_range if valid is None else ttnn.logical_and(valid, in_range)
        valid = ttnn.reshape(valid, tuple(valid.shape)[:-1])
        if self.score_threshold is None:
            return valid
        # The scores are sorted, so some score passes a threshold exactly when the first does.
        top_score = scores[:, 0:1]
        # The first fallback threshold the top score reaches: they decrease, so it is the largest
        # one at or below it; 0.0, every score passing, when none above 0.01 is.
        reached = ttnn.where(ttnn.le(self._fallback_tensor, top_score), self._fallback_tensor, 0.0)
        fallback_threshold = ttnn.max(reached, dim=-1, keepdim=True)
        passes = ttnn.where(
            ttnn.gt(top_score, self.score_threshold),
            ttnn.gt(scores, self.score_threshold),
            ttnn.ge(scores, fallback_threshold),
        )
        return ttnn.logical_and(valid, passes)

    @staticmethod
    def bottom_center(boxes):
        """Boxes with z moved from the box's gravity center to its bottom face, as upstream
        ``BEVFormerHead.get_bboxes`` sets it. On device."""
        bottom = ttnn.subtract(boxes[..., BOX_Z], ttnn.multiply(boxes[..., BOX_H], 0.5))
        return ttnn.concat([boxes[..., : BOX_Z.start], bottom, boxes[..., BOX_Z.stop :]], dim=-1)

    def finalize(self, scores, boxes):
        """:meth:`topk`'s scores and boxes to upstream ``get_bboxes``' boxes and their validity,
        tested before the move, as upstream's coder does."""
        return self.bottom_center(boxes), self.valid(scores, boxes)

    @staticmethod
    def compact(scores, labels, boxes, valid):
        """:meth:`topk`'s scores and labels, :meth:`topk`'s or :meth:`finalize`'s boxes and
        :meth:`valid`'s mask to one dict of host boxes, scores and labels per sample, the valid
        ones, as
        ``reference.nms_free_coder.NMSFreeCoder.decode`` returns them: the one step whose output
        size depends on the data."""
        scores, labels, boxes, valid = (ttnn.to_torch(t) for t in (scores, labels, boxes, valid))
        return [
            {"bboxes": b.float()[v.bool()], "scores": s.float()[v.bool()], "labels": lab.long()[v.bool()]}
            for s, lab, b, v in zip(scores, labels, boxes, valid)
        ]
