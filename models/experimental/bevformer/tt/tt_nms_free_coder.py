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
    BOX_SIZE,
    BOX_VELOCITY,
    BOX_YAW,
    BOX_Z,
    CENTERS_FIRST,
    CODER_SEARCH_ROUNDS,
    CODE_COS,
    CODE_H,
    CODE_SIN,
    CODE_SIZE,
    CODE_VELOCITY,
    CODE_WL,
    CODE_XY,
    CODE_Z,
    MAX_NUM,
    NUM_CLASSES,
    POST_CENTER_RANGE,
    SCORE_DTYPE,
)

# _gather_rows' padded row width, a whole tile.
_GATHER_WIDTH = ttnn.TILE_SIZE


def _gather_rows(table, rows, columns=None, compute_kernel_config=None):
    """Each sample's rows ``rows`` ``(bs, 1, 1, k)`` uint32 of ``table`` ``(bs, n, width)``, width at
    most ``_GATHER_WIDTH``, as ``(bs, k, width)`` in the table's dtype, exactly.

    ttnn.embedding gathers whole rows in one pass, far faster here than ttnn.gather, but only from a
    bfloat16 table. So a float32 table goes as three bfloat16 parts, each the next 8 significant
    bits of its values, whose float32 sum is the table again. Rows are padded to ``_GATHER_WIDTH``.

    ``columns``, a ``(_GATHER_WIDTH, W)`` bfloat16 0/1 matrix, rearranges the gathered columns:
    each part is multiplied by it, which with ``compute_kernel_config``'s HiFi4 and fp32
    accumulation copies bfloat16 values exactly, and the rows come out ``(bs, k, W)`` float32.
    Moving columns there costs one small matmul where slicing at columns off a tile boundary costs
    an untilize, a slice and a tilize each."""
    bs, n, width = table.shape
    k = rows.shape[-1]
    # Whole tiles both ways, so turning the parts to ROW_MAJOR strips no padding.
    padded_n = -(-n // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    padded = ttnn.pad(
        ttnn.reshape(table, (bs, 1, n, width)), [(0, 0), (0, 0), (0, padded_n - n), (0, _GATHER_WIDTH - width)], 0.0
    )
    rows = ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT)
    samples = []
    for b in range(bs):
        rest = padded if bs == 1 else padded[b : b + 1]
        sample_rows = rows if bs == 1 else rows[b : b + 1]
        out = None
        for _ in range(3):
            part = ttnn.typecast(rest, ttnn.bfloat16)
            rest = ttnn.subtract(rest, ttnn.typecast(part, ttnn.float32))
            gathered = ttnn.embedding(sample_rows, ttnn.to_layout(part, ttnn.ROW_MAJOR_LAYOUT))
            gathered = ttnn.to_layout(gathered, ttnn.TILE_LAYOUT)
            if columns is None:
                gathered = ttnn.typecast(gathered, ttnn.float32)
            else:
                gathered = ttnn.matmul(
                    gathered, columns, dtype=ttnn.float32, compute_kernel_config=compute_kernel_config
                )
            out = gathered if out is None else ttnn.add(out, gathered)
        if columns is None:
            out = out[..., :width]
        samples.append(ttnn.reshape(out, (1, k, out.shape[-1])))
    out = samples[0] if bs == 1 else ttnn.concat(samples, dim=0)
    return out if table.dtype == ttnn.float32 or columns is not None else ttnn.typecast(out, table.dtype)


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
        # The search's counts reach num_query * num_classes, which bfloat16 accumulation would
        # round; float32 holds them exactly. The decoding's column moves need it, and HiFi4, too.
        self._count_config = ttnn.init_device_compute_kernel_config(
            device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        # The decoded box's channels from the CENTERS_FIRST box code, moved by _gather_rows: (cx,
        # cy, cz, w, l, h, sin, vx, vy) in the first tile, w, l and h before their exp and sin
        # before the yaw's atan2, and cos in the second tile under the sin, so the decoding works
        # on whole tiles.
        codes = [*range(CODE_SIZE)]
        decoded = codes[CODE_XY] + codes[CODE_Z] + codes[CODE_WL] + codes[CODE_H] + codes[CODE_SIN]
        decoded += codes[CODE_VELOCITY]
        tile = ttnn.TILE_SIZE
        columns = [0.0] * (_GATHER_WIDTH * 2 * tile)
        for box_channel, code_channel in enumerate(decoded):
            columns[CENTERS_FIRST.index(code_channel) * 2 * tile + box_channel] = 1.0
        columns[CENTERS_FIRST.index(codes[CODE_COS][0]) * 2 * tile + tile + BOX_YAW.start] = 1.0
        self._decode_columns = ttnn.to_device(
            ttnn.Tensor(columns, [_GATHER_WIDTH, 2 * tile], ttnn.bfloat16, ttnn.TILE_LAYOUT), device
        )
        self._size_mask, self._yaw_mask = (
            ttnn.to_device(
                ttnn.Tensor(
                    [float(c in range(channels.start, channels.stop)) for c in range(tile)],
                    [1, 1, tile],
                    ttnn.float32,
                    ttnn.TILE_LAYOUT,
                ),
                device,
            )
            for channels in (BOX_SIZE, BOX_YAW)
        )
        # 1 .. 32 down dim 1: the search's thresholds, in steps of its bracket.
        self._search_steps = ttnn.to_device(
            ttnn.Tensor([float(i) for i in range(1, 33)], [1, 32, 1, 1], SCORE_DTYPE, ttnn.TILE_LAYOUT), device
        )
        # _sort_descending's tie order: an element ranks after the equal ones before it.
        padded_k = -(-max_num // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
        self._earlier = ttnn.to_device(
            ttnn.Tensor(
                [float(j < i) for i in range(padded_k) for j in range(padded_k)],
                [1, 1, padded_k, padded_k],
                ttnn.float32,
                ttnn.TILE_LAYOUT,
            ),
            device,
        )

    def _kth_logit(self, logits):
        """The exact k-th largest of ``logits`` ``(bs, 1, 1, n)``, ``(bs, 1, 1, 1)`` float32. A
        bfloat16 top-k gives it rounded, the pivot, as rounding is monotonic. The k-th logit then
        lies in ``[low, low + 2 * half_width)`` around the pivot, and each round splits that bracket
        into 32 steps and keeps the step holding it: of the thresholds ``low + i * step``, i = 1 ..
        32, those with at least k logits at or above them come first, and ``low`` moves up by their
        number of steps. The last threshold is the bracket's top, which never has k logits."""
        k = self.max_num
        bs = logits.shape[0]
        pivot, _ = ttnn.topk(ttnn.typecast(logits, ttnn.bfloat16), k=k, dim=-1)
        pivot = ttnn.typecast(pivot[..., k - 1 : k], SCORE_DTYPE)
        # Wider than the pivot's bfloat16 rounding step on either side; the floor covers a pivot of 0.
        half_width = ttnn.add(ttnn.multiply(ttnn.abs(pivot), 2.0**-6), 1e-30)
        low = ttnn.subtract(pivot, half_width)
        # The logits as whole tiles, (n / 32, 32), padded with -inf, which no threshold counts, and
        # the 32 thresholds down dim 1: a round's count reduces 32 such tile blocks, which spreads
        # over the cores where one row of n logits per threshold runs on one.
        n = logits.shape[-1]
        tile_area = ttnn.TILE_SIZE * ttnn.TILE_SIZE
        padded_n = -(-n // tile_area) * tile_area
        logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, padded_n - n)], float("-inf"))
        logits = ttnn.reshape(logits, (bs, 1, padded_n // ttnn.TILE_SIZE, ttnn.TILE_SIZE))
        steps = self._search_steps.shape[1]
        for round_ in range(CODER_SEARCH_ROUNDS):
            # The bracket is 2 * half_width and each round divides it by 32: powers of two, exactly.
            step = ttnn.multiply(half_width, 2.0 / steps ** (round_ + 1))
            thresholds = ttnn.add(low, ttnn.multiply(step, self._search_steps))
            count = ttnn.sum(
                ttnn.ge(logits, thresholds), dim=[-2, -1], keepdim=True, compute_kernel_config=self._count_config
            )
            # The counts fall as the thresholds rise, so the thresholds with enough logits come first.
            # Reduced along a row of one tile, where a reduction down dim 1 first fills the padding.
            passed = ttnn.reshape(ttnn.ge(count, float(k)), (bs, 1, 1, steps))
            passed = ttnn.sum(passed, dim=-1, keepdim=True, compute_kernel_config=self._count_config)
            low = ttnn.add(low, ttnn.multiply(step, passed))
        return low

    def _sort_descending(self, values, indexes):
        """``values`` ``(bs, 1, 1, k)`` float32 and their ``indexes`` uint32, sorted by value from the
        largest, equal values in their order, exactly, ROW_MAJOR. ttnn.topk ranks float32 on one
        core; here each value's place is the number of values ahead of it, larger or equal and
        earlier, from a k x k comparison, and a scatter moves it there."""
        k = values.shape[-1]
        padded_k = self._earlier.shape[-1]
        # Padded with -inf, which ranks after every value.
        row = ttnn.pad(values, [(0, 0), (0, 0), (0, 0), (0, padded_k - k)], float("-inf"))
        column = ttnn.transpose(row, -2, -1)
        ahead = ttnn.add(ttnn.gt(row, column), ttnn.multiply(ttnn.eq(row, column), self._earlier))
        place = ttnn.sum(ahead, dim=-1, keepdim=True, compute_kernel_config=self._count_config)
        place = ttnn.typecast(ttnn.transpose(place, -2, -1)[..., :k], ttnn.uint32)
        # Scatter takes float32 in ROW_MAJOR only. Every place is written once, so the input it
        # starts from is overwritten whole. The sorted rows stay ROW_MAJOR: tilizing a single row
        # costs more than the rest of the sort.
        place = ttnn.to_layout(place, ttnn.ROW_MAJOR_LAYOUT)
        return tuple(
            ttnn.scatter(t, -1, place, t) for t in (ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT) for t in (values, indexes))
        )

    def topk(self, cls_scores, bbox_preds):
        """``(bs, num_query, num_classes)`` ``SCORE_DTYPE`` logits and ``(bs, num_query, code_size)``
        box predictions, ``TTBEVFormerHead``'s, in ``CENTERS_FIRST`` order, to each sample's top
        ``max_num`` scores, sorted, as device tensors: scores ``(bs, max_num)`` float32, labels
        ``(bs, max_num)`` uint32 and query indexes ``(bs, max_num)`` uint32, all ROW_MAJOR, and
        boxes ``(bs, max_num, 9)`` in the box predictions' dtype.

        ttnn.topk ranks float32 on a single core, far slower than bfloat16, whose large k takes
        a multi-core route on Blackhole; bfloat16 alone ties many scores. So the logits, which
        rank as their sigmoid scores do, are ranked in three steps:

        1. The exact k-th logit, ``_kth_logit``.
        2. A bfloat16 top-k of the logits minus it: larger logits stay positive, smaller ones
           negative and equal ones zero, so it selects exactly the top-k set; it ties only
           logits equal in float32.
        3. A sort of those k logits, ``_sort_descending``, which carries their indexes along.
        """
        if cls_scores.dtype != SCORE_DTYPE:
            raise ValueError(f"cls_scores must be {SCORE_DTYPE}, got {cls_scores.dtype}")
        bs, num_query, num_classes = cls_scores.shape
        if num_classes != self.num_classes:
            raise ValueError(f"{num_classes} class logits, the coder has {self.num_classes} classes")
        if num_query * num_classes < self.max_num:
            raise ValueError(f"{num_query * num_classes} scores for a top-{self.max_num}")
        # The scores and boxes are tens of KB; in L1 the coder's many small ops, which keep their
        # input's memory, skip DRAM.
        cls_scores, bbox_preds = (ttnn.to_memory_config(t, ttnn.L1_MEMORY_CONFIG) for t in (cls_scores, bbox_preds))
        logits = ttnn.reshape(cls_scores, (bs, 1, 1, num_query * num_classes))
        shifted = ttnn.typecast(ttnn.subtract(logits, self._kth_logit(logits)), ttnn.bfloat16)
        _, selected = ttnn.topk(shifted, k=self.max_num, dim=-1)
        # The row gather takes uint32 indexes.
        selected = ttnn.typecast(selected, ttnn.uint32)
        selected_logits = _gather_rows(ttnn.reshape(logits, (bs, num_query * num_classes, 1)), selected)
        top_logits, indexes = self._sort_descending(ttnn.reshape(selected_logits, (bs, 1, 1, self.max_num)), selected)
        scores = ttnn.sigmoid(top_logits)
        labels = ttnn.remainder(indexes, num_classes)
        # Integer division, exact for any index; the row gather takes uint32 indexes.
        query_index = ttnn.div(ttnn.typecast(indexes, ttnn.int32), num_classes, rounding_mode="floor")
        query_index = ttnn.typecast(query_index, ttnn.uint32)
        boxes = self._decode(_gather_rows(bbox_preds, query_index, self._decode_columns, self._count_config))
        outputs = tuple(ttnn.reshape(t, (bs, self.max_num)) for t in (scores, labels, query_index)) + (boxes,)
        # The caller may hold them into the next frame, whose backbone needs the L1.
        return tuple(ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG) for t in outputs)

    def _decode(self, gathered):
        """``_gather_rows``' box codes under ``self._decode_columns`` to ``(cx, cy, cz, w, l, h, yaw,
        vx, vy)`` boxes, as the reference's ``denormalize_bbox``: w, l and h from their logs, the yaw
        from its sine and cosine."""
        tile = ttnn.TILE_SIZE
        codes, cos = gathered[..., :tile], gathered[..., tile:]
        boxes = ttnn.where(self._size_mask, ttnn.exp(codes), codes)
        boxes = ttnn.where(self._yaw_mask, ttnn.atan2(codes, cos), boxes)
        return boxes[..., : BOX_VELOCITY.stop]

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
        scores = ttnn.to_layout(scores, ttnn.TILE_LAYOUT)
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
