# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""VAD's NMS-free detection and map coders on device, after BEVFormer's ``TTNMSFreeCoder``.

The top-k selection, the decoding and the ``post_center_range`` test run on device, on fixed
``(1, max_num)`` tensors, with the test as a validity mask. Only dropping the invalid entries runs
on the host (:func:`compact_detections`, :func:`compact_map`): it leaves a data-dependent count.
The heads' logits are bfloat16, so the top-k ranks them in bfloat16; equal logits tie, and any of
them is a valid pick.
"""

import ttnn

# Gathered rows are padded to whole tiles, the width ttnn.embedding copies cleanly.
_TILE = ttnn.TILE_SIZE


def _padded_width(width):
    return -(-width // _TILE) * _TILE


def _top_logits(logits, max_num, num_classes):
    """(1, 1, num_query, num_classes) logits to the top max_num scores (1, max_num) float32 and
    their labels and query indexes (1, max_num) uint32."""
    num_query = logits.shape[-2]
    flat = ttnn.reshape(ttnn.to_layout(logits, ttnn.TILE_LAYOUT), (1, 1, 1, num_query * num_classes))
    if flat.dtype != ttnn.bfloat16:
        flat = ttnn.typecast(flat, ttnn.bfloat16)
    values, indexes = ttnn.topk(flat, k=max_num, dim=-1, stable=True)
    scores = ttnn.sigmoid(ttnn.typecast(values, ttnn.float32))
    indexes = ttnn.typecast(indexes, ttnn.uint32)
    labels = ttnn.remainder(indexes, num_classes)
    query_index = ttnn.typecast(
        ttnn.div(ttnn.typecast(indexes, ttnn.int32), num_classes, rounding_mode="floor"), ttnn.uint32
    )
    return tuple(ttnn.reshape(t, (1, max_num)) for t in (scores, labels, query_index))


def _gather_rows(table, query_index):
    """Rows query_index (1, k) uint32 of the (num_query, width) table, as (1, k, width) float32."""
    num_query, width = table.shape[-2], table.shape[-1]
    table = ttnn.reshape(ttnn.to_layout(table, ttnn.TILE_LAYOUT), (num_query, width))
    if table.dtype != ttnn.bfloat16:
        table = ttnn.typecast(table, ttnn.bfloat16)
    table = ttnn.pad(table, [(0, 0), (0, _padded_width(width) - width)], 0.0)
    rows = ttnn.embedding(
        ttnn.to_layout(query_index, ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(table, ttnn.ROW_MAJOR_LAYOUT)
    )
    rows = ttnn.typecast(ttnn.to_layout(rows, ttnn.TILE_LAYOUT), ttnn.float32)
    return rows[..., :width]


def _in_range(values, low, high):
    """1.0 where every one of values' (1, k, n) last-dim entries lies in [low[i], high[i]]."""
    valid = None
    for i in range(len(low)):
        column = values[..., i : i + 1]
        inside = ttnn.logical_and(ttnn.ge(column, low[i]), ttnn.le(column, high[i]))
        valid = inside if valid is None else ttnn.logical_and(valid, inside)
    return ttnn.reshape(valid, tuple(valid.shape)[:-1])


class TtVADDetectionCoder:
    """``reference.nms_free_coder.CustomNMSFreeCoder`` on device, with ``get_bboxes``' move of z
    to the box's bottom face. VAD sets no score threshold."""

    def __init__(self, pc_range, post_center_range, max_num=100, num_classes=10):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.num_classes = num_classes

    def __call__(self, all_cls_scores, all_bbox_preds, all_traj_preds):
        """The last decoder layer's (L, 1, num_query, ...) outputs to device tensors: scores
        (1, max_num) float32, labels and query indexes (1, max_num) uint32, boxes
        (1, max_num, 9) float32 (cx, cy, z of the bottom face, w, l, h, yaw, vx, vy), trajectories
        (1, max_num, fut_mode * fut_ts * 2) float32 and validity (1, max_num)."""
        num_layers, _, num_query = tuple(all_cls_scores.shape)[:3]
        last = num_layers - 1
        scores, labels, query_index = _top_logits(all_cls_scores[last : last + 1], self.max_num, self.num_classes)
        codes = _gather_rows(all_bbox_preds[last : last + 1], query_index)
        trajs = ttnn.reshape(all_traj_preds, (num_layers, num_query, -1))[last : last + 1]
        trajs = _gather_rows(trajs, query_index)

        # As denormalize_bbox: (cx, cy, w, l, cz, h, sin, cos, vx, vy) codes to boxes.
        cx, cy, w, l, cz, h = (codes[..., i : i + 1] for i in range(6))
        yaw = ttnn.atan2(codes[..., 6:7], codes[..., 7:8])
        w, l, h = ttnn.exp(w), ttnn.exp(l), ttnn.exp(h)
        velocity = codes[..., 8:10]
        valid = _in_range(ttnn.concat([cx, cy, cz], dim=-1), self.post_center_range[:3], self.post_center_range[3:])
        # get_bboxes reports the bottom face's z, after the coder's range test.
        bottom = ttnn.subtract(cz, ttnn.multiply(h, 0.5))
        boxes = ttnn.concat([cx, cy, bottom, w, l, h, yaw, velocity], dim=-1)
        return scores, labels, query_index, boxes, trajs, valid


class TtVADMapCoder:
    """``reference.nms_free_coder.MapNMSFreeCoder`` on device."""

    def __init__(self, pc_range, post_center_range, max_num=50, num_classes=3):
        self.pc_range = pc_range
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.num_classes = num_classes

    def __call__(self, map_all_cls_scores, map_all_bbox_preds, map_all_pts_preds):
        """The last decoder layer's outputs to device tensors: scores (1, max_num) float32, labels
        and query indexes (1, max_num) uint32, boxes (1, max_num, 4) float32 as (xmin, ymin,
        xmax, ymax) in metres, points (1, max_num, num_pts * 2) float32 in metres, and validity
        (1, max_num)."""
        num_layers, _, num_query = tuple(map_all_cls_scores.shape)[:3]
        last = num_layers - 1
        scores, labels, query_index = _top_logits(map_all_cls_scores[last : last + 1], self.max_num, self.num_classes)
        codes = _gather_rows(map_all_bbox_preds[last : last + 1], query_index)
        pts = ttnn.reshape(map_all_pts_preds, (num_layers, num_query, -1))[last : last + 1]
        pts = _gather_rows(pts, query_index)

        x_scale, x_shift = self.pc_range[3] - self.pc_range[0], self.pc_range[0]
        y_scale, y_shift = self.pc_range[4] - self.pc_range[1], self.pc_range[1]
        # As denormalize_2d_bbox: (cx, cy, w, h) to (xmin, ymin, xmax, ymax), then to metres.
        cx, cy, w, h = (codes[..., i : i + 1] for i in range(4))
        half_w, half_h = ttnn.multiply(w, 0.5), ttnn.multiply(h, 0.5)
        corners = [
            (ttnn.subtract(cx, half_w), x_scale, x_shift),
            (ttnn.subtract(cy, half_h), y_scale, y_shift),
            (ttnn.add(cx, half_w), x_scale, x_shift),
            (ttnn.add(cy, half_h), y_scale, y_shift),
        ]
        boxes = ttnn.concat([ttnn.add(ttnn.multiply(c, s), o) for c, s, o in corners], dim=-1)
        valid = _in_range(boxes, self.post_center_range[:4], self.post_center_range[4:])

        # As denormalize_2d_pts, on the points' interleaved (x, y).
        num_values = pts.shape[-1]
        pts = ttnn.reshape(pts, (1, self.max_num, num_values // 2, 2))
        pts = ttnn.concat(
            [
                ttnn.add(ttnn.multiply(pts[..., 0:1], x_scale), x_shift),
                ttnn.add(ttnn.multiply(pts[..., 1:2], y_scale), y_shift),
            ],
            dim=-1,
        )
        pts = ttnn.reshape(pts, (1, self.max_num, num_values))
        return scores, labels, query_index, boxes, pts, valid


def compact_detections(scores, labels, boxes, trajs, valid, fut_mode):
    """The coder's device tensors to the reference decoder's host dict, the valid entries only:
    the one step whose output size depends on the data."""
    scores, labels, boxes, trajs, valid = (ttnn.to_torch(t)[0] for t in (scores, labels, boxes, trajs, valid))
    keep = valid.bool()
    return {
        "bboxes": boxes.float()[keep],
        "scores": scores.float()[keep],
        "labels": labels.long()[keep],
        "trajs": trajs.float()[keep].reshape(-1, fut_mode, trajs.shape[-1] // fut_mode),
    }


def compact_map(scores, labels, boxes, pts, valid):
    """As :func:`compact_detections`, for the map coder."""
    scores, labels, boxes, pts, valid = (ttnn.to_torch(t)[0] for t in (scores, labels, boxes, pts, valid))
    keep = valid.bool()
    return {
        "map_bboxes": boxes.float()[keep],
        "map_scores": scores.float()[keep],
        "map_labels": labels.long()[keep],
        "map_pts": pts.float()[keep].reshape(-1, pts.shape[-1] // 2, 2),
    }
