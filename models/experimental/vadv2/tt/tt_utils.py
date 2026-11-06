# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.matmul_helpers import linear_flatten_batch


def grouped_softmax(logits, group_size):
    """Softmax over each group of group_size columns of TILE logits (..., num_groups * group_size).

    Each group gets its own row so other groups' logits cannot change its distribution.
    """
    grouped = ttnn.reshape(logits, (-1, group_size), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # softmax's additive -inf pad mask cannot clear NaN or +inf left in the tile padding.
    grouped = ttnn.fill_implicit_tile_padding(grouped, 0.0)
    weights = ttnn.softmax(grouped, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(weights, logits.shape)


def heads_packed_offsets(rows, num_heads):
    """Row-major sampling offsets (bs, Q, num_heads * L*P*2) as the fused op reads them: as they are
    when each head's values fill whole alignment units of the buffer, else a page per head."""
    per_head = rows.shape[-1] // num_heads
    in_l1 = rows.memory_config().buffer_type == ttnn.BufferType.L1
    alignment = ttnn.get_l1_alignment() if in_l1 else ttnn.get_dram_alignment()
    if (per_head * rows.element_size()) % alignment == 0:
        return rows
    return ttnn.reshape(rows, (rows.shape[0], rows.shape[1], num_heads, per_head))


def msda_operands(query, params, num_heads, samples_per_group):
    """The fused op's sampling offsets and attention weights from TILE query (bs, Q, C).

    The offsets stay raw, in feature-map pixels: the op scales them per level and adds them to
    the reference points in float32. The softmax runs over each group of samples_per_group
    attention logits, the (level, point) samples one head reads from one map.
    """
    bs, num_queries, _ = query.shape
    sampling_offsets = linear_flatten_batch(query, params.sampling_offsets.weight, bias=params.sampling_offsets.bias)
    sampling_offsets = heads_packed_offsets(
        ttnn.to_layout(sampling_offsets, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG), num_heads
    )
    attention_weights = linear_flatten_batch(query, params.attention_weights.weight, bias=params.attention_weights.bias)
    num_samples = attention_weights.shape[-1] // num_heads
    attention_weights = grouped_softmax(attention_weights, samples_per_group)
    attention_weights = ttnn.reshape(
        ttnn.to_layout(attention_weights, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        (bs, num_queries, num_heads, num_samples),
    )
    return sampling_offsets, attention_weights


def fused_msda(
    value,
    reference_points,
    sampling_offsets,
    attention_weights,
    spatial_shapes_hw,
    reference_mode,
    num_valid_queries_tensor=None,
):
    """Multi-scale deformable attention as one device op, sampling locations formed in float32.

    value (bs, S, C) is the projected value; reference_points (bs, Q, R, 2) in [0, 1]. Returns
    (bs, Q, C) ROW_MAJOR with the heads concatenated. With num_valid_queries_tensor, a (bs,) uint32
    device tensor, only that many rows of each batch are computed; the rest are left unwritten.
    """
    value = ttnn.to_layout(value, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    reference_points = ttnn.to_layout(reference_points, ttnn.ROW_MAJOR_LAYOUT)
    if reference_points.dtype != ttnn.float32:
        reference_points = ttnn.typecast(reference_points, ttnn.float32)
    return ttnn.experimental.fused_msda_from_offsets(
        value,
        reference_points,
        sampling_offsets,
        attention_weights,
        [(int(h), int(w)) for h, w in spatial_shapes_hw],
        reference_mode=reference_mode,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        num_valid_queries_tensor=num_valid_queries_tensor,
    )


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    # `1 - x` computed as a scalar reverse-subtract instead of materialising a
    # `ttnn.ones` tensor: ttnn.ones does a host->device fill that is forbidden
    # inside Metal-Trace capture. ttnn.rsub(x, 1.0) == 1.0 - x, device-only.
    x_temp = ttnn.rsub(x, 1.0)
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def bbox_xyxy_to_cxcywh(bbox):
    bbox = ttnn.unsqueeze(bbox, 0)
    bbox = ttnn.to_layout(bbox, layout=ttnn.ROW_MAJOR_LAYOUT)
    x1, y1, x2, y2 = ttnn.split(bbox, (1, 1, 1, 1), 2)
    bbox_new = [ttnn.div((x1 + x2), 2), ttnn.div((y1 + y2), 2), (x2 - x1), (y2 - y1)]
    return ttnn.concat(bbox_new, dim=-1)


def bbox_cxcywh_to_xyxy(bbox):
    cx, cy, w, h = ttnn.split(bbox, (1, 1, 1, 1), dim=-1)

    bbox_new = [ttnn.mul((cx - 0.5), w), ttnn.mul((cy - 0.5), h), ttnn.mul((cx + 0.5), w), ttnn.mul((cy + 0.5), h)]
    return ttnn.concat(bbox_new, dim=-1)


def tt_denormalize_2d_bbox(bboxes, pc_range):
    bboxes = bbox_cxcywh_to_xyxy(bboxes)

    bboxes_reshaped = ttnn.reshape(bboxes, (bboxes.shape[0], 2, 2))
    bboxes_even = bboxes_reshaped[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    bboxex_odd = bboxes_reshaped[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]

    bboxes_combined = ttnn.stack([bboxes_even, bboxex_odd], dim=-1)
    bboxes = ttnn.reshape(bboxes_combined, [bboxes.shape[0], -1])

    return bboxes


def tt_denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts_slice1 = new_pts[..., 0:1]
    new_pts_slice2 = new_pts[..., 1:2]
    new_pts_slice1 = pts[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    new_pts_slice2 = pts[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    new_pts = ttnn.concat([new_pts_slice1, new_pts_slice2], dim=-1)
    return new_pts
