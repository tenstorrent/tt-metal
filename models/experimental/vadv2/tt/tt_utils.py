# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def fold_offset_normalizer_into_weight(weight, bias, W, H, device):
    """Pre-scale a deformable-attention `sampling_offsets` Linear so its output is
    already divided by the offset_normalizer [W, H], eliminating the per-call
    broadcast SFPU DIV (`sampling_offsets / [W, H]`).

    The Linear output is laid out (..., num_points, 2) with the x/y pair
    innermost, so even output channels are x (÷W) and odd are y (÷H). Folding the
    static division into the weight is mathematically exact:
    `s · (Wx + b) == (Wx + b) / normalizer`. Computed once (offset_normalizer is
    static) and cached by the caller.
    """
    out_features = weight.shape[-1]  # preprocess_linear_weight stores (in, out)
    # The x/y pair is the innermost output dim, so out_features must be even for
    # the even=x / odd=y channel split below to align. Guard the layout contract.
    assert out_features % 2 == 0, f"sampling_offsets out_features must be even (x/y pairs), got {out_features}"
    sc = torch.ones(out_features, dtype=torch.float32)
    sc[0::2] = 1.0 / float(W)
    sc[1::2] = 1.0 / float(H)
    sc_t = ttnn.from_torch(sc.reshape(1, out_features), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    w2 = ttnn.mul(weight, sc_t)
    b2 = ttnn.mul(bias, sc_t) if bias is not None else None
    ttnn.deallocate(sc_t)
    return w2, b2


def build_folded_sampling_offsets(sampling_offsets, spatial_shapes, device):
    """Read [W, H] from spatial_shapes (a one-time host sync) and return the
    sampling_offsets Linear (weight, bias) pre-scaled by 1/[W, H] via
    fold_offset_normalizer_into_weight. Shared by the deformable-attention
    modules, which cache the result on their first call."""
    W = int(spatial_shapes[0, 1].item())
    H = int(spatial_shapes[0, 0].item())
    return fold_offset_normalizer_into_weight(sampling_offsets.weight, sampling_offsets.bias, W, H, device)


# Switch the num_levels==1 path over to the fused
# `ttnn.experimental.multi_scale_deformable_attn` op (grid_sample + weighted
# sum in one kernel) once there is enough batched work to amortize its launch.
# The kernel packs 32 queries per tile, so it only wins for large N*Q
# (N = bs*num_heads, Q = num_queries); below this the decomposed grid_sample +
# sum chain is faster. Threshold from the UniAD microbench sweep.
_MSDA_FUSED_MIN_NQ = 1024


def multi_scale_deformable_attn(value, value_spatial_shapes, sampling_locations, attention_weights, device, hw_py=None):
    """multi_scale_deformable_attn.

    Args:
        hw_py: optional list of (H, W) Python ints per level. When provided,
            the reshape on line below skips the `.item()` host syncs that would
            otherwise be needed to read H/W out of the `value_spatial_shapes`
            device tensor. Callers in this codebase populate this from a
            one-time `_hw_cache` they own so warm calls become trace-friendly.
    """
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_levels, num_points, _ = sampling_locations.shape
    value_list = []
    value_list.append(value)
    sampling_locations = ttnn.to_layout(sampling_locations, layout=ttnn.ROW_MAJOR_LAYOUT)
    sampling_grids = 2 * sampling_locations - 1

    # Fused fast path: when num_levels==1 and there's enough batched work,
    # replace the grid_sample + reshape + weighted-sum chain with the single
    # fused `multi_scale_deformable_attn` device op. The op packs grid_sample +
    # weighted sum in one kernel and matches mmcv (align_corners=False).
    if num_levels == 1 and (bs * num_heads) * num_queries >= _MSDA_FUSED_MIN_NQ:
        if hw_py is not None:
            H_int, W_int = hw_py[0]
        else:
            H_int, W_int = int(value_spatial_shapes[0, 0].item()), int(value_spatial_shapes[0, 1].item())

        # value (bs, num_value=H*W, num_heads, D) -> (bs*num_heads, H, W, D) ROW_MAJOR bf16.
        # A single permute moving num_heads next to bs, then (in ROW_MAJOR, where the
        # reshape is a free view) split H*W into H,W. Equivalent bit-for-bit to the old
        # reshape/permute/reshape/permute dance but two fewer materialising ops.
        value_l = ttnn.permute(value, (0, 2, 1, 3))  # (bs, num_heads, H*W, D)
        value_l = ttnn.to_layout(value_l, layout=ttnn.ROW_MAJOR_LAYOUT)
        value_l = ttnn.reshape(value_l, [bs * num_heads, H_int, W_int, embed_dims])

        # sampling_grids (bs, Q, num_heads, 1, num_points, 2) -> (N, Q*P, 1, 2)
        grid = sampling_grids[:, :, :, 0]  # (bs, Q, num_heads, num_points, 2)
        grid = ttnn.permute(grid, (0, 2, 1, 3, 4))  # (bs, num_heads, Q, num_points, 2)
        grid = ttnn.reshape(grid, [bs * num_heads, num_queries * num_points, 1, 2])

        # attention_weights (bs, Q, num_heads, 1, num_points) -> (N, Q, P)
        attn = attention_weights[:, :, :, 0, :]  # (bs, Q, num_heads, num_points)
        attn = ttnn.permute(attn, (0, 2, 1, 3))  # (bs, num_heads, Q, num_points)
        attn = ttnn.reshape(attn, [bs * num_heads, num_queries, num_points])
        attn = ttnn.to_layout(attn, layout=ttnn.ROW_MAJOR_LAYOUT)

        # The fused op requires all three inputs in bf16.
        if value_l.dtype != ttnn.bfloat16:
            value_l = ttnn.typecast(value_l, ttnn.bfloat16)
        if grid.dtype != ttnn.bfloat16:
            grid = ttnn.typecast(grid, ttnn.bfloat16)
        if attn.dtype != ttnn.bfloat16:
            attn = ttnn.typecast(attn, ttnn.bfloat16)

        output = ttnn.experimental.multi_scale_deformable_attn(value_l, grid, attn)  # (N, Q, D)
        ttnn.deallocate(value)
        ttnn.deallocate(sampling_grids)

        output = ttnn.reshape(output, [bs, num_heads, num_queries, embed_dims])
        output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
        output = ttnn.permute(output, (0, 2, 1, 3))
        output = ttnn.reshape(output, [bs, num_queries, num_heads * embed_dims])
        return output

    sampling_value_list = []

    for level, (H_, W_) in enumerate(value_spatial_shapes):
        value_l_ = value_list[level]
        value_l_ = ttnn.reshape(value_l_, [value_l_.shape[0], value_l_.shape[1], value_l_.shape[2] * value_l_.shape[3]])
        value_l_ = ttnn.permute(value_l_, (0, 2, 1))

        if hw_py is not None:
            H_int, W_int = hw_py[level]
        else:
            H_int, W_int = int(H_.item()), int(W_.item())
        value_l_ = ttnn.reshape(value_l_, [bs * num_heads, embed_dims, H_int, W_int])

        sampling_grid_l_ = sampling_grids[:, :, :, level]
        sampling_grid_l_ = ttnn.permute(sampling_grid_l_, (0, 2, 1, 3, 4))

        sampling_grid_l_ = ttnn.reshape(
            sampling_grid_l_,
            [
                sampling_grid_l_.shape[0] * sampling_grid_l_.shape[1],
                sampling_grid_l_.shape[2],
                sampling_grid_l_.shape[3],
                sampling_grid_l_.shape[4],
            ],
        )

        value_l_ = ttnn.permute(value_l_, (0, 2, 3, 1))
        value_l_ = ttnn.to_layout(value_l_, layout=ttnn.ROW_MAJOR_LAYOUT)
        sampling_value_l_ = ttnn.grid_sample(value_l_, sampling_grid_l_)
        ttnn.deallocate(value_l_)
        ttnn.deallocate(sampling_grid_l_)
        sampling_value_l_ = ttnn.permute(sampling_value_l_, (0, 3, 1, 2))

        sampling_value_list.append(sampling_value_l_)

        attention_weights = ttnn.permute(attention_weights, (0, 2, 1, 3, 4))

        attention_weights = ttnn.reshape(attention_weights, [bs * num_heads, 1, num_queries, num_levels * num_points])

    output = ttnn.stack(sampling_value_list, -2)
    ttnn.deallocate(sampling_grids)
    output = ttnn.reshape(
        output, [output.shape[0], output.shape[1], output.shape[2], output.shape[3] * output.shape[4]]
    )

    output = output * attention_weights
    output = ttnn.reallocate(output)
    for val in sampling_value_list:
        ttnn.deallocate(val)
    output = ttnn.sum(output, 3)
    output = ttnn.reallocate(output)
    output = ttnn.reshape(output, [bs, num_heads * embed_dims, num_queries])
    output = ttnn.reallocate(output)
    output = ttnn.permute(output, (0, 2, 1))

    ttnn.deallocate(attention_weights)
    ttnn.deallocate(sampling_value_l_)
    ttnn.deallocate(value)

    return output


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
