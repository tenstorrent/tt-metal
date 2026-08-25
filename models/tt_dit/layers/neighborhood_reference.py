# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Torch reference for 3D neighborhood attention. This is the definition of correct.

Written as a dense masked attention over the whole volume: obviously right, hopelessly slow,
and therefore only usable on test-sized volumes. The device op is judged against it by PCC.

The window rule -- the part that is easy to get subtly wrong -- is that the context window
keeps its SIZE and slides inward at a boundary, rather than truncating. A query at site 0
attends to ``[0, K)``, not to a half-empty ``[0, K/2]``. Consequences worth stating, because
each one is a test:

* every query attends to exactly the same number of keys, edges included;
* every query is inside its own context window;
* at ``context_window >= volume`` this degenerates exactly to full attention.

A truncating window satisfies none of those and looks plausible everywhere except near an
edge, which is where video artifacts live.

Terminology matches ``neighborhood_plan.hpp``: volume, site, context_window, query_group,
stride. Tensors here are in NATURAL order -- see ``neighborhood_permute`` for bricked.
"""

from __future__ import annotations

import math

import torch


def snap_extent(stride_extent: int, brick_extent: int) -> int:
    """The brick extent when the window origin may be snapped to a brick boundary, else 0.

    Transcribed from ``snap_extent_on_axis`` in ``neighborhood_window_rule.hpp``. Legal exactly
    when a whole brick lies inside one query group, so all 32 of its sites share a window.
    """
    if brick_extent == 0 or stride_extent % brick_extent != 0:
        return 0
    return brick_extent


def context_window_origin(
    query_group_index: int,
    stride_extent: int,
    window_extent: int,
    volume_extent: int,
    brick_extent: int = 0,
) -> int:
    """Where one query group's context window starts on one axis.

    Transcribed from ``window_origin_on_axis`` in ``neighborhood_plan.cpp``. Both are checked
    against the same independent search-based oracle in their respective tests, which is what
    keeps them from drifting apart.

    ``brick_extent`` enables brick snapping -- pass what ``snap_extent`` returns, never the
    brick directly. Centring is not the only legal placement: any origin keeping the window in
    bounds and still containing the group attends to a full window with the query inside it, and
    an aligned one makes the gathered region equal the window instead of straddling one extra
    brick per axis. That is worth 2.07x of the gather at a 12^3 window, so the op takes it and
    the reference has to agree about which placement was taken.
    """
    group_first_site = query_group_index * stride_extent
    group_last_site = min(group_first_site + stride_extent - 1, volume_extent - 1)
    group_centre_site = group_first_site + (group_last_site - group_first_site) // 2

    origin = group_centre_site - window_extent // 2
    highest_origin = volume_extent - window_extent
    origin = max(0, min(origin, highest_origin))

    if brick_extent <= 1:
        return origin

    # The window must still contain the whole group, which bounds how far the origin may move.
    lowest_containing = max(0, group_last_site + 1 - window_extent)
    highest_containing = min(group_first_site, highest_origin)

    snapped_down = (origin // brick_extent) * brick_extent
    if snapped_down >= lowest_containing:
        return snapped_down
    snapped_up = snapped_down + brick_extent
    if snapped_up <= highest_containing:
        return snapped_up
    return origin  # no aligned placement contains the group; keep the centred one


def validate(
    volume: tuple[int, int, int],
    context_window: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> None:
    """Reject configurations the device op also rejects, so the two cannot disagree.

    Mirrors ``validate`` in ``neighborhood_plan.cpp``.
    """
    for axis_name, volume_extent, window_extent, stride_extent in zip(
        ("time", "height", "width"), volume, context_window, stride
    ):
        if volume_extent < 1 or window_extent < 1 or stride_extent < 1:
            raise ValueError(f"{axis_name}: volume, context_window and stride must all be positive")
        if stride_extent > window_extent:
            raise ValueError(
                f"{axis_name}: stride {stride_extent} exceeds context_window {window_extent}, so a query "
                f"in the group would fall outside its own window"
            )


def neighborhood_mask(
    volume: tuple[int, int, int],
    context_window: tuple[int, int, int],
    stride: tuple[int, int, int],
    brick: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """``[site_count, site_count]`` bool in natural order; True where a query may attend.

    ``brick`` is the op's layout unit; passing it selects brick-snapped window placement, which
    is what the op does whenever the stride is a whole number of bricks. Leave it None for
    plain centred placement.
    """
    validate(volume, context_window, stride)
    snaps = (0, 0, 0) if brick is None else tuple(map(snap_extent, stride, brick))
    volume_time, volume_height, volume_width = volume
    window_extents = tuple(min(window, extent) for window, extent in zip(context_window, volume))

    site_count = volume_time * volume_height * volume_width
    mask = torch.zeros(site_count, site_count, dtype=torch.bool)

    for query_time in range(volume_time):
        for query_height in range(volume_height):
            for query_width in range(volume_width):
                origins = tuple(
                    context_window_origin(
                        query_site // stride_extent, stride_extent, window_extent, volume_extent, snap
                    )
                    for query_site, stride_extent, window_extent, volume_extent, snap in zip(
                        (query_time, query_height, query_width), stride, window_extents, volume, snaps
                    )
                )
                query_index = (query_time * volume_height + query_height) * volume_width + query_width

                for key_time in range(origins[0], origins[0] + window_extents[0]):
                    for key_height in range(origins[1], origins[1] + window_extents[1]):
                        base = (key_time * volume_height + key_height) * volume_width
                        key_start = base + origins[2]
                        mask[query_index, key_start : key_start + window_extents[2]] = True
    return mask


def neighborhood_attention_3d(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    volume: tuple[int, int, int],
    context_window: tuple[int, int, int],
    stride: tuple[int, int, int] = (1, 1, 1),
    brick: tuple[int, int, int] | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """``[batch, sites, heads, head_dim]`` -> ``[batch, sites, heads, head_dim]``, natural order.

    ``scale`` defaults to ``head_dim ** -0.5``. LTX passes 1.0 because Q arrives pre-scaled.
    """
    batch_count, site_count, head_count, head_dim = query.shape
    expected_site_count = volume[0] * volume[1] * volume[2]
    if site_count != expected_site_count:
        raise ValueError(f"query has {site_count} sites but volume {volume} implies {expected_site_count}")
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    mask = neighborhood_mask(volume, context_window, stride, brick).to(query.device)

    # [batch, heads, sites, head_dim]
    query_by_head = query.permute(0, 2, 1, 3).float()
    key_by_head = key.permute(0, 2, 1, 3).float()
    value_by_head = value.permute(0, 2, 1, 3).float()

    scores = torch.matmul(query_by_head, key_by_head.transpose(-2, -1)) * scale
    scores = scores.masked_fill(~mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)

    attended = torch.matmul(attention, value_by_head)
    return attended.permute(0, 2, 1, 3).to(query.dtype)
