# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Tests for the torch neighborhood-attention reference. Host only, no device.

The reference is what the device op will be judged against, so it has to be right before it
is useful. ``context_window_origin`` here and ``window_origin_on_axis`` in
``neighborhood_plan.cpp`` are two transcriptions of the same rule; each is checked against
the SAME independent, search-derived oracle, which is what stops them drifting apart.
"""

import pytest
import torch

from models.tt_dit.layers.neighborhood_reference import (
    context_window_origin,
    neighborhood_attention_3d,
    neighborhood_mask,
)


def oracle_window_origin(site_index: int, volume_extent: int, context_window_extent: int) -> int:
    """The NATTEN rule by brute force: of every in-bounds placement of a window of the right
    size that contains ``site_index``, take the one most centred on it.

    Derived by search rather than by formula, so it cannot share an off-by-one with the
    implementations it checks. This is the same oracle used by the C++ gtest.
    """
    window_extent = min(context_window_extent, volume_extent)
    best_origin = 0
    best_distance_to_centre = None
    for candidate_origin in range(volume_extent - window_extent + 1):
        if not (candidate_origin <= site_index < candidate_origin + window_extent):
            continue
        distance_to_centre = abs((candidate_origin + window_extent // 2) - site_index)
        if best_distance_to_centre is None or distance_to_centre < best_distance_to_centre:
            best_distance_to_centre = distance_to_centre
            best_origin = candidate_origin
    return best_origin


@pytest.mark.parametrize("volume_extent", [1, 2, 5, 7, 8, 11, 25])
@pytest.mark.parametrize("context_window_extent", [1, 3, 5, 11, 32])
def test_window_origin_matches_oracle_at_stride_one(volume_extent, context_window_extent):
    window_extent = min(context_window_extent, volume_extent)
    for site_index in range(volume_extent):
        assert context_window_origin(site_index, 1, window_extent, volume_extent) == oracle_window_origin(
            site_index, volume_extent, context_window_extent
        ), f"site {site_index} of {volume_extent}, window {context_window_extent}"


@pytest.mark.parametrize(
    "volume, context_window, stride",
    [
        ((4, 6, 6), (3, 3, 3), (1, 1, 1)),
        ((5, 7, 5), (3, 5, 5), (1, 1, 1)),
        ((2, 8, 8), (5, 3, 3), (1, 1, 1)),  # time axis shorter than the window
        ((4, 8, 8), (3, 5, 5), (2, 4, 4)),  # stride == a plausible brick
    ],
)
def test_every_query_attends_to_the_same_count(volume, context_window, stride):
    """The clamping rule's whole point: no query is short-changed at an edge."""
    mask = neighborhood_mask(volume, context_window, stride)
    expected_key_count = 1
    for window_extent, volume_extent in zip(context_window, volume):
        expected_key_count *= min(window_extent, volume_extent)

    key_counts = mask.sum(dim=-1)
    assert torch.all(
        key_counts == expected_key_count
    ), f"expected {expected_key_count} keys per query, saw {key_counts.min().item()}..{key_counts.max().item()}"


@pytest.mark.parametrize(
    "volume, context_window, stride",
    [
        ((4, 6, 6), (3, 3, 3), (1, 1, 1)),
        ((5, 7, 5), (3, 5, 5), (1, 1, 1)),
        ((4, 8, 8), (3, 5, 5), (2, 4, 4)),
    ],
)
def test_every_query_is_inside_its_own_window(volume, context_window, stride):
    mask = neighborhood_mask(volume, context_window, stride)
    assert torch.all(mask.diagonal()), "a query fell outside its own context window"


def test_full_window_degenerates_to_full_attention():
    """At context_window >= volume there is nothing to mask, so NA must equal plain attention."""
    volume = (2, 4, 4)
    site_count = volume[0] * volume[1] * volume[2]
    torch.manual_seed(0)
    query, key, value = (torch.randn(1, site_count, 2, 8) for _ in range(3))

    neighborhood = neighborhood_attention_3d(
        query, key, value, volume=volume, context_window=(8, 8, 8), stride=(1, 1, 1)
    )
    dense = torch.nn.functional.scaled_dot_product_attention(
        query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
    ).permute(0, 2, 1, 3)

    assert torch.allclose(neighborhood, dense, atol=1e-5), "full-window NA diverged from full attention"


def test_rejects_stride_wider_than_the_window(expect_error):
    """Must match ``validate`` in neighborhood_plan.cpp: a group of 4 queries sharing a
    3-wide window would leave a query outside its own context."""
    with expect_error(ValueError, "exceeds context_window"):
        neighborhood_mask((4, 8, 8), (3, 3, 3), (2, 4, 4))


def test_narrow_window_actually_restricts():
    """Guard against a reference that silently attends everywhere -- which would make every
    later PCC test pass for the wrong reason."""
    volume = (1, 1, 16)
    mask = neighborhood_mask(volume, (1, 1, 3), (1, 1, 1))
    assert mask.sum().item() == 16 * 3
    # The middle query sees exactly its three neighbours.
    assert mask[8].nonzero().flatten().tolist() == [7, 8, 9]
    # The edge query's window slid inward instead of truncating.
    assert mask[0].nonzero().flatten().tolist() == [0, 1, 2]
    assert mask[15].nonzero().flatten().tolist() == [13, 14, 15]
