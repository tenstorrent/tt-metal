# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical-layout transforms: box pieces must move every element to where the model would hold it at TP=1.

The oracle builds the device-layout array the way the model builds it, per TP shard, and checks the transform
returns the canonical order and back. No device is involved.
"""

import numpy as np
import pytest

from ttml.distributed_checkpointing._boxes import Box, boxes_tile
from ttml.distributed_checkpointing._errors import LayoutError
from ttml.distributed_checkpointing._layouts import (
    Compose,
    Identity,
    Pad,
    Segments,
    compose,
    declared_transform,
    fused_segments,
    padded_extent,
    transform_from_json,
)
from ttml.distributed_checkpointing._topology import Label, Placement, dist_coords


def _label(placements, dist_shape) -> Label:
    return Label(tuple(placements), tuple(dist_shape), tuple(dist_coords(dist_shape)))


TP4 = _label([Placement.shard(2)], (4,))
DP2_TP1 = _label([Placement.replicate(), Placement.shard(2)], (2, 1))  # a TP layer's label on a TP-1 axis
FSDP2_TP4_COLS = _label([Placement.shard(3), Placement.shard(2)], (2, 4))  # FSDP's default dim on a TP weight


def _per_shard_fused(canonical: np.ndarray, sizes, shards: int, dim: int) -> np.ndarray:
    """The device-layout array a model with ``shards`` TP shards holds for a weight fused from ``sizes`` segments:
    shard ``j`` holds piece ``j`` of every segment, in segment order, the way ColumnParallelLinear lays it out."""
    starts = np.cumsum((0,) + tuple(sizes[:-1]))
    pieces = []
    for j in range(shards):
        for g, size in enumerate(sizes):
            n = size // shards
            lo = starts[g] + j * n
            pieces.append(np.take(canonical, range(lo, lo + n), axis=dim))
    return np.concatenate(pieces, axis=dim)


def _to_canonical(transform, device: np.ndarray, canonical_shape) -> np.ndarray:
    out = np.full(canonical_shape, np.nan, dtype=device.dtype)
    for dev, can in transform.to_canonical(Box.whole(device.shape)):
        out[can.slices()] = device[dev.slices()]
    return out


def _from_canonical(transform, canonical: np.ndarray, device_shape) -> np.ndarray:
    out = np.full(device_shape, np.nan, dtype=canonical.dtype)
    for can, dev in transform.from_canonical(Box.whole(canonical.shape)):
        out[dev.slices()] = canonical[can.slices()]
    return out


@pytest.mark.parametrize("sizes, shards", [((4, 2, 2), 2), ((128, 64, 64), 8), ((96, 96), 4), ((8, 8), 1)])
def test_fused_segments_matches_the_per_shard_layout(sizes, shards):
    rows = sum(sizes)
    canonical = np.arange(rows * 3, dtype=np.float32).reshape(1, 1, rows, 3)
    device = _per_shard_fused(canonical, sizes, shards, dim=2)

    transform = fused_segments(2, sizes, shards)

    if shards == 1:
        assert isinstance(transform, Identity)
    np.testing.assert_array_equal(_to_canonical(transform, device, canonical.shape), canonical)
    np.testing.assert_array_equal(_from_canonical(transform, canonical, device.shape), device)


def test_fused_segments_refuses_sizes_that_do_not_split(expect_error):
    with expect_error(LayoutError, "not divisible"):
        fused_segments(2, (6, 4), 4)
    with expect_error(LayoutError, "bad sizes"):
        fused_segments(2, (0, 4), 2)


def test_segment_pieces_tile_any_sub_box_and_round_trip():
    transform = fused_segments(2, (16, 8, 8), 4)
    rng = np.random.default_rng(0)
    for _ in range(50):
        lo = int(rng.integers(0, 32))
        size = int(rng.integers(1, 32 - lo + 1))
        box = Box((0, 0, lo, 0), (1, 1, size, 5))
        pieces = transform.to_canonical(box)
        assert boxes_tile([p[0] for p in pieces], box)
        for dev, can in pieces:
            assert dev.sizes == can.sizes
            back = transform.from_canonical(can)
            assert len(back) == 1 and back[0][1] == dev


def test_a_segments_map_that_does_not_tile_names_the_side(expect_error):
    with expect_error(LayoutError, "does not tile the device extent"):
        Segments(dim=2, map=((0, 0, 4), (6, 4, 4)))  # a gap at device rows 4..6
    with expect_error(LayoutError, "does not tile the canonical extent"):
        Segments(dim=2, map=((0, 0, 4), (4, 4, 4), (8, 4, 4)))  # two segments at canonical row 4


def test_pad_clips_to_the_logical_extent_and_reports_padding(expect_error):
    pad = Pad(2, logical=130, padded=192)
    assert pad.canonical_shape((1, 1, 192, 64)) == (1, 1, 130, 64)
    assert pad.padding_boxes((1, 1, 192, 64)) == [Box((0, 0, 130, 0), (1, 1, 62, 64))]

    straddling = Box((0, 0, 96, 0), (1, 1, 96, 64))  # rows 96..192 of a 192-row array
    assert pad.to_canonical(straddling) == [(Box((0, 0, 96, 0), (1, 1, 34, 64)), Box((0, 0, 96, 0), (1, 1, 34, 64)))]
    assert pad.to_canonical(Box((0, 0, 160, 0), (1, 1, 32, 64))) == []
    assert pad.from_canonical(Box((0, 0, 0, 0), (1, 1, 130, 64))) == [(Box((0, 0, 0, 0), (1, 1, 130, 64)),) * 2]
    with expect_error(LayoutError, "past the logical extent"):
        pad.from_canonical(Box((0, 0, 0, 0), (1, 1, 131, 64)))
    with expect_error(LayoutError, "not the declared padded"):
        pad.canonical_shape((1, 1, 160, 64))
    assert isinstance(padded_extent(2, 64, 64), Identity)


def test_compose_pad_then_segments_round_trips(expect_error):
    # A vocab weight both padded and fused would be unusual; the point is that pieces chain through steps.
    transform = compose(Pad(2, logical=24, padded=32), fused_segments(3, (8, 8), 2))
    assert isinstance(transform, Compose)
    canonical = np.arange(24 * 16, dtype=np.float32).reshape(1, 1, 24, 16)
    device = np.full((1, 1, 32, 16), -1.0, dtype=np.float32)
    device[:, :, :24, :] = _per_shard_fused(canonical, (8, 8), 2, dim=3)

    np.testing.assert_array_equal(_to_canonical(transform, device, canonical.shape), canonical)
    back = _from_canonical(transform, canonical, device.shape)
    np.testing.assert_array_equal(back[:, :, :24, :], device[:, :, :24, :])
    assert np.isnan(back[:, :, 24:, :]).all(), "padding rows have no canonical source"
    assert transform.padding_boxes((1, 1, 32, 16)) == [Box((0, 0, 24, 0), (1, 1, 8, 16))]
    with expect_error(LayoutError, "first step"):
        Compose((fused_segments(3, (8, 8), 2), Pad(2, 24, 32))).padding_boxes((1, 1, 32, 16))


def test_compose_drops_identities():
    assert isinstance(compose(Identity(), Identity()), Identity)
    pad = Pad(2, 4, 8)
    assert compose(Identity(), pad) == pad


@pytest.mark.parametrize(
    "transform",
    [
        Identity(),
        Pad(2, 130, 192, "zeros"),
        fused_segments(2, (4, 2, 2), 2),
        compose(Pad(2, 24, 32), fused_segments(3, (8, 8), 2)),
    ],
)
def test_transforms_round_trip_through_json(transform):
    assert transform_from_json(transform.to_json()) == transform


def test_declared_transform_from_declarations(expect_error):
    assert isinstance(declared_transform((1, 1, 64, 64), TP4), Identity)
    assert declared_transform((1, 1, 192, 64), TP4, logical_extent=(2, 130)) == Pad(2, 130, 192)
    assert declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16, 16))) == fused_segments(2, (32, 16, 16), 4)
    assert isinstance(declared_transform((1, 1, 64, 32), DP2_TP1, fused=(2, (32, 16, 16))), Identity)
    assert declared_transform((1, 1, 64, 32), FSDP2_TP4_COLS, fused=(2, (32, 16, 16))) == fused_segments(
        2, (32, 16, 16), 4
    ), "FSDP on the columns leaves the fused rows to TP"
    # Fusion and padding along different dims compose.
    both = declared_transform((1, 1, 64, 192), TP4, fused=(2, (32, 16, 16)), logical_extent=(3, 130))
    assert both == compose(Pad(3, 130, 192), fused_segments(2, (32, 16, 16), 4))
    assert both.canonical_shape((1, 1, 64, 192)) == (1, 1, 64, 130)
    with expect_error(LayoutError, "sum to"):
        declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16)))


def test_declared_transform_refuses_layouts_it_cannot_describe(expect_error):
    # A fused weight padded along its fused dim: every TP shard would carry its own padding inside its segments.
    with expect_error(LayoutError, "both declared on dim 2"):
        declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16, 16)), logical_extent=(2, 60))
    # FSDP sharding the fused rows next to TP, which would make the label's count describe an interleave the weight
    # does not have: at TP 1 (fused_segments would claim 2 shards where the rows are canonical) and at TP 4.
    fsdp2_rows_tp1 = _label([Placement.shard(2), Placement.shard(2)], (2, 1))
    with expect_error(LayoutError, "model/qkv: fused_segments on dim 2, which 2 mesh axes shard"):
        declared_transform((1, 1, 64, 32), fsdp2_rows_tp1, fused=(2, (32, 16, 16)), key="model/qkv")
    fsdp2_rows_tp4 = _label([Placement.shard(2), Placement.shard(2)], (2, 4))
    with expect_error(LayoutError, "2 mesh axes shard"):
        declared_transform((1, 1, 64, 32), fsdp2_rows_tp4, fused=(2, (32, 16, 16)))
