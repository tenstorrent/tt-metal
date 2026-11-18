# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical-layout transforms: box pieces must move every element to where the model would hold it at TP=1.

The oracle builds the device-layout array the way the model builds it, per TP shard, and checks the transform
returns the canonical order and back. No device is involved.
"""

import numpy as np
import pytest
from dcp_oracle import canonical_of, device_global, per_shard_fused, world

from ttml.distributed_checkpointing._boxes import Box, boxes_tile
from ttml.distributed_checkpointing._errors import LayoutError
from ttml.distributed_checkpointing._layouts import (
    Identity,
    Pad,
    Segments,
    check_fused_segments,
    declared_transform,
    fused_segments,
    padded_extent,
    transform_from_json,
)
from ttml.distributed_checkpointing._topology import Placement

TP4 = world([Placement.shard(2)], (4,))
DP2_TP1 = world([Placement.replicate(), Placement.shard(2)], (2, 1))  # a TP layer's label on a TP-1 axis
FSDP2_TP4_COLS = world([Placement.shard(3), Placement.shard(2)], (2, 4))  # FSDP's default dim on a TP weight


@pytest.mark.parametrize("sizes, shards", [((4, 2, 2), 2), ((128, 64, 64), 8), ((96, 96), 4), ((8, 8), 1)])
def test_fused_segments_matches_the_per_shard_layout(sizes, shards):
    rows = sum(sizes)
    canonical = np.arange(rows * 3, dtype=np.float32).reshape(1, 1, rows, 3)
    device = per_shard_fused(canonical, sizes, shards, dim=2)

    transform = fused_segments(2, sizes, shards)

    if shards == 1:
        assert isinstance(transform, Identity)
    np.testing.assert_array_equal(canonical_of(device, transform, canonical.shape), canonical)
    np.testing.assert_array_equal(device_global(canonical, transform, device.shape, np.nan), device)


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
    with expect_error(LayoutError, "logical extent 200 must be within"):
        Pad(2, logical=200, padded=192)
    assert isinstance(padded_extent(2, 64, 64), Identity)
    # Zeros is the only fill a load implements, so another is refused where the transform is built or read.
    with expect_error(LayoutError, "pad: fill 'ones' is not one of"):
        Pad(2, 130, 192, "ones")
    with expect_error(LayoutError, "pad: fill 'ones' is not one of"):
        transform_from_json({"kind": "pad", "dim": 2, "logical": 130, "padded": 192, "fill": "ones"})


@pytest.mark.parametrize(
    "transform",
    [
        Identity(),
        Pad(2, 130, 192, "zeros"),
        fused_segments(2, (4, 2, 2), 2),
    ],
)
def test_transforms_round_trip_through_json(transform):
    assert transform_from_json(transform.to_json()) == transform


def test_an_unknown_transform_kind_is_refused(expect_error):
    with expect_error(LayoutError, "unknown transform kind 'rotate'"):
        transform_from_json({"kind": "rotate", "dim": 2})


def test_segments_refuse_boxes_and_shapes_past_their_extent(expect_error):
    seg = fused_segments(2, (4, 2, 2), 2)
    with expect_error(LayoutError, "reaches past the extent 8 on dim 2"):
        seg.to_canonical(Box((0, 0, 4, 0), (1, 1, 6, 4)))
    with expect_error(LayoutError, "global extent 16 on dim 2 is not the map's 8"):
        seg.canonical_shape((1, 1, 16, 4))


def test_declared_transform_from_declarations(expect_error):
    assert isinstance(declared_transform((1, 1, 64, 64), TP4), Identity)
    assert declared_transform((1, 1, 192, 64), TP4, logical_extent=(2, 130)) == Pad(2, 130, 192)
    assert declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16, 16))) == fused_segments(2, (32, 16, 16), 4)
    assert isinstance(declared_transform((1, 1, 64, 32), DP2_TP1, fused=(2, (32, 16, 16))), Identity)
    assert declared_transform((1, 1, 64, 32), FSDP2_TP4_COLS, fused=(2, (32, 16, 16))) == fused_segments(
        2, (32, 16, 16), 4
    ), "FSDP on the columns leaves the fused rows to TP"
    with expect_error(LayoutError, "sum to"):
        declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16)))


def test_declared_dims_count_from_the_end_and_are_range_checked(expect_error):
    # Labels hold non-negative dims; a declared -2 is dim 2, so it must get dim 2's interleave and padding.
    assert declared_transform((1, 1, 64, 32), TP4, fused=(-2, (32, 16, 16))) == fused_segments(2, (32, 16, 16), 4)
    assert declared_transform((1, 1, 192, 64), TP4, logical_extent=(-2, 130)) == Pad(2, 130, 192)
    fsdp2_rows_tp4 = world([Placement.shard(2), Placement.shard(2)], (2, 4))
    with expect_error(LayoutError, "2 mesh axes shard"):
        declared_transform((1, 1, 64, 32), fsdp2_rows_tp4, fused=(-2, (32, 16, 16)))
    with expect_error(LayoutError, r"fused_segments \(dim 2\) and logical_extent \(dim 2\) both declared"):
        declared_transform((1, 1, 64, 32), TP4, fused=(-2, (32, 16, 16)), logical_extent=(2, 60))
    with expect_error(LayoutError, "model/w: fused_segments declares dim 5, out of range for a rank-4 tensor"):
        declared_transform((1, 1, 64, 32), TP4, fused=(5, (32, 16, 16)), key="model/w")
    with expect_error(LayoutError, "logical_extent declares dim -5"):
        declared_transform((1, 1, 64, 32), TP4, logical_extent=(-5, 10))


def test_declaration_errors_name_the_key(expect_error):
    with expect_error(LayoutError, "model/fc/weight: pad: logical extent 200"):
        declared_transform((1, 1, 192, 64), TP4, logical_extent=(2, 200), key="model/fc/weight")
    with expect_error(LayoutError, "model/qkv: fused_segments: segment size 90"):
        declared_transform((1, 1, 96, 64), TP4, fused=(2, (90, 6)), key="model/qkv")


def test_declared_transform_refuses_layouts_it_cannot_describe(expect_error):
    # A weight both fused and padded, along one dim or two: no layer declares one, and no transform describes it.
    with expect_error(LayoutError, r"model/w: fused_segments \(dim 2\) and logical_extent \(dim 2\) both declared"):
        declared_transform((1, 1, 64, 32), TP4, fused=(2, (32, 16, 16)), logical_extent=(2, 60), key="model/w")
    with expect_error(LayoutError, r"fused_segments \(dim 2\) and logical_extent \(dim 3\) both declared"):
        declared_transform((1, 1, 64, 192), TP4, fused=(2, (32, 16, 16)), logical_extent=(3, 130))
    # FSDP sharding the fused rows next to TP, which would make the label's count describe an interleave the weight
    # does not have: at TP 1 (fused_segments would claim 2 shards where the rows are canonical) and at TP 4.
    fsdp2_rows_tp1 = world([Placement.shard(2), Placement.shard(2)], (2, 1))
    with expect_error(LayoutError, "model/qkv: fused_segments on dim 2, which 2 mesh axes shard"):
        declared_transform((1, 1, 64, 32), fsdp2_rows_tp1, fused=(2, (32, 16, 16)), key="model/qkv")
    fsdp2_rows_tp4 = world([Placement.shard(2), Placement.shard(2)], (2, 4))
    with expect_error(LayoutError, "2 mesh axes shard"):
        declared_transform((1, 1, 64, 32), fsdp2_rows_tp4, fused=(2, (32, 16, 16)))


def test_check_fused_segments_is_the_one_check_of_a_fused_declaration(expect_error):
    assert check_fused_segments([128, 64.0, 64], 256, 4) == (128, 64, 64), "sizes come back as ints"
    with expect_error(LayoutError, "sum to 254 but the fused dim has 256 rows"):
        check_fused_segments((128, 64, 62), 256, 2)
    with expect_error(LayoutError, "segment size 66 is not divisible by the 4 shards"):
        check_fused_segments((128, 66, 62), 256, 4)
    for sizes, shards in (((), 1), ((128, 0, 128), 1), ((256,), 0)):
        with expect_error(LayoutError, "bad sizes"):
            check_fused_segments(sizes, 256, shards)
