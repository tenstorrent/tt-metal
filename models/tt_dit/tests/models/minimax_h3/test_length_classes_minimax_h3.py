# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the dynamic-length classes: the class boundary and the warm-up points they imply."""

import pytest

import ttnn

from ....models.transformers.minimax_h3.length_classes import LENGTH_CLASSES, length_class, uncovered_class_lengths

SP = 8
SHORT_MAX = LENGTH_CLASSES[0].max_rows


def test_last_class_is_unbounded():
    assert LENGTH_CLASSES[-1].max_rows is None
    assert all(cls.max_rows is not None for cls in LENGTH_CLASSES[:-1])


@pytest.mark.parametrize(
    ("rows", "name"),
    [
        (1, "short"),
        (4736, "short"),  # 5 s at 768P on the 4x8
        (SHORT_MAX, "short"),
        (SHORT_MAX + 1, "long"),
        (13664, "long"),  # 15 s
        (40832, "long"),  # the ref2va cap
    ],
)
def test_length_class_boundary(rows, name):
    assert length_class(rows).name == name


def test_uncovered_class_lengths_empty():
    assert uncovered_class_lengths([], SP) == []


def test_two_ends_cover_both_classes():
    # The shipped warm-up: the shortest request (short) and the capacity (long) leave nothing to add.
    assert uncovered_class_lengths([SP * 4736, SP * 15040], SP) == []
    assert uncovered_class_lengths([SP * 15040, SP * 4736], SP) == []


def test_ends_in_one_class_add_nothing():
    # Both ends short: no class lies between them. Both ends long: the short class is below them, not between.
    assert uncovered_class_lengths([SP * 256, SP * 4736], SP) == []
    assert uncovered_class_lengths([SP * 9184, SP * 15040], SP) == []


def test_uncovered_middle_class_is_warmed_at_its_first_tile_row(monkeypatch):
    # A three-class table: the two ends land in the outer classes, so the middle one needs its own point,
    # at the first whole tile row past the previous bound, scaled to the packed length.
    from ....models.transformers.minimax_h3 import length_classes as lc

    classes = (lc.LengthClass("a", 1000), lc.LengthClass("b", 9000), lc.LengthClass("c", None))
    monkeypatch.setattr(lc, "LENGTH_CLASSES", classes)
    first_b = -(-(1000 + 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE * SP
    assert first_b == 1024 * SP
    assert lc.uncovered_class_lengths([SP * 512, SP * 15040], SP) == [first_b]
    # Covered once a warmed length lands in it.
    assert lc.uncovered_class_lengths([SP * 512, SP * 4736, SP * 15040], SP) == []
