# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Length classes: the best SDPA chunk pair and ff1 blocking differ between the ends of the envelope, so
each class freezes one program set; a class reaches the op as a config, never as a shape."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import ttnn


class LengthClass(NamedTuple):
    name: str
    max_rows: int | None  # per-device rows the class serves up to, None = unbounded


# Served 4x8 points: 5 s = 4736 rows/device, 10 s = 9184, 15 s = 13664, capacity 15040.
LENGTH_CLASSES: tuple[LengthClass, ...] = (LengthClass("short", 6144), LengthClass("long", None))


def length_class(rows: int) -> LengthClass:
    """The class serving `rows` per-device rows."""
    for cls in LENGTH_CLASSES:
        if cls.max_rows is None or rows <= cls.max_rows:
            return cls
    raise AssertionError("the last length class must be unbounded")


def uncovered_class_lengths(padded_lengths: Iterable[int], sp_factor: int) -> list[int]:
    """Extra warm-up points so that every length class a request can land in has compiled its programs.

    `padded_lengths` are the packed (all-device) lengths already warmed, normally the two ends of the served
    envelope. A class is covered when one of them falls in it. For each class that lies strictly between the
    shortest and the longest warmed length and that none of them covers, return the first tile-aligned packed
    length of that class, so one short generation at it warms the class. With the two shipped classes the two
    ends always cover both (the shortest request is `short`, the capacity is `long`), so this returns `[]`;
    it only adds points once a class sits between them.
    """
    lengths = sorted(padded_lengths)
    if not lengths:
        return []
    covered = {length_class(n // sp_factor).name for n in lengths}
    tile = ttnn.TILE_SIZE
    out = []
    for previous, cls in zip(LENGTH_CLASSES, LENGTH_CLASSES[1:]):
        # The class starts one row past the previous class's bound; round that to whole tiles on every device.
        first = -(-(previous.max_rows + 1) // tile) * tile * sp_factor
        if cls.name not in covered and lengths[0] < first < lengths[-1]:
            out.append(first)
    return out
