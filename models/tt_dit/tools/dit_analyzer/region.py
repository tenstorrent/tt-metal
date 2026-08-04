# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Logical tensor-region algebra.

The analyzer never touches values or buffers; it reasons about *which logical
regions of a tensor live on which device*. A region is a union of axis-aligned
boxes over the tensor's logical axes, e.g.

    rows [0:1024) x cols [0:2560)

Boxes inside a ``RegionSet`` are kept pairwise disjoint so ``volume()`` is a
plain sum, which is what the byte-savings estimates are built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Box:
    """Half-open hyper-rectangle: one ``(lo, hi)`` interval per tensor axis."""

    ranges: Tuple[Tuple[int, int], ...]

    @staticmethod
    def full(shape: Sequence[int]) -> "Box":
        return Box(tuple((0, int(d)) for d in shape))

    @property
    def ndim(self) -> int:
        return len(self.ranges)

    @property
    def is_empty(self) -> bool:
        return any(lo >= hi for lo, hi in self.ranges)

    @property
    def volume(self) -> int:
        if self.is_empty:
            return 0
        v = 1
        for lo, hi in self.ranges:
            v *= hi - lo
        return v

    def intersect(self, other: "Box") -> "Box":
        assert self.ndim == other.ndim, (self, other)
        return Box(tuple((max(a[0], b[0]), min(a[1], b[1])) for a, b in zip(self.ranges, other.ranges)))

    def contains(self, other: "Box") -> bool:
        if other.is_empty:
            return True
        if self.is_empty:
            return False
        return all(s[0] <= o[0] and o[1] <= s[1] for s, o in zip(self.ranges, other.ranges))

    def subtract(self, other: "Box") -> List["Box"]:
        """``self`` minus ``other`` as up to ``2 * ndim`` disjoint boxes."""
        inter = self.intersect(other)
        if inter.is_empty:
            return [self]
        out: List[Box] = []
        cur = list(self.ranges)
        for i in range(self.ndim):
            lo, hi = cur[i]
            ilo, ihi = inter.ranges[i]
            if lo < ilo:
                r = list(cur)
                r[i] = (lo, ilo)
                out.append(Box(tuple(r)))
            if ihi < hi:
                r = list(cur)
                r[i] = (ihi, hi)
                out.append(Box(tuple(r)))
            cur[i] = (ilo, ihi)
        return out

    def replace_axis(self, axis: int, lo: int, hi: int) -> "Box":
        r = list(self.ranges)
        r[axis] = (lo, hi)
        return Box(tuple(r))

    def __str__(self) -> str:
        return "x".join("[%d:%d)" % (lo, hi) for lo, hi in self.ranges)


class RegionSet:
    """A disjoint union of :class:`Box`es over one tensor's logical axes."""

    __slots__ = ("ndim", "boxes")

    def __init__(self, ndim: int, boxes: Iterable[Box] = ()):
        self.ndim = ndim
        kept: List[Box] = []
        for b in boxes:
            assert b.ndim == ndim, (b, ndim)
            if not b.is_empty:
                kept.append(b)
        self.boxes: Tuple[Box, ...] = tuple(_merge_adjacent(_disjoin(kept)))

    # -- constructors ---------------------------------------------------------
    @staticmethod
    def empty(ndim: int) -> "RegionSet":
        return RegionSet(ndim, ())

    @staticmethod
    def full(shape: Sequence[int]) -> "RegionSet":
        return RegionSet(len(shape), [Box.full(shape)])

    @staticmethod
    def of(box: Box) -> "RegionSet":
        return RegionSet(box.ndim, [box])

    @staticmethod
    def shard(shape: Sequence[int], axis: int, index: int, count: int) -> "RegionSet":
        """Even 1-D shard ``index`` of ``count`` along ``axis``."""
        axis = axis % len(shape)
        extent = int(shape[axis])
        step = -(-extent // count)  # ceil, mirrors ttnn's shard padding behaviour
        lo = min(index * step, extent)
        hi = min(lo + step, extent)
        return RegionSet.of(Box.full(shape).replace_axis(axis, lo, hi))

    # -- algebra --------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.boxes

    @property
    def volume(self) -> int:
        return sum(b.volume for b in self.boxes)

    def union(self, other: "RegionSet") -> "RegionSet":
        return RegionSet(self.ndim, list(self.boxes) + list(other.boxes))

    def intersect(self, other: "RegionSet") -> "RegionSet":
        out = [a.intersect(b) for a in self.boxes for b in other.boxes]
        return RegionSet(self.ndim, out)

    def subtract(self, other: "RegionSet") -> "RegionSet":
        cur = list(self.boxes)
        for cut in other.boxes:
            nxt: List[Box] = []
            for b in cur:
                nxt.extend(b.subtract(cut))
            cur = nxt
        return RegionSet(self.ndim, cur)

    def covers(self, other: "RegionSet") -> bool:
        """True when every element of ``other`` is present in ``self``."""
        return other.subtract(self).is_empty

    def bounds(self, axis: int) -> Optional[Tuple[int, int]]:
        """Min/max extent along ``axis`` across all boxes (None if empty)."""
        if self.is_empty:
            return None
        axis = axis % self.ndim
        lo = min(b.ranges[axis][0] for b in self.boxes)
        hi = max(b.ranges[axis][1] for b in self.boxes)
        return (lo, hi)

    def map_axis(self, axis: int, fn) -> "RegionSet":
        """Rewrite the interval on ``axis`` of every box via ``fn(lo, hi)``."""
        axis = axis % self.ndim
        out = []
        for b in self.boxes:
            lo, hi = fn(*b.ranges[axis])
            out.append(b.replace_axis(axis, lo, hi))
        return RegionSet(self.ndim, out)

    def reshaped(self, ndim: int, shape: Sequence[int]) -> "RegionSet":
        """Conservative rank change: keep coverage, drop precision to full."""
        if self.is_empty:
            return RegionSet.empty(ndim)
        return RegionSet.full(shape)

    # -- misc -----------------------------------------------------------------
    def __eq__(self, other) -> bool:
        return isinstance(other, RegionSet) and self.ndim == other.ndim and set(self.boxes) == set(other.boxes)

    def __hash__(self) -> int:
        return hash((self.ndim, frozenset(self.boxes)))

    def describe(self, shape: Optional[Sequence[int]] = None) -> str:
        if self.is_empty:
            return "<none>"
        if shape is not None and self == RegionSet.full(shape):
            return "full " + str(Box.full(shape))
        return " U ".join(str(b) for b in self.boxes)

    def __str__(self) -> str:
        return self.describe()

    def __repr__(self) -> str:
        return "RegionSet(%d, %s)" % (self.ndim, self.describe())


def _disjoin(boxes: Sequence[Box]) -> List[Box]:
    """Make ``boxes`` pairwise disjoint, preserving their union."""
    out: List[Box] = []
    for b in boxes:
        pieces = [b]
        for kept in out:
            nxt: List[Box] = []
            for p in pieces:
                nxt.extend(p.subtract(kept))
            pieces = nxt
            if not pieces:
                break
        out.extend(p for p in pieces if not p.is_empty)
    return out


def _merge_adjacent(boxes: List[Box]) -> List[Box]:
    """Glue boxes that differ on exactly one axis and touch there."""
    cur = list(boxes)
    changed = True
    while changed and len(cur) > 1:
        changed = False
        for i in range(len(cur)):
            for j in range(i + 1, len(cur)):
                merged = _try_merge(cur[i], cur[j])
                if merged is not None:
                    cur = [b for k, b in enumerate(cur) if k not in (i, j)] + [merged]
                    changed = True
                    break
            if changed:
                break
    return sorted(cur, key=lambda b: b.ranges)


def _try_merge(a: Box, b: Box) -> Optional[Box]:
    diff = [i for i in range(a.ndim) if a.ranges[i] != b.ranges[i]]
    if len(diff) != 1:
        return None
    i = diff[0]
    (alo, ahi), (blo, bhi) = a.ranges[i], b.ranges[i]
    if ahi == blo:
        return a.replace_axis(i, alo, bhi)
    if bhi == alo:
        return a.replace_axis(i, blo, ahi)
    return None
