# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Axis-aligned boxes over N-d arrays: the unit the planner reasons in.

A chunk is a box of a tensor's global array; a destination shard is a box; resharding intersects boxes. Nothing here
knows about tensors, devices or files.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Box:
    """``offsets`` and ``sizes`` along every dim of the array the box lives in."""

    offsets: tuple[int, ...]
    sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.sizes):
            raise ValueError(f"Box: {len(self.offsets)} offsets but {len(self.sizes)} sizes")
        object.__setattr__(self, "offsets", tuple(int(o) for o in self.offsets))
        object.__setattr__(self, "sizes", tuple(int(s) for s in self.sizes))
        if any(s < 0 for s in self.sizes) or any(o < 0 for o in self.offsets):
            raise ValueError(f"Box: negative offset or size in {self}")

    @staticmethod
    def whole(shape: Sequence[int]) -> "Box":
        return Box((0,) * len(shape), tuple(shape))

    @property
    def ndim(self) -> int:
        return len(self.sizes)

    @property
    def ends(self) -> tuple[int, ...]:
        return tuple(o + s for o, s in zip(self.offsets, self.sizes))

    @property
    def volume(self) -> int:
        return prod(self.sizes)

    def is_empty(self) -> bool:
        return any(s == 0 for s in self.sizes)

    def intersect(self, other: "Box") -> "Box | None":
        """The overlap of two boxes in the same array, or None when they do not overlap."""
        if other.ndim != self.ndim:
            raise ValueError(f"Box: cannot intersect {self} with {other}")
        offsets, sizes = [], []
        for lo_a, hi_a, lo_b, hi_b in zip(self.offsets, self.ends, other.offsets, other.ends):
            lo, hi = max(lo_a, lo_b), min(hi_a, hi_b)
            if hi <= lo:
                return None
            offsets.append(lo)
            sizes.append(hi - lo)
        return Box(tuple(offsets), tuple(sizes))

    def contains(self, other: "Box") -> bool:
        return all(lo <= o and o + s <= hi for lo, hi, o, s in zip(self.offsets, self.ends, other.offsets, other.sizes))

    def translate(self, delta: Sequence[int]) -> "Box":
        return Box(tuple(o + int(d) for o, d in zip(self.offsets, delta)), self.sizes)

    def relative_to(self, origin: "Box") -> "Box":
        """This box in coordinates local to ``origin`` (which must contain it)."""
        if not origin.contains(self):
            raise ValueError(f"Box: {self} is not inside {origin}")
        return Box(tuple(o - p for o, p in zip(self.offsets, origin.offsets)), self.sizes)

    def with_dim(self, dim: int, offset: int, size: int) -> "Box":
        offsets, sizes = list(self.offsets), list(self.sizes)
        offsets[dim], sizes[dim] = int(offset), int(size)
        return Box(tuple(offsets), tuple(sizes))

    def slices(self) -> tuple[slice, ...]:
        return tuple(slice(o, o + s) for o, s in zip(self.offsets, self.sizes))

    def to_json(self) -> dict:
        return {"offsets": list(self.offsets), "sizes": list(self.sizes)}

    @staticmethod
    def from_json(value: dict) -> "Box":
        return Box(tuple(value["offsets"]), tuple(value["sizes"]))


def boxes_tile(boxes: Iterable[Box], target: Box) -> bool:
    """True when ``boxes`` lie inside ``target``, do not overlap pairwise, and together fill it exactly."""
    boxes = [b for b in boxes if not b.is_empty()]
    if any(not target.contains(b) for b in boxes):
        return False
    if sum(b.volume for b in boxes) != target.volume:
        return False
    for i, a in enumerate(boxes):
        for b in boxes[i + 1 :]:
            if a.intersect(b) is not None:
                return False
    return True
