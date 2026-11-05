# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Canonical layout: what makes a checkpoint loadable onto a different parallelism.

Two things about a ttml tensor depend on the parallel configuration it was built for. The vocab dimension is padded
to a multiple of ``32 * tp``, so the global shape of the LM head changes with the TP size. A fused projection
(``[Q; K; V]`` or ``[gate; up]``) is built per TP shard, so its global row order ``[Q_0; K_0; V_0; Q_1; ...]`` changes
with the TP size too. The *canonical* array of a key is the array a single-device, TP=1 instance of the same model
holds, minus tile padding; the checkpoint records, per key, the transform from the saved device layout to it.

Every transform maps an axis-aligned box of one coordinate system to a short list of axis-aligned pieces of the
other, each piece being a translation of the matching sub-box, splitting only at known boundaries. No transform ever
needs a whole tensor, so resharding never materialises one.

The transforms are derived from declarations the fusing layer makes (``fused_segments`` on a column-parallel weight,
``logical_extent`` on a padded vocab weight); see ``declared_transform``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Sequence, Union

from ._boxes import Box
from ._errors import LayoutError

# (box in the coordinate system the call started from, box in the one it maps to); both have the same sizes.
Piece = tuple[Box, Box]


@dataclass(frozen=True)
class Identity:
    """Canonical layout equals device layout: every linear weight, norm, embedding without padding."""

    kind: ClassVar[str] = "identity"

    def to_canonical(self, box: Box) -> list[Piece]:
        return [(box, box)]

    def from_canonical(self, box: Box) -> list[Piece]:
        return [(box, box)]

    def canonical_shape(self, global_shape: Sequence[int]) -> tuple[int, ...]:
        return tuple(global_shape)

    def padding_boxes(self, global_shape: Sequence[int]) -> list[Box]:
        return []

    def to_json(self) -> dict:
        return {"kind": "identity"}


@dataclass(frozen=True)
class Pad:
    """Device rows ``[0, logical)`` along ``dim`` are canonical; rows ``[logical, padded)`` are padding.

    Padding rows are trained (the vocab-parallel loss normalises over them), so they are written to the checkpoint
    and copied 1:1 when the destination has the same padded extent. When it does not, destination padding rows are
    filled per ``fill``.
    """

    kind: ClassVar[str] = "pad"
    dim: int
    logical: int
    padded: int
    fill: str = "zeros"

    def __post_init__(self) -> None:
        if not 0 <= self.logical <= self.padded:
            raise LayoutError(f"pad: logical extent {self.logical} must be within [0, padded={self.padded}]")

    def to_canonical(self, box: Box) -> list[Piece]:
        lo = box.offsets[self.dim]
        hi = min(lo + box.sizes[self.dim], self.logical)
        if hi <= lo:
            return []
        piece = box.with_dim(self.dim, lo, hi - lo)
        return [(piece, piece)]

    def from_canonical(self, box: Box) -> list[Piece]:
        if box.offsets[self.dim] + box.sizes[self.dim] > self.logical:
            raise LayoutError(
                f"pad: canonical box {box} reaches past the logical extent {self.logical} on dim {self.dim}"
            )
        return [(box, box)]

    def canonical_shape(self, global_shape: Sequence[int]) -> tuple[int, ...]:
        shape = list(global_shape)
        if shape[self.dim] != self.padded:
            raise LayoutError(
                f"pad: global extent {shape[self.dim]} on dim {self.dim} is not the declared padded {self.padded}"
            )
        shape[self.dim] = self.logical
        return tuple(shape)

    def padding_boxes(self, global_shape: Sequence[int]) -> list[Box]:
        if self.padded == self.logical:
            return []
        return [Box.whole(global_shape).with_dim(self.dim, self.logical, self.padded - self.logical)]

    def to_json(self) -> dict:
        return {"kind": "pad", "dim": self.dim, "logical": self.logical, "padded": self.padded, "fill": self.fill}


@dataclass(frozen=True)
class Segments:
    """A piecewise-contiguous map along ``dim``: device range ``[d, d+n)`` is canonical range ``[c, c+n)``.

    ``map`` holds ``(device_start, canonical_start, length)`` triples that tile the extent on both sides. Fused
    projections use it: see ``fused_segments``.
    """

    kind: ClassVar[str] = "segments"
    dim: int
    map: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        triples = tuple((int(d), int(c), int(n)) for d, c, n in self.map)
        object.__setattr__(self, "map", triples)
        if not triples:
            raise LayoutError("segments: empty map")
        for side in (0, 1):
            cursor = 0
            for seg in sorted(triples, key=lambda s: s[side]):
                if seg[side] != cursor or seg[2] <= 0:
                    raise LayoutError(
                        f"segments: map does not tile the {'device' if side == 0 else 'canonical'} extent: {triples}"
                    )
                cursor += seg[2]

    @property
    def extent(self) -> int:
        return sum(n for _, _, n in self.map)

    def _split(self, box: Box, src: int, dst: int) -> list[Piece]:
        lo = box.offsets[self.dim]
        hi = lo + box.sizes[self.dim]
        if hi > self.extent:
            raise LayoutError(f"segments: box {box} reaches past the extent {self.extent} on dim {self.dim}")
        pieces: list[Piece] = []
        for seg in sorted(self.map, key=lambda s: s[src]):
            a, b = max(lo, seg[src]), min(hi, seg[src] + seg[2])
            if b > a:
                pieces.append(
                    (box.with_dim(self.dim, a, b - a), box.with_dim(self.dim, seg[dst] + (a - seg[src]), b - a))
                )
        return pieces

    def to_canonical(self, box: Box) -> list[Piece]:
        return self._split(box, 0, 1)

    def from_canonical(self, box: Box) -> list[Piece]:
        return self._split(box, 1, 0)

    def canonical_shape(self, global_shape: Sequence[int]) -> tuple[int, ...]:
        if global_shape[self.dim] != self.extent:
            raise LayoutError(
                f"segments: global extent {global_shape[self.dim]} on dim {self.dim} is not the map's {self.extent}"
            )
        return tuple(global_shape)

    def padding_boxes(self, global_shape: Sequence[int]) -> list[Box]:
        return []

    def to_json(self) -> dict:
        return {"kind": "segments", "dim": self.dim, "map": [list(t) for t in self.map]}


@dataclass(frozen=True)
class Compose:
    """``steps`` applied in order on the way to canonical, in reverse on the way back."""

    kind: ClassVar[str] = "compose"
    steps: tuple["Transform", ...]

    def _chain(self, box: Box, steps: Sequence["Transform"], forward: bool) -> list[Piece]:
        pieces: list[Piece] = [(box, box)]
        for step in steps:
            next_pieces: list[Piece] = []
            for origin, current in pieces:
                for current_sub, mapped in step.to_canonical(current) if forward else step.from_canonical(current):
                    # current_sub is a sub-box of current; the same sub-box of origin is its translation.
                    delta = tuple(a - b for a, b in zip(current_sub.offsets, current.offsets))
                    next_pieces.append(
                        (Box(tuple(o + d for o, d in zip(origin.offsets, delta)), current_sub.sizes), mapped)
                    )
            pieces = next_pieces
        return pieces

    def to_canonical(self, box: Box) -> list[Piece]:
        return self._chain(box, self.steps, forward=True)

    def from_canonical(self, box: Box) -> list[Piece]:
        return self._chain(box, tuple(reversed(self.steps)), forward=False)

    def canonical_shape(self, global_shape: Sequence[int]) -> tuple[int, ...]:
        shape = tuple(global_shape)
        for step in self.steps:
            shape = step.canonical_shape(shape)
        return shape

    def padding_boxes(self, global_shape: Sequence[int]) -> list[Box]:
        # Padding boxes are reported in device coordinates, so only the first step may declare padding: a later
        # step's padding would need mapping back through the steps before it.
        boxes = list(self.steps[0].padding_boxes(global_shape))
        shape = self.steps[0].canonical_shape(global_shape)
        for step in self.steps[1:]:
            if step.padding_boxes(shape):
                raise LayoutError("compose: padding must be declared on the first step")
            shape = step.canonical_shape(shape)
        return boxes

    def to_json(self) -> dict:
        return {"kind": "compose", "steps": [s.to_json() for s in self.steps]}


Transform = Union[Identity, Pad, Segments, Compose]


def compose(*transforms: Transform) -> Transform:
    """The composition of ``transforms`` with identities dropped: Identity, a single transform, or a Compose."""
    steps: list[Transform] = []
    for t in transforms:
        if isinstance(t, Compose):
            steps.extend(t.steps)
        elif not isinstance(t, Identity):
            steps.append(t)
    if not steps:
        return Identity()
    if len(steps) == 1:
        return steps[0]
    return Compose(tuple(steps))


def transform_from_json(value: dict) -> Transform:
    kind = value.get("kind")
    if kind == "identity":
        return Identity()
    if kind == "pad":
        return Pad(int(value["dim"]), int(value["logical"]), int(value["padded"]), str(value.get("fill", "zeros")))
    if kind == "segments":
        return Segments(int(value["dim"]), tuple(tuple(int(x) for x in t) for t in value["map"]))
    if kind == "compose":
        return Compose(tuple(transform_from_json(s) for s in value["steps"]))
    raise LayoutError(f"unknown transform kind {kind!r}")


def fused_segments(dim: int, sizes: Sequence[int], shards: int) -> Transform:
    """The transform of a fused weight whose canonical segments of global ``sizes`` are each split evenly across
    ``shards`` TP shards, shard ``j`` holding piece ``j`` of every segment in segment order.

    With ``sizes = (Q, K, V)`` and two shards the device rows are ``[Q_0; K_0; V_0; Q_1; K_1; V_1]`` and the
    canonical rows ``[Q; K; V]``. One shard is the identity.
    """
    sizes = tuple(int(s) for s in sizes)
    if shards < 1 or any(s <= 0 for s in sizes):
        raise LayoutError(f"fused_segments: bad sizes {sizes} or shard count {shards}")
    if shards == 1:
        return Identity()
    for s in sizes:
        if s % shards:
            raise LayoutError(f"fused_segments: segment size {s} is not divisible by the {shards} shards")
    total = sum(sizes)
    per_shard = total // shards
    canonical_starts = [sum(sizes[:g]) for g in range(len(sizes))]
    triples: list[tuple[int, int, int]] = []
    for j in range(shards):
        device_cursor = j * per_shard
        for g, size in enumerate(sizes):
            length = size // shards
            triples.append((device_cursor, canonical_starts[g] + j * length, length))
            device_cursor += length
    return Segments(dim, tuple(triples))


def padded_extent(dim: int, logical: int, padded: int, fill: str = "zeros") -> Transform:
    """``Pad`` when the extent is padded, the identity when it is not."""
    if padded == logical:
        return Identity()
    return Pad(dim, logical, padded, fill)


def declared_transform(
    global_shape: Sequence[int],
    shard_count_of_dim,
    fused: tuple[int, Sequence[int]] | None = None,
    logical_extent: tuple[int, int] | None = None,
) -> Transform:
    """The transform for a tensor from its declarations and its sharding.

    ``fused`` is ``(dim, segment sizes)`` from a fusing layer; ``logical_extent`` is ``(dim, logical rows)`` from a
    padded vocab weight; ``shard_count_of_dim(dim)`` gives how many shards the live label splits that dim into. The
    declarations are checked against ``global_shape``: a mismatch is a ``LayoutError``, never a silent identity.
    """
    steps: list[Transform] = []
    if logical_extent is not None:
        dim, logical = logical_extent
        steps.append(padded_extent(dim, logical, int(global_shape[dim])))
    if fused is not None:
        dim, sizes = fused
        if sum(sizes) != global_shape[dim]:
            raise LayoutError(
                f"fused_segments {tuple(sizes)} sum to {sum(sizes)} but the weight has {global_shape[dim]} rows on dim {dim}"
            )
        steps.append(fused_segments(dim, sizes, shard_count_of_dim(dim)))
    return compose(*steps)
