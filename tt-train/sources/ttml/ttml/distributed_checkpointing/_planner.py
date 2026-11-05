# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plans: which bytes each device file holds on save, and which bytes each destination shard reads on load.

Both plans are pure functions of labels, shapes and transforms, so the resharding math is tested without a device.

Save: every key's label gives one box per device; devices that hold the same box form a replica group, and one of
them, elected by a hash of the key, writes it. A file per device collects the chunks its device owns, largest first,
at offsets fixed before any byte moves. Before that, every key's unique boxes must tile its global shape: a label that
does not (a shard dim out of range, say) is refused at save instead of producing a checkpoint every load rejects.

Load: every destination shard's box is taken to canonical coordinates through the destination transform, back to
the checkpoint's device layout through the inverse of the saved transform, and intersected with the saved chunks.
Each overlap is one read: a sub-box of a chunk copied into a sub-box of the destination shard. When the saved and
destination layouts are the same (same transform and global shape) the boxes meet in device coordinates directly,
so a same-topology resume reads every shard whole, padding rows included. Destination padding rows that no chunk
covers are filled per the transform's ``fill``. The reads and fills must tile every destination shard exactly: the
saved chunks are checked to tile their global shape once per key, after which a shard's reads cannot overlap and
only their total volume is left to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import prod
from typing import Callable, Iterable

from ._boxes import Box, boxes_tile, grid_cell_size
from ._errors import (
    CoverageError,
    DtypeMismatch,
    MissingKeyError,
    RoleMismatch,
    ShapeMismatch,
    TopologyError,
    UnexpectedKeyError,
)
from ._index import ChunkRecord, Entry, Index
from ._layouts import Compose, Pad, Transform
from ._safetensors import ITEMSIZE, build_header
from ._topology import (
    Chunk,
    Label,
    chunk_boxes,
    global_shape as label_global_shape,
    owner as elect_owner,
    replica_groups,
)


@dataclass(frozen=True)
class TensorInfo:
    """What the planner needs to know about one live tensor."""

    key: str
    role: str
    dtype: str  # safetensors dtype name
    layout: str  # TILE | ROW_MAJOR, recorded only
    label: Label
    local_shape: tuple[int, ...]
    transform: Transform
    live_keys: tuple[str, ...] = ()
    param_ref: str | None = None
    state_path: tuple[str, ...] = ()

    @property
    def itemsize(self) -> int:
        return ITEMSIZE[self.dtype]

    @property
    def local_nbytes(self) -> int:
        return prod(self.local_shape) * self.itemsize

    @property
    def global_shape(self) -> tuple[int, ...]:
        return label_global_shape(self.label, self.local_shape)

    @property
    def canonical_shape(self) -> tuple[int, ...]:
        return self.transform.canonical_shape(self.global_shape)


# --- save -----------------------------------------------------------------------------------------------------------


@dataclass
class WriteItem:
    key: str
    chunk: Chunk  # the owner's chunk
    file: int
    nbytes: int
    data_offset: int = 0


@dataclass
class FilePlan:
    index: int
    coord: tuple[int, ...]
    path: str
    items: list[WriteItem] = field(default_factory=list)
    header: bytes = b""
    data_bytes: int = 0


@dataclass
class SavePlan:
    infos: dict[str, TensorInfo]
    files: list[FilePlan]
    items: dict[str, list[WriteItem]]  # per key, one item per unique box, in distribution order of the owner

    @property
    def bytes_written(self) -> int:
        return sum(f.data_bytes for f in self.files)

    def entries(self, crc32: dict[tuple[str, int], int] | None = None) -> dict[str, Entry]:
        """The index entries this plan produces; ``crc32[(key, file)]`` adds checksums when known."""
        entries: dict[str, Entry] = {}
        for key, info in self.infos.items():
            if not self.items.get(key):
                continue  # another rank writes every chunk of this key
            chunks = [
                ChunkRecord(
                    box=item.chunk.box,
                    file=item.file,
                    data_offset=item.data_offset,
                    nbytes=item.nbytes,
                    writer_coord=item.chunk.mesh_coord,
                    crc32=None if crc32 is None else crc32.get((key, item.file)),
                )
                for item in self.items[key]
            ]
            entries[key] = Entry(
                key=key,
                role=info.role,
                dtype=info.dtype,
                layout=info.layout,
                global_shape=info.global_shape,
                canonical_shape=info.canonical_shape,
                transform=info.transform,
                label=info.label,
                live_keys=info.live_keys or (key,),
                chunks=chunks,
                param_ref=info.param_ref,
                state_path=info.state_path,
            )
        return entries


@lru_cache(maxsize=4096)
def _replica_layout(label: Label, local_shape: tuple[int, ...]) -> tuple[dict[tuple[int, ...], tuple[Chunk, ...]], str]:
    """The replica groups a label gives a tensor of ``local_shape`` (each group's chunks as a tuple, so the cached
    value cannot be changed through it), and a description of what is wrong with them, empty when nothing is.

    A key's groups depend only on its label and shape, which do not change during training, and most keys of a model
    share a handful of them; caching here makes every save after the first, and every key after the first of each
    kind, skip the enumeration and the tiling check."""
    rank = len(local_shape)
    for axis, placement in enumerate(label.placements):
        if placement.kind == "shard" and not 0 <= placement.dim < rank:
            return {}, f"distribution axis {axis} shards dim {placement.dim}, out of range for a rank-{rank} tensor"
    groups = {group: tuple(chunks) for group, chunks in replica_groups(chunk_boxes(label, local_shape)).items()}
    boxes = [chunks[0].box for chunks in groups.values()]
    global_shape = label_global_shape(label, local_shape)
    if not boxes_tile(boxes, Box.whole(global_shape)):
        return groups, (
            f"the {len(boxes)} boxes of its label {label.to_json()} do not tile its global shape {list(global_shape)}"
        )
    return groups, ""


def _checked_groups(info: TensorInfo) -> dict[tuple[int, ...], tuple[Chunk, ...]]:
    """The replica groups of a live tensor, after checking that their boxes tile its global shape.

    This is the save-time assertion of the design: every shard dim lies within the tensor's rank, and the unique boxes
    (one per replica group, duplicates kept) tile the global shape exactly. The boxes are cells of one grid, so the
    check is linear in the device count, and it is cached per label and shape (``_replica_layout``).
    """
    groups, problem = _replica_layout(info.label, tuple(info.local_shape))
    if problem:
        error = TopologyError if problem.startswith("distribution axis") else CoverageError
        raise error(f"{info.key}: {problem}")
    return groups


def create_save_plan(
    infos: Iterable[TensorInfo],
    file_for_coord: Callable[[tuple[int, ...]], str],
    file_metadata: Callable[[tuple[int, ...]], dict] | None = None,
    owner_of: Callable[[str, list[Chunk]], Chunk | None] | None = None,
) -> SavePlan:
    """One file per mesh coordinate that writes something; each key's unique boxes assigned to their elected owners.

    ``owner_of(key, replica group)`` overrides the election: it returns the group's chunk this process writes, or
    ``None`` when another rank of a multi-rank world writes it. The default elects among the group's coordinates.
    """
    infos = {info.key: info for info in infos}
    owned: dict[str, list[Chunk]] = {}
    for key, info in infos.items():
        info.canonical_shape  # validates the transform against the global shape (raises LayoutError)
        chosen: list[Chunk] = []
        for group in _checked_groups(info).values():
            own = elect_owner(key, group) if owner_of is None else owner_of(key, group)
            if own is not None:
                chosen.append(own)
        owned[key] = chosen
    coords = sorted({own.mesh_coord for chunks in owned.values() for own in chunks})
    files = [FilePlan(i, c, file_for_coord(c)) for i, c in enumerate(coords)]
    file_of = {c: i for i, c in enumerate(coords)}
    items: dict[str, list[WriteItem]] = {}
    for key, chunks in owned.items():
        per_key: list[WriteItem] = []
        for own in chunks:
            item = WriteItem(key=key, chunk=own, file=file_of[own.mesh_coord], nbytes=infos[key].local_nbytes)
            per_key.append(item)
            files[item.file].items.append(item)
        items[key] = per_key
    for plan in files:
        plan.items.sort(key=lambda it: (-it.nbytes, it.key))
        tensors = [(it.key, infos[it.key].dtype, it.chunk.box.sizes, it.nbytes) for it in plan.items]
        metadata = file_metadata(plan.coord) if file_metadata else None
        plan.header, offsets, plan.data_bytes = build_header(tensors, metadata)
        for it in plan.items:
            it.data_offset = offsets[it.key][0]
    return SavePlan(infos, files, items)


# --- load -----------------------------------------------------------------------------------------------------------


@dataclass
class ReadItem:
    """Copy ``src`` (a box local to chunk ``chunk`` of the entry) into ``dst`` (a box local to the destination shard)."""

    key: str
    file: int
    chunk: int
    src: Box
    dst: Box
    group: tuple[int, ...]


@dataclass
class FillItem:
    key: str
    dst: Box
    group: tuple[int, ...]
    fill: str


@dataclass
class KeyPlan:
    key: str
    entry: Entry
    info: TensorInfo
    boxes: dict[tuple[int, ...], Box]  # destination box per replica group (global coordinates)
    reads: list[ReadItem]
    fills: list[FillItem]
    fast_path: bool  # every destination shard is exactly one saved chunk


@dataclass
class LoadPlan:
    keys: dict[str, KeyPlan]
    missing: list[str]
    unexpected: list[str]

    @property
    def fast_path_keys(self) -> list[str]:
        return [k for k, p in self.keys.items() if p.fast_path]


def _sub_box(outer: Box, inner: Box, inner_parent: Box) -> Box:
    """``inner`` is a sub-box of ``inner_parent``; ``outer`` is ``inner_parent`` in another coordinate system (same
    sizes, translated). Returns the sub-box of ``outer`` that corresponds to ``inner``."""
    delta = tuple(a - b for a, b in zip(inner.offsets, inner_parent.offsets))
    return Box(tuple(o + d for o, d in zip(outer.offsets, delta)), inner.sizes)


def _padding_fill(transform: Transform) -> str:
    if isinstance(transform, Pad):
        return transform.fill
    if isinstance(transform, Compose):
        for step in transform.steps:
            if isinstance(step, Pad):
                return step.fill
    return "zeros"


class _ChunkLookup:
    """The saved chunks of one entry, checked to tile its global shape, and found by the boxes they overlap.

    Chunks that are cells of one grid (every writer here produces such) are found by cell arithmetic; anything else
    is scanned.
    """

    def __init__(self, entry: Entry) -> None:
        self.chunks = entry.chunks
        boxes = [c.box for c in entry.chunks]
        if not boxes_tile(boxes, Box.whole(entry.global_shape)):
            raise CoverageError(
                f"{entry.key}: the checkpoint's {len(boxes)} chunks do not tile its global shape "
                f"{list(entry.global_shape)}; the index is inconsistent"
            )
        self.cell = grid_cell_size(boxes, (0,) * len(entry.global_shape))
        self.by_cell = (
            {tuple(o // s for o, s in zip(c.box.offsets, self.cell)): i for i, c in enumerate(entry.chunks)}
            if self.cell is not None
            else None
        )

    def overlapping(self, box: Box) -> list[int]:
        """Indices of the chunks that may overlap ``box`` (a superset when the chunks are not a grid)."""
        if self.by_cell is None:
            return list(range(len(self.chunks)))
        ranges = [range(o // s, (o + n - 1) // s + 1) for o, n, s in zip(box.offsets, box.sizes, self.cell)]
        found = []
        for cell in product(*ranges):
            index = self.by_cell.get(cell)
            if index is not None:
                found.append(index)
        return found


def plan_key(entry: Entry, info: TensorInfo, local_coords: set[tuple[int, ...]] | None = None) -> KeyPlan:
    """The reads and fills that build every destination shard of one key from the saved chunks.

    With ``local_coords`` (a multi-rank world), only the replica groups with a coordinate this rank holds are
    planned; the other ranks plan theirs.
    """
    if entry.dtype != info.dtype:
        raise DtypeMismatch(f"{info.key}: checkpoint holds {entry.dtype} but the live tensor is {info.dtype}")
    dest_global = info.global_shape
    dest_canonical = info.transform.canonical_shape(dest_global)
    if tuple(entry.canonical_shape) != tuple(dest_canonical):
        raise ShapeMismatch(
            f"{info.key}: canonical shape {list(entry.canonical_shape)} in the checkpoint vs {list(dest_canonical)} in the "
            f"model (global {list(dest_global)} through {info.transform.to_json()}); a different model configuration, "
            f"or a missing layout declaration"
        )
    same_layout = entry.transform == info.transform and tuple(entry.global_shape) == tuple(dest_global)
    groups = _checked_groups(info)
    if local_coords is not None:
        groups = {g: cs for g, cs in groups.items() if any(c.mesh_coord in local_coords for c in cs)}
    boxes = {group: chunks[0].box for group, chunks in groups.items()}
    reads: list[ReadItem] = []
    fills: list[FillItem] = []
    fast = same_layout
    lookup = _ChunkLookup(entry)
    saved_boxes = {c.box: i for i, c in enumerate(entry.chunks)}
    for group, dbox in boxes.items():
        group_reads: list[ReadItem] = []
        group_fills: list[FillItem] = []
        if same_layout:
            pieces = [(dbox, dbox)]
        else:
            pieces = []
            for d_piece, c_piece in info.transform.to_canonical(dbox):
                for c_sub, s_sub in entry.transform.from_canonical(c_piece):
                    pieces.append((_sub_box(d_piece, c_sub, c_piece), s_sub))
            for pad_box in info.transform.padding_boxes(dest_global):
                overlap = pad_box.intersect(dbox)
                if overlap is not None:
                    group_fills.append(
                        FillItem(info.key, overlap.relative_to(dbox), group, _padding_fill(info.transform))
                    )
        if fast and saved_boxes.get(dbox) is None:
            fast = False
        for d_piece, s_piece in pieces:
            for ci in lookup.overlapping(s_piece):
                chunk = entry.chunks[ci]
                overlap = s_piece.intersect(chunk.box)
                if overlap is None:
                    continue
                d_overlap = _sub_box(d_piece, overlap, s_piece)
                group_reads.append(
                    ReadItem(
                        info.key, chunk.file, ci, overlap.relative_to(chunk.box), d_overlap.relative_to(dbox), group
                    )
                )
        # The saved chunks tile their global shape (checked by the lookup) and a transform's pieces partition the
        # box they map (tested with the transforms), so the reads cannot overlap. Every read and fill lies inside the
        # shard by construction (relative_to raises otherwise), which leaves the total volume to check.
        covered = sum(r.dst.volume for r in group_reads) + sum(f.dst.volume for f in group_fills)
        if covered != prod(dbox.sizes):
            raise CoverageError(
                f"{info.key}: the saved chunks do not tile destination shard {dbox} (group {group}); "
                f"{covered} of {prod(dbox.sizes)} elements covered"
            )
        reads.extend(group_reads)
        fills.extend(group_fills)
    return KeyPlan(info.key, entry, info, boxes, reads, fills, fast)


def create_load_plan(
    index: Index,
    infos: Iterable[TensorInfo],
    strict: bool = True,
    roles: Iterable[str] | None = None,
    local_coords: set[tuple[int, ...]] | None = None,
) -> LoadPlan:
    """A ``KeyPlan`` per live tensor. Live names resolve through the entries' recorded live keys.

    What is restored is decided by ``infos``: every live tensor passed is planned. ``roles`` only scopes the
    unexpected-key report: entries whose role is not in it ("param", "buffer", "optimizer_state") are not reported
    as unexpected, so a model alone loads from a checkpoint that also holds optimizer state. ``None`` means every
    role. A live tensor that resolves to an entry of another role is a ``RoleMismatch``.
    """
    live_map = index.live_key_map()
    wanted = None if roles is None else set(roles)
    plans: dict[str, KeyPlan] = {}
    missing: list[str] = []
    used: set[str] = set()
    for info in infos:
        entry_key = live_map.get(info.key)
        if entry_key is None:
            for live in info.live_keys:
                entry_key = live_map.get(live)
                if entry_key is not None:
                    break
        if entry_key is None:
            missing.append(info.key)
            continue
        entry = index.entries[entry_key]
        if entry.role != info.role:
            raise RoleMismatch(
                f"{info.key}: resolves to checkpoint entry {entry_key!r}, a {entry.role!r}, but the live tensor is a "
                f"{info.role!r}"
            )
        used.add(entry_key)
        plans[info.key] = plan_key(entry, info, local_coords)
    unexpected = unexpected_keys(index, used, wanted)
    if strict and missing:
        raise MissingKeyError(f"{len(missing)} live tensor(s) have no entry in the checkpoint: {missing[:5]}")
    if strict and unexpected:
        raise UnexpectedKeyError(f"{len(unexpected)} checkpoint entr(ies) have no live tensor: {unexpected[:5]}")
    return LoadPlan(plans, missing, unexpected)


def unexpected_keys(index: Index, used: set[str], wanted: set[str] | None) -> list[str]:
    """The entries of the wanted roles no live tensor claimed. A multi-rank load judges this over the union of
    the ranks' claims."""
    return [key for key, entry in index.entries.items() if key not in used and (wanted is None or entry.role in wanted)]
