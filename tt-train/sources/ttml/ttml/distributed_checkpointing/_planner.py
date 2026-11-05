# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Plans: which bytes each device file holds on save, and which bytes each destination shard reads on load.

Both plans are pure functions of labels, shapes and transforms, so the resharding math is tested without a device.

Save: every key's label gives one box per device; devices that hold the same box form a replica group, and one of
them, elected by a hash of the key, writes it. A file per device collects the chunks its device owns, largest first,
at offsets fixed before any byte moves.

Load: every destination shard's box is taken to canonical coordinates through the destination transform, back to
the checkpoint's device layout through the inverse of the saved transform, and intersected with the saved chunks.
Each overlap is one read: a sub-box of a chunk copied into a sub-box of the destination shard. When the saved and
destination layouts are the same (same transform and global shape) the boxes meet in device coordinates directly,
so a same-topology resume reads every shard whole, padding rows included. Destination padding rows that no chunk
covers are filled per the transform's ``fill``. The reads and fills must tile every destination shard exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Callable, Iterable

from ._boxes import Box, boxes_tile
from ._errors import CoverageError, DtypeMismatch, MissingKeyError, ShapeMismatch, UnexpectedKeyError
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


def create_save_plan(
    infos: Iterable[TensorInfo],
    file_for_coord: Callable[[tuple[int, ...]], str],
    file_metadata: Callable[[tuple[int, ...]], dict] | None = None,
) -> SavePlan:
    """One file per mesh coordinate any tensor lives on; each key's unique boxes assigned to their elected owners."""
    infos = {info.key: info for info in infos}
    coords = sorted({c for info in infos.values() for c in info.label.mesh_coords})
    files = [FilePlan(i, c, file_for_coord(c)) for i, c in enumerate(coords)]
    file_of = {c: i for i, c in enumerate(coords)}
    items: dict[str, list[WriteItem]] = {}
    for key, info in infos.items():
        info.canonical_shape  # validates the transform against the global shape (raises LayoutError)
        per_key: list[WriteItem] = []
        for group in replica_groups(chunk_boxes(info.label, info.local_shape)).values():
            own = elect_owner(key, group)
            item = WriteItem(key=key, chunk=own, file=file_of[own.mesh_coord], nbytes=info.local_nbytes)
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

    @property
    def nbytes_needed(self) -> int:
        return self.src.volume


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


def plan_key(entry: Entry, info: TensorInfo) -> KeyPlan:
    """The reads and fills that build every destination shard of one key from the saved chunks."""
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
    groups = replica_groups(chunk_boxes(info.label, info.local_shape))
    boxes = {group: chunks[0].box for group, chunks in groups.items()}
    reads: list[ReadItem] = []
    fills: list[FillItem] = []
    fast = same_layout
    saved_boxes = {c.box: i for i, c in enumerate(entry.chunks)}
    for group, dbox in boxes.items():
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
                    fills.append(FillItem(info.key, overlap.relative_to(dbox), group, _padding_fill(info.transform)))
        if fast and saved_boxes.get(dbox) is None:
            fast = False
        for d_piece, s_piece in pieces:
            for ci, chunk in enumerate(entry.chunks):
                overlap = s_piece.intersect(chunk.box)
                if overlap is None:
                    continue
                d_overlap = _sub_box(d_piece, overlap, s_piece)
                reads.append(
                    ReadItem(
                        info.key, chunk.file, ci, overlap.relative_to(chunk.box), d_overlap.relative_to(dbox), group
                    )
                )
        local = Box.whole(dbox.sizes)
        covered = [r.dst for r in reads if r.group == group] + [f.dst for f in fills if f.group == group]
        if not boxes_tile(covered, local):
            raise CoverageError(
                f"{info.key}: the saved chunks do not tile destination shard {dbox} (group {group}); "
                f"{sum(b.volume for b in covered)} of {local.volume} elements covered"
            )
    return KeyPlan(info.key, entry, info, boxes, reads, fills, fast)


def create_load_plan(
    index: Index, infos: Iterable[TensorInfo], strict: bool = True, roles: Iterable[str] | None = None
) -> LoadPlan:
    """A ``KeyPlan`` per live tensor. Live names resolve through the entries' recorded live keys.

    ``roles`` are the entry roles the caller asked to restore ("param", "buffer", "optimizer_state"); entries of other
    roles are neither planned nor reported as unexpected, so a model can be loaded from a checkpoint that also holds
    optimizer state. ``None`` means every role.
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
        used.add(entry_key)
        plans[info.key] = plan_key(index.entries[entry_key], info)
    unexpected = [
        key for key, entry in index.entries.items() if key not in used and (wanted is None or entry.role in wanted)
    ]
    if strict and missing:
        raise MissingKeyError(f"{len(missing)} live tensor(s) have no entry in the checkpoint: {missing[:5]}")
    if strict and unexpected:
        raise UnexpectedKeyError(f"{len(unexpected)} checkpoint entr(ies) have no live tensor: {unexpected[:5]}")
    return LoadPlan(plans, missing, unexpected)
