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
covers are fills, which load as zeros. The reads and fills must tile every destination shard exactly: the
saved chunks are checked to tile their global shape once per key, after which a shard's reads cannot overlap and
only their total volume is left to check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from math import prod
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from ._boxes import Box, boxes_tile, grid_cell_size
from ._errors import (
    CheckpointError,
    CoverageError,
    DtypeMismatchError,
    RoleMismatchError,
    ShapeMismatchError,
    TopologyError,
)
from ._index import FORMAT, VERSION, ChunkRecord, Entry, Index
from ._layouts import Transform
from ._safetensors import ITEMSIZE, build_header
from ._topology import (
    Chunk,
    Label,
    chunk_boxes,
    global_shape as label_global_shape,
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
    tp_split: tuple[int, ...] = field(kw_only=True)  # recorded in the index entry; see ``Entry.tp_split``

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

    def entries(self, crc32: dict[tuple[str, int], int]) -> dict[str, Entry]:
        """The index entries this plan produces; ``crc32[(key, file)]`` is the checksum of the chunk of ``key`` written
        to file ``file``."""
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
                    crc32=crc32[(key, item.file)],
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
                tp_split=info.tp_split,
            )
        return entries


@lru_cache(maxsize=4096)
def _replica_layout(
    label: Label, local_shape: tuple[int, ...]
) -> tuple[Mapping[tuple[int, ...], tuple[Chunk, ...]], CheckpointError | None]:
    """The replica groups a label gives a tensor of ``local_shape``, read-only so the cached value cannot be changed
    through it, and the error that is wrong with them (``None`` when nothing is). The error is returned rather than
    raised so that a failure is cached too; callers raise it with the key prefixed.

    A key's groups depend only on its label and shape, which do not change during training, and most keys of a model
    share a handful of them; caching here makes every save after the first, and every key after the first of each
    kind, skip the enumeration and the tiling check."""
    rank = len(local_shape)
    for axis, placement in enumerate(label.placements):
        if placement.kind == "shard" and not 0 <= placement.dim < rank:
            return MappingProxyType({}), TopologyError(
                f"distribution axis {axis} shards dim {placement.dim}, out of range for a rank-{rank} tensor"
            )
    groups = {group: tuple(chunks) for group, chunks in replica_groups(chunk_boxes(label, local_shape)).items()}
    boxes = [chunks[0].box for chunks in groups.values()]
    global_shape = label_global_shape(label, local_shape)
    if not boxes_tile(boxes, Box.whole(global_shape)):
        return MappingProxyType(groups), CoverageError(
            f"the {len(boxes)} boxes of its label {label.to_json()} do not tile its global shape {list(global_shape)}"
        )
    return MappingProxyType(groups), None


def _checked_groups(info: TensorInfo) -> Mapping[tuple[int, ...], tuple[Chunk, ...]]:
    """The replica groups of a live tensor, after checking that their boxes tile its global shape.

    Save checks two things: every shard dim lies within the tensor's rank (a ``TopologyError`` otherwise), and the
    unique boxes (one per replica group, duplicates kept) tile the global shape exactly (a ``CoverageError``). A label
    whose shard dims are in range always tiles, so the second check only guards against a malformed label. The boxes
    are cells of one grid, so the check is linear in the device count, and it is cached per label and shape
    (``_replica_layout``).
    """
    groups, problem = _replica_layout(info.label, tuple(info.local_shape))
    if problem is not None:
        raise type(problem)(f"{info.key}: {problem}")
    return groups


def _by_key(infos: Iterable[TensorInfo], verb: str) -> dict[str, TensorInfo]:
    """``infos`` by key; two tensors under one key would leave one of them silently not ``verb``."""
    by_key: dict[str, TensorInfo] = {}
    for info in infos:
        if info.key in by_key:
            raise CheckpointError(f"{info.key}: two live tensors have this key; only one of them would be {verb}")
        by_key[info.key] = info
    return by_key


def create_save_plan(
    infos: Iterable[TensorInfo],
    file_for_coord: Callable[[tuple[int, ...]], str],
    owner_of: Callable[[str, list[Chunk]], Chunk | None],
) -> SavePlan:
    """One file per mesh coordinate that writes something; each key's unique boxes assigned to their elected owners.

    ``owner_of(key, replica group)`` returns the group's chunk this process writes, or ``None`` when another process
    writes it, so the election of the writers is the caller's. Every file's safetensors metadata names the format
    (``ttml.format``); which device wrote it is in the index and the file's name.
    """
    infos = _by_key(infos, "saved")
    owned: dict[str, list[Chunk]] = {}
    for key, info in infos.items():
        info.canonical_shape  # validates the transform against the global shape (raises LayoutError)
        chosen: list[Chunk] = []
        for group in _checked_groups(info).values():
            own = owner_of(key, group)
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
        plan.header, offsets, plan.data_bytes = build_header(tensors, {"ttml.format": f"{FORMAT}/{VERSION}"})
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
    """Zeros for ``dst`` (a box local to the destination shard): padding rows no saved chunk covers."""

    key: str
    dst: Box
    group: tuple[int, ...]


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
        raise DtypeMismatchError(f"{info.key}: checkpoint holds {entry.dtype} but the live tensor is {info.dtype}")
    dest_global = info.global_shape
    dest_canonical = info.transform.canonical_shape(dest_global)
    if tuple(entry.canonical_shape) != tuple(dest_canonical):
        raise ShapeMismatchError(
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
                    pieces.append((d_piece.corresponding(c_sub, c_piece), s_sub))
            for pad_box in info.transform.padding_boxes(dest_global):
                overlap = pad_box.intersect(dbox)
                if overlap is not None:
                    group_fills.append(FillItem(info.key, overlap.relative_to(dbox), group))
        if fast and saved_boxes.get(dbox) is None:
            fast = False
        for d_piece, s_piece in pieces:
            for ci in lookup.overlapping(s_piece):
                chunk = entry.chunks[ci]
                overlap = s_piece.intersect(chunk.box)
                if overlap is None:
                    continue
                d_overlap = d_piece.corresponding(overlap, s_piece)
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
    roles: Iterable[str] | None = None,
    local_coords: set[tuple[int, ...]] | None = None,
) -> LoadPlan:
    """A ``KeyPlan`` per live tensor. Live names resolve through the entries' recorded live keys.

    What is restored is decided by ``infos``: every live tensor passed is planned. The plan lists the live tensors
    without an entry (``missing``) and the entries without a live tensor (``unexpected``); ``load`` judges them under
    ``strict``, over the union of the ranks for the unexpected ones. ``roles`` only scopes the unexpected-key report:
    entries whose role is not in it ("param", "buffer", "optimizer_state") are not reported as unexpected, so a model
    alone loads from a checkpoint that also holds optimizer state. ``None`` means every role. A live tensor that
    resolves to an entry of another role is a ``RoleMismatchError``.
    """
    infos = list(_by_key(infos, "loaded").values())
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
            raise RoleMismatchError(
                f"{info.key}: resolves to checkpoint entry {entry_key!r}, a {entry.role!r}, but the live tensor is a "
                f"{info.role!r}"
            )
        used.add(entry_key)
        plans[info.key] = plan_key(entry, info, local_coords)
    return LoadPlan(plans, missing, unexpected_keys(index, used, wanted))


def unexpected_keys(index: Index, used: set[str], wanted: set[str] | None) -> list[str]:
    """The entries of the wanted roles no live tensor claimed. A multi-rank load judges this over the union of
    the ranks' claims."""
    return [key for key, entry in index.entries.items() if key not in used and (wanted is None or entry.role in wanted)]
