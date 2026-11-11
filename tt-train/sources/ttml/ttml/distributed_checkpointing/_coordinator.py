# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank coordination for save and load.

One communicator, the MPI world ``tt-run`` launches as seen through ttnn's distributed context, gives every rank its
rank and the group size. A single process is the group of one (``LocalGroup``): its phases just run, its transfers
return their input, so save and load take the same path on one process and on many. Every phase of a larger group
ends with a status exchange, an allgather of one int per rank carrying the error class of a failure, so an error on
any rank stops every rank with that class (``IncompleteCheckpoint``, ``MissingKeyError``, ...) and a message naming
the ranks that failed. An error raised by a collective itself, a peer that died under ULFM, is fatal: no further
collective is attempted and ``CheckpointFatal`` propagates. Metadata travels over MPI point-to-point (gathered to
rank 0, broadcast from rank 0) under the tag ``DCP_TAG``; no other point-to-point traffic between the ranks may be
outstanding when save or load is called. Every payload is serialized and size-checked inside a phase, so a payload
the transport cannot carry fails on every rank. Data files go through the shared filesystem and are only ever opened
by exact name.

Save. Every rank describes its tensors and the mesh coordinates it holds (its plan part); rank 0 merges the parts
into the union and broadcasts it; every rank runs the same owner election over the union, so a box held by several
ranks (a replica across hosts, or the same key on several meshes) is written by exactly one of them, the hash of
the key choosing among them in rank order; every rank writes its own files; the result parts (files, chunk records,
checksums, this rank's generators) are gathered; rank 0 merges them into ``index.json`` and ``header.json``, checks
every data file's size through the shared filesystem, and finalizes the directory.

Load. Every rank reads the index and restores the shards of the coordinates it holds. ``strict`` is judged over the
union of the ranks' live tensors, since on a multi-mesh world no single rank holds every key.
"""

from __future__ import annotations

import json
import re
import traceback
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from ._errors import (
    CheckpointError,
    CheckpointFatal,
    CoverageError,
    DtypeMismatch,
    FSDPStateError,
    IncompleteCheckpoint,
    LayoutError,
    LazyParameterError,
    MissingKeyError,
    RoleMismatch,
    ShapeMismatch,
    TopologyError,
    UnexpectedKeyError,
    VersionError,
)
from ._index import Entry, FileRecord, Index, _entry_from_json, _entry_to_json
from ._layouts import transform_from_json
from ._planner import TensorInfo, _replica_layout
from ._topology import Chunk, Label

DCP_TAG = 0x7DC0  # the MPI tag of every coordinator transfer, kept apart from tag 0 that application traffic uses
MAX_PAYLOAD = (1 << 31) - 1  # a payload length travels as a 32-bit int

# The error classes a phase carries across the ranks: status 0 is success, status k the class at index k - 1.
ERROR_CLASSES: tuple[type[CheckpointError], ...] = (
    CheckpointError,
    IncompleteCheckpoint,
    VersionError,
    LayoutError,
    ShapeMismatch,
    DtypeMismatch,
    MissingKeyError,
    UnexpectedKeyError,
    CoverageError,
    TopologyError,
    FSDPStateError,
    LazyParameterError,
    RoleMismatch,
)


def error_status(err: BaseException) -> int:
    """The status a failed phase reports: the most specific exported class of ``err``, ``CheckpointError`` for
    anything else."""
    for index, cls in enumerate(ERROR_CLASSES[1:], start=2):
        if isinstance(err, cls):
            return index
    return 1


def checked_payload(payload: bytes, what: str) -> bytes:
    """``payload`` when the transport can carry it; run inside a phase, so an oversized payload fails every rank."""
    if len(payload) > MAX_PAYLOAD:
        raise CheckpointError(f"{what} is {len(payload)} bytes, more than a transfer can carry ({MAX_PAYLOAD})")
    return payload


class Transport(Protocol):
    """What the coordinator needs from a communicator."""

    rank: int
    size: int
    world_rank: int

    def allgather_int(self, value: int) -> list[int]:
        ...

    def send_bytes(self, data: bytes, dest: int) -> None:
        ...

    def recv_bytes(self, size: int, source: int) -> bytes:
        ...


class MpiTransport:
    """ttnn's distributed context: the MPI world ``tt-run`` launched this process into."""

    def __init__(self) -> None:
        import ttnn

        self._ttnn = ttnn
        self.rank = int(ttnn.distributed_context_get_rank())
        self.size = int(ttnn.distributed_context_get_size())
        try:
            self.world_rank = int(ttnn.distributed_context_world_rank())
        except Exception:  # noqa: BLE001  (no sub-contexts: the world is the current context)
            self.world_rank = self.rank

    def allgather_int(self, value: int) -> list[int]:
        return [int(v) for v in self._ttnn.distributed_context_allgather_int(int(value))]

    def send_bytes(self, data: bytes, dest: int) -> None:
        self._ttnn.distributed_context_send_bytes(bytes(data), int(dest), DCP_TAG)

    def recv_bytes(self, size: int, source: int) -> bytes:
        return bytes(self._ttnn.distributed_context_recv_bytes(int(size), int(source), DCP_TAG))


def current_transport() -> Transport | None:
    """The MPI transport when this process is one of several ranks, else ``None`` (the single-process path)."""
    import ttnn

    if not ttnn.distributed_context_is_initialized():
        return None
    if int(ttnn.distributed_context_get_size()) <= 1:
        return None
    return MpiTransport()


class Group:
    """The ranks of one checkpoint operation and the collectives they agree through."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.rank = transport.rank
        self.size = transport.size
        self.world_rank = transport.world_rank
        self.failed = False  # a collective raised: no further collective may be attempted

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def allgather(self, value: int) -> list[int]:
        if self.failed:
            raise CheckpointFatal("a collective already failed in this group; no further exchange is attempted")
        try:
            return self.transport.allgather_int(int(value))
        except Exception as err:  # noqa: BLE001  (MPI_ERR_PROC_FAILED and friends surface as RuntimeError)
            self.failed = True
            raise CheckpointFatal(f"rank {self.rank}: collective failed ({err}); a peer may have died") from err

    def phase(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on this rank, then exchange its outcome: every rank raises when any rank failed."""
        status, err = 0, None
        try:
            result = fn()
        except Exception as caught:  # noqa: BLE001
            status, err, result = error_status(caught), caught, None
            traceback.print_exc()
        statuses = self.allgather(status)
        failed = [r for r, s in enumerate(statuses) if s]
        if failed:
            # Every rank raises the class the first failing rank reported, so callers can catch the typed error.
            code = statuses[failed[0]]
            cls = ERROR_CLASSES[code - 1] if 0 < code <= len(ERROR_CLASSES) else CheckpointError
            detail = f": {err}" if err is not None else f" ({cls.__name__} on rank {failed[0]})"
            raise cls(f"{name} failed on rank(s) {failed} of {self.size}{detail}") from err
        return result

    def gather(self, payload: bytes) -> list[bytes] | None:
        """Every rank's payload at rank 0 (in rank order); ``None`` elsewhere. Pure transfer: it cannot desync."""
        sizes = self.allgather(len(payload))
        try:
            if self.rank == 0:
                return [payload] + [self.transport.recv_bytes(sizes[r], r) for r in range(1, self.size)]
            self.transport.send_bytes(payload, 0)
            return None
        except Exception as err:  # noqa: BLE001
            self.failed = True
            raise CheckpointFatal(f"rank {self.rank}: point-to-point transfer failed ({err})") from err

    def broadcast(self, payload: bytes | None) -> bytes:
        """Rank 0's payload on every rank. Pure transfer."""
        size = self.allgather(len(payload) if self.rank == 0 else 0)[0]
        try:
            if self.rank == 0:
                assert payload is not None
                for r in range(1, self.size):
                    self.transport.send_bytes(payload, r)
                return payload
            return self.transport.recv_bytes(size, 0)
        except Exception as err:  # noqa: BLE001
            self.failed = True
            raise CheckpointFatal(f"rank {self.rank}: point-to-point transfer failed ({err})") from err

    def agree(self, name: str, value: int) -> list[int]:
        """Every rank's ``value``; raises when they differ."""
        values = self.allgather(value)
        if any(v != values[0] for v in values):
            raise CheckpointError(f"the ranks disagree on {name}: {values}")
        return values


def name_hash(text: str) -> int:
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


class LocalGroup:
    """The group of one process. Its phases run their function and let any error propagate as raised; its transfers
    return their input. Save and load use it on a single process, so one code path serves every world size."""

    rank = 0
    size = 1
    world_rank = 0
    is_coordinator = True
    failed = False

    def allgather(self, value: int) -> list[int]:
        return [int(value)]

    def phase(self, name: str, fn: Callable[[], Any]) -> Any:
        return fn()

    def gather(self, payload: bytes) -> list[bytes]:
        return [payload]

    def broadcast(self, payload: bytes | None) -> bytes:
        assert payload is not None
        return payload

    def agree(self, name: str, value: int) -> list[int]:
        return [int(value)]


def current_group() -> "Group | LocalGroup":
    """The group of this process: the MPI world under ``tt-run`` with several ranks, else the group of one."""
    transport = current_transport()
    return Group(transport) if transport is not None else LocalGroup()


def state_hash(value: Any) -> int:
    """A 31-bit hash of JSON state, the same on every rank that holds equal state."""
    return zlib.crc32(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")) & 0x7FFFFFFF


# --- plan parts ------------------------------------------------------------------------------------------------------


def info_to_json(info: TensorInfo) -> dict:
    return {
        "key": info.key,
        "role": info.role,
        "dtype": info.dtype,
        "layout": info.layout,
        "label": info.label.to_json(),
        "local_shape": list(info.local_shape),
        "transform": info.transform.to_json(),
        "live_keys": list(info.live_keys),
        "param_ref": info.param_ref,
        "state_path": list(info.state_path),
    }


def info_from_json(value: dict) -> TensorInfo:
    return TensorInfo(
        key=value["key"],
        role=value["role"],
        dtype=value["dtype"],
        layout=value["layout"],
        label=Label.from_json(value["label"]),
        local_shape=tuple(int(n) for n in value["local_shape"]),
        transform=transform_from_json(value["transform"]),
        live_keys=tuple(value.get("live_keys", ())),
        param_ref=value.get("param_ref"),
        state_path=tuple(value.get("state_path", ())),
    )


@dataclass
class RankRecord:
    """What the index records about one rank of the saving world."""

    rank: int
    world_rank: int
    mesh_id: int
    shape: list[int]
    axis_names: list[str]
    local_coords: list[tuple[int, ...]]

    def to_json(self) -> dict:
        return {
            "rank": self.rank,
            "world_rank": self.world_rank,
            "mesh_id": self.mesh_id,
            "shape": list(self.shape),
            "axis_names": list(self.axis_names),
            "local_coords": [list(c) for c in self.local_coords],
        }

    @staticmethod
    def from_json(value: dict) -> "RankRecord":
        return RankRecord(
            int(value["rank"]),
            int(value["world_rank"]),
            int(value["mesh_id"]),
            [int(n) for n in value["shape"]],
            [str(a) for a in value["axis_names"]],
            [tuple(int(x) for x in c) for c in value["local_coords"]],
        )


@dataclass
class PlanPart:
    """One rank's contribution to the save plan: its rank record, the tensors it holds, and a hash of its scheduler
    and dataloader state (the checkpoint stores one copy of those, which every rank restores, so they must agree)."""

    record: RankRecord
    infos: list[TensorInfo]
    train_state_hash: int = 0

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "record": self.record.to_json(),
                "infos": [info_to_json(i) for i in self.infos],
                "train_state_hash": self.train_state_hash,
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "PlanPart":
        value = json.loads(data.decode("utf-8"))
        return PlanPart(
            RankRecord.from_json(value["record"]),
            [info_from_json(i) for i in value["infos"]],
            int(value.get("train_state_hash", 0)),
        )


Owner = tuple[int, tuple[int, ...]]  # (rank, mesh coordinate)


@dataclass
class UnionPlan:
    """The merged parts: every rank's record, the agreed description of every key, and who writes each box."""

    records: list[RankRecord]
    infos: dict[str, TensorInfo]
    owners: dict[str, dict[tuple[int, ...], Owner]]  # key -> replica group -> owner

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "records": [r.to_json() for r in self.records],
                "infos": {k: info_to_json(i) for k, i in self.infos.items()},
                "owners": {
                    k: [[list(g), o[0], list(o[1])] for g, o in groups.items()] for k, groups in self.owners.items()
                },
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "UnionPlan":
        value = json.loads(data.decode("utf-8"))
        return UnionPlan(
            [RankRecord.from_json(r) for r in value["records"]],
            {k: info_from_json(i) for k, i in value["infos"].items()},
            {
                k: {tuple(g): (int(rank), tuple(int(x) for x in coord)) for g, rank, coord in groups}
                for k, groups in value["owners"].items()
            },
        )

    def owner_of(self, rank: int) -> Callable[[str, list[Chunk]], Chunk | None]:
        """The ``owner_of`` callback for ``create_save_plan`` on ``rank``: the group's chunk when this rank writes
        it, ``None`` when another rank does."""

        def pick(key: str, group: list[Chunk]) -> Chunk | None:
            owner_rank, coord = self.owners[key][group[0].group]
            if owner_rank != rank:
                return None
            for chunk in group:
                if chunk.mesh_coord == coord:
                    return chunk
            raise CheckpointError(f"{key}: rank {rank} is elected for coordinate {coord} but does not hold it")

        return pick

    def election(self) -> "Election":
        """The part every rank needs: the owners and the key count."""
        return Election(self.owners, len(self.infos))


@dataclass
class Election:
    """What every rank needs from the union to write its part: who writes each box, and how many keys the checkpoint
    holds. Rank 0 keeps the whole ``UnionPlan`` for the merge; only this is broadcast."""

    owners: dict[str, dict[tuple[int, ...], Owner]]
    keys: int

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "keys": self.keys,
                "owners": {
                    k: [[list(g), o[0], list(o[1])] for g, o in groups.items()] for k, groups in self.owners.items()
                },
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "Election":
        value = json.loads(data.decode("utf-8"))
        owners = {
            k: {tuple(g): (int(rank), tuple(int(x) for x in coord)) for g, rank, coord in groups}
            for k, groups in value["owners"].items()
        }
        return Election(owners, int(value["keys"]))

    def owner_of(self, rank: int) -> Callable[[str, list[Chunk]], Chunk | None]:
        """The ``owner_of`` callback for ``create_save_plan`` on ``rank``: the group's chunk when this rank writes
        it, ``None`` when another rank does."""

        def pick(key: str, group: list[Chunk]) -> Chunk | None:
            owner_rank, coord = self.owners[key][group[0].group]
            if owner_rank != rank:
                return None
            for chunk in group:
                if chunk.mesh_coord == coord:
                    return chunk
            raise CheckpointError(f"{key}: rank {rank} is elected for coordinate {coord} but does not hold it")

        return pick


def _same_description(a: TensorInfo, b: TensorInfo) -> bool:
    return (
        a.role == b.role
        and a.dtype == b.dtype
        and a.label == b.label
        and tuple(a.local_shape) == tuple(b.local_shape)
        and a.transform == b.transform
        and a.param_ref == b.param_ref
        and tuple(a.state_path) == tuple(b.state_path)
    )


def _groups_of(info: TensorInfo) -> tuple[dict[tuple[int, ...], tuple[Chunk, ...]], dict[tuple[int, ...], tuple]]:
    """A key's replica groups (cached per label and shape) and, per mesh coordinate, the group it holds."""
    groups, problem = _replica_layout(info.label, tuple(info.local_shape))
    if problem:
        raise CheckpointError(f"{info.key}: {problem}")
    by_coord = {chunk.mesh_coord: group for group, chunks in groups.items() for chunk in chunks}
    return groups, by_coord


def merge_plan_parts(parts: list[PlanPart]) -> UnionPlan:
    """The union of the ranks' parts and the owner election over it.

    A key held by several ranks must be described identically by all of them (label, per-device shape, dtype,
    transform), and the ranks' scheduler and dataloader states must agree, since the checkpoint stores one copy that
    every rank restores. Each key's replica groups come from its label; a group's members are the (rank, coordinate)
    pairs holding it; every group must have at least one member across the ranks. The owner is the member at the
    key's hash in (rank, coordinate) order, the same spread as the single-process election. The work is linear in
    the keys times the coordinates the ranks hold: a key's groups are computed once per label and shape.
    """
    parts = sorted(parts, key=lambda p: p.record.rank)
    differing = [p.record.rank for p in parts if p.train_state_hash != parts[0].train_state_hash]
    if differing:
        raise CheckpointError(
            f"the scheduler or dataloader state of rank(s) {differing} differs from rank {parts[0].record.rank}'s; a "
            f"checkpoint stores one copy of that state, which every rank restores"
        )
    infos: dict[str, TensorInfo] = {}
    first_rank: dict[str, int] = {}
    members: dict[str, dict[tuple[int, ...], list[Owner]]] = {}
    for part in parts:
        for info in part.infos:
            if info.key in infos:
                if not _same_description(infos[info.key], info):
                    raise CheckpointError(
                        f"{info.key}: rank {part.record.rank} describes it differently from rank "
                        f"{first_rank[info.key]} (label, per-device shape, dtype or transform)"
                    )
            else:
                infos[info.key] = info
                first_rank[info.key] = part.record.rank
                members[info.key] = {}
            _, by_coord = _groups_of(info)
            for coord in part.record.local_coords:
                group_key = by_coord.get(tuple(coord))
                if group_key is not None:
                    members[info.key].setdefault(group_key, []).append((part.record.rank, tuple(coord)))
    owners: dict[str, dict[tuple[int, ...], Owner]] = {}
    for key, info in infos.items():
        groups, _ = _groups_of(info)
        chosen: dict[tuple[int, ...], Owner] = {}
        spread = zlib.crc32(key.encode("utf-8"))
        for group_key in groups:
            holders = sorted(members[key].get(group_key, []))
            if not holders:
                raise CheckpointError(
                    f"{key}: no rank holds the box of replica group {group_key}; the ranks' local coordinates do not "
                    f"cover the label's mesh coordinates"
                )
            chosen[group_key] = holders[spread % len(holders)]
        owners[key] = chosen
    return UnionPlan([p.record for p in parts], infos, owners)


# --- result parts ----------------------------------------------------------------------------------------------------


@dataclass
class ResultPart:
    """What one rank wrote: its files, its chunk records per key, its generators and its bytes."""

    record: RankRecord
    files: list[FileRecord]
    entries: dict[str, Entry]
    rng: dict
    bytes_written: int
    scalars: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    train_state: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "record": self.record.to_json(),
                "files": [{"path": f.path, "nbytes": f.nbytes, "coord": list(f.coord)} for f in self.files],
                "entries": {k: _entry_to_json(e) for k, e in self.entries.items()},
                "rng": self.rng,
                "bytes_written": self.bytes_written,
                "scalars": self.scalars,
                "header": self.header,
                "train_state": self.train_state,
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "ResultPart":
        value = json.loads(data.decode("utf-8"))
        record = RankRecord.from_json(value["record"])
        files = [
            FileRecord(f["path"], int(f["nbytes"]), record.world_rank, record.mesh_id, tuple(f["coord"]))
            for f in value["files"]
        ]
        entries = {k: _entry_from_json(k, e) for k, e in value["entries"].items()}
        return ResultPart(
            record,
            files,
            entries,
            dict(value.get("rng", {})),
            int(value.get("bytes_written", 0)),
            dict(value.get("scalars", {})),
            dict(value.get("header", {})),
            dict(value.get("train_state", {})),
        )


def merge_results(parts: list[ResultPart], union: UnionPlan, checkpoint_id: str) -> tuple[Index, dict]:
    """Rank 0's merge: one file table, one entry per key with every rank's chunks, the world record, and the header
    (rank 0's user header and scheduler/dataloader state, every rank's generators)."""
    from ._index import FORMAT, VERSION

    parts = sorted(parts, key=lambda p: p.record.rank)
    files: list[FileRecord] = []
    entries: dict[str, Entry] = {}
    for part in parts:
        base = len(files)
        files.extend(part.files)
        for key, entry in part.entries.items():
            chunks = [
                type(c)(
                    box=c.box,
                    file=c.file + base,
                    data_offset=c.data_offset,
                    nbytes=c.nbytes,
                    writer_coord=c.writer_coord,
                    crc32=c.crc32,
                )
                for c in entry.chunks
            ]
            if key in entries:
                have = entries[key]
                if (
                    have.role != entry.role
                    or have.dtype != entry.dtype
                    or tuple(have.global_shape) != tuple(entry.global_shape)
                    or have.transform != entry.transform
                    or have.label != entry.label
                ):
                    raise CheckpointError(
                        f"{key}: rank {part.record.rank} reports a different entry than earlier ranks"
                    )
                have.chunks.extend(chunks)
            else:
                entry.chunks = chunks
                entries[key] = entry
    for key, info in union.infos.items():
        entry = entries.get(key)
        expected = len(union.owners[key])
        if entry is None or len(entry.chunks) != expected:
            got = 0 if entry is None else len(entry.chunks)
            raise CheckpointError(f"{key}: {got} chunk(s) reported by the ranks, {expected} elected")
    rank0 = parts[0]
    rng_ranks: dict[str, dict] = {}
    for part in parts:
        for rank_key, block in part.rng.get("ranks", {}).items():
            rng_ranks[rank_key] = block
    train_state = dict(rank0.train_state)
    if rng_ranks:
        train_state["rng"] = {"format": int(rank0.rng.get("format", 1)), "ranks": rng_ranks}
    world = {"group_size": len(parts), "ranks": [r.to_json() for r in union.records]}
    scalars = dict(rank0.scalars)
    # A param group's member list is per rank (each rank's optimizer holds the parameters of its own mesh); the index
    # records the union, which load checks every rank's live members against.
    for key in [k for k in scalars if _PARAM_GROUP_MEMBERS.search(k)]:
        names: set[str] = set()
        for part in parts:
            value = part.scalars.get(key)
            if isinstance(value, str) and value:
                names.update(value.split("\n"))
        scalars[key] = "\n".join(sorted(names))
    index = Index(
        checkpoint_id=checkpoint_id,
        world=world,
        files=files,
        entries=entries,
        scalars=scalars,
        writer_software={"ttml_distributed_checkpointing": "0.2"},
    )
    header = {
        "format": FORMAT,
        "version": VERSION,
        "checkpoint_id": checkpoint_id,
        "user": dict(rank0.header),
        "train_state": train_state,
    }
    return index, header


_PARAM_GROUP_MEMBERS = re.compile(r"(^|/)param_groups\.\d+\.params$")
