# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Multi-rank coordination for save and load.

One communicator, the MPI world ``tt-run`` launches as seen through ttnn's distributed context, gives every rank its
rank and the group size, so save and load span every rank of the world. The rules callers follow are in the package
docstring.

Save and load each run as a fixed sequence of phases. A phase is one named step of local work that every rank reaches at
the same point (collecting its live tensors, writing its shards or, on rank 0 alone, merging the ranks' plan parts),
followed by a status exchange: an allgather of one int per rank, 0 for success or the error class of a failure. Every
rank so learns whether any rank's step failed, and either all of them go on to the next phase or all of them raise, with
that class (``IncompleteCheckpointError``, ``MissingKeyError``, ...) and a message naming the phase and the ranks that
failed. Between phases the ranks only move bytes, rank 0 gathering the ranks' parts or broadcasting its own; those
transfers do no local work that could fail on one rank alone. A single process is the group of one, a ``Group`` on a
``LoopbackTransport``: its phases just run, and its transfers return their input, so save and load take the same path
on one process and on many. On any number of ranks a failure surfaces as a ``CheckpointError``: the class a failing
rank reported, or ``CheckpointError`` itself for an error from outside this package (an ``OSError``, a ``TypeError``
for a bad argument), which is then the ``__cause__`` on the rank where it was raised.

An error raised by a collective itself, a peer that died under ULFM, is fatal: ``CheckpointFatalError`` propagates and
the transport, one per process, records it, so neither this save or load nor any later one attempts another exchange.
MPI pairs collectives by order, so another exchange could hang or be paired with an unrelated one on a rank that moved
on. Metadata travels over MPI point-to-point (gathered to rank 0, broadcast from rank 0) under the tag ``DCP_TAG``.
Every payload is serialized and size-checked inside a phase, so a payload the transport cannot carry fails on every
rank. Data files go through the shared filesystem and are only ever opened by exact name.

Save. Every rank describes its tensors and the mesh coordinates it holds (its plan part); rank 0 merges the parts,
checking that the ranks describe every key they share alike and together hold every box, and elects one owner per box,
so a box held by several ranks (a replica across hosts, or the same key on several meshes) is written by exactly one of
them, chosen to even out the bytes each rank and device writes; rank 0 broadcasts the election; every rank writes the boxes
it owns into its own files; the result parts (files, chunk records, checksums, this rank's generators) are gathered;
rank 0 merges them into ``index.json`` and ``header.json``, checking that the ranks report one chunk per elected box,
and finalizes the directory. Its phases: collecting the
live tensors, checking the target directory, merging the plan parts, writing the shards, merging the index and
finalizing.

Load. Every rank reads the index and restores the shards of the coordinates it holds. ``strict`` is judged over the
union of the ranks' live tensors, since on a multi-mesh world no single rank holds every key. Its phases: checking the
arguments, reading the index and collecting the live tensors, planning the load, judging the unexpected entries, reading
and placing the shards, restoring the scalars and the train state.
"""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable, Protocol

import ttnn

from . import _errors
from ._errors import CheckpointError, CheckpointFatalError
from ._index import FORMAT, VERSION, Entry, FileRecord, Index, _entry_from_json, _entry_to_json, _gc_paused
from ._layouts import transform_from_json
from ._planner import TensorInfo, _checked_groups
from ._state_dict import is_structural_scalar
from ._topology import Chunk, Label, crc31
from ._train_state import DATALOADER_RANKS, RNG_FORMAT, this_world_rank

# The MPI tag of every coordinator transfer. It is reserved: the coordinator shares the world communicator with every
# other user of ttnn's point-to-point calls (tag 0 by default; tt-train's GRPO rollout uses 22010-22013 and
# 22101-22106), so no other protocol may use this tag while a save or load runs. A communicator of the coordinator's
# own (MPI_Comm_dup of the world) would make the reservation unnecessary.
DCP_TAG = 0x7DC0
# The largest payload one transfer can carry: its length travels through distributed_context_allgather_int, a C int,
# and MPI_Send / MPI_Recv take an int count (ttnn's MPI context throws past INT_MAX).
MAX_PAYLOAD = (1 << 31) - 1
# The writer version an index records.
WRITER_VERSION = "0.2"
# How much of a failing rank's message reaches the other ranks.
MESSAGE_LIMIT = 4096

# The error classes a phase carries across the ranks: status 0 is success, status k the class at index k - 1. They
# are every class _errors defines, in definition order, so a new error is carried without being listed here; all but
# CheckpointFatalError, which a failed exchange raises outside any status.
ERROR_CLASSES: tuple[type[CheckpointError], ...] = (CheckpointError,) + tuple(
    cls
    for cls in vars(_errors).values()
    if isinstance(cls, type) and issubclass(cls, CheckpointError) and cls not in (CheckpointError, CheckpointFatalError)
)


def error_status(err: BaseException) -> int:
    """The status a failed phase reports: the most specific exported class of ``err``, ``CheckpointError`` for
    anything else."""
    for index, cls in enumerate(ERROR_CLASSES[1:], start=2):
        if isinstance(err, cls):
            return index
    return 1


def describe_error(err: BaseException) -> str:
    """``"<class>: <message>"``; never raises: an exception whose ``__str__`` raises is named by its class alone."""
    try:
        return f"{type(err).__name__}: {err}"
    except Exception:  # noqa: BLE001
        return f"{type(err).__name__} (its message could not be rendered)"


def failure_message(rank: int, err: BaseException) -> bytes:
    """What a failing rank tells the others: UTF-8, at most ``MESSAGE_LIMIT`` bytes. It is built between the status
    exchange and the gather every rank then waits in, so it must not raise: a message that cannot be rendered is
    replaced, and a lone surrogate (a non-UTF-8 file name from ``os.fsdecode``) is escaped."""
    return f"rank {rank}: {describe_error(err)}".encode("utf-8", errors="backslashreplace")[:MESSAGE_LIMIT]


def checked_payload(payload: bytes, what: str) -> bytes:
    """``payload`` when the transport can carry it; run inside a phase, so an oversized payload fails every rank."""
    if len(payload) > MAX_PAYLOAD:
        raise CheckpointError(f"{what} is {len(payload)} bytes, more than a transfer can carry ({MAX_PAYLOAD})")
    return payload


class Transport(Protocol):
    """What the coordinator needs from a communicator. ``broken`` is the first failed exchange, or ``None``."""

    broken: str | None
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
        self.broken: str | None = None
        self.rank = int(ttnn.distributed_context_get_rank())
        self.size = int(ttnn.distributed_context_get_size())
        # its rank in MPI_COMM_WORLD; differs from rank only in a split launch with sub contexts
        self.world_rank = int(ttnn.distributed_context_world_rank())

    def allgather_int(self, value: int) -> list[int]:
        return [int(v) for v in ttnn.distributed_context_allgather_int(int(value))]

    def send_bytes(self, data: bytes, dest: int) -> None:
        ttnn.distributed_context_send_bytes(bytes(data), int(dest), DCP_TAG)

    def recv_bytes(self, size: int, source: int) -> bytes:
        return bytes(ttnn.distributed_context_recv_bytes(int(size), int(source), DCP_TAG))


_TRANSPORTS: dict[tuple[int, int], "MpiTransport"] = {}  # one per communicator (rank, size) of this process


def current_transport() -> Transport | None:
    """The MPI transport when this process is one of several ranks, else ``None`` (the single-process path). It is
    kept for the life of the process, so a failed exchange stops every later save and load too."""
    if not ttnn.distributed_context_is_initialized():
        return None
    size = int(ttnn.distributed_context_get_size())
    if size <= 1:
        return None
    key = (int(ttnn.distributed_context_get_rank()), size)
    if key not in _TRANSPORTS:
        _TRANSPORTS[key] = MpiTransport()
    return _TRANSPORTS[key]


class LoopbackTransport:
    """The transport of a single process, the group of one: an allgather returns its own value, and with one rank no
    point-to-point transfer ever happens. ``world_rank`` is the process's rank in the MPI world, which a one-rank sub
    context of a split launch has too."""

    broken: str | None = None
    rank = 0
    size = 1

    def __init__(self, world_rank: int = 0) -> None:
        self.world_rank = world_rank

    def allgather_int(self, value: int) -> list[int]:
        return [int(value)]

    def send_bytes(self, data: bytes, dest: int) -> None:
        raise AssertionError("the group of one has no peer to send to")

    def recv_bytes(self, size: int, source: int) -> bytes:
        raise AssertionError("the group of one has no peer to receive from")


class Group:
    """The ranks of one checkpoint operation and the collectives they agree through. A single process is the group
    of one, on a ``LoopbackTransport``."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.rank = transport.rank
        self.size = transport.size
        self.world_rank = transport.world_rank

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def _fatal(self, what: str) -> CheckpointFatalError:
        """Record a failed exchange on the transport, which outlives this group, and return the error to raise."""
        message = f"rank {self.rank}: {what}"
        self.transport.broken = message
        return CheckpointFatalError(message)

    def allgather(self, value: int) -> list[int]:
        if self.transport.broken:
            raise CheckpointFatalError(
                f"rank {self.rank}: an exchange between the ranks already failed in this process "
                f"({self.transport.broken}); no further exchange is attempted"
            )
        try:
            return self.transport.allgather_int(int(value))
        except Exception as err:  # noqa: BLE001  (MPI_ERR_PROC_FAILED and friends surface as RuntimeError)
            raise self._fatal(f"collective failed ({err}); a peer may have died") from err

    def phase(self, name: str, fn: Callable[[], Any]) -> Any:
        """Run ``fn`` on this rank, then exchange its outcome: every rank raises when any rank failed. The group of
        one has no other rank to tell, so its phases run ``fn``; a ``CheckpointError`` propagates as raised, and any
        other error is raised as a ``CheckpointError`` naming the phase, with the error as its cause, as it would be on
        many ranks, so one handler serves both.

        An interrupt (``KeyboardInterrupt``, ``SystemExit``) is exchanged like any failure, so the other ranks raise
        instead of waiting for this one in the next collective, and is then re-raised here as it was."""
        if self.size == 1:
            try:
                return fn()
            except CheckpointError:
                raise
            except Exception as err:  # noqa: BLE001
                raise CheckpointError(f"{name} failed: {describe_error(err)}") from err
        status, err = 0, None
        try:
            result = fn()
        except BaseException as caught:  # noqa: BLE001
            status, err, result = error_status(caught), caught, None
        statuses = self.allgather(status)
        failed = [r for r, s in enumerate(statuses) if s]
        if failed:
            # Every rank raises the class the first failing rank reported, so callers can catch the typed error, with
            # the failing ranks' own messages: one gather and one broadcast, on the failure path only.
            code = statuses[failed[0]]
            cls = ERROR_CLASSES[code - 1] if 0 < code <= len(ERROR_CLASSES) else CheckpointError
            parts = self.gather(failure_message(self.rank, err) if err is not None else b"")
            joined = b"\n".join(part for part in parts if part) if self.is_coordinator else None
            messages = self.broadcast(joined).decode("utf-8", errors="replace")
            if err is not None and not isinstance(err, Exception):
                raise err  # an interrupt, as it was, once the other ranks have its message too
            raise cls(f"{name} failed on rank(s) {failed} of {self.size}: {messages}") from err
        return result

    def gather(self, payload: Any) -> list[Any] | None:
        """Every rank's payload at rank 0 (in rank order); ``None`` elsewhere. Pure transfer: it cannot desync. The
        group of one returns its own payload, whatever it is, so a part ``outgoing`` kept as an object passes through.
        """
        if self.size == 1:
            return [payload]
        sizes = self.allgather(len(payload))
        try:
            if self.rank == 0:
                return [payload] + [self.transport.recv_bytes(sizes[r], r) for r in range(1, self.size)]
            self.transport.send_bytes(payload, 0)
            return None
        except Exception as err:  # noqa: BLE001
            raise self._fatal(f"point-to-point transfer failed ({err})") from err

    def broadcast(self, payload: Any) -> Any:
        """Rank 0's payload on every rank. Pure transfer. The group of one returns its payload as it is."""
        if self.size == 1:
            return payload
        size = self.allgather(len(payload) if self.rank == 0 else 0)[0]
        try:
            if self.rank == 0:
                assert payload is not None
                for r in range(1, self.size):
                    self.transport.send_bytes(payload, r)
                return payload
            return self.transport.recv_bytes(size, 0)
        except Exception as err:  # noqa: BLE001
            raise self._fatal(f"point-to-point transfer failed ({err})") from err

    def outgoing(self, part: Any, what: str) -> Any:
        """What this rank sends of ``part`` (a plan part, the election or a result part). On several ranks its bytes,
        size-checked, so call it inside a phase: a part too large to send then fails every rank. The group of one sends
        nothing and keeps the object, since encoding and decoding it would only cost time and a second copy of every
        tensor's description."""
        return part if self.size == 1 else checked_payload(part.to_bytes(), what)

    def incoming(self, data: Any, cls: type) -> Any:
        """A part ``outgoing`` sent, as a ``cls``: decoded on several ranks, the object itself on the group of one."""
        return data if self.size == 1 else cls.from_bytes(data)

    def agree(self, name: str, value: int, shown: Any = None) -> list[int]:
        """Every rank's ``value``; raises when they differ, naming what this rank has: ``shown``, the thing ``value``
        hashes when it is a hash, else ``value`` itself."""
        values = self.allgather(value)
        if any(v != values[0] for v in values):
            mine = value if shown is None else shown
            raise CheckpointError(f"the ranks disagree on {name}: rank {self.rank} has {mine!r} (all ranks: {values})")
        return values


def current_group() -> Group:
    """The group of this process: the MPI world under ``tt-run`` with several ranks, else the group of one, under the
    process's MPI world rank as ``this_world_rank`` reports it (nonzero in a one-rank sub context of a split launch),
    so save, load and the RNG helpers key its files and generators alike."""
    return Group(current_transport() or LoopbackTransport(this_world_rank()))


def state_hash(value: Any) -> int:
    """A 31-bit hash of JSON state, the same on every rank that holds equal state."""
    return crc31(json.dumps(value, sort_keys=True, separators=(",", ":")))


# --- plan parts ------------------------------------------------------------------------------------------------------


def info_to_json(info: TensorInfo, label_id: int) -> dict:
    """``info`` as JSON, its label named by ``label_id``, its position in the part's label table."""
    return {
        "key": info.key,
        "role": info.role,
        "dtype": info.dtype,
        "layout": info.layout,
        "label": label_id,
        "local_shape": list(info.local_shape),
        "transform": info.transform.to_json(),
        "live_keys": list(info.live_keys),
        "param_ref": info.param_ref,
        "state_path": list(info.state_path),
        "tp_split": list(info.tp_split),
    }


def info_from_json(value: dict, labels: list[Label]) -> TensorInfo:
    return TensorInfo(
        key=value["key"],
        role=value["role"],
        dtype=value["dtype"],
        layout=value["layout"],
        label=labels[value["label"]],
        local_shape=tuple(int(n) for n in value["local_shape"]),
        transform=transform_from_json(value["transform"]),
        live_keys=tuple(value["live_keys"]),
        param_ref=value["param_ref"],
        state_path=tuple(value["state_path"]),
        tp_split=tuple(int(n) for n in value["tp_split"]),
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
    """One rank's contribution to the save plan: its rank record, the tensors it holds, a hash of its scheduler state
    (the checkpoint stores one copy, which every rank restores, so the ranks must agree) and one of its dataloader
    state (stored once when the ranks agree, per rank when they differ)."""

    record: RankRecord
    infos: list[TensorInfo]
    scheduler_hash: int = 0
    dataloader_hash: int = 0

    def to_bytes(self) -> bytes:
        # Infos name their labels by position in a table, as result parts and index.json do: many keys share one.
        label_ids: dict[Label, int] = {}
        infos = [info_to_json(i, label_ids.setdefault(i.label, len(label_ids))) for i in self.infos]
        return json.dumps(
            {
                "record": self.record.to_json(),
                "labels": [label.to_json() for label in label_ids],
                "infos": infos,
                "scheduler_hash": self.scheduler_hash,
                "dataloader_hash": self.dataloader_hash,
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "PlanPart":
        # The cyclic GC paused: decoding allocates objects per key that all survive, which it would only rescan.
        with _gc_paused():
            value = json.loads(data.decode("utf-8"))
            labels = [Label.from_json(label) for label in value["labels"]]
            return PlanPart(
                RankRecord.from_json(value["record"]),
                [info_from_json(i, labels) for i in value["infos"]],
                int(value["scheduler_hash"]),
                int(value["dataloader_hash"]),
            )


Owner = tuple[int, tuple[int, ...]]  # (rank, mesh coordinate)


@dataclass
class Election:
    """Who writes each box of every key, how many keys the checkpoint holds, every live name any rank holds each key
    under, and whether the dataloader states are stored per rank (the ranks' differ). Rank 0 elects it over the
    ranks' plan parts and broadcasts it; every rank writes the boxes it owns, and rank 0 counts the result parts'
    chunks against it and records the names, which a rank that writes none of a key's boxes reports only here."""

    owners: dict[str, dict[tuple[int, ...], Owner]]
    keys: int
    live_keys: dict[str, tuple[str, ...]]
    dataloader_per_rank: bool = False

    def to_bytes(self) -> bytes:
        return json.dumps(
            {
                "keys": self.keys,
                "owners": {
                    k: [[list(g), o[0], list(o[1])] for g, o in groups.items()] for k, groups in self.owners.items()
                },
                "live_keys": {k: list(v) for k, v in self.live_keys.items()},
                "dataloader_per_rank": self.dataloader_per_rank,
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "Election":
        with _gc_paused():
            value = json.loads(data.decode("utf-8"))
            owners = {
                k: {tuple(g): (int(rank), tuple(int(x) for x in coord)) for g, rank, coord in groups}
                for k, groups in value["owners"].items()
            }
            live_keys = {k: tuple(v) for k, v in value["live_keys"].items()}
            return Election(owners, int(value["keys"]), live_keys, bool(value["dataloader_per_rank"]))

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
        and a.tp_split == b.tp_split
    )


def _groups_of(info: TensorInfo) -> tuple[dict[tuple[int, ...], tuple[Chunk, ...]], dict[tuple[int, ...], tuple]]:
    """A key's replica groups (cached per label and shape) and, per mesh coordinate, the group it holds."""
    groups = _checked_groups(info)  # a TopologyError naming the key
    by_coord = {chunk.mesh_coord: group for group, chunks in groups.items() for chunk in chunks}
    return groups, by_coord


def merge_plan_parts(parts: list[PlanPart]) -> Election:
    """The owner election over the ranks' plan parts.

    A key held by several ranks must be described identically by all of them (label, per-device shape, dtype,
    transform), and the ranks' scheduler states must agree, since the checkpoint stores one copy that every rank
    restores. Their dataloader states are stored once when they agree, and per rank when they differ (data-parallel
    loaders that keep a seed or a shard of their own). A key's label splits it into boxes (its replica groups); each box is held by one or more
    (rank, mesh coordinate) pairs, at least one of which must exist. The box's owner is the one pair that writes it.

    The owners even out the bytes written, since the slowest rank ends the write: a box with one holder is that
    holder's, and the others, largest first, go to the rank among their holders with the fewest bytes so far and to
    that rank's device with the fewest. A few large keys (a vocab matrix and its moments) would otherwise decide where
    most of the bytes land. Ties go to the lower rank and coordinate, so the same parts always elect the same owners.
    """
    parts = sorted(parts, key=lambda p: p.record.rank)
    differing = [p.record.rank for p in parts if p.scheduler_hash != parts[0].scheduler_hash]
    if differing:
        raise CheckpointError(
            f"the scheduler state of rank(s) {differing} differs from rank {parts[0].record.rank}'s; a checkpoint "
            f"stores one copy of it, which every rank restores"
        )
    dataloader_per_rank = any(p.dataloader_hash != parts[0].dataloader_hash for p in parts)
    infos: dict[str, TensorInfo] = {}
    first_rank: dict[str, int] = {}
    members: dict[str, dict[tuple[int, ...], list[Owner]]] = {}
    aliases: dict[str, set[str]] = {}
    named: dict[str, tuple[str, int]] = {}  # a live name -> (the key it names, the first rank naming it)
    for part in parts:
        for info in part.infos:
            # A rank may hold a key under names another does not (a layer tied to another name on that rank only);
            # the index keeps them all, so a name, the key's own included, must mean one key on every rank using it.
            names = {info.key, *info.live_keys}
            for live in sorted(names):
                key, rank = named.setdefault(live, (info.key, part.record.rank))
                if key != info.key:
                    raise CheckpointError(
                        f"{live} names {info.key} on rank {part.record.rank} but {key} on rank {rank}; a checkpoint "
                        f"maps each live name to one entry"
                    )
            aliases.setdefault(info.key, set()).update(names)
            if info.key in infos:
                if not _same_description(infos[info.key], info):
                    raise CheckpointError(
                        f"{info.key}: rank {part.record.rank} describes it differently from rank "
                        f"{first_rank[info.key]} (role, dtype, label, per-device shape, transform, TP split, or an "
                        f"optimizer state's parameter or state path)"
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
    owners: dict[str, dict[tuple[int, ...], Owner]] = {key: {} for key in infos}
    boxes: list[tuple[int, str, tuple[int, ...], list[Owner]]] = []  # (bytes, key, group, holders)
    for key, info in infos.items():
        groups, _ = _groups_of(info)
        nbytes = prod(info.local_shape) * info.itemsize
        for group_key in groups:
            holders = sorted(members[key].get(group_key, []))
            if not holders:
                raise CheckpointError(
                    f"{key}: no rank holds the box of replica group {group_key}; the ranks' local coordinates do not "
                    f"cover the label's mesh coordinates"
                )
            boxes.append((nbytes, key, group_key, holders))
    rank_bytes: dict[int, int] = defaultdict(int)
    device_bytes: dict[Owner, int] = defaultdict(int)
    forced = [box for box in boxes if len(box[3]) == 1]
    free = sorted((box for box in boxes if len(box[3]) > 1), key=lambda box: (-box[0], box[1], box[2]))
    for nbytes, key, group_key, holders in forced + free:
        rank = min({r for r, _ in holders}, key=lambda r: (rank_bytes[r], r))
        chosen = min((h for h in holders if h[0] == rank), key=lambda h: (device_bytes[h], h))
        owners[key][group_key] = chosen
        rank_bytes[chosen[0]] += nbytes
        device_bytes[chosen] += nbytes
    live_keys = {key: tuple(sorted(names)) for key, names in aliases.items()}
    return Election(owners, len(infos), live_keys, dataloader_per_rank)


# --- result parts ----------------------------------------------------------------------------------------------------


@dataclass
class ResultPart:
    """What one rank wrote: its files, its chunk records per key, its generators, its optimizer scalars and, when the
    election stores dataloader states per rank, its own (``{world rank: state}``, empty without a dataloader); on rank
    0 also the user header and the train state stored once (the scheduler, and the dataloader when the ranks agree)."""

    record: RankRecord
    files: list[FileRecord]
    entries: dict[str, Entry]
    rng: dict
    scalars: dict = field(default_factory=dict)
    header: dict = field(default_factory=dict)
    train_state: dict = field(default_factory=dict)
    dataloader: dict = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        # Entries name their labels by position in a table, as in index.json.
        label_ids: dict[Label, int] = {}
        entries = {k: _entry_to_json(e, label_ids.setdefault(e.label, len(label_ids))) for k, e in self.entries.items()}
        return json.dumps(
            {
                "record": self.record.to_json(),
                "files": [{"path": f.path, "nbytes": f.nbytes, "coord": list(f.coord)} for f in self.files],
                "labels": [label.to_json() for label in label_ids],
                "entries": entries,
                "rng": self.rng,
                "scalars": self.scalars,
                "header": self.header,
                "train_state": self.train_state,
                "dataloader": self.dataloader,
            }
        ).encode()

    @staticmethod
    def from_bytes(data: bytes) -> "ResultPart":
        with _gc_paused():
            value = json.loads(data.decode("utf-8"))
            record = RankRecord.from_json(value["record"])
            files = [
                FileRecord(f["path"], int(f["nbytes"]), record.world_rank, record.mesh_id, tuple(f["coord"]))
                for f in value["files"]
            ]
            labels = [Label.from_json(label) for label in value["labels"]]
            entries = {k: _entry_from_json(k, e, labels) for k, e in value["entries"].items()}
            return ResultPart(
                record,
                files,
                entries,
                dict(value["rng"]),
                dict(value["scalars"]),
                dict(value["header"]),
                dict(value["train_state"]),
                dict(value["dataloader"]),
            )


def merge_results(parts: list[ResultPart], election: Election, checkpoint_id: str) -> tuple[Index, dict]:
    """Rank 0's merge: one file table, one entry per key with every rank's chunks (exactly one per elected box), the
    world record, and the header (rank 0's user header and the train state stored once, every rank's generators and,
    when the election says so, every rank's dataloader state). The parts are left as they were: the index holds
    entries of its own."""
    parts = sorted(parts, key=lambda p: p.record.rank)
    files: list[FileRecord] = []
    entries: dict[str, Entry] = {}
    for part in parts:
        base = len(files)
        files.extend(part.files)
        for key, entry in part.entries.items():
            chunks = [dataclasses.replace(c, file=c.file + base) for c in entry.chunks]
            if key in entries:
                have = entries[key]
                if (
                    have.role != entry.role
                    or have.dtype != entry.dtype
                    or tuple(have.global_shape) != tuple(entry.global_shape)
                    or have.transform != entry.transform
                    or have.label != entry.label
                    or have.tp_split != entry.tp_split
                ):
                    raise CheckpointError(
                        f"{key}: rank {part.record.rank} reports a different entry than earlier ranks"
                    )
                have.chunks.extend(chunks)
            else:
                entries[key] = dataclasses.replace(entry, chunks=chunks)
    for key, entry in entries.items():
        # The writers' parts carry only their own names for a key; the election has every rank's.
        entry.live_keys = tuple(sorted(set(entry.live_keys) | set(election.live_keys.get(key, ()))))
    for key, owners in election.owners.items():
        entry = entries.get(key)
        expected = len(owners)
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
        train_state["rng"] = {"format": RNG_FORMAT, "ranks": rng_ranks}
    if election.dataloader_per_rank:
        train_state[DATALOADER_RANKS] = {k: v for part in parts for k, v in part.dataloader.items()}
    world = {"group_size": len(parts), "ranks": [p.record.to_json() for p in parts]}
    scalars = dict(rank0.scalars)
    # A param group's member list is per rank (each rank's optimizer holds the parameters of its own mesh); the index
    # records the union, which load checks every rank's live members against.
    for key in [k for k in scalars if is_structural_scalar(k)]:
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
        writer_software={"ttml_distributed_checkpointing": WRITER_VERSION},
    )
    header = {
        "format": FORMAT,
        "version": VERSION,
        "checkpoint_id": checkpoint_id,
        "user": dict(rank0.header),
        "train_state": train_state,
    }
    return index, header
