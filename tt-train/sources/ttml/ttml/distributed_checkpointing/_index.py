# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The on-disk layout of a checkpoint: ``index.json``, ``header.json``, directory naming and the staging directory.

A checkpoint is a directory. It is complete if and only if its name carries no ``.partial-<nonce>`` suffix and its
``index.json`` opens. Everything is written under the staging name first; the coordinator renames the directory
after ``index.json`` is fsynced, so a crashed attempt can never look complete and never mixes files into a later one.

``index.json`` is written by the coordinator only. It records, per canonical key, the global and canonical shapes,
the transform, the topology label the writer trusted, every live name the tensor was seen under, and a columnar
table of chunks: one row per unique shard, with its box in the global array, the file that holds it, its byte range
and its CRC. A model's keys share a handful of labels, each listing every device's mesh coordinate, so the index
holds each distinct label once, in a ``labels`` table, and an entry's ``label`` is its position there.

Readers refuse a newer ``VERSION`` and ignore fields they do not know. A new field that a reader may ignore needs no
bump. A field readers require, or one that changes how the rest is read (a reader ignoring it would load the wrong
data), bumps ``VERSION``, so that an older reader refuses the checkpoint instead of misreading it.
"""

from __future__ import annotations

import datetime
import errno
import gc
import json
import math
import os
import re
import secrets
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from ._boxes import Box
from ._errors import CheckpointError, IncompleteCheckpointError, LayoutError, VersionError
from ._layouts import Transform, transform_from_json
from ._topology import Label

FORMAT = "ttml-dcp"
VERSION = 1
INDEX_NAME = "index.json"
HEADER_NAME = "header.json"
DATA_DIR = "data"
PARTIAL_MARK = ".partial-"
REPLACED_MARK = ".replaced-"


@dataclass
class ChunkRecord:
    """One unique shard of a key: where it sits in the global array and in which file."""

    box: Box
    file: int  # index into Index.files
    data_offset: int  # within the file's data section
    nbytes: int
    crc32: int  # zlib.crc32 of the chunk's bytes


@dataclass
class Entry:
    key: str
    role: str  # "param" | "optimizer_state" | "buffer"
    dtype: str  # safetensors dtype name: BF16, F32, ...
    layout: str  # the live layout at save time: TILE or ROW_MAJOR
    global_shape: tuple[int, ...]
    canonical_shape: tuple[int, ...]
    transform: Transform
    label: Label
    live_keys: tuple[str, ...]
    chunks: list[ChunkRecord]
    param_ref: str | None = None  # optimizer state: the key of the parameter it belongs to
    state_path: tuple[str, ...] = ()  # optimizer state: the path of dict keys in the optimizer's state dict
    # Per dim, how many shards the mesh axes other than the tensor's FSDP axis split it into at save: the TP split,
    # which decides the row order of a fused weight (FSDP splits rows already in canonical order).
    tp_split: tuple[int, ...] = field(kw_only=True)


@dataclass
class FileRecord:
    path: str  # relative to the checkpoint directory
    nbytes: int
    world_rank: int
    mesh_id: int
    coord: tuple[int, ...]


@dataclass
class Index:
    checkpoint_id: str
    world: dict
    files: list[FileRecord]
    entries: dict[str, Entry]
    scalars: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    writer_software: dict = field(default_factory=dict)
    format: str = FORMAT
    version: int = VERSION

    def live_key_map(self) -> dict[str, str]:
        """Every live name any entry was seen under, mapped to its entry key."""
        mapping: dict[str, str] = {}
        for key, entry in self.entries.items():
            mapping[key] = key
            for live in entry.live_keys:
                mapping.setdefault(live, key)
        return mapping

    def to_json(self) -> dict:
        label_ids: dict[Label, int] = {}
        state = {
            key: _entry_to_json(entry, label_ids.setdefault(entry.label, len(label_ids)))
            for key, entry in self.entries.items()
        }
        return {
            "format": self.format,
            "version": self.version,
            "checkpoint_id": self.checkpoint_id,
            "created": self.created,
            "writer_software": self.writer_software,
            "world": self.world,
            "files": [
                {
                    "path": f.path,
                    "nbytes": f.nbytes,
                    "writer": {"world_rank": f.world_rank, "mesh_id": f.mesh_id, "coord": list(f.coord)},
                }
                for f in self.files
            ],
            "labels": [label.to_json() for label in label_ids],
            "state": state,
            "scalars": {key: _scalar_to_json(value) for key, value in self.scalars.items()},
        }

    @staticmethod
    def from_json(value: dict) -> "Index":
        if value.get("format") != FORMAT:
            raise VersionError(f"not a {FORMAT} index: format={value.get('format')!r}")
        version = int(value["version"])
        if version > VERSION:
            raise VersionError(f"index version {version} is newer than the supported {VERSION}")
        files = [
            FileRecord(
                f["path"],
                int(f["nbytes"]),
                int(f["writer"]["world_rank"]),
                int(f["writer"]["mesh_id"]),
                tuple(f["writer"]["coord"]),
            )
            for f in value["files"]
        ]
        labels = [Label.from_json(label) for label in value["labels"]]
        entries = {key: _entry_from_json(key, e, labels) for key, e in value["state"].items()}
        for key, entry in entries.items():
            for chunk in entry.chunks:
                if chunk.file >= len(files):
                    raise CheckpointError(
                        f"index entry {key!r}: chunk file {chunk.file} is not one of the index's {len(files)} files"
                    )
        scalars = {key: _scalar_from_json(v) for key, v in value["scalars"].items()}
        return Index(
            checkpoint_id=str(value["checkpoint_id"]),
            world=dict(value["world"]),
            files=files,
            entries=entries,
            scalars=scalars,
            created=str(value["created"]),
            writer_software=dict(value["writer_software"]),
            format=FORMAT,
            version=version,
        )


# --- columnar chunk table ---------------------------------------------------------------------------------------


def _broadcast(values: list) -> list:
    """A single element when every element is equal, else the list itself."""
    return values[:1] if values and all(v == values[0] for v in values) else values


def _expand(values: list, n: int) -> list:
    return values * n if len(values) == 1 and n != 1 else values


def _entry_to_json(entry: Entry, label_id: int) -> dict:
    chunks = entry.chunks
    table = {
        "n": len(chunks),
        "offsets": [list(c.box.offsets) for c in chunks],
        "sizes": _broadcast([list(c.box.sizes) for c in chunks]),
        "file": [c.file for c in chunks],
        "data_offset": [c.data_offset for c in chunks],
        "nbytes": _broadcast([c.nbytes for c in chunks]),
        "crc32": [f"{c.crc32:08x}" for c in chunks],
    }
    value = {
        "role": entry.role,
        "dtype": entry.dtype,
        "layout": entry.layout,
        "global_shape": list(entry.global_shape),
        "canonical_shape": list(entry.canonical_shape),
        "transform": entry.transform.to_json(),
        "label": label_id,
        "live_keys": list(entry.live_keys),
        "chunks": table,
    }
    if entry.param_ref is not None:
        value["param_ref"] = entry.param_ref
    if entry.state_path:
        value["state_path"] = list(entry.state_path)
    value["tp_split"] = list(entry.tp_split)
    return value


def _shape(key: str, what: str, value: Any, ndim: int | None = None) -> tuple[int, ...]:
    """``value`` from an index as a tuple of non-negative ints (``ndim`` of them when given), or a
    ``CheckpointError`` naming the entry."""
    if (
        not isinstance(value, list)
        or (ndim is not None and len(value) != ndim)
        or not all(type(v) is int and v >= 0 for v in value)
    ):
        raise CheckpointError(f"index entry {key!r}: malformed {what} {value!r}")
    return tuple(value)


def _transform(key: str, value: Any, global_shape: tuple[int, ...], canonical_shape: tuple[int, ...]) -> Transform:
    """An entry's transform, checked against its shapes: the dim it names within the tensor's rank (a negative or
    too large dim would index the wrong dim or none), and the canonical shape it gives from the global shape the one
    recorded."""
    transform = transform_from_json(value)
    dim = getattr(transform, "dim", None)
    if dim is not None and not 0 <= dim < len(global_shape):
        raise CheckpointError(
            f"index entry {key!r}: transform {transform.to_json()} names dim {dim} of a rank-{len(global_shape)} tensor"
        )
    try:
        canonical = transform.canonical_shape(global_shape)
    except LayoutError as err:
        raise LayoutError(f"index entry {key!r}: {err}") from None
    if tuple(canonical) != tuple(canonical_shape):
        raise CheckpointError(
            f"index entry {key!r}: its transform gives canonical shape {list(canonical)}, not the recorded "
            f"{list(canonical_shape)}"
        )
    return transform


def _entry_from_json(key: str, value: dict, labels: Sequence[Label]) -> Entry:
    # Boxes are built unchecked (``Box``), so every shape and box an index holds is checked here.
    global_shape = _shape(key, "global_shape", value["global_shape"])
    ndim = len(global_shape)
    table = value["chunks"]
    n = int(table["n"])
    offsets = table["offsets"]
    # A broadcast sizes column holds one row: checked once, then repeated.
    sizes = _expand([_shape(key, "chunk sizes", s, ndim) for s in table["sizes"]], n)
    files = _shape(key, "file column", table["file"])  # in range of the index's files: checked by Index.from_json
    data_offsets = table["data_offset"]
    nbytes = _expand(table["nbytes"], n)
    crcs = table["crc32"]
    if not (len(offsets) == len(sizes) == len(files) == len(data_offsets) == len(nbytes) == len(crcs) == n):
        raise CheckpointError(f"index entry {key!r}: chunk table columns disagree with n={n}")
    try:
        crc_values = [int(c, 16) for c in crcs]
    except (TypeError, ValueError):
        raise CheckpointError(f"index entry {key!r}: malformed crc32 column") from None
    chunks = [
        ChunkRecord(
            box=Box(_shape(key, "chunk offsets", offsets[i], ndim), sizes[i]),
            file=files[i],
            data_offset=int(data_offsets[i]),
            nbytes=int(nbytes[i]),
            crc32=crc_values[i],
        )
        for i in range(n)
    ]
    tp_split = value["tp_split"]
    if (
        not isinstance(tp_split, list)
        or len(tp_split) != ndim
        or not all(isinstance(s, int) and s >= 1 for s in tp_split)
    ):
        raise CheckpointError(f"index entry {key!r}: malformed tp_split {tp_split!r}")
    label_id = value["label"]
    if type(label_id) is not int or not 0 <= label_id < len(labels):
        raise CheckpointError(
            f"index entry {key!r}: label {label_id!r} is not a position in the index's table of {len(labels)} labels"
        )
    canonical_shape = _shape(key, "canonical_shape", value["canonical_shape"], ndim)
    transform = _transform(key, value["transform"], global_shape, canonical_shape)
    return Entry(
        key=key,
        role=str(value["role"]),
        dtype=str(value["dtype"]),
        layout=str(value["layout"]),
        global_shape=global_shape,
        canonical_shape=canonical_shape,
        transform=transform,
        label=labels[label_id],
        live_keys=tuple(value["live_keys"]),
        chunks=chunks,
        param_ref=value.get("param_ref"),
        state_path=tuple(value.get("state_path", [])),
        tp_split=tuple(tp_split),
    )


def _scalar_to_json(value: Any) -> dict:
    if isinstance(value, bool):
        return {"type": "bool", "value": value}
    if isinstance(value, int):
        return {"type": "int", "value": value}
    if isinstance(value, float):
        return {"type": "float", "value": value}
    if isinstance(value, str):
        return {"type": "str", "value": value}
    raise CheckpointError(f"scalar of type {type(value).__name__} cannot be stored in the index")


def _scalar_from_json(value: dict) -> Any:
    kind, raw = value["type"], value["value"]
    return {"bool": bool, "int": int, "float": float, "str": str}[kind](raw)


# --- naming -------------------------------------------------------------------------------------------------------

_DIR_NAME = re.compile(r"^(?:(?P<prefix>.+)_)?step_(?P<step>\d+)$")


def step_dir_name(step: int, prefix: str = "") -> str:
    """``step_0001000``, or ``<prefix>_step_0001000``. ``step`` must not be negative: ``parse_dir_name`` and
    ``find_latest`` would not recognize the name."""
    if int(step) < 0:
        raise ValueError(f"step_dir_name: step must not be negative, got {step}")
    name = f"step_{int(step):07d}"
    return f"{prefix}_{name}" if prefix else name


def parse_dir_name(name: str) -> tuple[str, int] | None:
    """``(prefix, step)`` of a checkpoint directory name, or None."""
    match = _DIR_NAME.match(os.path.basename(name.rstrip("/")))
    if not match:
        return None
    return match.group("prefix") or "", int(match.group("step"))


def is_complete(path: str) -> bool:
    """A final directory name (no staging or replaced suffix) that holds an ``index.json``.

    Only the file's presence is checked, so the call stays cheap for ``find_latest``; ``read_index`` raises
    ``IncompleteCheckpointError`` for an index that does not parse."""
    name = os.path.basename(path.rstrip("/"))
    if PARTIAL_MARK in name or REPLACED_MARK in name:
        return False
    return os.path.isfile(os.path.join(path, INDEX_NAME))


def find_latest(root: str, prefix: str = "", include_replaced: bool = False) -> str | None:
    """The complete checkpoint directory under ``root`` with the highest step, ordered by the step parsed from the
    name (never by mtime, which on NFS comes from the server clock).

    ``include_replaced`` also considers ``.replaced-`` directories, which ``finalize_staging`` leaves behind only if
    a process dies between moving the old checkpoint aside and moving the new one in. It is a recovery aid: a final
    directory always wins a step tie, and a replaced directory it returns must be renamed back to its final name
    before ``load`` (or ``is_complete``) accepts it.

    Directories left by saves that did not finish, ``.partial-`` ones of saves that died and ``.replaced-`` ones of
    interrupted overwrites, can each hold a whole checkpoint. Nothing removes them, since a ``.partial-`` directory may
    belong to a save still running in another job; this warns about those of ``prefix``, naming each."""
    best: tuple[int, int, str, str] | None = None
    leftovers: list[str] = []
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return None
    for name in names:
        candidate = name
        mark = REPLACED_MARK if REPLACED_MARK in name else PARTIAL_MARK if PARTIAL_MARK in name else None
        if mark is not None:
            candidate = name.split(mark, 1)[0]
            stem = parse_dir_name(candidate)
            if stem is not None and stem[0] == prefix:
                leftovers.append(name)
            if mark == PARTIAL_MARK or not include_replaced:
                continue
        parsed = parse_dir_name(candidate)
        if parsed is None or parsed[0] != prefix:
            continue
        path = os.path.join(root, name)
        if not os.path.isfile(os.path.join(path, INDEX_NAME)):
            continue
        rank = (parsed[1], 0 if REPLACED_MARK in name else 1, name, path)
        if best is None or rank > best:
            best = rank
    if leftovers:
        count = f"{len(leftovers)} director{'y' if len(leftovers) == 1 else 'ies'}"
        warnings.warn(
            f"{root} holds {count} left by saves that did not finish: {', '.join(sorted(leftovers))}. A .partial- "
            f"directory is the staging directory of a save that died, or of one still running; a .replaced- one is a "
            f"previous checkpoint that an interrupted overwrite moved aside. Remove them once no save into {root} is "
            f"running.",
            stacklevel=2,
        )
    return best[3] if best else None


# --- staging and JSON files ----------------------------------------------------------------------------------------


def staging_dir(final_dir: str, nonce: int) -> str:
    return f"{final_dir.rstrip('/')}{PARTIAL_MARK}{int(nonce)}"


def begin_staging(final_dir: str, nonce: int, shared: bool) -> str:
    """Create ``<final_dir>.partial-<nonce>/data`` and return the staging path. Save passes the nonce its ranks agreed
    on; with ``shared`` the directory may already exist, since every rank of a multi-rank save creates it."""
    staging = staging_dir(final_dir, nonce)
    os.makedirs(os.path.join(staging, DATA_DIR), exist_ok=shared)
    return staging


def finalize_staging(staging: str, final_dir: str, overwrite: bool = False) -> None:
    """Rename the staging directory to its final name after fsyncing it, then fsync the parent.

    Without ``overwrite``, a final directory that exists is refused, and the rename that commits the checkpoint is
    itself the last check: renaming onto a directory that is not empty fails, so a checkpoint another save commits
    meanwhile is refused here rather than replaced. With ``overwrite`` an existing final directory is first moved aside
    to ``<final>.replaced-<nonce>`` and deleted once the new one is in place; if it cannot be deleted (busy files on
    NFS), a warning names it. If moving the new one in fails, the old one is moved back before the error is raised, so
    a failed overwrite leaves the previous checkpoint where it was."""
    final_dir = final_dir.rstrip("/")
    if os.path.exists(final_dir) and not overwrite:
        raise CheckpointError(f"{final_dir} already exists; pass overwrite=True to replace it")
    _fsync_dir(staging)
    replaced = None
    if overwrite and os.path.exists(final_dir):
        replaced = f"{final_dir}{REPLACED_MARK}{secrets.randbits(31)}"
        os.rename(final_dir, replaced)
    try:
        os.rename(staging, final_dir)
    except BaseException as err:
        if replaced is not None:
            try:
                os.rename(replaced, final_dir)
            except OSError as rollback:
                raise CheckpointError(
                    f"could not move {staging} to {final_dir} ({err}), nor move the previous checkpoint back from "
                    f"{replaced} ({rollback}); rename it back by hand"
                ) from err
        if not overwrite and isinstance(err, OSError) and err.errno in (errno.EEXIST, errno.ENOTEMPTY, errno.ENOTDIR):
            raise CheckpointError(f"{final_dir} already exists; pass overwrite=True to replace it") from err
        raise
    _fsync_dir(os.path.dirname(final_dir) or ".")
    if replaced is not None:
        left: list[str] = []
        shutil.rmtree(replaced, onerror=lambda function, path, excinfo: left.append(path))
        if left:
            warnings.warn(
                f"{final_dir} replaced the previous checkpoint, but {len(left)} entr(ies) of it could not be removed "
                f"from {replaced} (first: {left[0]}); delete that directory by hand",
                stacklevel=2,
            )


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def check_json(name: str, value: Any) -> Any:
    """``value`` as plain JSON types (tuples become lists), or a ``CheckpointError`` naming ``name`` and the path of
    the first value that would not read back as written: a dict key that is not a string (JSON turns it into one), a
    float that is not finite (no standard JSON form), or a type JSON has no form for, which includes most numpy
    scalars (``np.int64``, ``np.float32``, ``np.bool_``). ``np.float64`` subclasses ``float``, so it is accepted and
    stored as a plain float. Checking before a save writes anything keeps such state from failing the save at its
    end, or from coming back changed on load."""

    def walk(node: Any, path: str) -> Any:
        where = path or "/"
        if node is None or isinstance(node, (bool, str)):
            return node
        if isinstance(node, int):
            return int(node)
        if isinstance(node, float):
            if not math.isfinite(node):
                raise CheckpointError(f"{name} at {where}: {node} is not finite and has no JSON form")
            return float(node)
        if isinstance(node, dict):
            out = {}
            for key, child in node.items():
                if not isinstance(key, str):
                    raise CheckpointError(
                        f"{name} at {where}: key {key!r} is a {type(key).__name__}, which JSON would turn into a string"
                    )
                out[key] = walk(child, f"{path}/{key}")
            return out
        if isinstance(node, (list, tuple)):
            return [walk(child, f"{path}[{i}]") for i, child in enumerate(node)]
        raise CheckpointError(f"{name} at {where}: a {type(node).__name__} has no JSON form")

    return walk(value, "")


def _atomic_write_text(path: str, text: str) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path) or ".")


@contextmanager
def _gc_paused() -> Iterator[None]:
    """The cyclic garbage collector paused, for building or parsing an index: both allocate millions of small objects
    that form no cycles, so the collections their allocation triggers would scan them for nothing."""
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def write_index(directory: str, index: Index) -> None:
    """``index.json`` as compact JSON: about a third of the indented size, and written by CPython's C encoder, which
    any ``indent`` replaces with the pure-Python one. Unlike ``header.json``, it is read by machines."""
    if not index.created:
        index.created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _gc_paused():
        text = json.dumps(index.to_json(), separators=(",", ":"))
    _atomic_write_text(os.path.join(directory, INDEX_NAME), text)


def _refuse_staging(directory: str) -> None:
    if PARTIAL_MARK in os.path.basename(os.path.normpath(directory)):
        raise IncompleteCheckpointError(
            f"{directory} is a staging directory; it becomes a checkpoint only when its save is finalized"
        )


def read_index(directory: str) -> Index:
    """The checkpoint's index as an ``Index``, the format's in-memory form, for inspecting a checkpoint without
    touching a device. A staging directory or one without a readable index is an ``IncompleteCheckpointError``."""
    _refuse_staging(directory)
    path = os.path.join(directory, INDEX_NAME)
    if not os.path.isfile(path):
        raise IncompleteCheckpointError(f"{directory} has no {INDEX_NAME}; it is not a complete checkpoint")
    try:
        with open(path, "r", encoding="utf-8") as f, _gc_paused():
            return Index.from_json(json.load(f))
    except CheckpointError as err:
        raise type(err)(f"{path}: {err}") from err
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as err:
        raise IncompleteCheckpointError(
            f"{path} is not a readable index ({type(err).__name__}: {err}); the checkpoint is corrupt or truncated"
        ) from err


def write_header(directory: str, header: dict) -> None:
    _atomic_write_text(os.path.join(directory, HEADER_NAME), json.dumps(header, indent=1))


def read_header_json(directory: str) -> dict:
    _refuse_staging(directory)
    path = os.path.join(directory, HEADER_NAME)
    if not os.path.isfile(path):
        raise IncompleteCheckpointError(f"{directory} has no {HEADER_NAME}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except ValueError as err:
        raise IncompleteCheckpointError(
            f"{path} is not readable JSON ({err}); the checkpoint is corrupt or truncated"
        ) from err


def _check_header(value: Any) -> dict:
    """``value`` if it is a header as save writes it: this format's name, a version no newer than the supported one and
    a ``user`` object. Raises VersionError for another format or a newer version, KeyError or TypeError otherwise."""
    if not isinstance(value, dict):
        raise TypeError(f"the header is a {type(value).__name__}, not an object")
    if value.get("format") != FORMAT:
        raise VersionError(f"not a {FORMAT} header: format={value.get('format')!r}")
    version = value["version"]
    if type(version) is not int:
        raise TypeError(f"header version {version!r} is not an integer")
    if version > VERSION:
        raise VersionError(f"header version {version} is newer than the supported {VERSION}")
    if not isinstance(value["user"], dict):
        raise TypeError(f"the user header is a {type(value['user']).__name__}, not an object")
    return value


def read_checked_header(directory: str) -> dict:
    """header.json, checked as ``read_index`` checks index.json: a header save would not write is refused."""
    value = read_header_json(directory)
    path = os.path.join(directory, HEADER_NAME)
    try:
        return _check_header(value)
    except CheckpointError as err:
        raise type(err)(f"{path}: {err}") from err
    except (KeyError, TypeError) as err:
        raise IncompleteCheckpointError(
            f"{path} is not a readable header ({type(err).__name__}: {err}); the checkpoint is corrupt or truncated"
        ) from err


def read_user_header(directory: str) -> dict:
    """The custom user-defined header dict passed to ``save``, without touching any device."""
    return read_checked_header(directory)["user"]


def data_file_name(world_rank: int, mesh_id: int, coord: Sequence[int]) -> str:
    """``data/w000_m0_d1-2.safetensors`` for world rank 0, mesh 0, coordinate (1, 2)."""
    return f"{DATA_DIR}/w{int(world_rank):03d}_m{int(mesh_id)}_d{'-'.join(str(int(c)) for c in coord)}.safetensors"
