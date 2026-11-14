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
and its CRC. Readers reject a higher major version and ignore unknown fields.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
import secrets
from dataclasses import dataclass, field
from typing import Any, Sequence

from ._boxes import Box
from ._errors import CheckpointError, IncompleteCheckpoint, VersionError
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
    writer_coord: tuple[int, ...]
    crc32: int | None = None


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
            "state": {key: _entry_to_json(entry) for key, entry in self.entries.items()},
            "scalars": {key: _scalar_to_json(value) for key, value in self.scalars.items()},
        }

    @staticmethod
    def from_json(value: dict) -> "Index":
        if value.get("format") != FORMAT:
            raise VersionError(f"not a {FORMAT} index: format={value.get('format')!r}")
        version = int(value.get("version", 0))
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
            for f in value.get("files", [])
        ]
        entries = {key: _entry_from_json(key, e) for key, e in value.get("state", {}).items()}
        scalars = {key: _scalar_from_json(v) for key, v in value.get("scalars", {}).items()}
        return Index(
            checkpoint_id=str(value.get("checkpoint_id", "")),
            world=dict(value.get("world", {})),
            files=files,
            entries=entries,
            scalars=scalars,
            created=str(value.get("created", "")),
            writer_software=dict(value.get("writer_software", {})),
            format=FORMAT,
            version=version,
        )


# --- columnar chunk table ---------------------------------------------------------------------------------------


def _broadcast(values: list) -> list:
    """A single element when every element is equal, else the list itself."""
    return values[:1] if values and all(v == values[0] for v in values) else values


def _expand(values: list, n: int) -> list:
    return values * n if len(values) == 1 and n != 1 else values


def _entry_to_json(entry: Entry) -> dict:
    chunks = entry.chunks
    table = {
        "n": len(chunks),
        "offsets": [list(c.box.offsets) for c in chunks],
        "sizes": _broadcast([list(c.box.sizes) for c in chunks]),
        "file": [c.file for c in chunks],
        "data_offset": [c.data_offset for c in chunks],
        "nbytes": _broadcast([c.nbytes for c in chunks]),
        "writer_coord": [list(c.writer_coord) for c in chunks],
    }
    if any(c.crc32 is not None for c in chunks):
        table["crc32"] = [None if c.crc32 is None else f"{c.crc32:08x}" for c in chunks]
    value = {
        "role": entry.role,
        "dtype": entry.dtype,
        "layout": entry.layout,
        "global_shape": list(entry.global_shape),
        "canonical_shape": list(entry.canonical_shape),
        "transform": entry.transform.to_json(),
        "label": entry.label.to_json(),
        "live_keys": list(entry.live_keys),
        "chunks": table,
    }
    if entry.param_ref is not None:
        value["param_ref"] = entry.param_ref
    if entry.state_path:
        value["state_path"] = list(entry.state_path)
    return value


def _entry_from_json(key: str, value: dict) -> Entry:
    table = value["chunks"]
    n = int(table["n"])
    offsets = table["offsets"]
    sizes = _expand(table["sizes"], n)
    files = table["file"]
    data_offsets = table["data_offset"]
    nbytes = _expand(table["nbytes"], n)
    coords = table["writer_coord"]
    crcs = table.get("crc32")
    if not (len(offsets) == len(sizes) == len(files) == len(data_offsets) == len(nbytes) == len(coords) == n) or (
        crcs is not None and len(crcs) != n
    ):
        raise CheckpointError(f"index entry {key!r}: chunk table columns disagree with n={n}")
    try:
        crc_values = [None if c is None else int(c, 16) for c in crcs] if crcs is not None else [None] * n
    except (TypeError, ValueError):
        raise CheckpointError(f"index entry {key!r}: malformed crc32 column") from None
    chunks = [
        ChunkRecord(
            box=Box(tuple(offsets[i]), tuple(sizes[i])),
            file=int(files[i]),
            data_offset=int(data_offsets[i]),
            nbytes=int(nbytes[i]),
            writer_coord=tuple(coords[i]),
            crc32=crc_values[i],
        )
        for i in range(n)
    ]
    return Entry(
        key=key,
        role=str(value["role"]),
        dtype=str(value["dtype"]),
        layout=str(value.get("layout", "TILE")),
        global_shape=tuple(value["global_shape"]),
        canonical_shape=tuple(value["canonical_shape"]),
        transform=transform_from_json(value["transform"]),
        label=Label.from_json(value["label"]),
        live_keys=tuple(value.get("live_keys", [key])),
        chunks=chunks,
        param_ref=value.get("param_ref"),
        state_path=tuple(value.get("state_path", [])),
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
    """``step_0001000``, or ``<prefix>_step_0001000``."""
    name = f"step_{int(step):07d}"
    return f"{prefix}_{name}" if prefix else name


def parse_dir_name(name: str) -> tuple[str, int] | None:
    """``(prefix, step)`` of a checkpoint directory name, legacy unpadded names included, or None."""
    match = _DIR_NAME.match(os.path.basename(name.rstrip("/")))
    if not match:
        return None
    return match.group("prefix") or "", int(match.group("step"))


def is_complete(path: str) -> bool:
    """A final directory name (no staging or replaced suffix) that holds an ``index.json``.

    Only the file's presence is checked, so the call stays cheap for ``find_latest``; ``read_index`` raises
    ``IncompleteCheckpoint`` for an index that does not parse."""
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
    before ``load`` (or ``is_complete``) accepts it."""
    best: tuple[int, int, str, str] | None = None
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        return None
    for name in names:
        candidate = name
        if REPLACED_MARK in name:
            if not include_replaced:
                continue
            candidate = name.split(REPLACED_MARK, 1)[0]
        elif PARTIAL_MARK in name:
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
    return best[3] if best else None


# --- staging and JSON files ----------------------------------------------------------------------------------------


def begin_staging(final_dir: str) -> tuple[str, int]:
    """Create ``<final_dir>.partial-<nonce>/data``; returns the staging path and the nonce."""
    nonce = secrets.randbits(31)
    staging = f"{final_dir.rstrip('/')}{PARTIAL_MARK}{nonce}"
    os.makedirs(os.path.join(staging, DATA_DIR), exist_ok=False)
    return staging, nonce


def finalize_staging(staging: str, final_dir: str, overwrite: bool = False) -> None:
    """Rename the staging directory to its final name after fsyncing it, then fsync the parent.

    With ``overwrite`` an existing final directory is first moved aside to ``<final>.replaced-<nonce>`` and deleted
    once the new one is in place. If moving the new one in fails, the old one is moved back before the error is
    raised, so a failed overwrite leaves the previous checkpoint where it was."""
    final_dir = final_dir.rstrip("/")
    if os.path.exists(final_dir) and not overwrite:
        raise CheckpointError(f"{final_dir} already exists; pass overwrite=True to replace it")
    _fsync_dir(staging)
    replaced = None
    if os.path.exists(final_dir):
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
        raise
    _fsync_dir(os.path.dirname(final_dir) or ".")
    if replaced is not None:
        import shutil

        shutil.rmtree(replaced, ignore_errors=True)


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


def _atomic_write_json(path: str, value: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=1, sort_keys=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(os.path.dirname(path) or ".")


def write_index(directory: str, index: Index) -> None:
    if not index.created:
        index.created = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _atomic_write_json(os.path.join(directory, INDEX_NAME), index.to_json())


def read_index(directory: str) -> Index:
    path = os.path.join(directory, INDEX_NAME)
    if not os.path.isfile(path):
        raise IncompleteCheckpoint(f"{directory} has no {INDEX_NAME}; it is not a complete checkpoint")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Index.from_json(json.load(f))
    except (ValueError, KeyError, TypeError, IndexError, AttributeError) as err:
        raise IncompleteCheckpoint(
            f"{path} is not a readable index ({type(err).__name__}: {err}); the checkpoint is corrupt or truncated"
        ) from err


def write_header(directory: str, header: dict) -> None:
    _atomic_write_json(os.path.join(directory, HEADER_NAME), header)


def read_header(directory: str) -> dict:
    path = os.path.join(directory, HEADER_NAME)
    if not os.path.isfile(path):
        raise IncompleteCheckpoint(f"{directory} has no {HEADER_NAME}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except ValueError as err:
        raise IncompleteCheckpoint(
            f"{path} is not readable JSON ({err}); the checkpoint is corrupt or truncated"
        ) from err


def read_user_header(directory: str) -> dict:
    """The custom user-defined header dict passed to ``save``, without touching any device."""
    return read_header(directory).get("user", {})


def data_file_name(world_rank: int, mesh_id: int, coord: Sequence[int]) -> str:
    """``data/w000_m0_d1-2.safetensors`` for world rank 0, mesh 0, coordinate (1, 2)."""
    return f"{DATA_DIR}/w{int(world_rank):03d}_m{int(mesh_id)}_d{'-'.join(str(int(c)) for c in coord)}.safetensors"
