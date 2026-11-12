# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F265: a ledger file that grows with every task, kept as chunk files below the repo's 500 KB limit.

The owner (2026-11-12, FIX.27): "split the file, if this is something that gets appended to split it into chunks, do
not compact". DeepSeek-V4.1 FIX.26's passed gate could not commit: state_metrics.json had reached 515,706 bytes.

A store is one JSON object ({task id: record}) that used to live in ``<name>.json``. It now lives in
``<name>.chunks/0001.json``, ``0002.json``, ...; every chunk is a JSON object, the chunks hold disjoint keys and their
union is the object. Nothing is compacted, rounded or dropped: a record is written exactly as the caller gives it.

The chunk rule (``write``):
- a key already in a chunk stays in that chunk (its record is rewritten in place);
- a new key goes to the newest chunk, or starts a new chunk when the newest would pass ``CHUNK_BYTES``;
- a chunk that passes ``CHUNK_BYTES`` because one of its records grew moves its last keys to a new chunk;
- a single record larger than ``CHUNK_BYTES`` gets a chunk of its own (a key is never split);
- a chunk left empty is deleted; numbers are never reused for different content in one write.
Only chunk files whose bytes change are rewritten.

Migration: a store with no chunk directory reads the old single file (legacy); its first ``write`` creates the chunks
and deletes the old file. ``migrate`` does that once without changing a record. Every reader and writer goes through
this class (core/ledger.py: state.json, state_metrics.json, history_state.json); the path patterns of the records
(core/stamps.py, orchestrator.py IGNORED, gate.stage_paths) name both the chunk files and the old file.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

CHUNK_BYTES = 256 * 1024  # owner: well below the repo's 500 KB hook (scripts/check_file_size.py)
REPO_LIMIT_BYTES = 500 * 1024  # scripts/check_file_size.py: ceil(size / 1024) > 500 KB is refused
CHUNK_SUFFIX = ".chunks"
_CHUNK = re.compile(r"^(\d{4,})\.json$")
MANIFEST = "MIGRATED.json"  # what migrate() moved: the old file's size and hash, a hash per key (selftest evidence)


def key_hash(value) -> str:
    """A record's content hash (canonical JSON), so a manifest proves a key unchanged without keeping its bytes."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def _git_knows(path: Path) -> bool:
    """The path is in git's index or in HEAD (False outside a work tree, or without git)."""
    import subprocess

    cwd = path.parent if path.parent.is_dir() else Path.cwd()
    for cmd in (
        ["git", "ls-files", "--error-unmatch", "--", str(path)],
        ["git", "cat-file", "-e", f"HEAD:./{path.name}"],
    ):
        try:
            r = subprocess.run(cmd, cwd=cwd, capture_output=True)
        except OSError:
            return False
        if r.returncode == 0:
            return True
    return False


class ChunkError(ValueError):
    pass


class ChunkedJson:
    def __init__(
        self,
        legacy: Path,
        indent: int | None = 1,
        bound: int = CHUNK_BYTES,
        merge_fields: tuple[str, ...] | None = None,
    ):
        """merge_fields: how a late old-format file merges over the chunks (``read``): None replaces a key's record;
        a tuple merges a record field by field, and the named fields are append-only lists (kept, then extended)."""
        self.legacy = Path(legacy)
        self.dir = self.legacy.with_name(self.legacy.stem + CHUNK_SUFFIX)
        self.manifest = self.dir / MANIFEST
        self.indent, self.bound, self.merge_fields = indent, bound, merge_fields

    # ---- layout
    def dump(self, obj: dict) -> str:
        """A chunk's text: the format the single file had (sorted keys, the store's indent)."""
        return json.dumps(obj, indent=self.indent, sort_keys=True) + "\n"

    def chunk_files(self) -> list[Path]:
        if not self.dir.is_dir():
            return []
        return sorted((p for p in self.dir.iterdir() if _CHUNK.match(p.name)), key=lambda p: int(p.stem))

    def files(self) -> list[Path]:
        """Every file that holds part of the store now (the chunks, or the old single file)."""
        return self.chunk_files() or ([self.legacy] if self.legacy.exists() else [])

    def paths(self) -> list[Path]:
        """What a commit stages for this store: the chunk directory, and the old file only while it is on disk or git
        still has it (in the index, or in HEAD with its deletion staged): its deletion is committed once, after which
        the old name is never listed again (OV-38: FIX.27's gate commit named a file git no longer knew)."""
        return [self.dir] + ([self.legacy] if self.legacy.exists() or _git_knows(self.legacy) else [])

    def exists(self) -> bool:
        return bool(self.files())

    # ---- read
    def chunks(self) -> list[tuple[Path, dict]]:
        out = []
        for p in self.chunk_files():
            obj = json.loads(p.read_text())
            if not isinstance(obj, dict):
                raise ChunkError(f"{p}: a chunk is a JSON object, got {type(obj).__name__}")
            out.append((p, obj))
        return out

    def read(self) -> dict:
        chunks = self.chunks()
        legacy = json.loads(self.legacy.read_text()) if self.legacy.exists() else None
        if not chunks:
            return legacy or {}
        merged, where = {}, {}
        for p, obj in chunks:
            for k, v in obj.items():
                if k in where:
                    raise ChunkError(f"{self.dir}: key {k!r} in both {where[k].name} and {p.name}")
                where[k] = p
                merged[k] = v
        if legacy:  # a process still running the pre-F265 code wrote the old file after the migration: newer, merged in
            merged = self._merge_late(merged, legacy)
        return merged

    def _merge_late(self, merged: dict, late: dict) -> dict:
        """The old code read no old file after the migration, so its file holds only what it wrote since: a record
        replaces the chunks' (merge_fields None), or its fields do, its append-only lists extending the chunks'."""
        out = dict(merged)
        for k, v in late.items():
            old = out.get(k)
            if self.merge_fields is None or not (isinstance(old, dict) and isinstance(v, dict)):
                out[k] = v
                continue
            rec = dict(old)
            for f, x in v.items():
                y = old.get(f)
                if f in self.merge_fields and isinstance(x, list) and isinstance(y, list):
                    rec[f] = x if x[: len(y)] == y else y + [e for e in x if e not in y]
                else:
                    rec[f] = x
            out[k] = rec
        return out

    # ---- write
    def _size(self, obj: dict) -> int:
        return len(self.dump(obj).encode())

    def plan(self, data: dict) -> list[tuple[str, dict]]:
        """(file name, chunk) for ``data`` given the chunks on disk (the rule in the module docstring): a kept chunk
        keeps its file name, a split-off tail or a new chunk takes the next free number."""
        old = self.chunks()
        nxt = max((int(p.stem) for p, _ in old), default=0) + 1
        placed = {k for _, obj in old for k in obj}
        out: list[list] = []
        for p, obj in old:
            c = {k: data[k] for k in obj if k in data}  # a kept key stays where it is
            name = p.name
            while len(c) > 1 and self._size(c) > self.bound:  # a grown record: the last keys move to a new chunk
                tail = {}
                while len(c) > 1 and self._size(c) > self.bound:
                    k = list(c)[-1]
                    tail = {k: c.pop(k), **tail}
                out.append([name, c])
                name, nxt, c = f"{nxt:04d}.json", nxt + 1, tail
            if c:
                out.append([name, c])
        out.sort(key=lambda nc: int(nc[0][:-5]))  # by number: the last one is the newest chunk
        for k in data:  # new keys, in the caller's order, onto the newest chunk
            if k in placed:
                continue
            if out and self._size({**out[-1][1], k: data[k]}) <= self.bound:
                out[-1][1][k] = data[k]
            else:
                out.append([f"{nxt:04d}.json", {k: data[k]}])
                nxt += 1
        return [(n, c) for n, c in out]

    def _put(self, path: Path, text: str) -> None:
        if path.exists() and path.read_text() == text:
            return  # only the chunks whose bytes change are rewritten
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text)
        tmp.replace(path)

    def write(self, data: dict) -> list[Path]:
        """Write ``data`` as chunks; the old single file, if any, is deleted after. Returns the chunk files."""
        if not isinstance(data, dict):
            raise ChunkError(f"{self.legacy.name}: a store holds a JSON object")
        plan = self.plan(data)
        self.dir.mkdir(parents=True, exist_ok=True)
        keep = set()
        for name, c in plan:
            self._put(self.dir / name, self.dump(c))
            keep.add(name)
        for p in self.chunk_files():
            if p.name not in keep:
                p.unlink()  # its keys were all removed by the caller
        if self.legacy.exists():
            self.legacy.unlink()
        return self.chunk_files()

    def migrate(self) -> list[Path]:
        """Turn the old single file into chunks once, without changing a record: the merged chunks must equal the
        old file (as parsed JSON, key order included) before it is deleted."""
        if self.chunk_files() or not self.legacy.exists():
            return self.chunk_files()
        raw = self.legacy.read_bytes()
        before = json.loads(raw)
        self.dir.mkdir(parents=True, exist_ok=True)
        for name, c in self.plan(before):
            self._put(self.dir / name, self.dump(c))
        after = self.read()
        if after != before:
            for p in self.chunk_files():
                p.unlink()
            raise ChunkError(f"{self.legacy}: the chunks do not equal the old file; nothing migrated")
        man = {
            "source": self.legacy.name,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "git_blob": "",
            "t": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "chunks": {p.name: p.stat().st_size for p in self.chunk_files()},
            "keys": {k: key_hash(v) for k, v in before.items()},
        }
        try:  # the old file's git object id: `git cat-file -p <id>` shows it while git keeps the object
            import subprocess

            man["git_blob"] = (
                subprocess.run(
                    ["git", "hash-object", "-w", "--stdin"], input=raw, capture_output=True, cwd=self.dir, check=True
                )
                .stdout.decode()
                .strip()
            )
        except (OSError, subprocess.CalledProcessError):
            pass
        self.manifest.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        self.legacy.unlink()
        return self.chunk_files()


def over_limit(paths: list[Path], limit: int = REPO_LIMIT_BYTES) -> list[tuple[Path, int]]:
    """The files the repo's large-file hook would refuse (ceil(size / 1024) > limit / 1024)."""
    out = []
    for p in paths:
        if p.is_file():
            size = p.stat().st_size
            if -(-size // 1024) > limit // 1024:
                out.append((p, size))
    return out


# ---- append-only text logs (BREADCRUMBS.md, supervision.md)
LOG_KEEP = CHUNK_BYTES // 4  # what the live file keeps after a roll: its newest sections
_SECTION = b"\n## "


def log_chunks(path: Path) -> list[Path]:
    """The rolled chunks of a log, oldest first: ``<stem>.0001<suffix>``, ``<stem>.0002<suffix>``, ..."""
    path = Path(path)
    pat = re.compile(rf"^{re.escape(path.stem)}\.(\d{{4,}}){re.escape(path.suffix)}$")
    if not path.parent.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in path.parent.iterdir() for m in [pat.match(p.name)] if m]
    return [p for _, p in sorted(found)]


def read_log(path: Path) -> str:
    """The whole log: its chunks in order, then the live file (the bytes as written, F265)."""
    path = Path(path)
    parts = [p.read_bytes() for p in log_chunks(path)] + ([path.read_bytes()] if path.exists() else [])
    return b"".join(parts).decode(errors="replace")


def _pieces(data: bytes, bound: int) -> list[bytes]:
    """``data`` cut at section starts (``\\n## ``, else line ends) into pieces of at most ``bound`` bytes where a cut
    exists; a single longer section stays one piece only when it has no line end to cut at."""
    out, start = [], 0
    while len(data) - start > bound:
        window = data[start : start + bound + 1]
        cut = window.rfind(_SECTION)
        if cut <= 0:
            cut = window.rfind(b"\n")
        if cut <= 0:
            nl = data.find(b"\n", start + bound)
            if nl < 0:
                break
            cut = nl - start
        cut += 1  # the newline stays with the earlier piece
        out.append(data[start : start + cut])
        start += cut
    out.append(data[start:])
    return out


def roll_log(path: Path, bound: int = CHUNK_BYTES, keep: int = LOG_KEEP) -> list[Path]:
    """F265: a log past ``bound`` keeps its newest sections (at most ``keep`` bytes where a section start allows) and
    moves everything before them, unchanged, to new chunk files below ``bound``. The chunks in order then the live
    file are the log's bytes exactly (``read_log``). Returns the chunk files written (none when under the bound)."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= bound:
        return []
    data = path.read_bytes()
    cut = len(data)
    i = data.rfind(_SECTION)
    while i > 0 and len(data) - (i + 1) <= keep:
        cut = i + 1
        i = data.rfind(_SECTION, 0, i)
    if cut == len(data):  # the newest section alone is longer than keep: it stays live, whole
        cut = data.rfind(_SECTION) + 1
    if cut <= 0:
        return []  # one section, nothing to cut at
    head, tail = data[:cut], data[cut:]
    nxt = max((int(p.name[len(path.stem) + 1 :].split(".")[0]) for p in log_chunks(path)), default=0) + 1
    written = []
    for piece in _pieces(head, bound):
        out = path.with_name(f"{path.stem}.{nxt:04d}{path.suffix}")
        out.write_bytes(piece)
        written.append(out)
        nxt += 1
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(tail)
    tmp.replace(path)
    return written
