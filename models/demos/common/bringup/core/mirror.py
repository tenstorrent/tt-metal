# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""A shared dir (the master: NFS, readable by other users on other boxes) and its node-local mirror (what runs read).
One module for both users of the pattern (owner 2026-11-06: one copy per thing, accessible in the shared data folder):

  F260 goldens (reference/golden.py): the shared copy is the source of truth, written only where the goldens are made;
       the mirror is read only when it matches the shared files exactly (mirror_problem), else the shared copy.
  F261 weight cache (core/weight_cache.py): either side may hold files the other lacks (a box builds a layer's cache,
       another pulls it); files are named by their content key, so the newer copy of a name wins and nothing is ever
       overwritten by an older one (plan: pull = what the master has that the mirror lacks, push = the reverse).

Shared pieces: ``listing`` (a stat walk: {relative path: [size, mtime_ns]}), the ``mirror.json`` manifest beside the
files (``write_manifest``; never part of a listing), ``rsync`` (nice + idle I/O class, temp file + rename per file, so
a reader never sees a partial file under its final name; optionally o+rX for other users) and ``compare``. Names a
writer has not renamed into place yet (``is_temp``: dot files, rsync's ``.name.XXXXXX``, ``*.tmp*``, ``*.part``) are
never listed when the caller filters (weight cache); the goldens keep their exact listing (no filter).
"""

from __future__ import annotations

import fnmatch
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

MANIFEST = "mirror.json"


def is_temp(rel: str) -> bool:
    """A name some writer has not renamed into place yet (or a lock / log of the mirror's own)."""
    name = os.path.basename(rel)
    return name.startswith(".") or ".tmp" in name or name.endswith(".part")


def matcher(patterns) -> callable:
    """keep(rel) for listing: the basename matches one of ``patterns`` (fnmatch) and is not a temp name."""
    pats = list(patterns or ["*"])
    return lambda rel: not is_temp(rel) and any(fnmatch.fnmatchcase(os.path.basename(rel), p) for p in pats)


def listing(d: Path, keep=None) -> dict[str, list[int]]:
    """{relative path: [size, mtime_ns]} of every file under d but the manifest (``keep(rel)``: only those). A missing
    dir lists nothing."""
    d = Path(d)
    out = {}
    for root, dirs, files in os.walk(d):
        dirs.sort()
        for f in sorted(files):
            p = Path(root) / f
            rel = str(p.relative_to(d))
            if rel == MANIFEST or (keep is not None and not keep(rel)):
                continue
            try:
                st = p.stat()
            except FileNotFoundError:  # (renamed or removed under the walk)
                continue
            out[rel] = [st.st_size, st.st_mtime_ns]
    return out


def read_json(p: Path) -> dict | None:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def write_json(p: Path, obj, readable: bool = False) -> None:
    """Atomic (tmp + rename); ``readable``: o+r, for other users of a shared dir."""
    p = Path(p)
    tmp = p.with_name(f".{p.name}.tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=0, sort_keys=True))
    if readable:
        tmp.chmod(tmp.stat().st_mode | stat.S_IRGRP | stat.S_IROTH)
    os.replace(tmp, p)


def write_manifest(d: Path, man: dict, readable: bool = False) -> bool:
    """mirror.json in d, rewritten only when it changed (an in-sync mirror stays in sync). False when d is not
    writable (another user's shared copy): the caller keeps a manifest on its own side."""
    d = Path(d)
    if read_json(d / MANIFEST) == json.loads(json.dumps(man)):
        return True
    try:
        write_json(d / MANIFEST, man, readable)
        return True
    except OSError:
        return False


def mirror_problem(shared: Path, mirror: Path, keep=None) -> str | None:
    """F260: why ``mirror`` cannot be read in place of ``shared`` (None: complete). Its mirror.json (written by the
    sync) must list the shared files as they are now, and every mirror file must match its line."""
    man = read_json(Path(mirror) / MANIFEST)
    if man is None:
        return "not synced" if not Path(mirror).exists() else "no mirror.json (not synced)"
    try:
        live = listing(shared, keep)
    except OSError as e:
        live = None
        if Path(shared).exists():
            return f"shared copy unreadable ({e})"
    if live is not None and man.get("files") != live:
        diff = sorted(set(live) ^ set(man.get("files") or {})) or sorted(
            k for k in live if live[k] != man["files"].get(k)
        )
        return f"stale: the shared copy changed since the last sync ({len(diff)} files, e.g. {diff[0]})"
    for rel, (size, mtime) in (man.get("files") or {}).items():
        try:
            st = (Path(mirror) / rel).stat()
        except OSError:
            return f"incomplete: {rel} missing"
        if st.st_size != size or st.st_mtime_ns != mtime:
            return f"incomplete: {rel} differs from its mirror.json line"
    return None


def compare(src: dict, dst: dict) -> dict[str, list[str]]:
    """What copying src -> dst would do, for files named by their content (F261): ``copy`` = in src only, or a
    different size with src newer; ``kept_newer`` = a different size with dst newer (never overwritten); ``same`` =
    the same size (the same name and size: the same content; a different mtime is two boxes' builds of it)."""
    out = {"copy": [], "kept_newer": [], "same": []}
    for rel, (size, mtime) in sorted(src.items()):
        d = dst.get(rel)
        if d is None or (d[0] != size and mtime > d[1]):
            out["copy"].append(rel)
        elif d[0] != size:
            out["kept_newer"].append(rel)
        else:
            out["same"].append(rel)
    return out


def rsync(src: Path, dst: Path, files: list[str] | None = None, *, update: bool = False, readable: bool = False):
    """rsync -a src/ -> dst/ (``files``: only these relative paths), at nice 10 and the idle I/O class (a running
    test's reads come first). Each file goes to a temp name in its dir and is renamed into place (rsync's default,
    never --inplace). ``update``: never overwrite a file that is newer on dst. ``readable``: g+rX and o+rX on what it
    writes (a shared dir other users read)."""
    Path(dst).mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-a"]
    if update:
        cmd.append("--update")
    if readable:
        cmd.append("--chmod=Dgo+rx,Fgo+r")
    if files is not None:
        if not files:
            return
        cmd.append("--files-from=-")
    cmd += [f"{src}/", f"{dst}/"]
    if shutil.which("ionice"):
        cmd = ["ionice", "-c", "3", *cmd]  # idle I/O class
    subprocess.run(
        ["nice", "-n", "10", *cmd], check=True, input="\n".join(files) + "\n" if files is not None else None, text=True
    )


def make_readable(d: Path, top: Path) -> list[str]:
    """o+rx on d and its parents up to top (the dirs a copy created), when owned; the parents above top that others
    cannot traverse (no o+x), as a note (F261: other users read the master)."""
    d, top = Path(d), Path(top)
    for p in [d, *d.parents]:
        try:
            if p.stat().st_uid == os.getuid():
                p.chmod(p.stat().st_mode | 0o055)
        except OSError:
            pass
        if p == top:
            break
    return [str(p) for p in top.parents if p != Path("/") and not (p.stat().st_mode & stat.S_IXOTH)]
