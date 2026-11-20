# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F261 (owner 2026-11-06): the weight cache has a shared master and a node-local mirror, like the goldens (F260).

    paths.tt_cache_shared   the master (e.g. /data/<owner>/bringup/<model>/tt_cache on NFS, readable by other users on
                            other boxes): spec.tt_cache_shared(), the same <mesh>/<version> below it as the local cache
    paths.tt_cache          the local cache under art (spec.tt_cache()): the only copy runs read and write

A model's loader never reads the master (NFS is ~100 MB/s; a DS41 layer is 7.6 GB). Instead:

  pull   master -> local: the files the master has that the local cache lacks (or holds at another size, older). At
         run / resume start (orchestrator), before any device task, blocking: a pull beats a rebuild from the
         checkpoint. Needs the room on the local disk (checked first).
  push   local -> master: the files a run built that the master lacks (or holds at another size, older). After a task
         passes, in the background (one push at a time a box: a lock file in the local cache), and as a CLI the owner
         runs once. Never overwrites a newer master file (rsync --update; the plan picks only local-newer files),
         atomic per file (rsync writes a temp name in the file's dir and renames it, so a reader of the master never
         sees a partial file under its final name), files g+r o+r and dirs g+rx o+rx (other users read the master).
         A box whose user cannot write the master (another user's) only pulls.

Which files: the model's hook ``weight_cache_patterns(spec)`` (basename globs; default every file), never a temp name
(core/mirror.is_temp: dot files, ``*.tmp*``, ``*.part``). Cache files are named by their content key (checkpoint hash,
dtype, layout hash), so a name at the same size on both sides is the same file (not copied, whatever the mtime) and
every file a model writes is renamed into place: a file under its final name is complete. A layer half-built when a
push runs has its finished files pushed and the rest on the next push; a box that pulls a partial layer misses on it
and rebuilds it. Each side keeps ``mirror.json`` (core/mirror.py): the master's listing after the last push / pull.

    python -m models.demos.common.bringup.core.weight_cache status --spec S
    python -m models.demos.common.bringup.core.weight_cache pull --spec S
    python -m models.demos.common.bringup.core.weight_cache push --spec S      # e.g. once, to create the master
"""

from __future__ import annotations

import argparse
import fcntl
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from models.demos.common.bringup.core import mirror as MR

LOCK = ".weight_cache_push.lock"  # in the local cache (a dot name: never listed, never copied)
NFS_MBPS = 100  # for the time estimates only


def roots(spec) -> tuple[Path, Path | None]:
    """(the local cache, its shared master or None)."""
    return Path(spec.tt_cache()), spec.tt_cache_shared()


def keep(spec):
    """The model's filter (hooks.weight_cache_patterns), else every file but temp names."""
    try:
        hooks = spec.hooks()
    except Exception:  # noqa: BLE001 -- a spec without importable hooks mirrors every file
        hooks = None
    pats = hooks.weight_cache_patterns(spec) if hasattr(hooks, "weight_cache_patterns") else ["*"]
    return MR.matcher(pats)


def plan(spec, patterns=None) -> dict:
    """What a pull and a push would copy: {"pull": [...], "push": [...], "kept_newer_local": [...],
    "kept_newer_master": [...], "bytes_pull", "bytes_push", "local", "master"} (relative paths)."""
    local, shared = roots(spec)
    k = MR.matcher(patterns) if patterns else keep(spec)
    lo = MR.listing(local, k)
    ma = MR.listing(shared, k) if shared is not None and shared.is_dir() else {}
    down, up = MR.compare(ma, lo), MR.compare(lo, ma)
    return {
        "pull": down["copy"],
        "push": up["copy"],
        "kept_newer_local": down["kept_newer"],
        "kept_newer_master": up["kept_newer"],
        "bytes_pull": sum(ma[r][0] for r in down["copy"]),
        "bytes_push": sum(lo[r][0] for r in up["copy"]),
        "local": lo,
        "master": ma,
    }


def _gb(n: int) -> str:
    return f"{n / 1e9:.1f} GB (~{n / (NFS_MBPS * 1e6) / 60:.0f} min at {NFS_MBPS} MB/s)"


def pull(spec, *, echo=print, patterns=None) -> str:
    """master -> local (see the module doc). Returns a one-line result ("FAILED: ..." on a failure)."""
    local, shared = roots(spec)
    if shared is None:
        return "no master (paths.tt_cache_shared unset)"
    if not shared.is_dir():
        return f"no master yet ({shared}): push creates it"
    p = plan(spec, patterns)
    if not p["pull"]:
        if local.is_dir():
            MR.write_manifest(local, {"files": p["master"]})
        return "local has every master file"
    local.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(local).free
    if free < p["bytes_pull"] * 1.01:
        need = p["bytes_pull"] / 1e9
        return f"FAILED: {len(p['pull'])} files, {need:.0f} GB to pull, {free / 1e9:.0f} GB free in {local}"
    echo(f"weight cache: pulling {len(p['pull'])} files, {_gb(p['bytes_pull'])}: {shared} -> {local}")
    t0 = time.time()
    try:
        MR.rsync(shared, local, p["pull"], update=True)
    except (OSError, subprocess.CalledProcessError) as e:
        return f"FAILED: {e}"
    left = plan(spec, patterns)["pull"]
    MR.write_manifest(local, {"files": MR.listing(shared, MR.matcher(patterns) if patterns else keep(spec))})
    if left:
        return f"FAILED: {len(left)} files still missing after the pull, e.g. {left[0]}"
    return f"pulled {len(p['pull'])} files in {time.time() - t0:.0f}s"


class _Lock:
    """One push at a time a box (flock on the local cache's lock file); held -> ``ok`` False."""

    def __init__(self, local: Path):
        local.mkdir(parents=True, exist_ok=True)
        self.f = open(local / LOCK, "a+")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.ok = True
        except OSError:
            self.ok = False

    def release(self):
        if self.ok:
            fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def pushing(spec) -> bool:
    """A push holds this box's lock now."""
    lk = _Lock(roots(spec)[0])
    held = not lk.ok
    lk.release()
    return held


def push(spec, *, echo=print, patterns=None) -> str:
    """local -> master (see the module doc): atomic per file, never over a newer master file, readable by others."""
    local, shared = roots(spec)
    if shared is None:
        return "no master (paths.tt_cache_shared unset)"
    base = Path(spec._art_sub("tt_cache_shared", ""))
    exist = next((a for a in [shared, *shared.parents] if a.exists()), None)
    if exist is None or not os.access(exist, os.W_OK | os.X_OK):
        return f"master not writable by {os.environ.get('USER', 'this user')} ({exist}): the local cache stays local"
    lk = _Lock(local)
    if not lk.ok:
        lk.release()
        return "another push is running on this box"
    try:
        p = plan(spec, patterns)
        if p["push"]:
            echo(f"weight cache: pushing {len(p['push'])} files, {_gb(p['bytes_push'])}: {local} -> {shared}")
            t0 = time.time()
            shared.mkdir(parents=True, exist_ok=True)
            MR.rsync(local, shared, p["push"], update=True, readable=True)
            res = f"pushed {len(p['push'])} files in {time.time() - t0:.0f}s"
        else:
            res = "the master has every local file"
        blocked = MR.make_readable(shared, base)
        man = {"files": MR.listing(shared, MR.matcher(patterns) if patterns else keep(spec))}
        MR.write_manifest(shared, man, readable=True)
        MR.write_manifest(local, man)
        if p["kept_newer_master"]:
            res += f"; {len(p['kept_newer_master'])} master files newer than the local ones kept"
        if blocked:
            res += f"; other users cannot traverse {blocked[0]} (no o+x)"
        return res
    except (OSError, subprocess.CalledProcessError) as e:
        return f"FAILED: {e}"
    finally:
        lk.release()


def spawn_push(spec, log: Path) -> int | None:
    """Start ``push`` in the background (detached, its own session; output to ``log``) when the plan has files to
    push and no push runs. Returns the pid, or None."""
    local, shared = roots(spec)
    if shared is None or spec.path is None or pushing(spec) or not plan(spec)["push"]:
        return None
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as out:
        pr = subprocess.Popen(
            [sys.executable, "-m", "models.demos.common.bringup.core.weight_cache", "push", "--spec", str(spec.path)],
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            cwd=str(spec.repo),
        )
    return pr.pid


def status(spec, patterns=None) -> list[str]:
    local, shared = roots(spec)
    if shared is None:
        return [f"weight cache {local}: no master (paths.tt_cache_shared unset)"]
    p = plan(spec, patterns)
    lines = [
        f"master {shared}: {len(p['master'])} files, {sum(v[0] for v in p['master'].values()) / 1e9:.1f} GB",
        f"local  {local}: {len(p['local'])} files, {sum(v[0] for v in p['local'].values()) / 1e9:.1f} GB",
        f"to pull: {len(p['pull'])} files, {_gb(p['bytes_pull'])}",
        f"to push: {len(p['push'])} files, {_gb(p['bytes_push'])}" + (" (a push is running)" if pushing(spec) else ""),
    ]
    for k in ("kept_newer_local", "kept_newer_master"):
        if p[k]:
            lines.append(f"{k}: {len(p[k])} files at another size, the newer one kept: {p[k][:3]}")
    man = MR.read_json(shared / MR.MANIFEST) if shared.is_dir() else None
    if shared.is_dir() and (man or {}).get("files") != p["master"]:
        lines.append("master mirror.json: missing or stale (a push rewrites it)")
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m models.demos.common.bringup.core.weight_cache")
    ap.add_argument("command", choices=["status", "pull", "push"])
    ap.add_argument("--spec")
    ap.add_argument("--glob", action="append", help="only these basename globs (repeatable); default: the model's")
    a = ap.parse_args(argv)
    from models.demos.common.bringup.reference.golden import load_spec

    spec = load_spec(a.spec)
    if a.command == "status":
        print("\n".join(status(spec, a.glob)))
        return 0
    res = (pull if a.command == "pull" else push)(spec, patterns=a.glob)
    print(f"weight cache {a.command}: {res}", flush=True)
    return 1 if res.startswith("FAILED") else 0


if __name__ == "__main__":
    sys.exit(main())
