# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F150: what a gate result was measured on (its stamp) and the results kept per stamp, so a gate never runs twice
on the same work.

A gate is one or more units: a fix job's segments of one source task each (F79), else the whole command. A unit's
stamp is the hash of its command as run, its source task's frozen block, the gate environment's BRINGUP_* / TT_*
variables, the content of every file a measurement can depend on (code_key: the tree as it is, committed or not, by
content, without the run's own records and the files that cannot change a measurement) and the goldens (goldens_key).
Every unit a gate runs is kept under runs/<run>/stamps/<unit>.json by stamp: verdict, metrics, side outputs. A unit
whose stamp has a PASS (rc 0, its metrics within the gate's limits) is not run again: its metrics and outputs are
restored ("reused: same code"). A FAIL or HANG is never reused; a changed stamp always runs.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

# F120: files that cannot change what a run measures (the orchestration, its docs and selftests, any markdown)
NO_MEASURE = tuple(
    f"models/demos/common/bringup/{p}"
    for p in (
        "orchestrator.py",
        "watch.py",
        "overseer.py",
        "agents/*",
        "briefs/*",
        "skill/*",
        "selftest/*",
        "dev/*",
        "dashboard/*",
        "knowledge/*",
    )
) + ("*.md",)
RECORDS = ("*state.json", "*/results/*", "*BREADCRUMBS.md", "*supervision.md", "*bringup/tasks.yaml")  # F105
SKIP = (
    RECORDS
    + NO_MEASURE
    + (
        "*bringup/overseer.yaml",  # notes and the fix queue
        "*bringup/dashboard/*",
        "models/demos/common/bringup/core/stamps.py",
        "models/demos/common/bringup/core/verify.py",
        "models/demos/common/bringup/__main__.py",
        "models/demos/common/bringup/timing.py",
        "*__pycache__*",
    )
)
_SKIP = re.compile("|".join(fnmatch.translate(g) for g in SKIP))
ENV_PREFIXES = ("BRINGUP_", "TT_")
ENV_IGNORED = {  # where a gate files its results, not what it measures
    "BRINGUP_TASK",
    "BRINGUP_RECORD_TASK",
    "BRINGUP_RESULTS_DIR",
    "BRINGUP_SPEC",
    "BRINGUP_DASHBOARD",
    "BRINGUP_AGENT_CMD",
    "TT_DEVICE_TIMING_LOG",
}
KEEP = 8  # stamps kept per unit


def sha16(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _git(repo: Path, *a, inp: str | None = None) -> str:
    return subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, input=inp).stdout


def code_key(repo: Path) -> str:
    """The content of the tree a gate runs on: committed, changed and untracked files alike (a commit of the same
    content keeps the key), without SKIP."""
    files = {}
    for entry in _git(repo, "ls-tree", "-r", "-z", "HEAD").split("\0"):
        if "\t" in entry:
            meta, path = entry.split("\t", 1)
            if not _SKIP.match(path):
                files[path] = meta.split()[2]
    changed = _git(repo, "diff", "HEAD", "--name-only", "-z").split("\0")
    changed += _git(repo, "ls-files", "-o", "--exclude-standard", "-z").split("\0")
    changed = sorted({p for p in changed if p and not _SKIP.match(p)})
    plain = [p for p in changed if (repo / p).is_file()]
    shas = _git(repo, "hash-object", "--stdin-paths", inp="\n".join(plain) + "\n").split() if plain else []
    files.update(zip(plain, shas))
    for p in set(changed) - set(plain):  # a deleted file, or a submodule at another commit
        files[p] = _git(repo / p, "rev-parse", "HEAD").strip() if (repo / p).is_dir() else "-"
    return sha16("\n".join(f"{p}\0{h}" for p, h in sorted(files.items())))


def goldens_key(spec) -> str:
    """The goldens' own records (manifest with its content_hash, drift and floor files) under the spec's golden root."""
    try:
        root = spec.golden_root
    except Exception:  # a spec without golden paths
        return ""
    if not root.is_dir():
        return ""
    parts = [
        f"{f.relative_to(root)}\0{hashlib.sha256(f.read_bytes()).hexdigest()}" for f in sorted(root.glob("*/*.json"))
    ]
    return sha16("\n".join(parts)) if parts else ""


def env_key(env: dict) -> str:
    keep = {k: v for k, v in env.items() if k.startswith(ENV_PREFIXES) and k not in ENV_IGNORED}
    return sha16(json.dumps(keep, sort_keys=True))


def stamp(cmd: str, frozen, env: dict, code: str, goldens: str) -> dict:
    return {
        "cmd": sha16(cmd),
        "frozen": sha16(json.dumps(frozen, sort_keys=True)),
        "env": env_key(env),
        "code": code,
        "goldens": goldens,
    }


def stamp_id(st: dict) -> str:
    return sha16(json.dumps(st, sort_keys=True))


def output_key(name: str, record_tid: str) -> str:
    """A gate output's name without its record task (FIX.5_profile.json -> _profile.json), so another gate restores it."""
    return name[len(record_tid) :] if name.startswith(f"{record_tid}_") else name


class Store:
    """runs/<run>/stamps/<unit>.json: {stamp id: record}; side outputs in runs/<run>/stamps/<unit>/<stamp id>/."""

    def __init__(self, run_dir: Path):
        self.dir = Path(run_dir) / "stamps"

    def _index(self, unit: str) -> Path:
        return self.dir / f"{unit}.json"

    def records(self, unit: str) -> dict:
        p = self._index(unit)
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except ValueError:
            return {}

    def get(self, unit: str, sid: str) -> dict | None:
        return self.records(unit).get(sid)

    def put(self, unit: str, sid: str, rec: dict, outputs: list[Path], record_tid: str) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        keep = self.dir / unit / sid
        shutil.rmtree(keep, ignore_errors=True)
        names = []
        for f in outputs:
            keep.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, keep / output_key(f.name, record_tid))
            names.append(output_key(f.name, record_tid))
        recs = self.records(unit)
        recs[sid] = dict(rec, outputs=names, t=rec.get("t") or time.strftime("%Y-%m-%dT%H:%M:%S"))
        for old in sorted(recs, key=lambda s: recs[s].get("t", ""))[:-KEEP]:
            recs.pop(old)
            shutil.rmtree(self.dir / unit / old, ignore_errors=True)
        tmp = self._index(unit).with_suffix(".tmp")
        tmp.write_text(json.dumps(recs, indent=1, sort_keys=True) + "\n")
        tmp.replace(self._index(unit))

    def restore(self, unit: str, sid: str, rec: dict, results_dir: Path, record_tid: str) -> None:
        for name in rec.get("outputs") or []:
            src = self.dir / unit / sid / name
            if src.exists():
                shutil.copy2(src, results_dir / (f"{record_tid}{name}" if name.startswith("_") else name))
