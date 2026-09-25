# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Frozen tests: content hashes that the gate checks before it runs anything.

A task's ``frozen`` block lists files (tests, their templates' fixtures, the golden manifest) with a sha256 each.
If any listed file changed, the gate fails without running the command, so an implement agent cannot pass a
gate by editing its test or its golden.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hash_paths(repo: Path, paths: list[str]) -> dict[str, str]:
    """Hash every file under the given repo-relative (or absolute) paths; directories expand recursively."""
    out = {}
    for p in paths:
        full = Path(p) if Path(p).is_absolute() else repo / p
        files = (
            sorted(x for x in full.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
            if full.is_dir()
            else [full]
        )
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"cannot freeze missing file {f}")
            key = str(f.relative_to(repo)) if f.is_relative_to(repo) else str(f)
            out[key] = sha256_file(f)
    return out


def verify(repo: Path, task: dict) -> list[str]:
    """Problems with the task's frozen files (empty = all match)."""
    frozen = (task.get("frozen") or {}).get("files") or {}
    errs = []
    for rel, want in frozen.items():
        f = Path(rel) if Path(rel).is_absolute() else repo / rel
        if not f.exists():
            errs.append(f"frozen file missing: {rel}")
        elif sha256_file(f) != want:
            errs.append(f"frozen file changed: {rel}")
    return errs
