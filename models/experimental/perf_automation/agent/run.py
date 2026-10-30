"""I-4 · Run directory + manifest (PLAN section 2 & section 5).

One self-contained directory per run:

    runs/<run_id>/
        manifest.json    # IMMUTABLE: write-once at PRECHECK
        state.json       # MUTABLE checkpoint
        ledger.jsonl     # APPEND-ONLY
        profiles/        # Tracy CSV outputs
    runs/latest -> <run_id>   # symlink; resume reads runs/latest/state.json
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .atomic import atomic_write


class ManifestExistsError(Exception):
    """Raised on a second write to a manifest (write-once, PLAN section 2)."""


class Manifest:
    """Write-once run manifest (env facts + file map + config)."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)

    def write(self, data: dict[str, Any]) -> None:
        if self.path.exists():
            raise ManifestExistsError(f"manifest already written: {self.path}")
        atomic_write(self.path, json.dumps(data, indent=2, sort_keys=True))

    def read(self) -> dict[str, Any]:
        return json.loads(self.path.read_text())


def _point_latest(runs_root, target) -> None:
    """Repoint ``<runs_root>/latest`` at ``target``. Best-effort; never raises.

    `target` may be a bare run id (the pointer and the run live in one tree, the ordinary case) or
    an absolute path (the run was produced in a linked worktree and the pointer belongs in the tree
    the operator has open). Both are symlinks, so nothing downstream has to know which it got.
    """
    from pathlib import Path as _P

    try:
        runs_root = _P(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        latest = runs_root / "latest"
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(str(target))
    except OSError:
        pass


def main_runs_root(runs_root):
    """The MAIN checkout's counterpart of ``runs_root``, or None when there is no separate one.

    A run in a linked worktree files its artifacts there, so the checkout an operator has open sees
    nothing of it. This names where those artifacts' durable twin belongs -- the same path, under the
    main working tree -- so the pointer and the report land beside each other exactly as they do in
    the worktree. Returns None for an ordinary checkout, where the run is already in the only tree.
    """
    from pathlib import Path as _P

    from . import gitio

    try:
        runs_root = _P(runs_root).resolve()
        here = gitio.repo_root(runs_root)
        main = gitio.main_worktree_root(runs_root)
        if main == here:
            return None
        return main / runs_root.relative_to(here)
    except Exception:  # noqa: BLE001
        return None


class Run:
    """A single run directory and its referenced artifacts."""

    def __init__(self, runs_root: str | os.PathLike[str], run_id: str):
        self.runs_root = Path(runs_root)
        self.run_id = run_id
        self.dir = self.runs_root / run_id

    @property
    def manifest(self) -> Manifest:
        return Manifest(self.dir / "manifest.json")

    @property
    def state_path(self) -> Path:
        return self.dir / "state.json"

    @property
    def ledger_path(self) -> Path:
        return self.dir / "ledger.jsonl"

    @property
    def profiles_dir(self) -> Path:
        return self.dir / "profiles"

    @classmethod
    def create(
        cls,
        runs_root: str | os.PathLike[str],
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
        label: str | None = None,
    ) -> "Run":
        """Create runs/<id>/ (+ profiles/), point runs/latest at it, write manifest.

        ``label`` names the model in the directory itself. Every model's runs land in one flat
        ``runs/``, so a bare timestamp meant opening manifest.json to tell two models' runs apart.
        Appended, never prefixed: the id stays timestamp-first, so name-sorting is still
        chronological and globs that assume a leading year (demo_route) keep matching. Only the
        leaf is used, so a path-shaped label cannot introduce a directory level. Ignored when
        ``run_id`` is passed explicitly.
        """
        runs_root = Path(runs_root)
        if not run_id:
            run_id = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            leaf = Path(label).name if label else ""
            if leaf:
                run_id = f"{run_id}-{leaf}"
        run = cls(runs_root, run_id)
        run.profiles_dir.mkdir(parents=True, exist_ok=True)

        _point_latest(runs_root, run_id)

        if config is not None:
            run.manifest.write(config)
        return run

    @classmethod
    def open(cls, runs_root: str | os.PathLike[str], run_id: str) -> "Run":
        return cls(runs_root, run_id)

    @classmethod
    def latest(cls, runs_root: str | os.PathLike[str]) -> "Run | None":
        """Open the run pointed to by runs/latest, or None if absent."""
        runs_root = Path(runs_root)
        latest = runs_root / "latest"
        if not latest.is_symlink() and not latest.exists():
            return None
        target = os.readlink(latest)
        return cls(runs_root, Path(target).name)
