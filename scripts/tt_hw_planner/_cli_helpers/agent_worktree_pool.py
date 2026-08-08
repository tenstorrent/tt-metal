"""Per-agent git worktree isolation for the parallel bring-up path.

Each concurrent agent runs in its own git worktree instead of the single shared
bring-up tree, eliminating the concurrent read/write collision on stubs, the
_handoff dir, and the claude session storage that caused the multi-agent stall.

Reuses the proven worktree recipe (git worktree add --detach HEAD + symlinks
for the big shared host dirs). Because the demo dir is UNTRACKED, it is copied
from the main tree into each worktree and baselined with ``git add -A`` so that
``git status`` afterward reports only the agent's own delta. After the agent
finishes, the whole delta is copied back into the main tree (the facebook
capture mechanism), excluding orchestrator-owned and session-local files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional

from ..worktree import _SHARED_HOST_DIRS, _base_dir, _slug

_EXCLUDE_FROM_HARVEST_NAMES = (
    "bringup_status.json",
    "locked_modules.json",
    "skip_diagnosis.json",
    "harness_skipped.json",
    ".tt_hw_planner_session.json",
)
_EXCLUDE_FROM_HARVEST_SUFFIXES = (
    ".best_native",
    ".preiter_native",
    ".last_good_native",
    ".stable_native",
)


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip()}")
    return proc.stdout


class AgentWorktreePool:
    """Creates and tears down one git worktree per concurrent agent slot.

    main_repo:    the bring-up tree (harvest destination, working-state source)
    model_id:     used to name worktrees

    Each worktree is created at HEAD and then has the main tree's FULL
    uncommitted working state replicated into it (tracked diff + untracked
    files), because the demo dir is untracked and the bring-up also carries
    uncommitted edits to tracked files the model depends on.
    """

    _SKIP_UNTRACKED_TOPDIRS = frozenset(set(_SHARED_HOST_DIRS) | {"runs", "build", "build_Release"})
    _MAX_UNTRACKED_FILE_BYTES = 50 * 1024 * 1024

    def __init__(self, main_repo: Path, model_id: str):
        self.main_repo = Path(main_repo)
        self.model_id = model_id
        self._lock = threading.Lock()
        self._slots: Dict[int, Path] = {}
        self._counter = 0

    def _excluded(self, rel: str) -> bool:
        name = rel.rsplit("/", 1)[-1]
        if name in _EXCLUDE_FROM_HARVEST_NAMES:
            return True
        return any(rel.endswith(s) for s in _EXCLUDE_FROM_HARVEST_SUFFIXES)

    def _unique_path(self, slot: int) -> Path:
        with self._lock:
            self._counter += 1
            n = self._counter
        return _base_dir() / f"tt_hw_planner_{_slug(self.model_id)}_agent{slot}_{os.getpid()}_{n}"

    def _symlink_shared(self, wt: Path) -> None:
        for d in _SHARED_HOST_DIRS:
            src = self.main_repo / d
            if not src.exists():
                continue
            dst = wt / d
            if dst.exists() and not dst.is_symlink():
                continue
            if dst.is_symlink():
                dst.unlink()
            try:
                dst.symlink_to(src)
            except OSError:
                pass

    def _replicate_working_state(self, wt: Path) -> None:
        diff = _git(self.main_repo, "diff", "HEAD", "--binary", check=False)
        if diff.strip():
            subprocess.run(
                ["git", "-C", str(wt), "apply", "--whitespace=nowarn"],
                input=diff,
                capture_output=True,
                text=True,
                check=False,
            )
        others = _git(self.main_repo, "ls-files", "--others", "--exclude-standard", check=False)
        for rel in others.splitlines():
            rel = rel.strip()
            if not rel:
                continue
            top = rel.split("/", 1)[0]
            if top in self._SKIP_UNTRACKED_TOPDIRS:
                continue
            src = self.main_repo / rel
            if not src.is_file() or src.is_symlink():
                continue
            try:
                if src.stat().st_size > self._MAX_UNTRACKED_FILE_BYTES:
                    continue
            except OSError:
                continue
            dst = wt / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        _git(wt, "add", "-A", check=False)
        subprocess.run(
            [
                "git",
                "-c",
                "user.email=tt-hw-planner@local",
                "-c",
                "user.name=tt-hw-planner",
                "-C",
                str(wt),
                "commit",
                "--no-verify",
                "-q",
                "-m",
                "agent-worktree baseline",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def acquire(self, slot: int) -> Path:
        path = self._unique_path(slot)
        _git(self.main_repo, "worktree", "add", "--detach", "-f", str(path), "HEAD")
        with self._lock:
            self._slots[slot] = path
        self._symlink_shared(path)
        self._replicate_working_state(path)
        return path

    def changed_files(self, slot: int) -> List[str]:
        path = self._slots.get(slot)
        if not path:
            return []
        out = _git(path, "status", "--porcelain", "--untracked-files=all", check=False)
        files: List[str] = []
        for line in out.splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            rel = parts[1]
            if " -> " in rel:
                rel = rel.split(" -> ", 1)[1]
            files.append(rel.strip())
        return files

    def harvest(self, slot: int) -> List[str]:
        path = self._slots.get(slot)
        if not path:
            return []
        copied: List[str] = []
        for rel in self.changed_files(slot):
            if self._excluded(rel):
                continue
            src = path / rel
            if not src.is_file() or src.is_symlink():
                continue
            dst = self.main_repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)
        return copied

    def handoff_dir(self, slot: int) -> Optional[Path]:
        path = self._slots.get(slot)
        if not path:
            return None
        return path / "_handoff"

    def release(self, slot: int) -> None:
        with self._lock:
            path = self._slots.pop(slot, None)
        if not path:
            return
        proc = subprocess.run(
            ["git", "-C", str(self.main_repo), "worktree", "remove", "--force", str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            shutil.rmtree(path, ignore_errors=True)
            subprocess.run(
                ["git", "-C", str(self.main_repo), "worktree", "prune"],
                capture_output=True,
                text=True,
                check=False,
            )

    def cleanup(self) -> None:
        for slot in list(self._slots.keys()):
            self.release(slot)
