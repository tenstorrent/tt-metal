from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .discovery import REPO_ROOT


_WORKTREE_BASE_DEFAULT = Path("/tmp")
_WORKTREE_PREFIX = "tt_hw_planner_"
_SESSION_FILE = ".tt_hw_planner_session.json"
_SHARED_HOST_DIRS = ("generated", "model_cache", "models_cache")


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_").replace(" ", "_")


def _base_dir() -> Path:
    return Path(os.environ.get("TT_HW_PLANNER_WORKTREE_BASE", str(_WORKTREE_BASE_DEFAULT)))


@dataclass
class WorktreeSession:
    path: Path
    model_id: str
    created_ts: float
    creator_pid: int
    source_repo: Path

    def write_marker(self) -> None:
        meta = {
            "model_id": self.model_id,
            "created_ts": self.created_ts,
            "creator_pid": self.creator_pid,
            "source_repo": str(self.source_repo),
        }
        (self.path / _SESSION_FILE).write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, worktree_path: Path) -> Optional["WorktreeSession"]:
        marker = worktree_path / _SESSION_FILE
        if not marker.is_file():
            return None
        try:
            meta = json.loads(marker.read_text())
        except Exception:
            return None
        return cls(
            path=worktree_path,
            model_id=meta.get("model_id", "?"),
            created_ts=float(meta.get("created_ts", 0)),
            creator_pid=int(meta.get("creator_pid", 0)),
            source_repo=Path(meta.get("source_repo", str(REPO_ROOT))),
        )


def create(model_id: str) -> WorktreeSession:
    base = _base_dir()
    base.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    path = base / f"{_WORKTREE_PREFIX}{_slug(model_id)}_{ts}"
    if path.exists():
        path = base / f"{_WORKTREE_PREFIX}{_slug(model_id)}_{ts}_{os.getpid()}"

    proc = subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()}")

    session = WorktreeSession(
        path=path,
        model_id=model_id,
        created_ts=time.time(),
        creator_pid=os.getpid(),
        source_repo=REPO_ROOT,
    )
    session.write_marker()

    for d in _SHARED_HOST_DIRS:
        src = REPO_ROOT / d
        if not src.exists():
            continue
        dst = path / d
        if dst.exists() and not dst.is_symlink():
            continue
        if dst.is_symlink():
            dst.unlink()
        try:
            dst.symlink_to(src)
        except OSError as exc:
            print(
                f"[worktree] WARN could not symlink {dst} -> {src}: {exc}",
                file=sys.stderr,
            )

    return session


def destroy(session: WorktreeSession, *, force: bool = True) -> bool:
    if not session.path.exists():
        return True
    args = ["git", "worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(session.path))
    proc = subprocess.run(args, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        try:
            import shutil

            shutil.rmtree(session.path, ignore_errors=True)
            subprocess.run(
                ["git", "worktree", "prune"],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            return not session.path.exists()
        except Exception:
            return False
    return True


def list_active() -> List[WorktreeSession]:
    sessions: List[WorktreeSession] = []
    proc = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return sessions
    current_path: Optional[Path] = None
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            current_path = Path(line[len("worktree ") :].strip())
        elif line.strip() == "" and current_path is not None:
            if _is_ours(current_path):
                s = WorktreeSession.load(current_path)
                if s is not None:
                    sessions.append(s)
            current_path = None
    if current_path is not None and _is_ours(current_path):
        s = WorktreeSession.load(current_path)
        if s is not None:
            sessions.append(s)
    return sessions


def _is_ours(p: Path) -> bool:
    return p.name.startswith(_WORKTREE_PREFIX) and (p / _SESSION_FILE).is_file()


def list_orphans() -> List[WorktreeSession]:
    out: List[WorktreeSession] = []
    for s in list_active():
        if not _pid_alive(s.creator_pid):
            out.append(s)
    return out


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def cleanup_orphans(*, prompt: bool = True) -> int:
    orphans = list_orphans()
    if not orphans:
        return 0
    removed = 0
    for s in orphans:
        age_h = (time.time() - s.created_ts) / 3600.0
        msg = (
            f"  orphan worktree: {s.path} " f"(model={s.model_id}, creator-pid={s.creator_pid} dead, age={age_h:.1f}h)"
        )
        if prompt:
            print(msg)
            answer = input("  remove? [y/N] ").strip().lower()
            if answer != "y":
                continue
        else:
            print(msg + "  -> removing")
        if destroy(s):
            removed += 1
    return removed
