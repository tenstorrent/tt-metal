"""F201: a device hang's evidence survives the next run and the overseer.

A hang's evidence is run_safe_pytest's tt-triage report dir (generated/tt-triage) and Metal's Inspector logs
(generated/inspector, rewritten by every mesh open). Both live at fixed repo paths, so the next unit, gate or re-run
wipes them, and a kill while tt-triage writes leaves nothing (DeepSeek-V4.1 galaxy_8x4, 2026-10-27: lost twice).
- keep(): copied whole into the run's logs, the moment a unit returns HANG (the safe runner returns only after triage);
- wait_triage(): every kill by the framework (overseer abort, an agent's timeout, a stopped orchestrator) waits first
  for a running tt-triage to finish, up to TRIAGE_WAIT_S.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

DIRS = ("generated/tt-triage", "generated/inspector")
TRIAGE_WAIT_S = 300


def keep(repo, dest) -> Path | None:
    """Copy the hang dump dirs of `repo` into `dest` (created); None when there is nothing to keep."""
    src = [Path(repo) / d for d in DIRS if (Path(repo) / d).exists()]
    if not src:
        return None
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    for s in src:
        shutil.copytree(s, dest / s.name, dirs_exist_ok=True, symlinks=False, ignore_dangling_symlinks=True)
    return dest


def dest_for(log: Path, label: str) -> Path:
    """<logs>/<log stem>.hang-<label>-<time>/ (a new dir per hang: a later hang never overwrites an earlier one)."""
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in label)
    return log.parent / f"{log.stem}.hang-{safe}-{time.strftime('%Y%m%d_%H%M%S')}"


def triage_running() -> bool:
    """A tt-triage of this user is running (tools/tt-triage.py, started by the safe runner on a hang)."""
    r = subprocess.run(["pgrep", "-u", str(_uid()), "-f", "tt-triage.py"], capture_output=True)
    return r.returncode == 0


def wait_triage(timeout_s: float = TRIAGE_WAIT_S, running=triage_running, sleep=time.sleep, say=print) -> bool:
    """Block while tt-triage runs (at most timeout_s); True if none is running at the end."""
    t0, told = time.time(), False
    while running():
        if not told:
            say(f"tt-triage is writing a hang dump: waiting for it (up to {timeout_s:.0f} s) before the kill (F201)")
            told = True
        if time.time() - t0 > timeout_s:
            return False
        sleep(5)
    return True


def _uid() -> int:
    import os

    return os.getuid()
