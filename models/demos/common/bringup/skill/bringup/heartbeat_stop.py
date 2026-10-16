#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Claude Code Stop hook (F93, owner): an overseer session never goes idle on a bring-up run nobody watches.

In a session that used the bringup skill, for each run of a model the session named whose log moved in the last day:
- the orchestrator runs and no watch of that model is armed: block ("arm the watch");
- the orchestrator is not running and the run neither finished nor waits for the owner: block once (act on the stop,
  or tell the owner why it stays stopped).
Stdlib only; installed in ~/.claude/settings.json by the skill (section "Always be watching")."""

import getpass
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

RECENT = 24 * 3600
SKILL = re.compile(r"skills/bringup(?![-\w])")
SPEC = re.compile(r"models/demos/(\w+)/bringup/spec\.yaml")
FINISHED = re.compile(r"WAITING FOR A PERSON|nothing left to run|complete with")


def alive(run: Path) -> bool:
    try:
        os.kill(int((run / "ORCHESTRATOR.pid").read_text().split()[0]), 0)
        return True
    except (OSError, ValueError, IndexError):
        return False


def watched(model: str) -> bool:
    ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout.splitlines()
    return any("bringup.watch" in a and f"/{model}/" in a for a in ps)


def last_line(log: Path) -> str:
    with log.open("rb") as f:
        f.seek(max(0, log.stat().st_size - 4096))
        lines = [
            ln for ln in f.read().decode(errors="replace").splitlines() if ln.strip() and "ORCHESTRATOR_EXIT" not in ln
        ]
    return lines[-1] if lines else ""


def problems(transcript: str) -> list[str]:
    if not SKILL.search(transcript):
        return []
    root = Path(f"/localdev/{getpass.getuser()}/bringup")
    out = []
    for model in sorted(set(SPEC.findall(transcript))):
        for log in root.glob(f"{model}/runs/*/orchestrator.log"):
            if time.time() - log.stat().st_mtime > RECENT:
                continue
            run = log.parent
            if alive(run):
                if not watched(model):
                    out.append(f"{model} {run.name}: the orchestrator runs and no watch is armed: arm the watch now")
            elif not FINISHED.search(last_line(log)):
                out.append(
                    f"{model} {run.name}: the orchestrator is not running (last line: {last_line(log)[:160]!r}); "
                    "act on the stop (resume or relaunch, then arm the watch), or tell the owner why it stays stopped"
                )
    return out


def main() -> int:
    data = json.load(sys.stdin)
    if data.get("stop_hook_active"):
        return 0  # one forced look per stop, never a loop
    try:
        transcript = Path(data["transcript_path"]).read_text(errors="replace")
    except (KeyError, OSError):
        return 0
    bad = problems(transcript)
    if bad:
        print(json.dumps({"decision": "block", "reason": "Bring-up heartbeat (F93): " + "; ".join(bad)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
