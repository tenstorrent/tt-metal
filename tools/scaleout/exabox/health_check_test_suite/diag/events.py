# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The galaxy load's event points and the caller's --hook.

stage-start, stage-end  around the bus-bar stage (step "busbar")
before, after           around each bus-bar part (after: its check in <outdir>/check.json), and the links sweep (step
                        "sweep"); a non-zero rc from before skips it
run                     a part named in --order that is not a bus-bar part: the hook runs it (rc 0 PASS, HOOK_SKIP SKIP,
                        else FAIL)

emit() calls the hook as ``<hook> <event> <step> <status> <outdir>`` and waits for it, outside each part's timer.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from .common import log

EVENTS = ("stage-start", "before", "after", "stage-end", "run")
HOOK_SKIP = 3  # a hook's run rc for a part that loaded nothing
hook: str | None = None
dry_run = False


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--hook",
        metavar="SCRIPT",
        help="Galaxy load: called as SCRIPT <event> <step> <status> <outdir> at each event point (diag/events.py), "
        "synchronously, outside each part's timer.",
    )


def setup(script: str | None, dry: bool = False) -> None:
    global hook, dry_run
    hook, dry_run = script, dry


def emit(event: str, step: str, status: str, outdir: Path) -> int:
    """The hook's rc for *event*; 0 with no hook or in a dry run."""
    if not hook:
        return 0
    if dry_run:
        log(f"[dry-run] {hook} {event} {step} {status} {outdir}")
        return 0
    outdir.mkdir(parents=True, exist_ok=True)
    return subprocess.run([hook, event, step, status, str(outdir)], check=False).returncode
