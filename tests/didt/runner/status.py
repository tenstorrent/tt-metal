# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""status.json: the run's only result. Top-level keys follow gwdt's so the gwdt sweep page renders it unchanged."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .checks import Status

HEADER_KEYS = ("tool", "host", "run", "mode", "commit", "run_dir", "config", "chips")


@dataclass
class Attempt:
    name: str | None
    env: dict
    opts: list
    status: Status
    reason: str
    details: list[str]
    devices: list[dict]
    log: str
    start: float | None
    end: float | None


@dataclass
class StepState:
    key: str
    first: Attempt
    fallbacks: list[Attempt]


def _result(a: Attempt) -> dict:
    return {
        "status": Status(a.status).value,
        "reason": a.reason,
        "details": list(a.details),
        "devices": list(a.devices),
        "log": a.log,
        "start": a.start,
        "end": a.end,
    }


def _step(s: StepState) -> dict:
    r = _result(s.first)
    fallbacks = [{"name": f.name, "env": dict(f.env), "opts": list(f.opts), **_result(f)} for f in s.fallbacks]
    return {
        "key": s.key,
        **{k: r[k] for k in ("status", "reason", "details", "log", "start", "end", "devices")},
        "fallbacks": fallbacks,
    }


def write(path: Path, header: dict, steps: list[StepState], current: str | None) -> None:
    """Temp file + rename: a runner killed mid-write still leaves the previous, valid status.json."""
    doc = {k: header[k] for k in HEADER_KEYS} | {"current": current, "steps": [_step(s) for s in steps]}
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".status.", suffix=".tmp")
    umask = os.umask(0)
    os.umask(umask)
    try:
        # mkstemp creates the file owner-only; status.json is read by others on the shared filesystem.
        os.fchmod(fd, 0o666 & ~umask)
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise
