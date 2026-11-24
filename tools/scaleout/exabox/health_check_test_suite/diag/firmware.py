# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The firmware reads before the tests (galaxy/fw_reads.py) as one check; SIGTERM stops a read at FW_READ_S."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

from .busbar import GALAXY_DIR
from .common import start_group, stop_group
from .galaxy import parse_bdf_physical
from .model import FAIL, PASS, SKIP, WARN, Check

FW_READ_S = 20
SYS = Path("/sys")


def _step(argv: list[str], out: Path) -> None:
    with open(out, "w") as f:
        p = start_group(argv, cwd=GALAXY_DIR, stdout=f, stderr=subprocess.DEVNULL)
        try:
            p.wait(timeout=FW_READ_S)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                p.wait(timeout=5)
        finally:
            stop_group(p)


def firmware_check(d: Path) -> Check:
    d.mkdir(parents=True, exist_ok=True)
    chips = {(p / "device").resolve().name: p.name.split("!")[1] for p in SYS.glob("class/tenstorrent/tenstorrent!*")}
    (d / "chips.txt").write_text("".join(f"{b} {n}\n" for b, n in sorted(chips.items())))
    (d / "trays.txt").write_text("".join(f"{b[5:]} tray {(parse_bdf_physical(b) or '?')[0]}\n" for b in sorted(chips)))
    prog = str(GALAXY_DIR / "fw_reads.py")
    for step in ("spi", "eth"):
        _step([sys.executable, prog, step, str(d / "chips.txt")], d / f"{step}.jsonl")
    argv = [sys.executable, prog, "judge", *(str(d / f) for f in ("trays.txt", "spi.jsonl", "eth.jsonl"))]
    judged = subprocess.run(argv, capture_output=True, text=True, timeout=FW_READ_S).stdout.splitlines()
    found = [ln for ln in judged if " FAIL " in ln or " WARN " in ln]
    summary = next((ln.split(": ", 1)[1] for ln in judged if ln.startswith("C1 firmware reads: ")), "not checked")
    status = FAIL if any(" FAIL " in ln for ln in found) else WARN if found else PASS if "ok" in summary else SKIP
    data = {"findings": found, "not_checked": [ln for ln in judged if " INFO " in ln], "dir": str(d)}
    return Check(name="firmware_reads", status=status, details="; ".join([summary, *found])[:400], data=data, ip="fw")
