# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The firmware reads before the tests (galaxy/fw_reads.py, read-only): each chip's SPI boot-fs checksums, its ETH cores'
mailbox (firmware version, heartbeat) and its TLB readback, voted across the chips. A checksum that does not match is a
FAIL, a chip off the others' firmware or a stuck heartbeat a WARN, a read that could not be made an INFO line; one
check, its finding lines in data["findings"]. Each read stops at FW_READ_S with a SIGTERM, so the chip it holds is
released before it is killed."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
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
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        finally:
            stop_group(p)


def firmware_check(d: Path) -> Check:
    """Read every chip on /dev/tenstorrent into *d* and judge the reads."""
    d.mkdir(parents=True, exist_ok=True)
    chips = {}
    for p in (SYS / "class" / "tenstorrent").glob("tenstorrent!*"):
        chips[(p / "device").resolve().name] = p.name.rsplit("!", 1)[1]
    (d / "chips.txt").write_text("".join(f"{b} {n}\n" for b, n in sorted(chips.items())))
    (d / "trays.txt").write_text(
        "".join(f"{b[5:]} tray {(parse_bdf_physical(b) or ('?',))[0]}\n" for b in sorted(chips))
    )
    prog = str(GALAXY_DIR / "fw_reads.py")
    for step in ("spi", "eth"):
        _step([sys.executable, prog, step, str(d / "chips.txt")], d / f"{step}.jsonl")
    judged = subprocess.run(
        [sys.executable, prog, "judge", *(str(d / f) for f in ("trays.txt", "spi.jsonl", "eth.jsonl"))],
        capture_output=True,
        text=True,
        timeout=FW_READ_S,
    ).stdout.splitlines()
    found = [ln for ln in judged if " FAIL " in ln or " WARN " in ln]
    summary = next((ln.split(": ", 1)[1] for ln in judged if ln.startswith("C1 firmware reads: ")), "not checked")
    status = FAIL if any(" FAIL " in ln for ln in found) else WARN if found else PASS if "ok" in summary else SKIP
    return Check(
        name="firmware_reads",
        status=status,
        details="; ".join([summary, *found])[:400],
        data={"findings": found, "not_checked": [ln for ln in judged if " INFO " in ln], "dir": str(d)},
        ip="fw",
    )
