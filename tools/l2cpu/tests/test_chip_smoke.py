# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-FileCopyrightText: © 2026 Kansei Motoe (kinginu)
# SPDX-License-Identifier: Apache-2.0
"""Chip smoke test (bring-up, heartbeats, mailbox, trap, restarts) through scripts/l2cpu_run.sh.

Destructive for the machine's other users (it resets the chips): runs only with L2CPU_CHIP_TESTS=1 on a machine
with a Blackhole device and the firmware built (make -C tools/l2cpu/fw all). Lock / reset are configured with
L2CPU_LOCK and L2CPU_RESET_CMD (see scripts/l2cpu_run.sh)."""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.skipif(
    os.environ.get("L2CPU_CHIP_TESTS") != "1" or not os.path.exists("/dev/tenstorrent"),
    reason="chip tests disabled (set L2CPU_CHIP_TESTS=1 on a machine with a Blackhole device)",
)
def test_chip_smoke():
    restarts = os.environ.get("L2CPU_SMOKE_RESTARTS", "200")
    cmd = [
        os.path.join(ROOT, "scripts", "l2cpu_run.sh"),
        "pytest l2cpu smoke",
        sys.executable,
        os.path.join(ROOT, "scripts", "bringup_smoke.py"),
        "--restarts",
        restarts,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0 and "SMOKE PASS" in r.stdout, r.stdout[-6000:] + r.stderr[-2000:]
