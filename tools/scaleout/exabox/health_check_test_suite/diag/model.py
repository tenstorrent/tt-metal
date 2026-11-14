# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The result model every phase records into and the JSON report serialises."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIP"

# Check.ip groups, in the order the console summary prints them.
IP_ORDER = ("board", "pcie", "gddr", "eth", "asic", "fw", "thermal", "other")


@dataclass
class Check:
    name: str
    status: str
    details: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ip: str = "other"  # pcie / gddr / eth / fw / asic / thermal / other — used for console grouping
    console_visible: bool = True  # False → kept in JSON but hidden from console summary


@dataclass
class Phase:
    name: str
    status: str = SKIP
    duration_s: float = 0.0
    checks: list[Check] = field(default_factory=list)
    error: str = ""
    # False → the phase's findings are recorded at their real status but do not
    # reach the run's verdict: not the exit code, not the JIRA gate, not the CSV
    # rollup. For a tool still being validated against the fleet, which has to
    # report honestly without being able to take nodes out of service. The flag
    # travels in the JSON so the consumers downstream apply one decision rather
    # than each keeping its own list of which phases count.
    gates: bool = True

    def add(self, c: Check) -> None:
        self.checks.append(c)

    def rollup(self) -> None:
        if any(c.status == FAIL for c in self.checks):
            self.status = FAIL
        elif any(c.status == WARN for c in self.checks):
            self.status = WARN
        elif self.checks:
            self.status = PASS
