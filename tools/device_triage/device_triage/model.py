# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field


@dataclass
class Event:
    host: str | None
    ts: float | None
    source: str
    origin: str
    category: str
    signal: str
    severity: str
    detail: str
    chips: str = ""
    scope: str = "unknown"
    run_id: str | None = None
    path: str = ""
    line: int | None = None
    count: int = 1


@dataclass
class Run:
    run_id: str
    source: str
    kind: str
    hosts: list[str]
    ts_start: float | None
    ts_end: float | None
    status: str
    summary: str
    path: str
    artifact: str = ""
    extra: dict = field(default_factory=dict)


@dataclass
class UnitResult:
    """Everything one parsed unit (a run directory or a log file) contributes."""

    unit: str
    runs: list[Run] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    chip_map: list[tuple[str, str, str, int | None]] = field(default_factory=list)
    error: str = ""


@dataclass
class HostData:
    """One host's parsed runs: events keyed by the unit they came from (a run counts once per category)."""

    runs: list[Run] = field(default_factory=list)
    events: list[tuple[str, Event]] = field(default_factory=list)
    chip_map: list[tuple[str, str, str, int | None]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def add(self, res: UnitResult) -> None:
        self.runs += res.runs
        self.events += [(res.unit, ev) for ev in res.events]
        self.chip_map += res.chip_map
        if res.error:
            self.errors.append(f"{res.unit}: {res.error}")
