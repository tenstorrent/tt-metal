# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Lock-phase breakdown of one device-test log from its timing events (models/common/timing_events.py).

Usage::

    python models/demos/deepseek_v3_d_p/tests/v41/lock_phases.py <test.log> [<runner result .json>]

Prints, per test (device.open .. device.close): time from device.open to the first compute phase, seconds per
phase kind (reference, oracle, weights, compute), and cache hit/miss counts per cache; with the safe runner's
JSON result, also the runner's wall time (it includes waiting for the device lock; the lock is then held for the
whole pytest process, so runner time minus the queue wait bounds the lock-held time).
"""

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime

PREFIX = "TT_EVENT "


def events(log_path: str) -> list[dict]:
    out = []
    with open(log_path, errors="ignore") as log:
        for line in log:
            at = line.find(PREFIX)
            if at >= 0:
                try:
                    out.append(json.loads(line[at + len(PREFIX) :]))
                except json.JSONDecodeError:
                    continue
    return sorted(out, key=lambda e: e["ts"])  # pytest prints captured output per test, not chronologically


def _t(event: dict) -> datetime:
    return datetime.fromisoformat(event["ts"])


def summarize(evts: list[dict]) -> list[dict]:
    """One record per device.open .. device.close span (events outside a span belong to no test)."""
    runs, current = [], None
    for e in evts:
        if e["event"] == "device.open":
            current = {"test": e.get("test"), "open": _t(e), "phases": defaultdict(float), "cache": Counter()}
            current["first_compute"] = None
            runs.append(current)
        elif current is None:
            continue
        elif e["event"] == "phase.begin" and e.get("phase") == "compute" and current["first_compute"] is None:
            current["first_compute"] = _t(e)
        elif e["event"] == "phase.end":
            current["phases"][e["phase"]] += e.get("seconds", 0.0)
        elif e["event"] in ("cache.hit", "cache.miss"):
            current["cache"][(e["cache"], e["event"].split(".")[1])] += 1
        elif e["event"] == "device.close":
            current["close"] = _t(e)
            current = None
    return runs


def main(log_path: str, runner_json: str | None = None) -> None:
    for run in summarize(events(log_path)):
        to_compute = (run["first_compute"] - run["open"]).total_seconds() if run["first_compute"] else None
        held = (run["close"] - run["open"]).total_seconds() if "close" in run else None
        print(run["test"])
        print(f"  device.open -> first compute: {to_compute if to_compute is None else f'{to_compute:.1f}s'}")
        print(f"  device.open -> device.close: {held if held is None else f'{held:.1f}s'}")
        for kind in ("reference", "oracle", "weights", "compute"):
            print(f"  {kind:9s}: {run['phases'].get(kind, 0.0):8.1f}s")
        for (cache, kind), n in sorted(run["cache"].items()):
            print(f"  cache {cache} {kind}: {n}")
    if runner_json:
        run = json.load(open(runner_json))["command_run"]
        print(
            f"runner wall time (incl. lock wait): {run['duration_seconds']:.1f}s, {run['started_at']} .. {run['ended_at']}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:3])
