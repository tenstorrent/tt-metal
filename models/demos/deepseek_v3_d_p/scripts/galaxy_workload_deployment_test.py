#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""galaxy-workload-deployment-test: hardware gates, then per-model block -> smoke -> soak on a Blackhole Galaxy.

Usage: galaxy_workload_deployment_test.py [--co] [--skip KEY[,KEY]] [--models kimi_k2_7,glm53]
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from gwdt_checks import TOOL_NAME, Status, dur, log_root, run_name
from gwdt_steps import MODEL_ENV, MODEL_LABEL, SOAK_LOOPS, Ctx, Step, build_steps, run_all

try:
    from colorama import Fore, Style
except ImportError:
    print(
        f"colorama is missing: uv pip install --python {Path(__file__).resolve().parents[4]}/python_env/bin/python colorama"
    )
    sys.exit(1)

REDRAW_SECS = 15
COLOR = {
    Status.PASS: Fore.GREEN,
    Status.FAIL: Style.BRIGHT + Fore.RED,
    Status.HANG: Style.BRIGHT + Fore.RED,
    Status.WARN: Fore.YELLOW,
    Status.SKIP: Fore.YELLOW,
    Status.RUN: Fore.CYAN,
    Status.PENDING: Style.DIM,
}


class Screen:
    def __init__(self, steps: list[Step], header: list[str], status_json: Path):
        self.steps, self.header, self.status_json = steps, header, status_json
        self.tty = sys.stdout.isatty()
        self.last_draw = 0.0
        self.done = False

    def paint(self, status: Status, text: str) -> str:
        return f"{COLOR[status]}{text}{Style.RESET_ALL}" if self.tty else text

    def cell(self, step: Step) -> str:
        s = step.state
        took = f" {dur((s.end or time.time()) - s.start)}" if s.start else ""
        return self.paint(s.result.status, f"{s.result.status.value}{took}")

    def lines(self) -> list[str]:
        gates = [s for s in self.steps if s.group == "gate"]
        out = self.header + ["", "GATES"]
        for i, s in enumerate(gates):
            took = dur((s.state.end or time.time()) - s.state.start) if s.state.start else ""
            status = s.state.result.status
            out.append(
                f" {i} {s.title:<10} {self.paint(status, f'{status.value:<6}')} {took:>8}    {s.state.result.reason}"
            )
            out += [
                f"   {d}" if "MISSING" not in d else self.paint(Status.FAIL, f"   {d}") for d in s.state.result.details
            ]
        out += ["", f"{'MODELS':<13}" + "".join(f"{c:<16}" for c in self.columns())]
        for m in dict.fromkeys(s.model for s in self.steps if s.model):
            out.append(f" {MODEL_LABEL[m]:<12}" + "".join(f"{c:<16}" for c in self.model_cells(m)))
        running = next((s for s in self.steps if s.state.result.status == Status.RUN), None)
        if running:
            out += ["", f"NOW   {running.title}  {running.state.now or dur(time.time() - running.state.start)}"]
        errors = [s for s in self.steps if s.state.result.status in (Status.FAIL, Status.HANG)]
        if errors:
            out += ["", "ERRORS"]
            for s in errors:
                out.append(
                    f" {s.title:<20} {self.paint(s.state.result.status, s.state.result.status.value)}  {s.state.result.reason}"
                )
                out += [f"   {d}" for d in s.state.result.details]
                if s.state.log:
                    out.append(f"   log  {s.state.log}")
        if self.done:
            first = errors[0].title if errors else None
            result = Status.FAIL if errors else Status.PASS
            out += ["", f"RESULT  {self.paint(result, result.value)}" + (f" at {first}" if first else "")]
        return out

    def columns(self) -> list[str]:
        groups = list(dict.fromkeys(s.group for s in self.steps if s.model and s.group != "soak"))
        return [g for g in groups] + [f"soak {i}/{SOAK_LOOPS}" for i in range(1, SOAK_LOOPS + 1)]

    def model_cells(self, model: str) -> list[str]:
        cells = []
        for g in dict.fromkeys(s.group for s in self.steps if s.model and s.group != "soak"):
            step = next((s for s in self.steps if s.model == model and s.group == g), None)
            cells.append(self.cell(step) if step else "-")
        soak = next((s for s in self.steps if s.model == model and s.group == "soak"), None)
        loops = soak.state.loops if soak and soak.state.loops else [soak.state.result] * SOAK_LOOPS if soak else []
        cells += [self.paint(r.status, r.status.value) for r in loops]
        return cells

    def update(self, step: Step | None = None) -> None:
        self.write_status()
        if self.tty:
            if step is not None or time.time() - self.last_draw >= REDRAW_SECS:
                print("\033[2J\033[H" + "\n".join(self.lines()), flush=True)
                self.last_draw = time.time()
        elif step is not None:
            r = step.state.result
            print(f"{datetime.now():%H:%M:%S} {step.title:<20} {r.status.value:<5} {r.reason}", flush=True)

    def write_status(self) -> None:
        running = next((s.key for s in self.steps if s.state.result.status == Status.RUN), None)
        doc = {
            "header": self.header,
            "current": running,
            "steps": [
                {
                    "key": s.key,
                    "status": s.state.result.status.name,
                    "reason": s.state.result.reason,
                    "details": s.state.result.details,
                    "loops": [r.status.name for r in s.state.loops],
                    "log": s.state.log,
                    "start": s.state.start,
                    "end": s.state.end,
                }
                for s in self.steps
            ],
        }
        tmp = self.status_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(self.status_json)


def main() -> int:
    ap = argparse.ArgumentParser(description=TOOL_NAME)
    ap.add_argument("--co", action="store_true", help="collect-only mockup: no device work")
    ap.add_argument("--skip", default="", help="comma-separated step keys, e.g. soak:kimi_k2_7,didt")
    ap.add_argument("--models", default=",".join(MODEL_ENV), help="comma-separated: " + ",".join(MODEL_ENV))
    args = ap.parse_args()

    repo = Path(os.environ.get("TT_METAL_HOME") or Path(__file__).resolve().parents[4])
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    host = socket.gethostname().split(".")[0]
    run = run_name(datetime.now(), host, commit or "nogit")
    run_dir = log_root(os.environ.get("USER", ""), Path.home()) / run
    run_dir.mkdir(parents=True, exist_ok=True)

    steps = build_steps(args.models.split(","))
    unknown = set(filter(None, args.skip.split(","))) - {s.key for s in steps}
    if unknown:
        print(f"unknown --skip keys: {', '.join(sorted(unknown))}; valid: {', '.join(s.key for s in steps)}")
        return 2
    header = [f"{TOOL_NAME}  {host}  run {run}  mode={'co' if args.co else 'real'}", f"logs  {run_dir}/"]
    print("\n".join(header), flush=True)
    screen = Screen(steps, header, run_dir / "status.json")
    ctx = Ctx(repo, run_dir, args.co, tick=screen.update)
    run_all(steps, ctx, set(filter(None, args.skip.split(","))), screen.update)
    screen.done = True
    if screen.tty:
        screen.update(steps[-1])
    else:
        print("\n".join(screen.lines()), flush=True)
    return 1 if any(s.state.result.status in (Status.FAIL, Status.HANG) for s in steps) else 0


if __name__ == "__main__":
    sys.exit(main())
