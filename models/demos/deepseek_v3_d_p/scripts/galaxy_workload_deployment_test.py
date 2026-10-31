#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""galaxy-workload-deployment-test: gates, matmul ND runs, then per-model block -> smoke -> soak on a Blackhole Galaxy.

Usage: galaxy_workload_deployment_test.py [--co] [--skip KEY[,KEY] | --only KEY[,KEY]] [--models kimi_k2_7,glm53]
"""

import argparse
import contextlib
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from gwdt_checks import TOOL_NAME, Result, Status, dur, layer_label, log_root, run_name
from gwdt_steps import MODEL_ENV, MODEL_LABEL, MODEL_LAYERS, Ctx, Step, build_steps, run_all

try:
    from colorama import Fore, Style
    from rich.live import Live
    from rich.text import Text
except ImportError as e:
    package = e.name.split(".")[0]
    print(
        f"{package} is missing: uv pip install --python {Path(__file__).resolve().parents[4]}/python_env/bin/python {package}"
    )
    sys.exit(1)

REDRAW_SECS = 15
NOTE = 33
COLOR = {
    Status.PASS: Fore.GREEN,
    Status.FAIL: Style.BRIGHT + Fore.RED,
    Status.HANG: Style.BRIGHT + Fore.RED,
    Status.WARN: Fore.YELLOW,
    Status.SKIP: Fore.YELLOW,
    Status.RUN: Fore.CYAN,
    Status.PENDING: Style.DIM,
}


def device_rows(r: Result) -> list[tuple[str, Result]]:
    rows = [d.partition(": ") for d in r.details]
    return [(label, Result(Status.FAIL, why)) for label, _, why in rows]


class Screen:
    def __init__(self, steps: list[Step], header: list[str], status_json: Path, meta: dict[str, str]):
        self.steps, self.header, self.status_json, self.meta = steps, header, status_json, meta
        self.tty = sys.stdout.isatty()
        self.last_draw = 0.0
        self.done = False
        self.live: Live | None = None

    def paint(self, status: Status, text: str, width: int = 0) -> str:
        text = f"{text:<{width}}"
        return f"{COLOR[status]}{text}{Style.RESET_ALL}" if self.tty else text

    def lines(self) -> list[str]:
        gates = [s for s in self.steps if s.group == "gate"]
        out = self.header + ["", f"{'GATES':<20} {'time':>8}    note"]
        for i, s in enumerate(gates):
            took = dur((s.state.end or time.time()) - s.state.start) if s.state.start else ""
            status = s.state.result.status
            out.append(
                f"{i:>2} {s.title:<10} {self.paint(status, status.value, 6)} {took:>8}    {s.state.result.reason}"
            )
            out += [self.gate_detail(s, d) for d in s.state.result.details]
        matmul = [s for s in self.steps if s.group == "matmul"]
        if matmul:
            out += ["", "MATMUL ND"]
        for s in matmul:
            out.append(self.model_row(3, s.key.split(":")[1], s.state.result, s.state.start, s.state.end))
            out += [self.model_row(5, label, r) for label, r in self.sub_rows(s)]
            if rerun := s.state.rerun:
                env = " ".join(f"{k}={v}" for k, v in rerun.env.items())
                r = Result(rerun.result.status, f"{env}  {rerun.result.reason}".rstrip())
                out.append(self.model_row(5, "rerun", r, rerun.start, rerun.end))
                out += [self.model_row(7, label, r) for label, r in device_rows(rerun.result)]
        models = list(dict.fromkeys(s.model for s in self.steps if s.model))
        if models:
            out += ["", "MODELS"]
        for m in models:
            out.append(f" {MODEL_LABEL[m]} ({MODEL_LAYERS[m]})")
            for s in (s for s in self.steps if s.model == m):
                out.append(self.model_row(3, s.group, s.state.result, s.state.start, s.state.end))
                out += [self.model_row(5, label, r) for label, r in self.sub_rows(s)]
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

    def gate_detail(self, step: Step, detail: str) -> str:
        if step.key != "paths":
            return " " * NOTE + detail
        if detail.endswith(" MISSING"):
            return f"   {detail.removesuffix('MISSING')}{self.paint(Status.FAIL, 'MISSING')}"
        return f"   {detail}"

    def model_row(
        self, indent: int, label: str, r: Result, start: float | None = None, end: float | None = None
    ) -> str:
        took = dur((end or time.time()) - start) if start else ""
        return f"{' ' * indent}{label:<{23 - indent}}{self.paint(r.status, r.status.value, 6)} {took:>8}    {r.reason}".rstrip()

    def sub_rows(self, step: Step) -> list[tuple[str, Result]]:
        """Layer-type rows for pytest steps, loop rows for multi-loop stress steps, device rows for matmul runs."""
        if step.group == "matmul":
            return device_rows(step.state.result)
        if len(step.state.loops) > 1:
            return [(f"loop {i}/{len(step.state.loops)}", r) for i, r in enumerate(step.state.loops, 1)]
        if step.collect is None or step.group in ("smoke", "soak"):
            return []
        ids = step.collect(None)
        rows = step.state.rows or [Result(Status.PENDING)] * len(ids)
        return [(layer_label(i), r) for i, r in zip(ids, rows)]

    def update(self, step: Step | None = None) -> None:
        self.write_status()
        if self.tty:
            if step is not None or time.time() - self.last_draw >= REDRAW_SECS:
                self.live.update(Text.from_ansi("\n".join(self.lines())), refresh=True)
                self.last_draw = time.time()
        elif step is not None:
            r = step.state.result
            print(f"{datetime.now():%H:%M:%S} {step.title:<20} {r.status.value:<5} {r.reason}", flush=True)

    def write_status(self) -> None:
        running = next((s.key for s in self.steps if s.state.result.status == Status.RUN), None)
        doc = {
            **self.meta,
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
                    "rerun": s.state.rerun
                    and {
                        "env": s.state.rerun.env,
                        "status": s.state.rerun.result.status.name,
                        "reason": s.state.rerun.result.reason,
                        "details": s.state.rerun.result.details,
                        "log": s.state.rerun.log,
                        "start": s.state.rerun.start,
                        "end": s.state.rerun.end,
                    },
                }
                for s in self.steps
            ],
        }
        tmp = self.status_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(self.status_json)


def skipped_keys(keys: list[str], skip: str, only: str) -> set[str]:
    """Step keys not to run: the --skip list, or everything outside the --only list."""
    flag, listed = ("--only", only) if only else ("--skip", skip)
    picked = set(filter(None, listed.split(",")))
    if unknown := picked - set(keys):
        raise ValueError(f"unknown {flag} keys: {', '.join(sorted(unknown))}; valid: {', '.join(keys)}")
    return set(keys) - picked if only else picked


def main(argv: list[str] | None = None) -> int:
    keys = "\n".join(f"  {s.key}" for s in build_steps(list(MODEL_ENV)))
    ap = argparse.ArgumentParser(
        description=TOOL_NAME,
        epilog=f"step keys for --skip / --only, in run order:\n{keys}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--co", action="store_true", help="collect-only mockup: no device work")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--skip", default="", help="comma-separated step keys, e.g. soak:kimi_k2_7,matmul:ff1")
    pick.add_argument("--only", default="", help="run exactly these step keys, e.g. reset,block:kimi_k2_7")
    ap.add_argument("--models", default=",".join(MODEL_ENV), help="comma-separated: " + ",".join(MODEL_ENV))
    args = ap.parse_args(argv)

    repo = Path(os.environ.get("TT_METAL_HOME") or Path(__file__).resolve().parents[4])
    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    host = socket.gethostname().split(".")[0]
    run = run_name(datetime.now(), host, commit or "nogit")
    steps = build_steps(args.models.split(","))
    try:
        skip = skipped_keys([s.key for s in steps], args.skip, args.only)
    except ValueError as e:
        print(e)
        return 2
    run_dir = log_root(os.environ.get("USER", ""), Path.home()) / run
    run_dir.mkdir(parents=True, exist_ok=True)
    header = [f"{TOOL_NAME}  {host}  run {run}  mode={'co' if args.co else 'real'}", f"logs  {run_dir}/"]
    meta = {"host": host, "run": run, "mode": "co" if args.co else "real", "commit": commit, "run_dir": str(run_dir)}
    screen = Screen(steps, header, run_dir / "status.json", meta)
    ctx = Ctx(repo, run_dir, args.co, tick=screen.update)
    # Live redraws in place, so scrollback keeps only the command and the final table.
    with (Live(Text("\n".join(header)), auto_refresh=False) if screen.tty else contextlib.nullcontext()) as live:
        screen.live = live
        if not screen.tty:
            print("\n".join(header), flush=True)
        run_all(steps, ctx, skip, screen.update)
        screen.done = True
        if screen.tty:
            screen.update(steps[-1])
        else:
            print("\n".join(screen.lines()), flush=True)
    return 1 if any(s.state.result.status in (Status.FAIL, Status.HANG) for s in steps) else 0


if __name__ == "__main__":
    sys.exit(main())
