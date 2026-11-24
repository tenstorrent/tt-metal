# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The galaxy load's event points: the hardware captures, the recovery and the caller's --hook.

stage-start, stage-end  around the bus-bar stage (step "busbar") and the links test (step "links")
before, after           around each bus-bar part (after: its check in <outdir>/check.json), and the links sweep (step
                        "sweep"); a non-zero rc from before skips it
run                     a part named in --order that is not a bus-bar part: the hook runs it (rc 0 PASS, HOOK_SKIP SKIP,
                        else FAIL)

emit() runs the captures for the event (diag/captures.py, one state for the run in its directory), then the hook as
``<hook> <event> <step> <status> <outdir>``, waiting for it, outside each part's timer, then after a part that lost a
chip, hung or failed a CCL the recovery glx_reset (diag/resets.py recover), so the hook still sees the part's outcome.
Every finding line goes to the run's findings.txt and finding_stage.tsv, which diag/report.py reads.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import suppress
from pathlib import Path

from .common import log
from .model import PASS

EVENTS = ("stage-start", "before", "after", "stage-end", "run")
HOOK_SKIP = 3  # a hook's run rc for a part that loaded nothing
STAGE = {"links": "C5", "sweep": "sweep", "end": "end"}  # report.py's stage code by step; else the bus-bar stage's
PHASE_STAGE = {"snapshot": "P1", "reset_loop": "C2", "tests": "busbar", "post_test_reset": "C9"}  # else the end
hook: str | None = None
args: argparse.Namespace | None = None  # the run's flags: the captures', the resets' and --output's run directory


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--hook",
        metavar="SCRIPT",
        help="Galaxy load: called as SCRIPT <event> <step> <status> <outdir> at each event point (diag/events.py), "
        "synchronously, outside each part's timer.",
    )


def setup(a: argparse.Namespace | None) -> None:
    global hook, args
    hook, args = a and a.hook, a


def run_dir() -> Path:
    """The run's directory, --output's."""
    d = Path(args.output).resolve().parent
    d.mkdir(parents=True, exist_ok=True)
    return d


def finding(lines: list[str], stage: str) -> None:
    """*lines* into the run's findings.txt, and each with its *stage* code (diag/report.py STAGE) into
    finding_stage.tsv."""
    if not lines or args is None:
        return
    d = run_dir()
    with open(d / "findings.txt", "a") as f, open(d / "finding_stage.tsv", "a") as s:
        for ln in lines:
            log(ln)
            f.write(f"{ln}\n")
            s.write(f"{stage}\t{ln}\n")


def _check_lines(c: dict) -> list[str]:
    """A FAIL or WARN check as finding lines: one per failed link (both ends by BDF, tray and ETH channel), else its
    data's findings or FAIL lines, else '<CHECK> <STATUS> <BDF> (tray N), ...: <details>' with the chips it blames
    (links.name_chips)."""
    d = c.get("data") or {}
    end = lambda e: f"{e.get('bdf')} (tray {e.get('tray')}) e{e.get('channel')}"  # noqa: E731
    links = [
        f"LINK {'HANG' if f.get('why') == 'hung' else 'FAIL'} {end(x['a'])} <-> {end(x['b'])}: {f.get('why')}"
        for f in d.get("failed") or []
        if isinstance(f, dict)
        for x in f.get("ends") or []
    ]
    found = [str(f.get("text", "")) for f in d.get("findings") or [] if isinstance(f, dict)]
    who = ", ".join(f"{b} (tray {t})" for b, t in zip(c.get("bdf") or [], c.get("tray") or []))
    one = f"{str(c.get('name')).upper().replace('_', '-')} {c.get('status')}{' ' + who if who else ''}"
    return links or found or d.get("fails") or [f"{one}: {c.get('details') or ''}"]


def check_findings(report: dict) -> None:
    """Every FAIL or WARN check of a phase that gates as finding lines (_check_lines), so diag/report.py's tray lines
    name them."""
    for name, ph in report.get("phases", {}).items():
        for c in ph.get("checks", []) if ph.get("gates", True) else []:
            if c.get("status") in ("FAIL", "WARN"):
                links = str(c.get("name", "")).startswith("links_")
                finding(_check_lines(c), "C5" if links else PHASE_STAGE.get(name, "end"))


def capture(event: str, step: str, status: str = "-", stage: str | None = None, load_pct: float | None = None) -> None:
    """The captures for *event* (none in a dry run or without the run's flags), their finding lines kept; *load_pct*:
    the stage's load reached, % of its plan, for the HSC judge (the run's --load-pct otherwise)."""
    if args is not None and not args.dry_run:
        from . import captures

        a = args if load_pct is None else argparse.Namespace(**{**vars(args), "load_pct": load_pct})
        finding(captures.run(event, step, status, run_dir(), a), stage or STAGE.get(step, "busbar"))


def vr_pg(t0: float) -> str:
    """A tray's VR power-good fall since *t0* (the captures' power poll), why a running part stops; "" when none."""
    if args is None or args.dry_run:
        return ""
    from .captures import vr_pg_fell

    return "; ".join(f"vr_pg {b} on tray {t}" for t, (_, b) in sorted(vr_pg_fell(run_dir() / "captures", t0).items()))


def gone() -> list[str]:
    """The chips on PCIe at the stage's start (the captures' record) that are off it now; none in a dry run."""
    if args is None or args.dry_run:
        return []
    from . import captures

    st = {}
    with suppress(OSError, ValueError):
        st = json.loads((run_dir() / "captures" / "state.json").read_text())
    return sorted(set((st.get("stage") or {}).get("chips", [])) - set(captures.chips()))


def _lost(outdir: Path) -> bool:
    """The part in *outdir* hung or failed a CCL (its check.json), or a chip left PCIe in the stage."""
    c = {}
    with suppress(OSError, ValueError):
        c = json.loads((outdir / "check.json").read_text())
    return bool(c.get("hung") or c.get("link_fault") or gone())


def emit(event: str, step: str, status: str, outdir: Path, load_pct: float | None = None) -> int:
    """The captures for *event*, the hook, then after a lost part the recovery; the hook's rc (0 with no hook or in a
    dry run)."""
    dry, rc = args is not None and args.dry_run, 0
    capture(event, step, status, load_pct=load_pct)
    if hook and dry:
        log(f"[dry-run] {hook} {event} {step} {status} {outdir}")
    elif hook:
        outdir.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run([hook, event, step, status, str(outdir)], check=False).returncode
    if event == "after" and step != "sweep" and status != PASS and args is not None and not dry and _lost(outdir):
        from .resets import recover

        finding(recover(args, outdir), STAGE.get(step, "busbar"))
    return rc
