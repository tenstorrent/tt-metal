# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The links test of the galaxy load (tiers.py LOAD_TESTS): the pair link test, the sweep on a fault and the fabric
open/close.

pair    every chip-to-chip Ethernet link UMD discovers, tested as a 2-chip all-gather of each linked pair (data
        checked bit for bit); pairs run in rounds, no chip in two pairs of a round
sweep   after a link/CCL failure or a hang (in the pair test, a bus-bar step or, with --link-fault, the caller's own
        earlier call; not a numeric fault): the hook's before event (a non-zero rc skips it), glx_reset, then the pair
        test again, --pair-s per pair, no new round past --sweep-s in total, then the hook's after event
fabric  one whole-box fabric open and close
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from . import events
from .busbar import GALAXY_DIR, SETUP_S, Step, run_step
from .common import emit_result, emit_running, log, start_group, stop_group
from .model import FAIL, PASS, SKIP, Check, Phase
from .naming import chip, link_end

# each pair runs PAIR_S s of all-gathers, every op cut at OP_TIMEOUT_S; a round waits at most PAIR_S + OP_TIMEOUT_S + COMPILE_S
# (a pair past it is hung); SWEEP_S caps the whole sweep, not one round
PAIR_S, SWEEP_S, OP_TIMEOUT_S, COMPILE_S = 10, 300, 30, 60
DISCOVER = """import json, tt_umd as u
cd = u.TopologyDiscovery.create_cluster_descriptor()
print(json.dumps({"bdfs": {int(k): v for k, v in cd.get_chip_pci_bdfs().items()},
                  "links": {int(c): {int(ch): [int(rc), int(rch)] for ch, (rc, rch) in v.items()}
                            for c, v in cd.get_ethernet_connections().items()}}))"""


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--pair-s", type=int, default=PAIR_S, help="Galaxy load: seconds of all-gathers per linked pair.")
    ap.add_argument("--sweep-s", type=int, default=SWEEP_S, help="Galaxy load: no new sweep round past this, s.")
    ap.add_argument(
        "--link-fault",
        action="store_true",
        help="Galaxy load: the caller's earlier load, in a call of its own, had a link or CCL failure or a hang: "
        "the links test sweeps after its pair test.",
    )


def discover() -> dict:
    """UMD's chips and links in a child process, so this one holds no device: {"bdfs": {chip: bdf}, "links": ...}."""
    out = subprocess.run([sys.executable, "-c", DISCOVER], capture_output=True, text=True, timeout=SETUP_S, check=True)
    d = json.loads(out.stdout.strip().splitlines()[-1])
    return {
        "bdfs": {int(k): v for k, v in d["bdfs"].items()},
        "links": {int(c): {int(ch): v for ch, v in m.items()} for c, m in d["links"].items()},
    }


def pairs(links: dict) -> list[list[tuple[int, int, list[int]]]]:
    """Linked chip pairs as rounds of (a, b, a's channels): greedy edge colouring, most-connected chips first."""
    chans: dict[tuple[int, int], set[int]] = {}
    for c, m in links.items():
        for ch, (rc, rch) in m.items():
            a, b = sorted((int(c), int(rc)))
            chans.setdefault((a, b), set()).add(int(ch) if int(c) == a else int(rch))
    deg: dict[int, int] = {}
    for a, b in chans:
        deg[a], deg[b] = deg.get(a, 0) + 1, deg.get(b, 0) + 1
    rounds: list[list[tuple[int, int, list[int]]]] = []
    busy: list[set[int]] = []
    for a, b in sorted(chans, key=lambda e: (-(deg[e[0]] + deg[e[1]]), e)):
        i = next((i for i, r in enumerate(busy) if a not in r and b not in r), len(rounds))
        if i == len(rounds):
            rounds.append([])
            busy.append(set())
        rounds[i].append((a, b, sorted(chans[(a, b)])))
        busy[i].update((a, b))
    return rounds


def umd_index(bdfs: list[str], sys_root: str = "/sys") -> list[int] | None:
    """The TT_VISIBLE_DEVICES index of each BDF: its rank among the PCI devices with a tt-kmd node, sorted by
    address (UMD's own order, not the /dev number). None when one has no node (off the bus)."""
    root = Path(sys_root, "bus/pci/devices")
    live = (
        sorted(d.name for d in root.iterdir() if any((d / "tenstorrent").glob("tenstorrent!*")))
        if root.is_dir()
        else []
    )
    full = [b if b.count(":") == 2 else f"0000:{b}" for b in bdfs]
    return [live.index(b) for b in full] if all(b in live for b in full) else None


def _short(bdf: str) -> str:
    return bdf.replace("0000:", "")


def link_ends(disc: dict, a: int, b: int, channels: list[int]) -> list[dict]:
    """The links of pair (a, b) by their *channels* on chip a, each end named by tray, port, ASIC and BDF."""
    return [
        {"a": link_end(disc["bdfs"][a], ch), "b": link_end(disc["bdfs"][b], disc["links"][a][ch][1])} for ch in channels
    ]


def name_chips(report: dict) -> None:
    """Add to every check of the *report*: ``bdf``, ``tray`` and ``asic`` (parallel lists, one entry per Galaxy chip the check
    blames, empty when it blames none) and ``link_ends`` (the links it names). A check blames chips in its data:
    ``blamed_chips`` (only the failing or warned ones, as naming.chip) and the ends of ``link_ends``; nothing is read
    from its text.
    """
    for phase in report["phases"].values():
        for c in phase["checks"]:
            d = c["data"]
            ends = d.get("link_ends", [])
            bdfs = [m["bdf"] for m in d.get("blamed_chips", [])] + [e["bdf"] for l in ends for e in l.values()]
            named = {b: n for b in bdfs if (n := chip(b))["tray"] is not None}
            c["bdf"], c["tray"], c["asic"] = ([n[k] for n in named.values()] for k in ("bdf", "tray", "asic"))
            c["link_ends"] = ends


def _start_pair(d: Path, lab: str, idx: list[int], seconds: int) -> subprocess.Popen:
    """The 2-chip all-gather of the chips at UMD indices *idx*, logging to <d>/<lab>.log."""
    env = {
        **os.environ,
        "TT_VISIBLE_DEVICES": ",".join(map(str, idx)),
        "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(OP_TIMEOUT_S),
    }
    # Run as an argv list, no shell: a galaxy/ program, ints, paths under *d* and a label of UMD's BDFs.
    argv = [
        sys.executable,
        str(GALAXY_DIR / "fabric_repro.py"),
        "--n",
        "2",
        "--seconds",
        str(seconds),
        "--rows",
        str(d / f"{lab}.rows"),
        "--beat",
        str(d / f"{lab}.beat"),
        "--label",
        lab,
    ]
    with open(d / f"{lab}.log", "w") as f:
        return start_group(argv, cwd=GALAXY_DIR, env=env, stdout=f, stderr=subprocess.STDOUT)


def pair_test(
    phase: Phase,
    out: Path,
    name: str,
    seconds: int,
    disc: dict,
    dry_run: bool = False,
    sys_root: str = "/sys",
    cap_s: int | None = None,
) -> bool:
    """One 2-chip all-gather per linked pair, *seconds* each, a round at a time; with *cap_s*, no round starts after
    cap_s s and its pairs are reported not reached. Adds one check (FAIL on a failed or hung pair, else SKIP when a pair
    was not reached, else PASS); True when a pair failed or hung."""
    rounds, failed, missed, check = pairs(disc["links"]), [], [], f"links_{name}"
    n = sum(len(r) for r in rounds)
    if dry_run:
        cap = f", {cap_s} s in total" if cap_s is not None else ""
        log(f"[dry-run] {name}: {n} pairs in {len(rounds)} rounds, {seconds} s each{cap}")
        print(f"  {check:30} (dry-run)")
        phase.add(Check(name=check, status=SKIP, details="dry run", ip="eth"))
        return False
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    emit_running(check)
    t0, start = time.time(), time.monotonic()
    for r in rounds:
        if cap_s is not None and time.monotonic() - start >= cap_s:
            missed += [
                {
                    "pair": f"{_short(disc['bdfs'][a])}-{_short(disc['bdfs'][b])}",
                    "channels": ch,
                    "ends": link_ends(disc, a, b, ch),
                }
                for a, b, ch in r
            ]
            continue
        jobs = []
        try:
            for a, b, ch in r:
                ba, bb, ends = disc["bdfs"][a], disc["bdfs"][b], link_ends(disc, a, b, ch)
                lab, idx = f"{_short(ba)}-{_short(bb)}", umd_index([ba, bb], sys_root)
                if idx is None:
                    failed.append(
                        {"pair": lab, "channels": ch, "why": "a chip has no tt-kmd node: not opened", "ends": ends}
                    )
                    continue
                jobs.append((lab, ch, ends, _start_pair(d, lab, idx, seconds)))
            deadline = (
                time.monotonic() + seconds + OP_TIMEOUT_S + COMPILE_S
            )  # the round's pairs run together: one limit
            for lab, ch, ends, p in jobs:
                try:
                    rc = p.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    rc = None
                    stop_group(p)
                if rc:
                    failed.append(
                        {"pair": lab, "channels": ch, "why": "data mismatch" if rc == 4 else f"rc {rc}", "ends": ends}
                    )
                elif rc is None:
                    failed.append({"pair": lab, "channels": ch, "why": "hung", "ends": ends})
        finally:
            for *_, p in jobs:
                stop_group(p)
    what = "; ".join(f"{f['pair']} (channels {','.join(map(str, f['channels']))}): {f['why']}" for f in failed)
    if missed:
        what = "; ".join(filter(None, (what, f"{len(missed)} of {n} pairs not reached in the {cap_s} s cap")))
    status = FAIL if failed else SKIP if missed else PASS
    emit_result(check, status, suffix=f"({time.time() - t0:.1f}s, {n} pairs)")
    phase.add(
        Check(
            name=check,
            status=status,
            ip="eth",
            details=what[:400] or f"{n} linked pairs, {len(rounds)} rounds",
            data={
                "pairs": n,
                "rounds": len(rounds),
                "failed": failed,
                "not_reached": missed,
                "link_ends": [e for f in failed for e in f["ends"]],
                "dir": str(d),
            },
        )
    )
    return bool(failed)


def fabric_open(phase: Phase, out: Path, dry_run: bool = False) -> None:
    phase.add(
        run_step(
            Step(
                "fabric",
                "whole-box fabric open and close",
                ("fabric_open.py", "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat", "--label", "open"),
                60,
            ),
            out,
            dry_run,
            test="links",
        )
    )


def _discover(phase: Phase, check: str, dry_run: bool) -> dict | None:
    """discover(), or None after adding a FAIL *check* that says why."""
    if dry_run:
        return {"bdfs": {}, "links": {}}
    try:
        return discover()
    except Exception as e:
        phase.add(Check(name=check, status=FAIL, details=f"UMD discovery failed: {e!r}"[:400], ip="eth"))
        return None


def run_links(
    phase: Phase, out: Path, fault: bool, reset, dry_run: bool = False, pair_s: int = PAIR_S, sweep_s: int = SWEEP_S
) -> None:
    """Pair test; on a failed or hung pair here, or *fault* (a bus-bar hang or CCL failure: busbar.link_fault), the hook's
    before event, *reset* (the glx_reset; "" when it ran clean or was skipped on purpose, else why not) and the sweep
    (*pair_s* per pair, *sweep_s* in total), which is skipped with the hook's rc or that reason; then the fabric
    open/close. The sweep tests the links discovered after the reset, which can bring back a tray that dropped under
    the load."""
    disc = _discover(phase, "links_pair", dry_run)
    if disc is not None:
        fault |= pair_test(phase, out / "links", "pair", pair_s, disc, dry_run)
    if fault and disc is not None:
        sd = out / "links" / "sweep"
        if rc := events.emit("before", "sweep", "-", sd):
            phase.add(Check(name="links_sweep", status=SKIP, details=f"held back by the hook (rc {rc})", ip="eth"))
        else:
            if why := reset():
                phase.add(Check(name="links_sweep", status=SKIP, details=why, ip="eth"))
            elif (disc := _discover(phase, "links_sweep", dry_run)) is not None:
                pair_test(phase, out / "links", "sweep", pair_s, disc, dry_run, cap_s=sweep_s)
            events.emit("after", "sweep", phase.checks[-1].status, sd)
    fabric_open(phase, out / "links", dry_run)
