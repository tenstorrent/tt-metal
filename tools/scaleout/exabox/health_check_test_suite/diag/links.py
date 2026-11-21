# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The galaxy tier's links phase: the pair link test, the sweep on fault and the fabric open/close.

pair    every chip-to-chip Ethernet link UMD discovers, tested as a 2-chip all-gather of each linked pair (data
        checked bit for bit); pairs run in rounds, no chip in two pairs of a round
sweep   after a bus-bar or pair fault: glx_reset, then the pair test again with a longer budget per pair
fabric  one whole-box fabric open and close
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from .busbar import GALAXY_DIR, SETUP_S, Step, run_step
from .common import log
from .model import FAIL, PASS, SKIP, Check, Phase
from .naming import chip, link_end

PAIR_S, SWEEP_S, OP_TIMEOUT_S = 10, 300, 30
DISCOVER = """import json, tt_umd as u
cd = u.TopologyDiscovery.create_cluster_descriptor()
print(json.dumps({"bdfs": {int(k): v for k, v in cd.get_chip_pci_bdfs().items()},
                  "links": {int(c): {int(ch): [int(rc), int(rch)] for ch, (rc, rch) in v.items()}
                            for c, v in cd.get_ethernet_connections().items()}}))"""


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


def pair_test(
    phase: Phase, out: Path, name: str, seconds: int, disc: dict, dry_run: bool = False, sys_root: str = "/sys"
) -> bool:
    """One 2-chip all-gather per linked pair, a round at a time. Adds one check; True when a pair failed or hung."""
    rounds, failed = pairs(disc["links"]), []
    n = sum(len(r) for r in rounds)
    if dry_run:
        log(f"[dry-run] {name}: {n} pairs in {len(rounds)} rounds, {seconds} s each")
        phase.add(Check(name=f"galaxy_{name}", status=SKIP, details="dry run", ip="eth"))
        return False
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    for r in rounds:
        jobs = []
        for a, b, ch in r:
            ba, bb, ends = disc["bdfs"][a], disc["bdfs"][b], link_ends(disc, a, b, ch)
            lab, idx = f"{_short(ba)}-{_short(bb)}", umd_index([ba, bb], sys_root)
            if idx is None:
                failed.append(
                    {"pair": lab, "channels": ch, "why": "a chip has no tt-kmd node: not opened", "ends": ends}
                )
                continue
            env = {
                **os.environ,
                "TT_VISIBLE_DEVICES": ",".join(map(str, idx)),
                "TT_METAL_OPERATION_TIMEOUT_SECONDS": str(OP_TIMEOUT_S),
            }
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
                jobs.append(
                    (
                        lab,
                        ch,
                        ends,
                        subprocess.Popen(
                            argv, cwd=GALAXY_DIR, env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True
                        ),
                    )
                )
        for lab, ch, ends, p in jobs:
            try:
                rc = p.wait(timeout=seconds + SETUP_S)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                rc = None
            if rc:
                failed.append(
                    {"pair": lab, "channels": ch, "why": "data mismatch" if rc == 4 else f"rc {rc}", "ends": ends}
                )
            elif rc is None:
                failed.append({"pair": lab, "channels": ch, "why": "hung", "ends": ends})
    what = "; ".join(f"{f['pair']} (channels {','.join(map(str, f['channels']))}): {f['why']}" for f in failed)
    phase.add(
        Check(
            name=f"galaxy_{name}",
            status=FAIL if failed else PASS,
            ip="eth",
            details=what[:400] or f"{n} linked pairs, {len(rounds)} rounds",
            data={
                "pairs": n,
                "rounds": len(rounds),
                "failed": failed,
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
        )
    )


def run_links(phase: Phase, out: Path, fault: bool, reset, dry_run: bool = False) -> None:
    """Pair test; on a fault here or in the bus-bar phase, *reset* (the glx_reset; "" when it ran clean, else why not) and the
    sweep, which is skipped with that reason; then the fabric open/close."""
    try:
        disc = discover() if not dry_run else {"bdfs": {}, "links": {}}
    except Exception as e:
        phase.add(Check(name="galaxy_pair", status=FAIL, details=f"UMD discovery failed: {e!r}"[:400], ip="eth"))
        disc = None
    if disc is not None:
        fault |= pair_test(phase, out / "links", "pair", PAIR_S, disc, dry_run)
    if fault and disc is not None:
        why = reset()
        if why:
            phase.add(Check(name="galaxy_sweep", status=SKIP, details=why, ip="eth"))
        else:
            pair_test(phase, out / "links", "sweep", SWEEP_S, disc, dry_run)
    fabric_open(phase, out / "links", dry_run)
