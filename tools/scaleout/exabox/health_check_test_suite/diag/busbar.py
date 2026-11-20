# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The galaxy tier's bus-bar phase: the bus-bar stage's parts in order, each one galaxy/ program.

qb     2 x (60 s clock-checked quiet + 30 s full-load burst), one process
mix    the chunked load: 1, 2, 3, 4 min chunks (full: 1, 2, 3, 25), a quiet and a numeric readback after each, and
       after chunk 1 G3, G3b (DRAM byte/DBI patterns), N1 (L1 march), N2 (NoC all-to-all) and the register-file walk
step   10 s quiet + 90 s burst, numeric readback after it
heavy  the DIDT ops back to back, each output checked bit-exact and voted across chips (90 s; full 750 s)
code   code readback: each core's firmware and kernel code in L1, voted across chips (exalens, no load)

--mode smoke keeps every part and cuts each length to min(smoke_s, plain's). A part that fails does not end the
phase: the next parts still run and are judged on the chips that are left.
"""
from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .common import log
from .model import FAIL, PASS, SKIP, Check, Phase

GALAXY_DIR = Path(__file__).resolve().parent.parent / "galaxy"
MODES = ("plain", "full", "smoke")
SMOKE_S = 2
SETUP_S = 300  # mesh open, upload and close on top of a part's planned length; past both it counts as hung
FAIL_LINE = re.compile(r"\b[A-Z][A-Z0-9]* FAIL\b")  # NUMERIC FAIL, QUIET FAIL, G3 FAIL, ...

BURST = ("--heavy", "lofi_all", "--burst-gap-ms", "50", "--burst-steps", "8", "--sfpu-par", "1", "--sfpu-act", "gelu:1")
DIDT_OPS = "16k_hifi2,16k_bf8_lofi"
DIDT_X_OPS = "16k_hifi4,16k_bfp4_lofi,16k_bf8_hifi2,16k_bfp4_hifi3,8k_fp32_hifi4,8k_int32_eltwise"


def _load(parts: str = "heavy,h3,mm,sfpu,dram,pcie") -> tuple[str, ...]:
    return ("--parts", parts, "--cqs", "2", "--heavy-n", "16", "--heavy-pcie-mb", "16", "--pcie-every", "1", *BURST)


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    argv: tuple[str, ...]  # a galaxy/ program and its args; {out} is the step's own output directory
    seconds: int  # planned length

    def line(self) -> str:
        return f"{self.name:<7} {self.seconds:>5} s  {self.what}"


def plan(mode: str = "plain", smoke_s: int = SMOKE_S, mix_s: tuple[int, ...] | None = None) -> list[Step]:
    """The bus-bar steps for *mode*. *mix_s* overrides the mix's chunk lengths (seconds)."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of: {', '.join(MODES)}")
    full = mode == "full"

    def s(plain: int, full_s: int | None = None) -> int:
        return min(smoke_s, plain) if mode == "smoke" else full_s if full and full_s is not None else plain

    qq, qb, gap = s(60), s(30), s(10)
    chunks = mix_s or (s(60), s(120), s(180), s(240, 1500))
    dbi, l1, noc, rf, rw = s(10, 120), s(12, 60), s(5, 100), s(3, 15), s(35)
    sq, sb, hv = s(10), s(90), s(90, 750)
    xs, code = min(s(10, 750), hv), {"smoke": 1, "plain": 5, "full": 60}[mode]
    rows = ("--gap-s", str(gap), "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat")

    def ub0(q: int, b: int, *extra: str) -> tuple[str, ...]:
        return (
            "ub0_repro.py",
            "--load",
            "ub0",
            *BURST,
            *extra,
            "--wait-quiet-min",
            "0",
            "--idle-s",
            str(q),
            "--burst-s",
            str(b),
            "--result",
            "{out}/result.json",
            "--telem",
            "{out}/telem.csv",
        )

    mins = ",".join(f"{c / 60:g}" for c in chunks)
    return [
        Step(
            "qb",
            f"2 x ({qq} s clock-checked quiet + {qb} s lofi_all burst)",
            ub0(qq, qb, "--numeric-every", "-1", "--cycles", "2"),
            2 * (qq + qb),
        ),
        Step(
            "mix",
            f"chunks of {mins} min, {gap} s quiet + numeric after each; G3, G3b {dbi} s, N1 {l1} s, "
            f"N2 {noc} s, register files {rf} s after chunk 1",
            (
                "chunk_load.py",
                "--chunk-min",
                mins,
                *_load(),
                "--heavy-check-each",
                "1",
                "--heavy-cols-rotate",
                "512,511",
                "--rw-after",
                "1",
                "--rw-s",
                str(s(15)),
                "--rw-wall-s",
                str(rw),
                "--dbi-wall-s",
                str(dbi),
                "--l1-wall-s",
                str(l1),
                *(("--l1-full",) if full else ()),
                "--noc-wall-s",
                str(noc),
                "--rf-wall-s",
                str(rf),
                *rows,
            ),
            sum(chunks) + gap * len(chunks) + rw + dbi + l1 + noc + rf,
        ),
        Step(
            "step",
            f"{sq} s quiet + {sb} s lofi_all burst, numeric after it",
            ub0(sq, sb, "--numeric-every", "500"),
            sq + sb,
        ),
        Step(
            "heavy",
            f"{hv} s of {DIDT_OPS} (last {xs} s also {DIDT_X_OPS}), each output checked",
            (
                "chunk_load.py",
                "--chunk-min",
                f"{hv / 60:g}",
                *_load("heavy"),
                "--heavy",
                DIDT_OPS,
                "--sfpu-par",
                "0",
                "--heavy-check-each",
                "1",
                "--sfpu-vote",
                "1",
                "--heavy-x",
                DIDT_X_OPS,
                "--heavy-x-s",
                str(xs),
                *rows,
            ),
            hv + gap,
        ),
        Step(
            "code",
            f"code readback, {code} s budget, {mode} scope",
            (
                "exalens_probe.py",
                "--code",
                "--budget-s",
                str(code),
                "--scope",
                mode,
                "--cursor-file",
                "{out}/cursor",
                "--out",
                "{out}/result.json",
            ),
            code + 25,
        ),
    ]


def run_step(step: Step, out: Path, dry_run: bool = False, env: dict | None = None) -> Check:
    """Run one step's program in its own process group; FAIL on a non-zero exit, a '<X> FAIL' line or a hang."""
    d = out / step.name
    argv = [sys.executable, str(GALAXY_DIR / step.argv[0]), *(a.format(out=d) for a in step.argv[1:])]
    if dry_run:
        log(f"[dry-run] {' '.join(argv)}")
        return Check(name=f"galaxy_{step.name}", status=SKIP, details="dry run", data={"argv": argv})
    d.mkdir(parents=True, exist_ok=True)
    t0, rc, limit = time.time(), None, step.seconds + SETUP_S
    with open(d / "run.log", "w") as f:
        p = subprocess.Popen(argv, cwd=GALAXY_DIR, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    text = (d / "run.log").read_text(errors="replace").splitlines()
    bad = [ln.strip() for ln in text if FAIL_LINE.search(ln)]
    why = (f"hung: no exit in {limit} s" if rc is None else f"rc {rc}" if rc else "") or (bad[0] if bad else "")
    if not why:
        return Check(
            name=f"galaxy_{step.name}",
            status=PASS,
            details=step.what,
            data={"seconds": round(time.time() - t0), "log": str(d / "run.log")},
        )
    last = next((ln.strip() for ln in reversed(text) if ln.strip()), "")
    return Check(
        name=f"galaxy_{step.name}",
        status=FAIL,
        details=f"{why}: {(bad or [last])[0][:200]}",
        data={
            "rc": rc,
            "hung": rc is None,
            "fails": bad[:20],
            "seconds": round(time.time() - t0),
            "log": str(d / "run.log"),
        },
    )


def run_busbar(
    phase: Phase,
    out: Path,
    mode: str,
    dry_run: bool = False,
    smoke_s: int = SMOKE_S,
    mix_s: tuple[int, ...] | None = None,
) -> bool:
    """Every bus-bar step, in order, into *phase*. True when one hung or failed (the links phase then sweeps)."""
    fault = False
    for step in plan(mode, smoke_s, mix_s):
        log(f"busbar {step.line()}")
        c = run_step(step, out / "busbar", dry_run)
        phase.add(c)
        fault |= c.status == FAIL
    return fault
