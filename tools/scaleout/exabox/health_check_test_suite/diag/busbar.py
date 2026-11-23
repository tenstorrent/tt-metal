# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""The bus-bar test of the galaxy load (tiers.py LOAD_TESTS): the bus-bar stage's parts in order, each one
galaxy/ program.

qb     2 x (60 s clock-checked quiet + 30 s full-load burst), one process
mix    the chunked load: 1, 2, 3, 4 min chunks (full: 1, 2, 3, 25), a quiet and a numeric readback after each, and
       after chunk 1 G3, G3b (DRAM byte/DBI patterns), N1 (L1 march), N2 (NoC all-to-all) and the register-file walk
step   10 s quiet + 90 s burst, numeric readback after it
heavy  the DIDT ops back to back, each output checked bit-exact and voted across chips (90 s; full 750 s)
code   code readback: each core's firmware and kernel code in L1, voted across chips (exalens, no load)

Each tier that runs the load names its length, one of MODES (tiers.py load_mode). smoke keeps every part and cuts
each length to min(SMOKE_S, plain's). A part that fails does not end the test: the next parts still run and are
judged on the chips that are left. OVERRIDES set one value of the parts that have it, whatever the mode says.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from .common import emit_result, emit_running, log, start_group, stop_group
from .model import FAIL, PASS, SKIP, Check, Phase

GALAXY_DIR = Path(__file__).resolve().parent.parent / "galaxy"
# What a tier lists under tests for the galaxy load, after its other tests: this bus-bar test, then the links test
# (diag/links.py), which sweeps the links after a bus-bar fault.
LOAD_TESTS = ("busbar", "links")
MODES = ("smoke", "plain", "full")
STEPS = ("qb", "mix", "step", "heavy", "code")
# plan() keyword -> (diag_runner flag, type, what it sets). Unset by default: the plan is then unchanged.
OVERRIDES = {
    "heavy_s": ("--heavy-s", int, "heavy: seconds of DIDT ops"),
    "ops": ("--heavy-ops", str, "heavy: the DIDT ops, comma-separated"),
    "x_s": ("--heavy-x-s", int, "heavy: seconds at the end that also run the extra DIDT ops (0: none)"),
    "x_ops": ("--heavy-x-ops", str, "heavy: the extra DIDT ops, comma-separated"),
    "heavy_n": ("--heavy-n", int, "mix, heavy: heavy ops per launch"),
    "gap_ms": ("--burst-gap-ms", int, "qb, mix, step, heavy: pause after each burst of heavy ops, ms"),
    "trays": ("--trays", str, "mix, heavy: load only these trays, 1-4 comma-separated"),
    "mm_throttle": ("--mm-throttle", int, "qb, mix, step, heavy: matmul di/dt throttle, TT_MM_THROTTLE_PERF 1-5"),
}
SMOKE_S = 2
SETUP_S = 300  # mesh open, upload and close on top of a part's planned length; past both it counts as hung
FAIL_LINE = re.compile(r"\b[A-Z][A-Z0-9]* FAIL\b")  # NUMERIC FAIL, QUIET FAIL, G3 FAIL, ...
OP_NAME = re.compile(r"[A-Za-z0-9_]+")
CCL_PARTS = {"h3", "h3-ref"}  # chunk_load parts whose output mixes every chip's data (chunk_sched.CCL_PARTS)
# an error line from a CCL op: the active Ethernet-core timeout, a critical line naming an ERISC, or a traceback, RuntimeError
# or TT_THROW line with a CCL word; info, warning, debug and trace lines never count (fabric set-up lines are in every log)
CCL_ERR = re.compile(
    r"Timed out while waiting for active ethernet core|\| *critical *\|.*erisc"
    r"|(Traceback|^\s*File \"|RuntimeError|TT_THROW).*(all_gather|reduce_scatter|fabric|erisc|ccl)"
    r"|(all_gather|reduce_scatter|fabric|erisc|ccl).*(Traceback|RuntimeError|TT_THROW)",
    re.I,
)
# the op timeout's hang line (TT_METAL_OPERATION_TIMEOUT_SECONDS set: a hung op exits with it instead of being killed)
HANG_ERR = re.compile(r"device timeout|potential hang detected", re.I)
LOG_LEVEL = re.compile(r"\| *(info|warning|debug|trace) *\|", re.I)

GAP_MS, HEAVY_N = 50, 16  # the burst's pause after each 8 heavy ops; heavy ops per launch
DIDT_OPS = "16k_hifi2,16k_bf8_lofi"
DIDT_X_OPS = "16k_hifi4,16k_bfp4_lofi,16k_bf8_hifi2,16k_bfp4_hifi3,8k_fp32_hifi4,8k_int32_eltwise"


def _burst(gap_ms: int) -> tuple[str, ...]:
    return (
        "--heavy",
        "lofi_all",
        "--burst-gap-ms",
        str(gap_ms),
        "--burst-steps",
        "8",
        "--sfpu-par",
        "1",
        "--sfpu-act",
        "gelu:1",
    )


def _load(parts: str, n: int, gap_ms: int) -> tuple[str, ...]:
    return (
        "--parts",
        parts,
        "--cqs",
        "2",
        "--heavy-n",
        str(n),
        "--heavy-pcie-mb",
        "16",
        "--pcie-every",
        "1",
        *_burst(gap_ms),
    )


def _csv(name: str, v: str, empty_ok: bool = False) -> str:
    v = v.replace(" ", "").strip(",")
    if not v and not empty_ok:
        raise ValueError(f"{name}: empty")
    return v


def _overrides(over: dict) -> dict:
    """The set overrides, checked."""
    if unknown := set(over) - set(OVERRIDES):
        raise ValueError(f"unknown override {', '.join(sorted(unknown))}; expected: {', '.join(OVERRIDES)}")
    ov = {k: v for k, v in over.items() if v is not None}
    for k, lo in (("heavy_s", 1), ("heavy_n", 1), ("x_s", 0), ("gap_ms", 0)):
        if k in ov and ov[k] < lo:
            raise ValueError(f"{OVERRIDES[k][0]}: at least {lo} (got {ov[k]})")
    if "mm_throttle" in ov and ov["mm_throttle"] not in range(1, 6):
        raise ValueError(f"--mm-throttle: 1-5 (got {ov['mm_throttle']})")
    for k in ("ops", "x_ops", "trays"):
        if k in ov:
            ov[k] = _csv(OVERRIDES[k][0], ov[k], empty_ok=k == "x_ops")
    if "trays" in ov and not all(t in ("1", "2", "3", "4") for t in ov["trays"].split(",")):
        raise ValueError(f"--trays: tray numbers 1-4, comma-separated (got {ov['trays']!r})")
    for k in ("ops", "x_ops"):
        if k in ov and not all(OP_NAME.fullmatch(o) for o in ov[k].split(",") if o):
            raise ValueError(f"{OVERRIDES[k][0]}: op names (letters, digits, _), comma-separated (got {ov[k]!r})")
    return ov


@dataclass(frozen=True)
class Step:
    name: str
    what: str
    argv: tuple[str, ...]  # a galaxy/ program and its args; {out} is the step's own output directory
    seconds: int  # planned length
    env: tuple[tuple[str, str], ...] = ()  # set in the program's environment

    def line(self) -> str:
        return f"{self.name:<7} {self.seconds:>5} s  {self.what}"


def plan(mode: str = "plain", **over: int | str | None) -> list[Step]:
    """The bus-bar steps for *mode*, in order; *over* is OVERRIDES."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of: {', '.join(MODES)}")
    ov = _overrides(over)
    full = mode == "full"

    def s(plain: int, full_s: int | None = None) -> int:
        return min(SMOKE_S, plain) if mode == "smoke" else full_s if full and full_s is not None else plain

    qq, qb, gap = s(60), s(30), s(10)
    chunks = (s(60), s(120), s(180), s(240, 1500))
    dbi, l1, noc, rf, rw = s(10, 120), s(12, 60), s(5, 100), s(3, 15), s(35)
    sq, sb, hv = s(10), s(90), ov.get("heavy_s", s(90, 750))
    xs, code = min(ov.get("x_s", s(10, 750)), hv), {"smoke": 1, "plain": 5, "full": 60}[mode]
    ops, x_ops = ov.get("ops", DIDT_OPS), ov.get("x_ops", DIDT_X_OPS)
    n, gap_ms, trays = ov.get("heavy_n", HEAVY_N), ov.get("gap_ms", GAP_MS), ov.get("trays")
    burst = _burst(gap_ms)
    env = (("TT_MM_THROTTLE_PERF", str(ov["mm_throttle"])),) if "mm_throttle" in ov else ()
    on_trays = ("--trays", trays) if trays else ()
    rows = ("--gap-s", str(gap), "--rows", "{out}/rows.jsonl", "--beat", "{out}/beat")

    def ub0(q: int, b: int, *extra: str) -> tuple[str, ...]:
        return (
            "ub0_repro.py",
            "--load",
            "ub0",
            *burst,
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
    parts = [
        Step(
            "qb",
            f"2 x ({qq} s clock-checked quiet + {qb} s lofi_all burst)",
            ub0(qq, qb, "--numeric-every", "-1", "--cycles", "2"),
            2 * (qq + qb),
            env,
        ),
        Step(
            "mix",
            f"chunks of {mins} min, {gap} s quiet + numeric after each; G3, G3b {dbi} s, N1 {l1} s, "
            f"N2 {noc} s, register files {rf} s after chunk 1",
            (
                "chunk_load.py",
                "--chunk-min",
                mins,
                *_load("heavy,h3,mm,sfpu,dram,pcie", n, gap_ms),
                *on_trays,
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
            env,
        ),
        Step(
            "step",
            f"{sq} s quiet + {sb} s lofi_all burst, numeric after it",
            ub0(sq, sb, "--numeric-every", "500"),
            sq + sb,
            env,
        ),
        Step(
            "heavy",
            f"{hv} s of {ops} (last {xs} s also {x_ops}), each output checked",
            (
                "chunk_load.py",
                "--chunk-min",
                f"{hv / 60:g}",
                *_load("heavy", n, gap_ms),
                *on_trays,
                "--heavy",
                ops,
                "--sfpu-par",
                "0",
                "--heavy-check-each",
                "1",
                "--sfpu-vote",
                "1",
                "--heavy-x",
                x_ops,
                "--heavy-x-s",
                str(xs),
                *rows,
            ),
            hv + gap,
            env,
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
    return parts


def run_step(step: Step, out: Path, dry_run: bool = False, test: str = "busbar") -> Check:
    """Run one step's program in its own process group as the check ``<test>_<step>``; FAIL on a non-zero exit, a
    '<X> FAIL' line or a hang."""
    name, d = f"{test}_{step.name}", out / step.name
    argv = [sys.executable, str(GALAXY_DIR / step.argv[0]), *(a.format(out=d) for a in step.argv[1:])]
    if dry_run:
        log(f"[dry-run] {' '.join(argv)}")
        print(f"  {name:30} (dry-run)")
        return Check(
            name=name,
            status=SKIP,
            details="dry run",
            data={"argv": argv, **({"env": dict(step.env)} if step.env else {})},
        )
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / f"{name}.log"
    emit_running(name)
    t0, rc, limit = time.time(), None, step.seconds + SETUP_S
    with open(log_path, "w") as f:
        # An argv list, no shell: a galaxy/ program and plan()'s args, every override checked by _overrides().
        p = start_group(argv, cwd=GALAXY_DIR, env={**os.environ, **dict(step.env)}, stdout=f, stderr=subprocess.STDOUT)
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            pass
        finally:
            stop_group(p)
    dt = time.time() - t0
    text = log_path.read_text(errors="replace").splitlines()
    bad = [ln.strip() for ln in text if FAIL_LINE.search(ln)]
    why = (f"hung: no exit in {limit} s" if rc is None else f"rc {rc}" if rc else "") or (bad[0] if bad else "")
    emit_result(name, FAIL if why else PASS, suffix=f"({dt:.1f}s)")
    if not why:
        return Check(
            name=name,
            status=PASS,
            details=step.what,
            data={"seconds": round(dt), "log": str(log_path)},
        )
    last = next((ln.strip() for ln in reversed(text) if ln.strip()), "")
    return Check(
        name=name,
        status=FAIL,
        details=f"{why}: {(bad or [last])[0][:200]}",
        data={
            "rc": rc,
            "hung": rc is None,
            "fails": bad[:20],
            "seconds": round(dt),
            "log": str(log_path),
        },
    )


def link_fault(c: Check) -> bool:
    """True when a step's check is a hang or a CCL failure: killed with no exit, an op-timeout hang line or a CCL error in
    the last 40 lines of its log, or a mismatch in a CCL part (its rows' chunk-end verify; h3-only chips beside a chip
    with a chip-local fault are its spread, as in chunk_sched). A numeric fault in another part or a plain error is
    not one."""
    if c.status != FAIL:
        return False
    if c.data.get("hung"):
        return True
    log_path = Path(c.data.get("log", ""))
    rows = log_path.parent / "rows.jsonl"
    if rows.is_file():
        for ln in rows.read_text(errors="replace").splitlines():
            if '"verify"' not in ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("event") != "verify":
                continue
            bad = [set(x.get("bad_parts") or []) for x in (r.get("chips") or {}).values()]
            local = any(
                b - CCL_PARTS for b in bad
            )  # a chip failing a chip-local part spreads it to the others' h3 output
            if any(b & CCL_PARTS and not (local and b <= CCL_PARTS) for b in bad):
                return True
    tail = log_path.read_text(errors="replace").splitlines()[-40:] if log_path.is_file() else []
    return any((CCL_ERR.search(ln) or HANG_ERR.search(ln)) and not LOG_LEVEL.search(ln) for ln in tail)


def run_busbar(
    phase: Phase, out: Path, mode: str, dry_run: bool = False, steps: tuple[str, ...] = STEPS, **over
) -> bool:
    """The bus-bar steps (see plan) named in *steps*, in order, into *phase*. True when one hung or had a CCL failure
    (link_fault; the links test then sweeps); a numeric fault alone is not one."""
    fault = False
    for step in plan(mode, **over):
        if step.name not in steps:
            continue
        log(f"busbar {step.line()}")
        c = run_step(step, out / "busbar", dry_run)
        phase.add(c)
        fault |= link_fault(c)
    return fault
