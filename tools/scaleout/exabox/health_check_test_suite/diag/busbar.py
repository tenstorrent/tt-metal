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

--order runs the parts in its order; a name outside STEPS is a part the caller runs itself, through the --hook's run
event (diag/events.py: rc 0 PASS, 3 SKIP, else FAIL). The stage emits stage-start and stage-end (step "busbar") and
before and after each part; a non-zero rc from before skips that part. Before each part but code the chips get the
clock check of every quiet (galaxy/quiet_clock.py): the quiet counts only with every chip at <= 800 MHz, at most
5 min, then a QUIET FAIL finding names the chips by BDF and tray.
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

from . import events
from .common import emit_result, emit_running, log, start_group, stop_group
from .model import FAIL, PASS, SKIP, Check, Phase

GALAXY_DIR = Path(__file__).resolve().parent.parent / "galaxy"
sys.path.append(str(GALAXY_DIR))
import chunk_sched  # noqa: E402
import numeric_report  # noqa: E402
from chunk_sched import CCL_PARTS  # noqa: E402  the parts whose output mixes every chip's data

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
    "order": ("--order", str, "the parts in order, comma-separated, each once; names outside STEPS run through --hook"),
    "chunks": ("--chunks", str, "mix: chunk minutes, comma-separated (full: the last one is --long-min)"),
    "long_min": ("--long-min", int, "mix: minutes of full's last chunk (25)"),
    "qb_n": ("--qb-n", int, "qb: clock-checked quiet + burst cycles (2)"),
    "shape": ("--shape", str, "mix: a tray-drop load shape (galaxy/drop_shapes.py list); needs --trays"),
    "shape_tray": ("--shape-tray", int, "mix: the tray --shape aims at (2)"),
}
SMOKE_S = 2
SETUP_S = 300  # mesh open, upload and close on top of a part's planned length; past both it counts as hung
PG_POLL_S = 2  # a running part's check for a VR power-good fall (the captures' power poll period)
FAIL_LINE = re.compile(r"\b[A-Z][A-Z0-9]* FAIL\b")  # NUMERIC FAIL, QUIET FAIL, G3 FAIL, ...
OP_NAME = re.compile(r"[A-Za-z0-9_]+")
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
    for k, lo in (("heavy_s", 1), ("heavy_n", 1), ("x_s", 0), ("gap_ms", 0), ("long_min", 1), ("qb_n", 1)):
        if k in ov and ov[k] < lo:
            raise ValueError(f"{OVERRIDES[k][0]}: at least {lo} (got {ov[k]})")
    if "mm_throttle" in ov and ov["mm_throttle"] not in range(1, 6):
        raise ValueError(f"--mm-throttle: 1-5 (got {ov['mm_throttle']})")
    for k in ("ops", "x_ops", "trays", "order", "chunks"):
        if k in ov:
            ov[k] = _csv(OVERRIDES[k][0], ov[k], empty_ok=k == "x_ops")
    if ov.get("shape_tray", 2) not in range(1, 5):
        raise ValueError(f"--shape-tray: 1-4 (got {ov['shape_tray']})")
    if "shape" in ov and "trays" not in ov:
        raise ValueError("--shape runs parts on some trays only: it needs --trays")
    if "trays" in ov and not all(t in ("1", "2", "3", "4") for t in ov["trays"].split(",")):
        raise ValueError(f"--trays: tray numbers 1-4, comma-separated (got {ov['trays']!r})")
    for k in ("ops", "x_ops"):
        if k in ov and not all(OP_NAME.fullmatch(o) for o in ov[k].split(",") if o):
            raise ValueError(f"{OVERRIDES[k][0]}: op names (letters, digits, _), comma-separated (got {ov[k]!r})")
    if "order" in ov:
        names = ov["order"].split(",")
        if not all(OP_NAME.fullmatch(n) for n in names) or len(set(names)) != len(names):
            raise ValueError(
                f"--order: part names (letters, digits, _), each once, comma-separated (got {ov['order']!r})"
            )
    if "chunks" in ov:
        try:
            ok = all(float(m) > 0 for m in ov["chunks"].split(","))
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(f"--chunks: minutes above 0, comma-separated (got {ov['chunks']!r})")
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

    qq, qb, gap, qn = s(60), s(30), s(10), ov.get("qb_n", 2)
    cm = [round(float(m) * 60) for m in ov["chunks"].split(",")] if "chunks" in ov else [60, 120, 180, 240]
    chunks = (*(s(c) for c in cm[:-1]), s(cm[-1], 60 * ov.get("long_min", 25)))
    dbi, l1, noc, rf, rw = s(10, 120), s(12, 60), s(5, 100), s(3, 15), s(35)
    sq, sb, hv = s(10), s(90), ov.get("heavy_s", s(90, 750))
    xs, code = min(ov.get("x_s", s(10, 750)), hv), {"smoke": 1, "plain": 5, "full": 60}[mode]
    ops, x_ops = ov.get("ops", DIDT_OPS), ov.get("x_ops", DIDT_X_OPS)
    n, gap_ms, trays = ov.get("heavy_n", HEAVY_N), ov.get("gap_ms", GAP_MS), ov.get("trays")
    burst = _burst(gap_ms)
    env = (("TT_MM_THROTTLE_PERF", str(ov["mm_throttle"])),) if "mm_throttle" in ov else ()
    on_trays = ("--trays", trays) if trays else ()
    shape = ("--shape", ov["shape"], "--shape-tray", str(ov.get("shape_tray", 2))) if "shape" in ov else ()
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
            f"{qn} x ({qq} s clock-checked quiet + {qb} s lofi_all burst)",
            ub0(qq, qb, "--numeric-every", "-1", "--cycles", str(qn)),
            qn * (qq + qb),
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
                *shape,
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
    by = {p.name: p for p in parts}
    order = ov["order"].split(",") if "order" in ov else STEPS
    return [by.get(n) or Step(n, "run by the hook", (), 0) for n in order]


def readback_lines(step: Step, d: Path) -> list[str]:
    """The per-chip lines of a part's readbacks, which its log does not print: chunk_sched's report of its rows (NUMERIC,
    QUIET, CHUNK, L1 FAIL by BDF and tray) and, for a ub0_repro.py part, numeric_report's of its result."""
    rows, res = d / "rows.jsonl", d / "result.json"
    try:
        out = chunk_sched.report(chunk_sched.load(str(rows))) if rows.is_file() else []
        return out + (numeric_report.lines([str(res)]) if step.argv[0] == "ub0_repro.py" and res.is_file() else [])
    except Exception as e:  # an unreadable row file is no finding; the part's rc still judges it
        return [f"readback report not read: {e!r}"]


def run_step(step: Step, out: Path, dry_run: bool = False, test: str = "busbar") -> Check:
    """Run one step's program in its own process group as the check ``<test>_<step>``; FAIL on a non-zero exit, a
    '<X> FAIL' line, a hang or a tray's VR power-good fall while it runs, which stops it (events.vr_pg)."""
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
    t0, rc, limit, cut = time.time(), None, step.seconds + SETUP_S, ""
    with open(log_path, "w") as f:
        # An argv list, no shell: a galaxy/ program and plan()'s args, every override checked by _overrides().
        p = start_group(argv, cwd=GALAXY_DIR, env={**os.environ, **dict(step.env)}, stdout=f, stderr=subprocess.STDOUT)
        try:
            while rc is None and time.time() - t0 < limit and not (cut := events.vr_pg(t0)):
                try:
                    rc = p.wait(timeout=PG_POLL_S)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            stop_group(p)
    dt = time.time() - t0
    text = log_path.read_text(errors="replace").splitlines()
    bad = [ln.strip() for ln in text + (readback_lines(step, d) if test == "busbar" else []) if FAIL_LINE.search(ln)]
    why = (f"stopped: {cut}" if cut else f"hung: no exit in {limit} s" if rc is None else f"rc {rc}" if rc else "") or (
        bad[0] if bad else ""
    )
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
            "hung": rc is None and not cut,
            **({"vr_pg": cut} if cut else {}),
            "fails": bad[:20],
            "seconds": round(dt),
            "log": str(log_path),
        },
    )


def link_fault(c: Check) -> bool:
    """True when a step's check is a hang or a CCL failure: killed with no exit, an op-timeout hang line or a CCL error in
    the last 40 lines of its log, or a mismatch in CCL parts alone (its rows' chunk-end verify, every chunk joined per
    chip). A chip that fails a chip-local part, with or without h3, is a numeric fault and spreads h3 to the others, as
    in chunk_sched; a plain error is not a link fault either."""
    if c.status != FAIL:
        return False
    if c.data.get("hung"):
        return True
    log_path = Path(c.data.get("log", ""))
    rows = log_path.parent / "rows.jsonl"
    if rows.is_file():
        bad: dict[str, set[str]] = {}  # each chip's bad parts over all the step's chunks, as chunk_sched joins them
        for ln in rows.read_text(errors="replace").splitlines():
            if '"verify"' not in ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("event") != "verify":
                continue
            for k, x in (r.get("chips") or {}).items():
                bad.setdefault(k, set()).update(x.get("bad_parts") or [])
        if any(bad.values()) and all(b <= CCL_PARTS for b in bad.values()):
            return True
    tail = log_path.read_text(errors="replace").splitlines()[-40:] if log_path.is_file() else []
    return any((CCL_ERR.search(ln) or HANG_ERR.search(ln)) and not LOG_LEVEL.search(ln) for ln in tail)


QUIET_T = 420  # s: quiet_clock.py settle's 5 min cap, its reads and a margin


def quiet(step: Step, out: Path, dry_run: bool = False, test: str = "busbar") -> Check | None:
    """The clock check of the quiet before *step* (quiet_clock.py settle): None when every chip reached <= 800 MHz,
    else the check ``<test>_<step>_quiet`` naming the chips (its QUIET FAIL line) or the reason it could not read."""
    d = out / step.name
    argv = [sys.executable, str(GALAXY_DIR / "quiet_clock.py"), "settle", f"before {test} {step.name}"]
    if dry_run:
        log(f"[dry-run] {' '.join(argv)}")
        return None
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "quiet.log"
    with open(log_path, "w") as f:
        try:  # the settle caps itself at 5 min; a read stuck on a hung chip must not stall the stage
            rc = subprocess.run(
                argv, cwd=GALAXY_DIR, stdout=f, stderr=subprocess.STDOUT, check=False, timeout=QUIET_T
            ).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    text = log_path.read_text(errors="replace").splitlines()
    for ln in text:
        if ln.startswith("QUIET CLOCK") and "after" in ln:
            log(ln)
    bad = [ln.strip() for ln in text if FAIL_LINE.search(ln)]
    if not bad and not rc:
        return None
    last = next((ln.strip() for ln in reversed(text) if ln.strip()), "")
    return Check(
        name=f"{test}_{step.name}_quiet",
        status=FAIL,
        details=(bad[0] if bad else f"clock check rc {rc}: {last}")[:300],
        data={"rc": rc, "fails": bad[:20], "log": str(log_path)},
    )


def run_busbar(
    phase: Phase, out: Path, mode: str, dry_run: bool = False, steps: tuple[str, ...] = STEPS, **over
) -> None:
    """The bus-bar steps (see plan) named in *steps*, in --order, into *phase*, with the quiet's clock check before each
    and the hook events (diag/events.py) around them. A check that is a link fault (link_fault) says so in its data."""
    d, held, pg = out / "busbar", len(phase.checks), ""
    # the quiet's clock check runs for the stage; one part run alone (busbar_heavy as a caller's own step) keeps its caller's quiet
    clocks = events.hook is not None or len(steps) > 1
    events.emit("stage-start", "busbar", "-", d)
    for step in plan(mode, **over):
        if step.argv and step.name not in steps:
            continue
        log(f"busbar {step.line()}")
        name, sd = f"busbar_{step.name}", d / step.name
        if pg:  # no more load after a tray's power-good fell or with chips gone
            phase.add(Check(name=name, status=SKIP, details=f"not run: {pg} in an earlier part"))
            continue
        if rc := events.emit("before", step.name, "-", sd):
            phase.add(Check(name=name, status=SKIP, details=f"held back by the hook (rc {rc})"))
            continue
        if clocks and step.name != "code" and (q := quiet(step, d, dry_run)):
            phase.add(q)
        if step.argv:
            c = run_step(step, d, dry_run)
        elif events.hook is None:
            c = Check(name=name, status=SKIP, details="no --hook to run it")
        else:
            t0 = time.time()
            rc = events.emit("run", step.name, "-", sd)
            st = SKIP if dry_run or rc == events.HOOK_SKIP else PASS if rc == 0 else FAIL
            c = Check(
                name=name, status=st, details=f"run by the hook (rc {rc})", data={"seconds": round(time.time() - t0)}
            )
        if link_fault(c):
            c.data["link_fault"] = True
        phase.add(c)
        pg = c.data.get("vr_pg", "")
        if sd.is_dir():  # the hook's after event reads the part's check here
            (sd / "check.json").write_text(json.dumps({"status": c.status, "details": c.details, **c.data}))
        events.emit("after", step.name, c.status, sd)
        if not pg and (g := events.gone()):  # its recovery left chips off PCIe: the full mesh cannot open
            pg = f"chips still off PCIe after the recovery: {', '.join(g)}"
    events.emit("stage-end", "busbar", FAIL if any(c.status == FAIL for c in phase.checks[held:]) else PASS, d)
