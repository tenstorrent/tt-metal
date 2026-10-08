#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""One driver for every tt-ember sweep, configured by a YAML file.

A sweep is a list of runs and a list of analyses. Every run is one auto.py invocation: reset the
board, export the workload knobs into the environment, run the workload under the telemetry
sampler into <output-root>/<subdir>, parse. Every analysis reads the program_intervals.csv files
those runs produced. The configs under sweeps/ pin down one paper result each, so a figure is one
command:

    python3 run_sweep.py --config sweeps/paper_fig1_stage_breakdown.yaml \\
        --telemetry-exe $TT_METAL_HOME/build_Release/tools/umd/telemetry \\
        --app-exe $TT_METAL_HOME/build_Release/programming_examples/metal_example_long_matmul \\
        --tt-venv-activate ~/venv/bin/activate --tt-metal-root $TT_METAL_HOME \\
        --output-root ./out_fig1

Config keys:

    name:              free text, recorded in PROVENANCE.md
    workload:          long_matmul | ttnn_ops
    app_args:          positional args for the workload (M N K iters [fixed_blocks_per_core] for
                       long_matmul; the ttnn_ops_workload.py flags for ttnn_ops)
    telemetry_freq_hz: sampling rate, default 50
    trim_ms, slot_ms:  passed to parser.py, defaults 1.0 and 1
    aiclk_mhz:         optional; pin AICLK to this after every reset (Blackhole, via aiclk/set_aiclk.py)
    runs:              long_matmul: {power_cases: [..], ops: [..], blocks: [[M, N], ..]} --
                       one run per combination, into <op>/<case>[_b<M>x<N>]
                       ttnn_ops: {subdir: <name>} -- one run
    analyses:          list of step names, or {name: {options}}; see ANALYSES below

Existing output directories are skipped unless --force, so an interrupted sweep resumes.
--dry-run prints every command without touching the board. Every command and its environment
goes to <output-root>/sweep.log; the config, the tt-metal commit and the board list go to
<output-root>/PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis import power_cases as pc  # noqa: E402

RESET_COOLDOWN_S = 10
AICLK_SETTLE_S = 5

WORKLOADS = ("long_matmul", "ttnn_ops")


@dataclass
class RunSpec:
    subdir: str
    app_args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    op: Optional[str] = None
    power_case: Optional[int] = None
    block: Tuple[int, int] = (1, 1)


@dataclass
class Sweep:
    name: str
    workload: str
    app_args: List[str]
    telemetry_freq_hz: int
    trim_ms: float
    slot_ms: int
    aiclk_mhz: Optional[int]
    runs: List[RunSpec]
    ops: List[str]
    cases: List[Tuple[str, str]]
    analyses: List[Tuple[str, dict]]
    raw: dict


class Log:
    def __init__(self, path: Optional[Path]):
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(f"{dt.datetime.now().isoformat(timespec='seconds')} {msg}\n")


def shell_join(args: List[str]) -> str:
    return " ".join(shlex.quote(a) for a in args)


def load_sweep(path: Path) -> Sweep:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top level must be a mapping")

    workload = raw.get("workload", "long_matmul")
    if workload not in WORKLOADS:
        raise SystemExit(f"{path}: workload must be one of {WORKLOADS}, got {workload!r}")

    app_args = [str(a) for a in raw.get("app_args", [])]
    runs_cfg = raw.get("runs") or {}
    runs: List[RunSpec] = []
    ops: List[str] = []
    cases: List[Tuple[str, str]] = []

    if workload == "long_matmul":
        power_cases = [int(c) for c in runs_cfg.get("power_cases", [0])]
        for c in power_cases:
            if c not in pc.POWER_CASE_SPECS:
                raise SystemExit(f"{path}: unknown POWER_CASE {c}")
        ops = [str(o) for o in runs_cfg.get("ops", ["matmul"])]
        for o in ops:
            if o not in pc.ALL_OPS:
                raise SystemExit(f"{path}: unknown op {o!r}, expected one of {pc.ALL_OPS}")
        blocks = [tuple(int(v) for v in b) for b in runs_cfg.get("blocks", [[1, 1]])]
        for b in blocks:
            if len(b) != 2 or b[0] < 1 or b[1] < 1 or b[0] * b[1] > 8:
                raise SystemExit(f"{path}: block must be [M, N] with 1 <= M*N <= 8, got {b}")
        cases = [pc.POWER_CASE_SPECS[c] for c in power_cases]
        for op in ops:
            for c in power_cases:
                case_name, _label = pc.POWER_CASE_SPECS[c]
                for bm, bn in blocks:
                    suffix = "" if (bm, bn) == (1, 1) else f"_b{bm}x{bn}"
                    env = {"POWER_CASE": str(c), "LONG_MATMUL_OP": op}
                    if (bm, bn) != (1, 1):
                        env["LONG_MATMUL_BLOCK_M"] = str(bm)
                        env["LONG_MATMUL_BLOCK_N"] = str(bn)
                    runs.append(
                        RunSpec(
                            subdir=f"{op}/{case_name}{suffix}",
                            app_args=app_args,
                            env=env,
                            op=op,
                            power_case=c,
                            block=(bm, bn),
                        )
                    )
    else:
        subdir = str(runs_cfg.get("subdir", "decoder_block"))
        runs.append(RunSpec(subdir=subdir, app_args=app_args))

    analyses: List[Tuple[str, dict]] = []
    for item in raw.get("analyses") or []:
        if isinstance(item, str):
            analyses.append((item, {}))
        elif isinstance(item, dict) and len(item) == 1:
            ((name, opts),) = item.items()
            analyses.append((str(name), dict(opts or {})))
        else:
            raise SystemExit(f"{path}: analyses entries must be a name or {{name: {{options}}}}, got {item!r}")
    for name, _ in analyses:
        if name not in ANALYSES:
            raise SystemExit(f"{path}: unknown analysis {name!r}, expected one of {sorted(ANALYSES)}")

    return Sweep(
        name=str(raw.get("name", path.stem)),
        workload=workload,
        app_args=app_args,
        telemetry_freq_hz=int(raw.get("telemetry_freq_hz", 50)),
        trim_ms=float(raw.get("trim_ms", 1.0)),
        slot_ms=int(raw.get("slot_ms", 1)),
        aiclk_mhz=(int(raw["aiclk_mhz"]) if raw.get("aiclk_mhz") else None),
        runs=runs,
        ops=ops,
        cases=cases,
        analyses=analyses,
        raw=raw,
    )


def matmul_flops(app_args: List[str]) -> Optional[float]:
    if len(app_args) < 4:
        return None
    try:
        m, n, k, iters = (int(app_args[i]) for i in range(4))
    except ValueError:
        return None
    return 2.0 * m * n * k * iters


# --- board control -------------------------------------------------------------------------------


def run_cmd(cmd: List[str], log: Log, dry_run: bool, env: Optional[Dict[str, str]] = None) -> int:
    extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    log(f"[CMD] {extra + ' ' if extra else ''}{shell_join(cmd)}")
    if dry_run:
        return 0
    full_env = dict(os.environ)
    full_env.update(env or {})
    return subprocess.run(cmd, env=full_env).returncode


def reset_hardware(tt_smi: str, log: Log, dry_run: bool) -> int:
    rc = run_cmd([tt_smi, "-r"], log, dry_run)
    if rc != 0:
        log(f"[RESET] ERROR: {tt_smi} -r failed with {rc}")
        return rc
    if not dry_run:
        log(f"[RESET] waiting {RESET_COOLDOWN_S}s for the device to re-initialize")
        time.sleep(RESET_COOLDOWN_S)
    return 0


def pin_aiclk(mhz: int, device_id: int, activate: Path, log: Log, dry_run: bool) -> int:
    # A reset clears the ARC FORCE_AICLK override, so this runs after every reset. Same check as
    # aiclk/tt-smi-aiclk1000.sh: the value the chip reports back has to be the one requested.
    setter = HERE / "aiclk" / "set_aiclk.py"
    bash = f"sleep {AICLK_SETTLE_S} && source {shlex.quote(str(activate))} && python {shlex.quote(str(setter))} {mhz} --busy --device-id {device_id}"
    log(f"[AICLK] pin to {mhz} MHz: bash -c {shlex.quote(bash)}")
    if dry_run:
        return 0
    res = subprocess.run(["bash", "-c", bash], capture_output=True, text=True)
    log(res.stdout.rstrip())
    got = None
    for line in res.stdout.splitlines():
        if "after force:" in line and "'AICLK':" in line:
            got = line.split("'AICLK':")[1].split(",")[0].strip().strip("}")
    if res.returncode != 0 or got != str(mhz):
        log(f"[AICLK] ERROR: could not pin AICLK to {mhz} MHz (read back {got!r})\n{res.stderr}")
        return 1
    return 0


# --- runs ----------------------------------------------------------------------------------------


def auto_py_command(args: argparse.Namespace, sweep: Sweep, spec: RunSpec, output_root: Path) -> List[str]:
    if sweep.workload == "long_matmul":
        app_exe = args.app_exe
        if app_exe is None:
            raise SystemExit("--app-exe is required for the long_matmul workload")
    else:
        app_exe = args.app_exe or (HERE / "op_power_breakdown" / "ttnn_ops_workload.py")

    # <output-root>/<op>/<case> is produced by giving auto.py <output-root>/<op> as its root and
    # <case> as the subdir, which keeps its own launcher.log/telemetry.txt layout intact.
    sub = Path(spec.subdir)
    run_root = output_root / sub.parent if sub.parent != Path(".") else output_root

    cmd = [
        args.python_exe,
        str(HERE / "auto.py"),
        "--telemetry-exe",
        str(args.telemetry_exe),
        "--telemetry-freq",
        str(sweep.telemetry_freq_hz),
        "--app-exe",
        str(app_exe),
        "--parser-script",
        str(args.parser_script),
        "--tt-venv-activate",
        str(args.tt_venv_activate),
        "--tt-metal-root",
        str(args.tt_metal_root),
        "--output-root",
        str(run_root),
        "--subdir",
        sub.name,
        "--slot-ms",
        str(sweep.slot_ms),
        "--trim-ms",
        str(sweep.trim_ms),
        "--python-exe",
        args.python_exe,
    ]
    if args.device_id is not None:
        cmd += ["--device-id", str(args.device_id)]
    # --app-args must be last (argparse.REMAINDER in auto.py)
    cmd += ["--app-args"] + spec.app_args
    return cmd


def run_sweep(args: argparse.Namespace, sweep: Sweep, output_root: Path, log: Log) -> int:
    for i, spec in enumerate(sweep.runs, start=1):
        run_dir = output_root / spec.subdir
        head = f"[RUN {i}/{len(sweep.runs)}] {spec.subdir}"
        if run_dir.exists() and not args.force:
            log(f"{head}: exists, skipping (use --force to re-run)")
            continue
        if run_dir.exists() and args.force and not args.dry_run:
            shutil.rmtree(run_dir)

        log(head)
        rc = reset_hardware(args.tt_smi, log, args.dry_run)
        if rc != 0:
            return rc
        if sweep.aiclk_mhz is not None:
            rc = pin_aiclk(sweep.aiclk_mhz, args.device_id or 0, args.tt_venv_activate, log, args.dry_run)
            if rc != 0:
                return rc

        rc = run_cmd(auto_py_command(args, sweep, spec, output_root), log, args.dry_run, env=spec.env)
        if rc != 0:
            log(f"{head}: FAILED with {rc}, aborting")
            return rc
    return 0


# --- analyses ------------------------------------------------------------------------------------


def _grid_order(output_root: Path, sweep: Sweep) -> List[str]:
    for op in sweep.ops:
        rows = pc._load_program_intervals(output_root / op / pc.ALL_ACTIVE_CASE / "program_intervals.csv")
        if rows:
            return list(rows.keys())
    for op in sweep.ops:
        for case_name, _ in sweep.cases:
            rows = pc._load_program_intervals(output_root / op / case_name / "program_intervals.csv")
            if rows:
                return list(rows.keys())
    return []


def _script(cmd: List[str], log: Log, dry_run: bool) -> int:
    return run_cmd(cmd, log, dry_run)


def an_compare_runs2(args, sweep, output_root, opts, log) -> int:
    for op in sweep.ops:
        cases_file = output_root / op / "power_cases.txt"
        if not args.dry_run:
            pc.write_cases_file(cases_file, sweep.cases)
        rc = _script(
            [
                args.python_exe,
                str(HERE / "compare_runs2.py"),
                "-i",
                str(output_root / op),
                "-c",
                str(cases_file),
                "--dpi",
                str(args.dpi),
            ],
            log,
            args.dry_run,
        )
        if rc != 0:
            return rc
    return 0


def an_energy_by_engine(args, sweep, output_root, opts, log) -> int:
    if args.dry_run:
        log("[ANALYSIS] energy_by_engine (dry-run) skipped")
        return 0
    grids = _grid_order(output_root, sweep)
    for op in sweep.ops:
        per_engine = pc.compute_energy_by_engine(output_root / op)
        if per_engine is not None:
            out = output_root / op / "compare_runs_out"
            out.mkdir(parents=True, exist_ok=True)
            pc.plot_energy_by_engine_stacked(per_engine, grids, out / "energy_by_engine_stacked.png", args.dpi)
    return 0


def an_energy_per_flop_by_engine(args, sweep, output_root, opts, log) -> int:
    flops = matmul_flops(sweep.app_args)
    if flops is None:
        log("[ANALYSIS] energy_per_flop_by_engine needs M N K iters in app_args, skipping")
        return 0
    if args.dry_run:
        log("[ANALYSIS] energy_per_flop_by_engine (dry-run) skipped")
        return 0
    grids = _grid_order(output_root, sweep)
    for op in sweep.ops:
        out = output_root / op / "compare_runs_out"
        out.mkdir(parents=True, exist_ok=True)
        per_engine = pc.compute_energy_per_flop_by_engine(output_root / op, flops)
        if per_engine is not None:
            pc.plot_energy_per_flop_by_engine(per_engine, grids, out / "energy_per_flop_by_engine.png", args.dpi)
        per_case = pc.compute_energy_per_flop_per_core(output_root / op, sweep.cases, flops)
        if per_case:
            pc.plot_energy_per_flop_per_core(
                per_case, sweep.cases, grids, out / "energy_per_flop_per_core.png", args.dpi
            )
    return 0


def an_cross_op(args, sweep, output_root, opts, log) -> int:
    if len(sweep.ops) < 2:
        log("[ANALYSIS] cross_op needs at least two ops, skipping")
        return 0
    if args.dry_run:
        log("[ANALYSIS] cross_op (dry-run) skipped")
        return 0
    cases_file = output_root / "power_cases_cross_op.txt"
    entries = pc.write_cross_op_cases_file(cases_file, sweep.ops, sweep.cases, output_root)
    if len(entries) >= 2:
        rc = _script(
            [
                args.python_exe,
                str(HERE / "compare_runs2.py"),
                "-i",
                str(output_root),
                "-c",
                str(cases_file),
                "--dpi",
                str(args.dpi),
            ],
            log,
            args.dry_run,
        )
        if rc != 0:
            return rc
    grids = _grid_order(output_root, sweep)
    out = output_root / "compare_runs_out"
    out.mkdir(parents=True, exist_ok=True)
    pc.plot_cross_op_energy_by_engine_stacked(
        sweep.ops, output_root, grids, out / "energy_by_engine_stacked_cross_op.png", args.dpi
    )
    pc.plot_cross_op_energy_share_by_engine_stacked(
        sweep.ops, output_root, grids, out / "energy_share_by_engine_stacked_cross_op.png", args.dpi
    )
    flops = matmul_flops(sweep.app_args)
    if flops is not None:
        pc.plot_total_energy_per_flop_by_op(
            sweep.ops, output_root, grids, flops, out / "total_energy_per_flop_by_op.png", args.dpi
        )
        pc.plot_cross_op_energy_per_flop_by_engine_stacked(
            sweep.ops, output_root, grids, flops, out / "energy_per_flop_by_engine_stacked_cross_op.png", args.dpi
        )
    return 0


def an_cross_op_corrected(args, sweep, output_root, opts, log) -> int:
    if len(sweep.app_args) < 4:
        log("[ANALYSIS] cross_op_corrected needs M N K iters in app_args, skipping")
        return 0
    cmd = [
        args.python_exe,
        str(HERE / "analysis" / "crossop_corrected_flops.py"),
        str(output_root),
        "--app-args",
        *sweep.app_args[:4],
        "--dpi",
        str(args.dpi),
        "--out",
        str(output_root / "compare_runs_out" / "energy_per_flop_by_engine_stacked_cross_op_corrected.png"),
    ]
    if "board" in opts:
        cmd += ["--board", str(opts["board"])]
    if not args.dry_run:
        (output_root / "compare_runs_out").mkdir(parents=True, exist_ok=True)
    return _script(cmd, log, args.dry_run)


def an_pj_per_flop(args, sweep, output_root, opts, log) -> int:
    if len(sweep.app_args) < 4:
        log("[ANALYSIS] pj_per_flop needs M N K iters in app_args, skipping")
        return 0
    m, n, k, iters = sweep.app_args[:4]
    for op in sweep.ops:
        cmd = [
            args.python_exe,
            str(HERE / "analysis" / "make_pj_per_flop.py"),
            str(output_root / op),
            "--label",
            str(opts.get("label", f"{sweep.name} ({op})")),
            "--seq",
            m,
            "--hidden",
            n,
            "--k",
            k,
            "--iters",
            iters,
            "--dpi",
            str(args.dpi),
            "--out",
            str(output_root / op / "compare_runs_out" / "pj_per_flop.png"),
        ]
        if not args.dry_run:
            (output_root / op / "compare_runs_out").mkdir(parents=True, exist_ok=True)
        rc = _script(cmd, log, args.dry_run)
        if rc != 0:
            return rc
    return 0


def an_pj_per_flop_by_engine(args, sweep, output_root, opts, log) -> int:
    # The published Fig. 1: peak-charge ablation against writer_amp, with regular (POWER_CASE=0)
    # as the writer stand-in. Needs power cases 0, 1, 2 and 4.
    if len(sweep.app_args) < 4:
        log("[ANALYSIS] pj_per_flop_by_engine needs M N K iters in app_args, skipping")
        return 0
    m, n, k, iters = sweep.app_args[:4]
    for op in sweep.ops:
        out = output_root / op / "compare_runs_out" / "pj_per_flop_by_engine.png"
        cmd = [
            args.python_exe,
            str(HERE / "analysis" / "make_pj_per_flop_by_engine.py"),
            str(output_root / op),
            "--label",
            str(opts.get("label", f"{sweep.name} ({op})")),
            "--seq",
            m,
            "--hidden",
            n,
            "--k",
            k,
            "--iters",
            iters,
            "--dpi",
            str(args.dpi),
            "--out",
            str(out),
        ]
        if not args.dry_run:
            out.parent.mkdir(parents=True, exist_ok=True)
        rc = _script(cmd, log, args.dry_run)
        if rc != 0:
            return rc
    return 0


def an_naive_vs_blocked(args, sweep, output_root, opts, log) -> int:
    op = sweep.ops[0] if sweep.ops else "matmul"
    case = opts.get("case", "regular")
    naive = output_root / str(opts.get("naive", f"{op}/{case}")) / "program_intervals.csv"
    blocked_default = next(
        (f"{op}/{case}_b{r.block[0]}x{r.block[1]}" for r in sweep.runs if r.block != (1, 1)), f"{op}/{case}_b2x4"
    )
    blocked = output_root / str(opts.get("blocked", blocked_default)) / "program_intervals.csv"
    out = output_root / "compare_runs_out" / "pj_per_flop_naive_vs_blocked.png"
    if len(sweep.app_args) < 4:
        log("[ANALYSIS] naive_vs_blocked needs M N K iters in app_args, skipping")
        return 0
    if not args.dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.python_exe,
        str(HERE / "analysis" / "make_pj_per_flop_naive_vs_blocked.py"),
        str(naive),
        str(blocked),
        "--app-args",
        *sweep.app_args[:4],
        "--dpi",
        str(args.dpi),
        "--out",
        str(out),
    ]
    if "board" in opts:
        cmd += ["--board", str(opts["board"])]
    return _script(cmd, log, args.dry_run)


def an_compare_runs(args, sweep, output_root, opts, log) -> int:
    # compare_runs.py auto-discovers every program_intervals.csv under a root and overlays the
    # runs as line plots; this is what run_all.py used for its prefill / fixed-tpc comparisons.
    cmd = [
        args.python_exe,
        str(HERE / "compare_runs.py"),
        "--out-root",
        str(output_root),
        "--output-dir",
        str(output_root / "comparison"),
        "--dpi",
        str(args.dpi),
    ]
    if "filter" in opts:
        cmd += [
            "--filter",
            *[str(f) for f in (opts["filter"] if isinstance(opts["filter"], list) else [opts["filter"]])],
        ]
    return _script(cmd, log, args.dry_run)


def an_op_breakdown(args, sweep, output_root, opts, log) -> int:
    rc = 0
    for spec in sweep.runs:
        rc = _script(
            [
                args.python_exe,
                str(HERE / "op_power_breakdown" / "preview_op_breakdown.py"),
                str(output_root / spec.subdir),
            ],
            log,
            args.dry_run,
        )
        if rc != 0:
            return rc
    return rc


ANALYSES = {
    "compare_runs2": an_compare_runs2,
    "energy_by_engine": an_energy_by_engine,
    "energy_per_flop_by_engine": an_energy_per_flop_by_engine,
    "cross_op": an_cross_op,
    "cross_op_corrected": an_cross_op_corrected,
    "pj_per_flop": an_pj_per_flop,
    "pj_per_flop_by_engine": an_pj_per_flop_by_engine,
    "naive_vs_blocked": an_naive_vs_blocked,
    "compare_runs": an_compare_runs,
    "op_breakdown": an_op_breakdown,
}


# --- provenance ----------------------------------------------------------------------------------


def _git(root: Path, *argv: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_provenance(args: argparse.Namespace, sweep: Sweep, config_path: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_root / "sweep.yaml")
    boards = "unknown"
    if not args.dry_run:
        try:
            boards = subprocess.run([args.tt_smi, "-ls"], capture_output=True, text=True, timeout=60).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    lines = [
        f"# {sweep.name}",
        "",
        f"- date: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- host: {platform.node()}",
        f"- command: {shell_join(sys.argv)}",
        f"- config: sweep.yaml (copied from {config_path})",
        f"- tt-metal: {_git(args.tt_metal_root, 'rev-parse', 'HEAD')} ({_git(args.tt_metal_root, 'describe', '--always', '--dirty')})",
        f"- tools/tt-ember: {_git(HERE, 'rev-parse', 'HEAD')} ({_git(HERE, 'describe', '--always', '--dirty')})",
        f"- workload: {sweep.workload} {shell_join(sweep.app_args)}",
        f"- telemetry: {sweep.telemetry_freq_hz} Hz, trim {sweep.trim_ms} ms, slot {sweep.slot_ms} ms"
        + (f", AICLK pinned to {sweep.aiclk_mhz} MHz" if sweep.aiclk_mhz else ""),
        f"- runs: {len(sweep.runs)} -> " + ", ".join(r.subdir for r in sweep.runs),
        "",
        "## Boards (tt-smi -ls)",
        "",
        "```",
        boards,
        "```",
        "",
    ]
    (output_root / "PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")


# --- main ----------------------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path, help="Sweep config (YAML); see sweeps/.")
    ap.add_argument("--telemetry-exe", required=True, type=Path)
    ap.add_argument(
        "--app-exe",
        type=Path,
        default=None,
        help="Workload executable. Required for long_matmul; defaults to op_power_breakdown/ttnn_ops_workload.py for ttnn_ops.",
    )
    ap.add_argument("--parser-script", type=Path, default=HERE / "parser.py")
    ap.add_argument(
        "--tt-venv-activate",
        required=True,
        type=Path,
        help="Activation script of a Python environment with numpy/matplotlib (and ttnn for ttnn_ops).",
    )
    ap.add_argument("--tt-metal-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python used for auto.py, parser.py and the analysis scripts. Default: the one running this script.",
    )
    ap.add_argument("--tt-smi", default="tt-smi", help="tt-smi executable. Default: tt-smi")
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--force", action="store_true", help="Re-run runs whose output directory already exists.")
    ap.add_argument("--dry-run", action="store_true", help="Print every command without running anything.")
    ap.add_argument(
        "--analyses-only",
        action="store_true",
        help="Skip the runs; only (re)generate the analyses from existing output.",
    )
    args = ap.parse_args(argv)

    sweep = load_sweep(args.config)
    output_root = args.output_root.expanduser().resolve()
    log = Log(None if args.dry_run else output_root / "sweep.log")

    log(f"[SWEEP] {sweep.name}: workload={sweep.workload} app_args={shell_join(sweep.app_args)}")
    log(f"[SWEEP] {len(sweep.runs)} run(s): " + ", ".join(r.subdir for r in sweep.runs))
    log(f"[SWEEP] analyses: {[n for n, _ in sweep.analyses]}")
    log(f"[SWEEP] output root: {output_root}")

    if not args.dry_run:
        write_provenance(args, sweep, args.config, output_root)

    if not args.analyses_only:
        rc = run_sweep(args, sweep, output_root, log)
        if rc != 0:
            return rc

    for name, opts in sweep.analyses:
        log(f"[ANALYSIS] {name} {opts if opts else ''}")
        rc = ANALYSES[name](args, sweep, output_root, opts, log)
        if rc != 0:
            log(f"[ANALYSIS] {name} FAILED with {rc}")
            return rc

    log(f"[SWEEP] done: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
