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

Config keys (unknown keys are rejected):

    name:              free text, recorded in PROVENANCE.md
    workload:          long_matmul | ttnn_ops
    app_args:          args for the workload (M N K iters [fixed_blocks_per_core] for
                       long_matmul; the ttnn_ops_workload.py flags for ttnn_ops)
    mode:              updated (default) or legacy: exported as LONG_MATMUL_MODE to every run and
                       used by the analyses; legacy reproduces how the published results were made
    telemetry_freq_hz: sampling rate, default 50
    trim_ms, slot_ms:  passed to parser.py, defaults 1.0 and 1
    aiclk_mhz:         optional; pin AICLK to this after every reset (Blackhole, via aiclk/set_aiclk.py)
    runs:              long_matmul: {power_cases: [..], ops: [..], blocks: [[M, N], ..]} --
                       one run per combination, into <op>/<case>[_b<M>x<N>]
                       ttnn_ops: {subdir: <name>} -- one run
    analyses:          list of step names, or {name: {options}}; see ANALYSES below

A run is skipped when it is complete with the same settings (auto.py's run_complete.json), so an
interrupted sweep resumes; a run made with other settings stops the sweep, before the first
reset, unless --force. The analyses read only runs complete with this sweep's settings, except
compare_runs, which overlays every run under <op>/. --dry-run prints every command without
touching the board. Every command, the resets and the environment go to <output-root>/sweep.log;
the first run that executes records the config, the tt-metal commit and the board list in
<output-root>/PROVENANCE.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from analysis import power_cases as pc  # noqa: E402
from common import (  # noqa: E402
    DEFAULT_DEVICE_ID,
    DEFAULT_DPI,
    DEFAULT_SLOT_MS,
    DEFAULT_TELEMETRY_FREQ_HZ,
    POWER_CASES_TRIM_MS,
    RUN_COMPLETE_FILE,
    auto_py_command,
    check_long_matmul_device,
    environ_with,
    exit_reason,
    long_matmul_args,
    reset_hardware,
    run_settings,
    run_status,
    shell_join,
    subdir_name,
)

AICLK_SETTLE_S = 5
# long_matmul's kDestTileBudget: a block's M*N output tiles are held in DEST at once.
DEST_BLOCK_TILES = 8
TILE = 32

WORKLOADS = ("long_matmul", "ttnn_ops")
MODES = ("updated", "legacy")
CONFIG_KEYS = {
    "name",
    "workload",
    "mode",
    "app_args",
    "telemetry_freq_hz",
    "trim_ms",
    "slot_ms",
    "aiclk_mhz",
    "runs",
    "analyses",
}
RUNS_KEYS = {"long_matmul": {"power_cases", "ops", "blocks"}, "ttnn_ops": {"subdir"}}


@dataclass
class RunSpec:
    subdir: str
    app_args: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    block: Tuple[int, int] = (1, 1)


@dataclass
class Sweep:
    name: str
    workload: str
    mode: str
    app_args: List[str]
    telemetry_freq_hz: int
    trim_ms: float
    slot_ms: int
    aiclk_mhz: Optional[int]
    runs: List[RunSpec]
    ops: List[str]
    cases: List[Tuple[str, str]]
    analyses: List[Tuple[str, dict]]


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


def _reject_unknown(path: Path, where: str, got, allowed) -> None:
    unknown = sorted(set(got) - set(allowed))
    if unknown:
        raise SystemExit(f"{path}: unknown key(s) in {where}: {unknown}; expected some of {sorted(allowed)}")


def _positive(path: Path, key: str, value: float) -> float:
    if not value > 0:
        raise SystemExit(f"{path}: {key} must be positive, got {value}")
    return value


def load_sweep(path: Path) -> Sweep:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top level must be a mapping")
    _reject_unknown(path, "the config", raw, CONFIG_KEYS)

    workload = raw.get("workload", "long_matmul")
    if workload not in WORKLOADS:
        raise SystemExit(f"{path}: workload must be one of {WORKLOADS}, got {workload!r}")

    mode = str(raw.get("mode", "updated"))
    if mode not in MODES:
        raise SystemExit(f"{path}: mode must be one of {MODES}, got {mode!r}")
    # An empty value means updated to long_matmul, parser.py and the ttnn workload, and is left
    # out of run_complete.json, so updated runs match runs made with the variable unset.
    mode_env = "legacy" if mode == "legacy" else ""

    aiclk_mhz = int(_positive(path, "aiclk_mhz", raw["aiclk_mhz"])) if raw.get("aiclk_mhz") else None
    # Recorded in run_complete.json (common.WORKLOAD_ENV_VARS), so runs at another clock do not count
    # as done; empty when not pinned.
    base_env = {"LONG_MATMUL_MODE": mode_env, "TT_EMBER_AICLK_MHZ": str(aiclk_mhz or "")}

    app_args = [str(a) for a in raw.get("app_args", [])]
    runs_cfg = raw.get("runs") or {}
    _reject_unknown(path, "runs", runs_cfg, RUNS_KEYS[workload])
    runs: List[RunSpec] = []
    ops: List[str] = []
    cases: List[Tuple[str, str]] = []

    if workload == "long_matmul":
        try:
            m, n, _k, _iters, _fixed = long_matmul_args(app_args)
        except ValueError as e:
            raise SystemExit(f"{path}: app_args: {e}")
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
            if len(b) != 2 or b[0] < 1 or b[1] < 1 or b[0] * b[1] > DEST_BLOCK_TILES:
                raise SystemExit(f"{path}: block must be [M, N] with 1 <= M*N <= {DEST_BLOCK_TILES}, got {b}")
            if (m // TILE) % b[0] or (n // TILE) % b[1]:
                raise SystemExit(f"{path}: block {b} must divide the output tiles ({m // TILE} x {n // TILE})")
        cases = [pc.POWER_CASE_SPECS[c] for c in power_cases]
        for op in ops:
            for c in power_cases:
                case_name, _label = pc.POWER_CASE_SPECS[c]
                for bm, bn in blocks:
                    blocked = (bm, bn) != (1, 1)
                    env = dict(base_env, POWER_CASE=str(c), LONG_MATMUL_OP=op)
                    # Empty for 1x1, so a block size exported in the shell cannot leak in.
                    env["LONG_MATMUL_BLOCK_M"] = str(bm) if blocked else ""
                    env["LONG_MATMUL_BLOCK_N"] = str(bn) if blocked else ""
                    suffix = f"_b{bm}x{bn}" if blocked else ""
                    runs.append(RunSpec(f"{op}/{case_name}{suffix}", app_args, env, (bm, bn)))
    else:
        try:
            subdir = subdir_name(str(runs_cfg.get("subdir", "decoder_block")))
        except argparse.ArgumentTypeError as e:
            raise SystemExit(f"{path}: runs.subdir: {e}")
        runs.append(RunSpec(subdir, app_args, dict(base_env)))

    analyses: List[Tuple[str, dict]] = []
    for item in raw.get("analyses") or []:
        if isinstance(item, str):
            analyses.append((item, {}))
        elif isinstance(item, dict) and len(item) == 1:
            ((name, opts),) = item.items()
            analyses.append((str(name), dict(opts or {})))
        else:
            raise SystemExit(f"{path}: analyses entries must be a name or {{name: {{options}}}}, got {item!r}")
    for name, opts in analyses:
        if name not in ANALYSES:
            raise SystemExit(f"{path}: unknown analysis {name!r}, expected one of {sorted(ANALYSES)}")
        _reject_unknown(path, f"the {name} options", opts, ANALYSIS_OPTIONS[name])

    return Sweep(
        name=str(raw.get("name", path.stem)),
        workload=workload,
        mode=mode,
        app_args=app_args,
        telemetry_freq_hz=int(
            _positive(path, "telemetry_freq_hz", raw.get("telemetry_freq_hz", DEFAULT_TELEMETRY_FREQ_HZ))
        ),
        trim_ms=float(raw.get("trim_ms", POWER_CASES_TRIM_MS)),
        slot_ms=int(_positive(path, "slot_ms", raw.get("slot_ms", DEFAULT_SLOT_MS))),
        aiclk_mhz=aiclk_mhz,
        runs=runs,
        ops=ops,
        cases=cases,
        analyses=analyses,
    )


@dataclass
class Context:
    """What every analysis step gets: the command line, the sweep, and its runs."""

    args: argparse.Namespace
    sweep: Sweep
    output_root: Path
    log: Log
    runs: pc.Runs
    _energy: Dict[str, Optional[Dict[str, Dict[str, float]]]] = field(default_factory=dict)

    def energy_by_engine(self, op: str) -> Optional[Dict[str, Dict[str, float]]]:
        """pc.compute_energy_by_engine, once per op for the whole sweep."""
        if op not in self._energy:
            self._energy[op] = pc.compute_energy_by_engine(self.runs, op)
        return self._energy[op]

    def flops(self) -> Optional[float]:
        """FLOPs per interval: the same on every grid in split mode. None in fixed mode, where
        total work grows with the core count, or for a workload that is not long_matmul."""
        if self.sweep.workload != "long_matmul":
            return None
        m, n, k, iterations, fixed_blocks_per_core = long_matmul_args(self.sweep.app_args)
        return None if fixed_blocks_per_core else 2.0 * m * n * k * iterations

    def grid_order(self) -> List[str]:
        return self.runs.grid_order([(op, case) for op in self.sweep.ops for case, _ in self.sweep.cases])

    def unusable(self, op: str, cases: List[str]) -> List[str]:
        """Of the cases a standalone script reads under <op>/, those with a program_intervals.csv
        that is not a complete run of this sweep. The scripts cannot tell, so they are not run."""
        return [
            f"{op}/{case}"
            for case in cases
            if (self.output_root / op / case / "program_intervals.csv").exists() and not self.runs.complete(op, case)
        ]


def run_settings_for(args: argparse.Namespace, sweep: Sweep, spec: RunSpec) -> Dict[str, object]:
    """The settings auto.py records for this run (it sees os.environ plus spec.env)."""
    return run_settings(
        app_exe_for(args, sweep),
        spec.app_args,
        environ_with(spec.env),
        sweep.telemetry_freq_hz,
        sweep.trim_ms,
        sweep.slot_ms,
        args.device_id,
    )


# --- board control -------------------------------------------------------------------------------


def run_cmd(cmd: List[str], log: Log, dry_run: bool, env: Optional[Dict[str, str]] = None) -> int:
    extra = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    log(f"[CMD] {extra + ' ' if extra else ''}{shell_join(cmd)}")
    if dry_run:
        return 0
    return subprocess.run(cmd, env=environ_with(env or {})).returncode


def pin_aiclk(mhz: int, device_id: int, activate: Path, log: Log, dry_run: bool) -> int:
    # A reset clears the ARC FORCE_AICLK override, so this runs after every reset. Same check as
    # set_aiclk.py's own readback: the value the chip reports has to be the one requested. Whether
    # the force survives tt-metal opening the device is not checked here; see aiclk/set_aiclk.py.
    setter = HERE / "aiclk" / "set_aiclk.py"
    # Fixed script; the values arrive as positional arguments, so none of them is parsed as shell.
    script = 'sleep "$1" && source "$2" && exec python "$3" "$4" --busy --device-id "$5"'
    cmd = ["bash", "-c", script, "bash", str(AICLK_SETTLE_S), str(activate), str(setter), str(mhz), str(device_id)]
    log(f"[AICLK] pin to {mhz} MHz: {shell_join(cmd)}")
    if dry_run:
        return 0
    res = subprocess.run(cmd, capture_output=True, text=True)
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


def app_exe_for(args: argparse.Namespace, sweep: Sweep) -> Path:
    if sweep.workload == "long_matmul":
        return args.app_exe
    return args.app_exe or (HERE / "op_power_breakdown" / "ttnn_ops_workload.py")


def auto_command(args: argparse.Namespace, sweep: Sweep, spec: RunSpec, output_root: Path) -> List[str]:
    # <output-root>/<op>/<case> is produced by giving auto.py <output-root>/<op> as its root and
    # <case> as the subdir, which keeps its own launcher.log/telemetry.txt layout intact.
    sub = Path(spec.subdir)
    return auto_py_command(
        auto_script=HERE / "auto.py",
        telemetry_exe=args.telemetry_exe,
        telemetry_freq=sweep.telemetry_freq_hz,
        app_exe=app_exe_for(args, sweep),
        app_args=spec.app_args,
        parser_script=args.parser_script,
        tt_venv_activate=args.tt_venv_activate,
        tt_metal_root=args.tt_metal_root,
        output_root=output_root / sub.parent,
        subdir=sub.name,
        slot_ms=sweep.slot_ms,
        device_id=args.device_id,
        trim_ms=sweep.trim_ms,
        python_exe=args.python_exe,
    )


def run_sweep(args: argparse.Namespace, sweep: Sweep, output_root: Path, log: Log) -> int:
    statuses = [run_status(output_root / spec.subdir, run_settings_for(args, sweep, spec)) for spec in sweep.runs]
    conflicts = [spec.subdir for spec, status in zip(sweep.runs, statuses) if status == "different"]
    if conflicts and not args.force:
        log(
            f"ERROR: these runs were made with different settings (see their {RUN_COMPLETE_FILE}): "
            f"{', '.join(conflicts)}. Use another --output-root, or --force to replace them."
        )
        return 1

    provenance_written = False
    for i, (spec, status) in enumerate(zip(sweep.runs, statuses), start=1):
        head = f"[RUN {i}/{len(sweep.runs)}] {spec.subdir}"
        if status == "done" and not args.force:
            log(f"{head}: complete, skipping (use --force to re-run)")
            continue

        log(head)
        # PROVENANCE.md describes the measurement, so only a sweep that measures writes it.
        if not args.dry_run and not provenance_written:
            write_provenance(args, sweep, output_root)
            provenance_written = True
        log(f"[RESET] {args.tt_smi} -r")
        rc = reset_hardware(args.tt_smi, args.dry_run)
        if rc != 0:
            log(f"[RESET] ERROR: {args.tt_smi} -r failed with return code {rc}")
            return rc
        if sweep.aiclk_mhz is not None:
            rc = pin_aiclk(sweep.aiclk_mhz, args.device_id, args.tt_venv_activate, log, args.dry_run)
            if rc != 0:
                return rc

        rc = run_cmd(auto_command(args, sweep, spec, output_root), log, args.dry_run, env=spec.env)
        if rc != 0:
            log(f"{head}: FAILED with return code {exit_reason(rc)}, aborting")
            return rc
    return 0


# --- analyses ------------------------------------------------------------------------------------


def _out_dir(ctx: Context, *parts: str) -> Path:
    out = ctx.output_root.joinpath(*parts, "compare_runs_out")
    if not ctx.args.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    return out


def _compare_cases(ctx: Context, input_dir: Path, cases_file: Path) -> int:
    cmd = [ctx.args.python_exe, str(HERE / "compare_cases.py"), "-i", str(input_dir), "-c", str(cases_file)]
    return run_cmd(cmd + ["--dpi", str(ctx.args.dpi)], ctx.log, ctx.args.dry_run)


def an_compare_cases(ctx: Context, opts: dict) -> int:
    for op in ctx.sweep.ops:
        cases = [(s, label) for s, label in ctx.sweep.cases if ctx.args.dry_run or ctx.runs.complete(op, s)]
        if not cases:
            ctx.log(f"[ANALYSIS] compare_cases: no complete run for {op}, skipping")
            continue
        cases_file = ctx.output_root / op / "power_cases.txt"
        if not ctx.args.dry_run:
            pc.write_cases_file(cases_file, cases)
        rc = _compare_cases(ctx, ctx.output_root / op, cases_file)
        if rc != 0:
            return rc
    return 0


def an_energy_by_engine(ctx: Context, opts: dict) -> int:
    if ctx.args.dry_run:
        ctx.log("[ANALYSIS] energy_by_engine (dry-run) skipped")
        return 0
    grids = ctx.grid_order()
    for op in ctx.sweep.ops:
        per_engine = ctx.energy_by_engine(op)
        if per_engine is not None:
            pc.plot_energy_by_engine_stacked(
                per_engine, grids, _out_dir(ctx, op) / "energy_by_engine_stacked.png", ctx.args.dpi
            )
    return 0


def an_energy_per_flop_by_engine(ctx: Context, opts: dict) -> int:
    flops = ctx.flops()
    if flops is None:
        ctx.log("[ANALYSIS] energy_per_flop_by_engine needs long_matmul in split mode, skipping")
        return 0
    if ctx.args.dry_run:
        ctx.log("[ANALYSIS] energy_per_flop_by_engine (dry-run) skipped")
        return 0
    grids = ctx.grid_order()
    for op in ctx.sweep.ops:
        out = _out_dir(ctx, op)
        per_engine = ctx.energy_by_engine(op)
        if per_engine is not None:
            pc.plot_energy_per_flop_by_engine(
                pc.to_pj_per_flop(per_engine, flops), grids, out / "energy_per_flop_by_engine.png", ctx.args.dpi
            )
        per_case = pc.compute_energy_per_flop_per_core(ctx.runs, op, ctx.sweep.cases, flops)
        if per_case:
            pc.plot_energy_per_flop_per_core(
                per_case, ctx.sweep.cases, grids, out / "energy_per_flop_per_core.png", ctx.args.dpi
            )
    return 0


def an_cross_op(ctx: Context, opts: dict) -> int:
    if len(ctx.sweep.ops) < 2:
        ctx.log("[ANALYSIS] cross_op needs at least two ops, skipping")
        return 0
    ops = ctx.sweep.ops
    cases_file = ctx.output_root / "power_cases_cross_op.txt"
    if ctx.args.dry_run:
        return _compare_cases(ctx, ctx.output_root, cases_file)
    entries = pc.write_cross_op_cases_file(cases_file, ops, ctx.sweep.cases, ctx.runs)
    if len(entries) >= 2:
        rc = _compare_cases(ctx, ctx.output_root, cases_file)
        if rc != 0:
            return rc

    grids = ctx.grid_order()
    out = _out_dir(ctx)
    per_op_engine = {op: e for op in ops if (e := ctx.energy_by_engine(op)) is not None}
    if not per_op_engine:
        ctx.log("[ANALYSIS] cross_op: no op has energy-by-engine data, skipping its charts")
    else:
        pc.plot_cross_op_stacked_by_engine(
            per_op_engine,
            ops,
            grids,
            out / "energy_by_engine_stacked_cross_op.png",
            ctx.args.dpi,
            ylabel="Dynamic energy [J]",
            title="Total dynamic energy by engine",
            value_unit="J",
        )
        pc.plot_cross_op_stacked_by_engine(
            per_op_engine,
            ops,
            grids,
            out / "energy_share_by_engine_stacked_cross_op.png",
            ctx.args.dpi,
            ylabel="Share of dynamic energy [%]",
            title="Reader/Writer/Compute share of dynamic energy",
            value_unit="J",
            normalize_pct=True,
        )

    flops = ctx.flops()
    if flops is not None:
        if per_op_engine:
            pc.plot_cross_op_stacked_by_engine(
                {op: pc.to_pj_per_flop(e, flops) for op, e in per_op_engine.items()},
                ops,
                grids,
                out / "energy_per_flop_by_engine_stacked_cross_op.png",
                ctx.args.dpi,
                ylabel="Energy per FLOP [pJ]",
                title="Energy per FLOP by engine (matmul's 2*M*N*K FLOPs for every op)",
                value_unit="pJ/FLOP",
            )
        per_op = pc.compute_total_energy_per_flop_by_op(ctx.runs, ops, pc.ALL_ACTIVE_CASE, flops)
        if per_op:
            pc.plot_total_energy_per_flop_by_op(
                per_op, ops, grids, out / "total_energy_per_flop_by_op.png", ctx.args.dpi
            )
    return 0


def _mnk_iters(ctx: Context, step: str) -> Optional[List[str]]:
    """M N K iters for the standalone analysis scripts, or None (with a note) in fixed mode."""
    if ctx.flops() is None:
        ctx.log(f"[ANALYSIS] {step} needs long_matmul in split mode, skipping")
        return None
    return ctx.sweep.app_args[:4]


def _script(ctx: Context, name: str, argv: List[str]) -> int:
    cmd = [ctx.args.python_exe, str(HERE / "analysis" / name), *argv]
    return run_cmd(cmd + ["--mode", ctx.sweep.mode, "--dpi", str(ctx.args.dpi)], ctx.log, ctx.args.dry_run)


def _skip_unusable(ctx: Context, step: str, op: str, cases: List[str]) -> bool:
    bad = [] if ctx.args.dry_run else ctx.unusable(op, cases)
    if bad:
        ctx.log(f"[ANALYSIS] {step}: skipping {op}, {', '.join(bad)} is not a complete run of this sweep")
    return bool(bad)


def an_cross_op_corrected(ctx: Context, opts: dict) -> int:
    mnk = _mnk_iters(ctx, "cross_op_corrected")
    if mnk is None:
        return 0
    cases = [pc.ALL_ACTIVE_CASE] + [idle for _engine, idle, _label in pc.ENGINE_ABLATIONS]
    ops = [
        op
        for op in ctx.sweep.ops
        if not _skip_unusable(ctx, "cross_op_corrected", op, cases)
        and (ctx.args.dry_run or ctx.runs.complete(op, pc.ALL_ACTIVE_CASE))
    ]
    if not ops:
        ctx.log(f"[ANALYSIS] cross_op_corrected: no op has a complete {pc.ALL_ACTIVE_CASE} run, skipping")
        return 0
    out = _out_dir(ctx) / "energy_per_flop_by_engine_stacked_cross_op_corrected.png"
    argv = [str(ctx.output_root), "--ops", *ops, "--app-args", *mnk, "--out", str(out)]
    if "board" in opts:
        argv += ["--board", str(opts["board"])]
    return _script(ctx, "crossop_corrected_flops.py", argv)


# The cases each per-op script reads under <op>/, and whether it needs all of them (else any).
PJ_PER_FLOP_CASES = ["regular", "writer_amp"]
PJ_PER_FLOP_BY_ENGINE_CASES = ["regular", "writer_amp", "compute_idle", "reader_idle2"]


def _per_op_script(
    ctx: Context, opts: dict, step: str, script: str, cases: List[str], needs_all: bool, png: str
) -> int:
    mnk = _mnk_iters(ctx, step)
    if mnk is None:
        return 0
    m, n, k, iters = mnk
    for op in ctx.sweep.ops:
        if _skip_unusable(ctx, step, op, cases):
            continue
        complete = [case for case in cases if ctx.args.dry_run or ctx.runs.complete(op, case)]
        if (len(complete) < len(cases)) if needs_all else not complete:
            missing = [f"{op}/{case}" for case in cases if case not in complete]
            ctx.log(f"[ANALYSIS] {step}: skipping {op}, no complete run for {', '.join(missing)}")
            continue
        argv = [str(ctx.output_root / op), "--label", str(opts.get("label", f"{ctx.sweep.name} ({op})"))]
        argv += ["--seq", m, "--hidden", n, "--k", k, "--iters", iters, "--out", str(_out_dir(ctx, op) / png)]
        rc = _script(ctx, script, argv)
        if rc != 0:
            return rc
    return 0


def an_pj_per_flop(ctx: Context, opts: dict) -> int:
    return _per_op_script(ctx, opts, "pj_per_flop", "make_pj_per_flop.py", PJ_PER_FLOP_CASES, False, "pj_per_flop.png")


def an_pj_per_flop_by_engine(ctx: Context, opts: dict) -> int:
    # The published Fig. 1: peak-charge ablation against writer_amp, with regular (POWER_CASE=0)
    # as the writer stand-in. Needs power cases 0, 1, 2 and 4.
    return _per_op_script(
        ctx,
        opts,
        "pj_per_flop_by_engine",
        "make_pj_per_flop_by_engine.py",
        PJ_PER_FLOP_BY_ENGINE_CASES,
        True,
        "pj_per_flop_by_engine.png",
    )


def an_naive_vs_blocked(ctx: Context, opts: dict) -> int:
    mnk = _mnk_iters(ctx, "naive_vs_blocked")
    if mnk is None:
        return 0
    sweep = ctx.sweep
    op = sweep.ops[0] if sweep.ops else "matmul"
    case = str(opts.get("case", "regular"))
    blocked_run = next((r for r in sweep.runs if r.block != (1, 1)), None)
    block = blocked_run.block if blocked_run else (2, 4)
    naive = str(opts.get("naive", f"{op}/{case}"))
    blocked = str(opts.get("blocked", f"{op}/{case}_b{block[0]}x{block[1]}"))
    if not ctx.args.dry_run:
        for rel in (naive, blocked):
            if not ctx.runs.complete_run(rel):
                ctx.log(f"[ANALYSIS] naive_vs_blocked: {rel} is not a complete run of this sweep, skipping")
                return 0
    argv = [
        str(ctx.output_root / naive / "program_intervals.csv"),
        str(ctx.output_root / blocked / "program_intervals.csv"),
        "--app-args",
        *mnk,
        "--block",
        str(block[0]),
        str(block[1]),
        "--case",
        case,
        "--out",
        str(_out_dir(ctx) / "pj_per_flop_naive_vs_blocked.png"),
    ]
    if "board" in opts:
        argv += ["--board", str(opts["board"])]
    return _script(ctx, "make_pj_per_flop_naive_vs_blocked.py", argv)


def an_compare_runs(ctx: Context, opts: dict) -> int:
    # compare_runs.py overlays every run directly under its --out-root as line plots (what the
    # former run_all.py used for its prefill / fixed-tpc comparisons), without checking their
    # settings. Runs here sit under <op>/, so it runs once per op, into <op>/comparison.
    for root in [ctx.output_root / op for op in ctx.sweep.ops] or [ctx.output_root]:
        cmd = [ctx.args.python_exe, str(HERE / "compare_runs.py"), "--out-root", str(root)]
        cmd += ["--output-dir", str(root / "comparison"), "--dpi", str(ctx.args.dpi)]
        if "filter" in opts:
            filters = opts["filter"] if isinstance(opts["filter"], list) else [opts["filter"]]
            cmd += ["--filter", *[str(f) for f in filters]]
        # The overcompute charts assume total work grows with the core count (fixed mode).
        if opts.get("no_overcompute", ctx.flops() is not None):
            cmd += ["--no-overcompute"]
        rc = run_cmd(cmd, ctx.log, ctx.args.dry_run)
        if rc != 0:
            return rc
    return 0


def an_op_breakdown(ctx: Context, opts: dict) -> int:
    for spec in ctx.sweep.runs:
        if not ctx.args.dry_run and not ctx.runs.complete_run(spec.subdir):
            ctx.log(f"[ANALYSIS] op_breakdown: {spec.subdir} is not a complete run of this sweep, skipping")
            continue
        cmd = [ctx.args.python_exe, str(HERE / "op_power_breakdown" / "preview_op_breakdown.py")]
        cmd += [str(ctx.output_root / spec.subdir), "--mode", ctx.sweep.mode]
        if "board" in opts:
            cmd += ["--board", str(opts["board"])]
        rc = run_cmd(cmd, ctx.log, ctx.args.dry_run)
        if rc != 0:
            return rc
    return 0


ANALYSES = {
    "compare_cases": an_compare_cases,
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
# The options each analysis step accepts.
ANALYSIS_OPTIONS = {
    "compare_cases": set(),
    "energy_by_engine": set(),
    "energy_per_flop_by_engine": set(),
    "cross_op": set(),
    "cross_op_corrected": {"board"},
    "pj_per_flop": {"label"},
    "pj_per_flop_by_engine": {"label"},
    "naive_vs_blocked": {"case", "naive", "blocked", "board"},
    "compare_runs": {"filter", "no_overcompute"},
    "op_breakdown": {"board"},
}


# --- provenance ----------------------------------------------------------------------------------


def _git(root: Path, *argv: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *argv], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_provenance(args: argparse.Namespace, sweep: Sweep, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    config_copy = output_root / "sweep.yaml"
    # Re-running from the recorded config passes that very file as --config.
    if args.config.resolve() != config_copy.resolve():
        shutil.copy2(args.config, config_copy)
    boards = "unknown"
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
        f"- config: sweep.yaml (copied from {args.config})",
        f"- tt-metal: {_git(args.tt_metal_root, 'rev-parse', 'HEAD')} ({_git(args.tt_metal_root, 'describe', '--always', '--dirty')})",
        f"- tools/tt-ember: {_git(HERE, 'rev-parse', 'HEAD')} ({_git(HERE, 'describe', '--always', '--dirty')})",
        f"- workload: {sweep.workload} {shell_join(sweep.app_args)}, mode {sweep.mode}",
        f"- telemetry: {sweep.telemetry_freq_hz} Hz, trim {sweep.trim_ms} ms, slot {sweep.slot_ms} ms"
        + (f", AICLK {sweep.aiclk_mhz} MHz requested after every reset" if sweep.aiclk_mhz else ""),
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
    ap.add_argument("--device-id", type=int, default=DEFAULT_DEVICE_ID)
    ap.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-run runs that are already complete, or were made with different settings.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print every command without running anything.")
    ap.add_argument(
        "--analyses-only",
        action="store_true",
        help="Skip the runs; only (re)generate the analyses from existing output.",
    )
    args = ap.parse_args(argv)

    sweep = load_sweep(args.config)
    # Also for --analyses-only: run_complete.json records the app, so the analyses need it to
    # tell this sweep's runs from others.
    if sweep.workload == "long_matmul" and args.app_exe is None:
        ap.error("--app-exe is required for the long_matmul workload")
    if not args.analyses_only:
        try:
            check_long_matmul_device(app_exe_for(args, sweep), args.device_id)
        except ValueError as e:
            ap.error(str(e))
        if sweep.workload == "ttnn_ops" and args.device_id != 0:
            ap.error("ttnn_ops_workload.py always runs on device 0")

    output_root = args.output_root.expanduser().resolve()
    log = Log(None if args.dry_run else output_root / "sweep.log")

    log(f"[SWEEP] {sweep.name}: workload={sweep.workload} mode={sweep.mode} app_args={shell_join(sweep.app_args)}")
    log(f"[SWEEP] {len(sweep.runs)} run(s): " + ", ".join(r.subdir for r in sweep.runs))
    log(f"[SWEEP] analyses: {[n for n, _ in sweep.analyses]}")
    log(f"[SWEEP] output root: {output_root}")

    if not args.analyses_only:
        rc = run_sweep(args, sweep, output_root, log)
        if rc != 0:
            return rc

    runs = pc.Runs(
        output_root,
        {spec.subdir: run_settings_for(args, sweep, spec) for spec in sweep.runs},
        legacy=sweep.mode == "legacy",
    )
    ctx = Context(args, sweep, output_root, log, runs)
    for name, opts in sweep.analyses:
        log(f"[ANALYSIS] {name} {opts if opts else ''}")
        rc = ANALYSES[name](ctx, opts)
        if rc != 0:
            log(f"[ANALYSIS] {name} FAILED with {rc}")
            return rc

    log(f"[SWEEP] done: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
