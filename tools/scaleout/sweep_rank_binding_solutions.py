#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Run a command across every solution produced by ``generate_rank_bindings --all-solutions``.

This is a thin orchestrator on top of ``tt-run``: it accepts the **same arguments as
``tt-run``** (so a working ``tt-run`` invocation ports over directly) plus a few sweep-specific
extras, then:

  1. (new mode) runs ``generate_rank_bindings --all-solutions`` to enumerate every valid
     placement into ``<solutions-output-dir>/`` -- OR consumes an existing solutions directory
     via ``--solutions-dir``;
  2. for **each** solution, launches the trailing ``<program>`` via ``tt-run`` legacy mode bound
     to that solution's ``rank_bindings.yaml`` (+ ``rankfile`` / ``phase2_mock_mapping.yaml``);
  3. if a launch fails or times out, runs ``--recover-command``, sleeps, and retries the same
     ``tt-run`` (default 3 attempts). Recover is **not** run after a successful launch;
  4. records per-solution pass/fail into ``sweep_report.yaml`` (pass if any attempt succeeds)
     and returns non-zero only if a solution still failed after all retries.

See tools/scaleout/README_sweep_rank_binding_solutions.md for the design.
"""

import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import click
import yaml

# Reuse tt-run internals rather than reimplementing MPI/mock launch.
from ttnn.distributed.ttrun import (
    build_generate_rank_bindings_mpi_cmd,
    find_generate_rank_bindings_executable,
    get_generate_rank_bindings_output_paths,
    load_mock_rank_to_descriptors,
    run_generate_rank_bindings,
)

PREFIX = "[tt-sweep]"
# Sleep between every failed command and the next recover / tt-run retry.
RETRY_DELAY_S = 5
DEFAULT_RETRIES = 3


def _repo_root() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", ".")).resolve()


def _find_tt_run() -> str:
    """Locate the tt-run entrypoint (installed console script, or run the module directly)."""
    exe = shutil.which("tt-run")
    if exe:
        return exe
    # Fall back to `python -m ttnn.distributed.ttrun` semantics is not available (no __main__),
    # so use the venv-adjacent script if present.
    candidate = Path(sys.executable).parent / "tt-run"
    if candidate.exists():
        return str(candidate)
    raise click.ClickException("Could not find the `tt-run` executable on PATH or next to the Python interpreter.")


def _inject_solution_flags(cmd: List[str], extra: List[str]) -> List[str]:
    """Insert generate_rank_bindings sweep flags after every ``--output-dir <value>``.

    ``build_generate_rank_bindings_mpi_cmd`` emits ``--mesh-graph-descriptor``/``--output-dir`` once
    (real cluster) or once per MPMD rank segment (mock). Inserting after each ``--output-dir`` value
    puts the flags in the right place for both layouts.
    """
    out: List[str] = []
    i = 0
    while i < len(cmd):
        out.append(cmd[i])
        if cmd[i] == "--output-dir" and i + 1 < len(cmd):
            out.append(cmd[i + 1])
            out.extend(extra)
            i += 2
            continue
        i += 1
    return out


def _generate_solutions(
    *,
    mesh_graph_descriptor: Path,
    hosts: Optional[List[str]],
    mock_cluster_rank_binding: Optional[Path],
    output_dir: Path,
    max_solutions: int,
    distinct_host_sets: bool,
    allow_shape_permutations: bool,
    mpi_args: Optional[List[str]],
    dry_run: bool,
) -> Path:
    """Phase 1: run generate_rank_bindings --all-solutions. Returns the solutions dir."""
    executable = find_generate_rank_bindings_executable()
    repo_root = _repo_root()
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_rank_to_desc: Optional[Dict[int, Path]] = None
    if mock_cluster_rank_binding is not None:
        mock_rank_to_desc = load_mock_rank_to_descriptors(mock_cluster_rank_binding.resolve())

    cmd = build_generate_rank_bindings_mpi_cmd(
        executable=executable,
        mgd_path=mesh_graph_descriptor,
        hosts=hosts,
        output_dir=output_dir,
        mock_rank_to_desc=mock_rank_to_desc,
        mpi_args=mpi_args,
    )

    extra = ["--all-solutions"]
    if max_solutions:
        extra += ["--max-solutions", str(max_solutions)]
    if distinct_host_sets:
        extra += ["--distinct-host-sets"]
    if allow_shape_permutations:
        # hidden: turn OFF generate_rank_bindings' always-on solver unique_shapes dedup
        extra += ["--allow-shape-permutations"]
    cmd = _inject_solution_flags(cmd, extra)

    click.echo(f"{PREFIX} Phase 1 (generate solutions):\n  {' '.join(shlex.quote(c) for c in cmd)}")
    if dry_run:
        click.echo(f"{PREFIX} --dry-run: skipping Phase 1 execution.")
        return output_dir

    rc = run_generate_rank_bindings(cmd, cwd=repo_root)
    if rc != 0:
        raise click.ClickException(f"generate_rank_bindings failed (exit {rc}).")
    return output_dir


def _load_index(solutions_dir: Path) -> dict:
    index_path = solutions_dir / "solutions_index.yaml"
    if not index_path.is_file():
        raise click.ClickException(f"No solutions_index.yaml under {solutions_dir}. Run with --all-solutions first.")
    with open(index_path) as f:
        return yaml.safe_load(f)


def _select_solutions(index: dict, select: Optional[str], limit: Optional[int]) -> List[dict]:
    solutions = list(index.get("solutions", []))
    if select:
        wanted = {s.strip() for s in select.split(",") if s.strip()}
        solutions = [s for s in solutions if s.get("id") in wanted]
    if limit is not None:
        solutions = solutions[:limit]
    return solutions


@dataclass
class _RetryResult:
    """Outcome of ``_run_with_retries`` (one solution's tt-run, including recover)."""

    status: str  # pass | fail | timeout
    returncode: Optional[int]
    attempts: int
    recover_returncode: Optional[int] = None
    recover_exhausted: bool = False


def _run_once(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: Optional[int],
    append: bool,
    header: Optional[str] = None,
) -> Tuple[str, Optional[int]]:
    """Run ``cmd`` once. Returns ``(status, returncode)`` where status is pass/fail/timeout."""
    mode = "a" if append else "w"
    try:
        with open(log_path, mode) as log:
            if header:
                log.write(header)
                log.flush()
            proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        rc = proc.returncode
        return ("pass" if rc == 0 else "fail", rc)
    except subprocess.TimeoutExpired:
        return ("timeout", None)


def _run_with_retries(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: Optional[int],
    retries: int,
    recover_cmd: Optional[str] = None,
    label: str = "tt-run",
    append_log: bool = False,
    cmd_display: Optional[str] = None,
) -> _RetryResult:
    """Run ``cmd`` up to ``retries`` times, recovering only after a failure.

    * Pass on any attempt is recorded as pass; recover is **not** run after success.
    * Fail/timeout: sleep ``RETRY_DELAY_S``, run ``recover_cmd`` (itself retried),
      sleep again, then retry ``cmd``. If ``recover_cmd`` is omitted, just sleep
      and retry ``cmd`` (used for recover itself).
    * Fail/timeout is returned only if every attempt fails. Recover exhausting
      its retries sets ``recover_exhausted`` and stops immediately.
    """
    attempts = max(1, retries)
    last_status = "fail"
    last_rc: Optional[int] = None
    recover_rc: Optional[int] = None

    for attempt in range(1, attempts + 1):
        append = append_log or attempt > 1
        header = None
        display = cmd_display or " ".join(shlex.quote(c) for c in cmd)
        if label == "tt-run" and attempt > 1:
            header = f"\n{PREFIX} --- tt-run retry ---\n"
        elif label == "recover":
            header = ""
            if attempt == 1:
                header = f"\n{PREFIX} --- recover ---\n{display}\n"
            header += f"{PREFIX} recover attempt {attempt}/{attempts}\n"

        click.echo(f"{PREFIX}   {label} [{attempt}/{attempts}]")
        last_status, last_rc = _run_once(cmd, cwd=cwd, log_path=log_path, timeout=timeout, append=append, header=header)
        if last_status == "pass":
            if label == "recover":
                click.echo(f"{PREFIX}   WARNING: recover succeeded (rc=0) on attempt {attempt}/{attempts}; continuing")
            else:
                click.echo(f"{PREFIX}   -> {last_status} (rc={last_rc}) attempt {attempt}/{attempts}")
            return _RetryResult(status=last_status, returncode=last_rc, attempts=attempt, recover_returncode=recover_rc)

        if label == "recover":
            click.echo(f"{PREFIX}   recover -> fail (rc={last_rc}) on attempt {attempt}/{attempts}")
        else:
            click.echo(f"{PREFIX}   -> {last_status} (rc={last_rc}) attempt {attempt}/{attempts}")

        if recover_cmd:
            click.echo(f"{PREFIX}   recover: {recover_cmd}")
            click.echo(f"{PREFIX}   recovering in {RETRY_DELAY_S}s...")
            time.sleep(RETRY_DELAY_S)
            rec = _run_with_retries(
                ["/bin/bash", "-c", recover_cmd],
                cwd=cwd,
                log_path=log_path,
                timeout=None,
                retries=retries,
                recover_cmd=None,
                label="recover",
                append_log=True,
                cmd_display=recover_cmd,
            )
            recover_rc = 0 if rec.status == "pass" else (rec.returncode if rec.returncode is not None else 1)
            if recover_rc != 0:
                return _RetryResult(
                    status=last_status,
                    returncode=last_rc,
                    attempts=attempt,
                    recover_returncode=recover_rc,
                    recover_exhausted=True,
                )

        if attempt < attempts:
            click.echo(f"{PREFIX}   retrying {label} in {RETRY_DELAY_S}s...")
            time.sleep(RETRY_DELAY_S)

    return _RetryResult(
        status=last_status,
        returncode=last_rc,
        attempts=attempts,
        recover_returncode=recover_rc,
    )


def _build_tt_run_cmd(
    *,
    tt_run: str,
    solutions_dir: Path,
    sol: dict,
    program: List[str],
    mock: bool,
    mpi_args: Optional[List[str]],
    passthrough: List[str],
) -> List[str]:
    """Build the per-solution ``tt-run`` legacy invocation for one solution."""
    sol_dir = solutions_dir / sol["dir"]
    rank_bindings, rankfile = get_generate_rank_bindings_output_paths(sol_dir)

    cmd = [tt_run, "--rank-binding", str(rank_bindings.resolve())]

    effective_mpi_args = list(mpi_args or [])
    if mock:
        phase2_mock = sol_dir / "phase2_mock_mapping.yaml"
        if phase2_mock.is_file():
            cmd += ["--mock-cluster-rank-binding", str(phase2_mock.resolve())]
    else:
        # Real cluster: place ranks via the solution's rankfile.
        effective_mpi_args += ["--map-by", f"rankfile:file={rankfile.resolve()}"]

    if effective_mpi_args:
        cmd += ["--mpi-args", " ".join(effective_mpi_args)]

    cmd += passthrough
    cmd += program
    return cmd


def _make_result(
    *,
    sol: dict,
    solutions_dir: Path,
    label: str,
    cmd_str: str,
    status: str,
    returncode: Optional[int],
    duration_seconds: float,
    tt_run_attempts: int,
    recover_command: Optional[str],
    recover_returncode: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> dict:
    """Build one per-solution row for ``sweep_report.yaml``."""
    return {
        "solution_id": label,  # content hash = the solution subdirectory name
        "status": status,  # pass | fail | timeout | dry-run  (pass if any attempt succeeded)
        "returncode": returncode,  # last / successful attempt (null if timed out)
        "duration_seconds": duration_seconds,  # wall-clock including recover + retries
        "num_hosts": sol.get("num_hosts"),  # distinct physical hosts this solution occupies
        "host_set": sol.get("host_set"),  # the hosts (per-host cluster descriptor / hostname)
        "tt_run_command": cmd_str,  # exact tt-run command (copy-paste to re-run)
        "tt_run_attempts": tt_run_attempts,  # how many tt-run attempts were made (1 if first pass)
        "rank_binding_path": str((solutions_dir / sol["dir"] / "rank_bindings.yaml").resolve()),
        "log_path": str(log_path) if log_path else None,  # full stdout+stderr of all attempts
        "recover_command": recover_command,  # set only when --recover-command ran (or dry-run)
        "recover_returncode": recover_returncode,
    }


def _write_sweep_report(
    *,
    results: List[dict],
    solutions: List[dict],
    index: dict,
    sol_dir: Path,
    program: List[str],
    recover_command: str,
    stopped_early: bool,
    sweep_report: Optional[Path],
    dry_run: bool,
) -> Tuple[Path, int, int, int]:
    """Build, optionally write, and print ``sweep_report.yaml``.

    Returns ``(report_path, passed, failed, timed_out)``.
    """
    passed = sum(r["status"] == "pass" for r in results)
    failed = sum(r["status"] == "fail" for r in results)
    timed_out = sum(r["status"] == "timeout" for r in results)
    report = {
        "mesh_graph_desc_path": index.get("mesh_graph_desc_path"),
        "solutions_dir": str(sol_dir),
        # The workload run once per solution, as a single-line command string.
        "workload_command": " ".join(shlex.quote(p) for p in program),
        "recover_command": recover_command,  # recovery command run after fail/timeout (then tt-run is retried)
        # Enumeration metadata copied from solutions_index.yaml:
        #   mode           = all | distinct-host-sets
        #   max_solutions  = requested cap (0 = all up to the solver safety cap)
        #   found          = number of distinct solutions generated
        #   truncated      = true if the cap bounded the result (more solutions may exist)
        "enumeration": index.get("enumeration"),
        # Tally across the solutions actually swept this run.
        "summary": {
            "total": len(results),  # solutions attempted (swept)
            "found": len(solutions),  # solutions selected for the sweep (0 => the run already errored out)
            "passed": passed,  # workload exit code 0
            "failed": failed,  # workload non-zero exit
            "timed_out": timed_out,  # killed by --per-solution-timeout
            "stopped_early": stopped_early,  # true => --sweep-timeout budget stopped the sweep before all were run
        },
        "results": results,
    }
    report_path = Path(sweep_report).resolve() if sweep_report else (sol_dir / "sweep_report.yaml")
    if not dry_run:
        with open(report_path, "w") as f:
            # width=inf keeps long values (tt_run_command, paths) on a single line instead of YAML-wrapping them.
            yaml.safe_dump(report, f, sort_keys=False, default_flow_style=False, width=float("inf"))
        click.echo(f"\n{PREFIX} Report: {report_path}")

    click.echo(
        f"{PREFIX} SUMMARY: {passed}/{len(results)} passed"
        + (f", {failed} failed" if failed else "")
        + (f", {timed_out} timed out" if timed_out else "")
        + (f" (stopped early by --sweep-timeout; {len(solutions) - len(results)} not run)" if stopped_early else "")
    )
    return report_path, passed, failed, timed_out


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    help="Run <program> across every generate_rank_bindings --all-solutions solution. "
    "Accepts the same arguments as tt-run, plus sweep extras.",
)
# ---- tt-run compatible options (same names/semantics) ----
# NOTE: the sweep is new-mode only. tt-run's legacy --rank-binding (a single explicit binding) is
# intentionally NOT exposed -- a sweep needs an MGD + hosts (or a mock mapping) to enumerate solutions.
@click.option(
    "--mesh-graph-descriptor",
    type=click.Path(path_type=Path),
    default=None,
    help="(tt-run) MGD to solve; enables generate-then-sweep. Requires --hosts or --mock-cluster-rank-binding.",
)
@click.option("--hosts", type=str, default=None, help="(tt-run) Comma-separated hostnames (real cluster).")
@click.option(
    "--mock-cluster-rank-binding",
    type=click.Path(path_type=Path),
    default=None,
    help="(tt-run) Mock rank->descriptor mapping YAML (mock cluster).",
)
@click.option("--mpi-args", default=None, help="(tt-run) Extra MPI args (quoted); forwarded to each launch.")
@click.option(
    "--rankfile-syntax",
    type=click.Choice(["auto", "rankfile", "map-by", "mca"]),
    default="auto",
    help="(tt-run) Rankfile syntax; forwarded to each launch.",
)
@click.option("--tcp-interface", type=str, default=None, help="(tt-run) MPI TCP interface; forwarded.")
@click.option("--bare", is_flag=True, help="(tt-run) Disable tt-run defaults; forwarded.")
@click.option("--tracy", "tracy_args", type=str, default=None, help="(tt-run) Tracy profiling args; forwarded.")
@click.option("--debug-gdbserver", is_flag=True, help="(tt-run) Launch under gdbserver; forwarded.")
@click.option("--skip-executable-check", is_flag=True, help="(tt-run) forwarded.")
@click.option("--skip-mgd-check", is_flag=True, help="(tt-run) forwarded.")
@click.option("-v", "--verbose", is_flag=True, help="(tt-run) Verbose; forwarded.")
@click.option("--dry-run", is_flag=True, help="Print per-solution tt-run commands without executing.")
# ---- sweep-specific extras ----
@click.option(
    "--solutions-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="EXTRA: sweep an existing solutions dir (with solutions_index.yaml). Skips Phase 1.",
)
@click.option(
    "--solutions-output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="EXTRA: where Phase 1 writes solutions (default generated/ttrun/sweep).",
)
@click.option(
    "--max-solutions",
    type=int,
    default=0,
    help="EXTRA: cap solutions generated in Phase 1 (0 = all). Forwarded to generate_rank_bindings.",
)
@click.option(
    "--distinct-host-sets",
    is_flag=True,
    help="EXTRA: keep only one solution per unique set of HOSTS (real host-set dedup). "
    "Forwarded to generate_rank_bindings.",
)
@click.option(
    "--allow-shape-permutations",
    is_flag=True,
    hidden=True,
    help="(advanced/hidden) Disable generate_rank_bindings' always-on solver unique_shapes dedup.",
)
@click.option("--select", type=str, default=None, help="EXTRA: only sweep these solution ids (comma-separated).")
@click.option("--limit", type=int, default=None, help="EXTRA: sweep at most the first N solutions (index order).")
@click.option(
    "--per-solution-timeout", type=int, default=None, help="EXTRA: kill a launch after N seconds (=> timeout)."
)
@click.option(
    "--sweep-timeout",
    type=int,
    default=None,
    help="EXTRA: total wall-clock budget (seconds) for the whole sweep. When exceeded, stop launching "
    "further solutions and return what was swept so far WITH A WARNING (not an error). Checked before "
    "each launch, so it never interrupts a solve already running. Only a sweep that found 0 solutions "
    "exits non-zero.",
)
@click.option(
    "--stop-on-failure/--continue-on-failure",
    default=False,
    help="EXTRA: stop the sweep on the first failing *workload* after all tt-run retries. "
    "Default: continue. Does NOT apply to --recover-command: if recover fails all retries, "
    "the sweep always aborts as an unrecoverable hardware error.",
)
@click.option(
    "--recover-command",
    type=str,
    required=True,
    help="REQUIRED: arbitrary recovery command (bash -c) run after a solution's tt-run fails or "
    "times out, then the same tt-run is retried. Use this to recover the machine/cluster. "
    "Quoted string, e.g. --recover-command './recover.sh --force' or --recover-command 'sudo reboot'. "
    f"Recover and tt-run are each retried {DEFAULT_RETRIES} times by default with {RETRY_DELAY_S}s "
    "between attempts (see hidden --retries). "
    "Success is a warning and the sweep continues; exhausting recover retries is an unrecoverable "
    "error (hardware cannot recover) and always aborts, ignoring --stop-on-failure / "
    "--continue-on-failure. Not run on pass. Still run after the last failure so the machine is left clean.",
)
@click.option(
    "--retries",
    type=int,
    default=DEFAULT_RETRIES,
    hidden=True,
    help=f"(advanced/hidden) Attempts for both tt-run and --recover-command (default {DEFAULT_RETRIES}). "
    f"{RETRY_DELAY_S}s between failed attempts. A later tt-run pass is recorded as pass; fail/timeout "
    "only if every tt-run attempt fails. Exhausting recover retries is unrecoverable and always aborts "
    "(--stop-on-failure / --continue-on-failure are ignored).",
)
@click.option(
    "--sweep-report",
    type=click.Path(path_type=Path),
    default=None,
    help="EXTRA: path for sweep_report.yaml (default <solutions-dir>/sweep_report.yaml).",
)
@click.option(
    "--log-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="EXTRA: directory for per-solution logs (default <solutions-dir>/sweep_logs). "
    "Each solution's stdout+stderr is saved to <log-dir>/<solution_id>.log.",
)
@click.pass_context
def main(
    ctx,
    mesh_graph_descriptor,
    hosts,
    mock_cluster_rank_binding,
    mpi_args,
    rankfile_syntax,
    tcp_interface,
    bare,
    tracy_args,
    debug_gdbserver,
    skip_executable_check,
    skip_mgd_check,
    verbose,
    dry_run,
    solutions_dir,
    solutions_output_dir,
    max_solutions,
    distinct_host_sets,
    allow_shape_permutations,
    select,
    limit,
    per_solution_timeout,
    sweep_timeout,
    stop_on_failure,
    recover_command,
    retries,
    sweep_report,
    log_dir,
):
    program = list(ctx.args)
    if not program:
        raise click.ClickException("No <program> to run. Pass it after the options, e.g. `... -- ./my_app`.")
    recover_command = recover_command.strip()
    if not recover_command:
        raise click.ClickException("--recover-command must be a non-empty command string.")
    if retries < 1:
        raise click.ClickException("--retries must be >= 1.")

    parsed_hosts = [h for h in hosts.split(",") if h] if hosts else None
    parsed_mpi_args = shlex.split(mpi_args) if mpi_args else None
    mock = mock_cluster_rank_binding is not None

    # --sweep-timeout budget covers enumeration (Phase 1) + the per-solution sweep (Phase 2).
    sweep_start = time.time()

    # 1. Obtain the solutions directory.
    if solutions_dir is not None:
        sol_dir = Path(solutions_dir).resolve()
    else:
        if mesh_graph_descriptor is None:
            raise click.ClickException("Provide --solutions-dir, or --mesh-graph-descriptor to generate solutions.")
        if not mock and not parsed_hosts:
            raise click.ClickException("New mode needs --hosts (real cluster) or --mock-cluster-rank-binding (mock).")
        out = Path(solutions_output_dir).resolve() if solutions_output_dir else (_repo_root() / "generated/ttrun/sweep")
        sol_dir = _generate_solutions(
            mesh_graph_descriptor=Path(mesh_graph_descriptor),
            hosts=parsed_hosts,
            mock_cluster_rank_binding=Path(mock_cluster_rank_binding) if mock else None,
            output_dir=out,
            max_solutions=max_solutions,
            distinct_host_sets=distinct_host_sets,
            allow_shape_permutations=allow_shape_permutations,
            mpi_args=parsed_mpi_args,
            dry_run=dry_run,
        )
        if dry_run:
            click.echo(f"{PREFIX} --dry-run: no solutions generated; nothing to sweep.")
            return

    index = _load_index(sol_dir)
    solutions = _select_solutions(index, select, limit)
    if not solutions:
        raise click.ClickException(f"No solutions to sweep in {sol_dir} (after --select/--limit).")

    # Auto-detect mock mode when sweeping an existing solutions dir (per-solution phase2_mock_mapping.yaml).
    if not mock and (sol_dir / solutions[0]["dir"] / "phase2_mock_mapping.yaml").is_file():
        mock = True

    tt_run = _find_tt_run()

    # Args forwarded verbatim to every per-solution tt-run launch.
    passthrough: List[str] = []
    if rankfile_syntax and rankfile_syntax != "auto":
        passthrough += ["--rankfile-syntax", rankfile_syntax]
    if tcp_interface:
        passthrough += ["--tcp-interface", tcp_interface]
    if bare:
        passthrough += ["--bare"]
    if tracy_args is not None:
        passthrough += ["--tracy", tracy_args]
    if debug_gdbserver:
        passthrough += ["--debug-gdbserver"]
    if skip_executable_check:
        passthrough += ["--skip-executable-check"]
    if skip_mgd_check:
        passthrough += ["--skip-mgd-check"]
    if verbose:
        passthrough += ["-v"]

    logs_root = Path(log_dir).resolve() if log_dir else (sol_dir / "sweep_logs")
    if not dry_run:
        logs_root.mkdir(parents=True, exist_ok=True)

    click.echo(f"{PREFIX} Sweeping {len(solutions)} solution(s) in {sol_dir}")
    click.echo(f"{PREFIX} Per-solution logs -> {logs_root}")
    click.echo(f"{PREFIX} Recover on failure: {recover_command} " f"(retries={retries}, delay={RETRY_DELAY_S}s)")
    results = []
    stopped_early = False
    recover_exhausted = False
    for i, sol in enumerate(solutions):
        # Total-budget check (before launching, so we never interrupt a running solve).
        if sweep_timeout is not None:
            elapsed = time.time() - sweep_start
            if elapsed >= sweep_timeout:
                click.echo(
                    f"{PREFIX} WARNING: --sweep-timeout ({sweep_timeout}s) reached after {round(elapsed, 1)}s; "
                    f"stopping with {len(results)}/{len(solutions)} solution(s) swept "
                    f"({len(solutions) - len(results)} not run)."
                )
                stopped_early = True
                break

        cmd = _build_tt_run_cmd(
            tt_run=tt_run,
            solutions_dir=sol_dir,
            sol=sol,
            program=program,
            mock=mock,
            mpi_args=parsed_mpi_args,
            passthrough=passthrough,
        )
        label = sol.get("id", sol["dir"])
        cmd_str = " ".join(shlex.quote(c) for c in cmd)  # exact, copy-paste-reproducible tt-run command
        click.echo(
            f"\n{PREFIX} [{i + 1}/{len(solutions)}] solution {label} "
            f"({sol.get('num_hosts', '?')} hosts)\n  {cmd_str}"
        )

        if dry_run:
            click.echo(
                f"{PREFIX}   --dry-run: would retry tt-run up to {retries} time(s); "
                f"recover on fail/timeout then sleep {RETRY_DELAY_S}s:\n  {recover_command}"
            )
            results.append(
                _make_result(
                    sol=sol,
                    solutions_dir=sol_dir,
                    label=label,
                    cmd_str=cmd_str,
                    status="dry-run",
                    returncode=None,
                    duration_seconds=0.0,
                    tt_run_attempts=0,
                    recover_command=recover_command,
                )
            )
            continue

        log_path = logs_root / f"{label}.log"
        t0 = time.time()
        outcome = _run_with_retries(
            cmd,
            cwd=_repo_root(),
            log_path=log_path,
            timeout=per_solution_timeout,
            retries=retries,
            recover_cmd=recover_command,
            label="tt-run",
            cmd_display=cmd_str,
        )
        dur = round(time.time() - t0, 1)
        click.echo(
            f"{PREFIX}   -> {outcome.status} (rc={outcome.returncode}, {dur}s, "
            f"attempts={outcome.attempts})\n     log={log_path}"
        )

        results.append(
            _make_result(
                sol=sol,
                solutions_dir=sol_dir,
                label=label,
                cmd_str=cmd_str,
                status=outcome.status,
                returncode=outcome.returncode,
                duration_seconds=dur,
                tt_run_attempts=outcome.attempts,
                recover_command=recover_command if outcome.recover_returncode is not None else None,
                recover_returncode=outcome.recover_returncode,
                log_path=log_path,
            )
        )
        # Recover failure is unrecoverable: always abort, ignoring --stop-on-failure/--continue-on-failure.
        # Those flags only control whether a *workload* fail/timeout (after all tt-run retries) continues.
        if outcome.recover_exhausted:
            recover_exhausted = True
            click.echo(
                f"{PREFIX} UNRECOVERABLE: hardware cannot recover with this command "
                f"after {retries} attempt(s) (last rc={outcome.recover_returncode}). "
                f"--stop-on-failure/--continue-on-failure do not apply. Halting after {label}."
            )
            break
        if outcome.status != "pass" and stop_on_failure:
            click.echo(f"{PREFIX} --stop-on-failure: halting after {label}.")
            break

    # 3. Report.
    report_path, _passed, failed, timed_out = _write_sweep_report(
        results=results,
        solutions=solutions,
        index=index,
        sol_dir=sol_dir,
        program=program,
        recover_command=recover_command,
        stopped_early=stopped_early,
        sweep_report=sweep_report,
        dry_run=dry_run,
    )
    # Stopping early on --sweep-timeout is NOT an error -- return what we swept. A sweep that found 0
    # solutions has already exited non-zero above (_select_solutions / _generate_solutions). Real
    # per-solution failures/timeouts still surface as a non-zero exit. Recover exhausting its retries
    # is a hard error: the machine could not be brought back, so later solutions would be untrustworthy.
    if recover_exhausted:
        raise click.ClickException(
            f"UNRECOVERABLE: hardware cannot recover with this command "
            f"(--recover-command failed after {retries} attempt(s)). Sweep aborted. See {report_path}."
        )
    if not dry_run and (failed or timed_out):
        sys.exit(1)


if __name__ == "__main__":
    main()
