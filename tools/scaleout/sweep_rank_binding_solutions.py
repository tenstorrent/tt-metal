#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Async, interleaved generate-and-sweep across every topology solution.

It's ``tt-run``, but across every valid placement of one MGD + host set:

  1. launches ``generate_rank_bindings --all-solutions`` ONCE in the background (it streams each
     solution to disk + rewrites ``solutions_index.yaml`` as it finds them);
  2. the moment a solution is ready, runs your workload on it via ``tt-run`` -- ONE workload at a
     time, while generation races ahead producing the next ones;
  3. writes a self-contained folder per solution (rank bindings + ``run.sh`` reproducer +
     ``workload.log`` + ``result.yaml``) and an aggregate ``sweep/sweep_report.yaml``.

Exactly one producer (generation) + one consumer (workload). See
tools/scaleout/README_sweep_rank_binding_solutions.md for the full design.
"""

import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
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
)

PREFIX = "[tt-sweep]"
POLL_INTERVAL_S = 1.0  # how often the consumer re-reads the index / polls the running workload
HEARTBEAT_INTERVAL_S = 15.0  # periodic status while waiting on the producer or a long-running workload
# Give up searching for the NEXT solution if the producer goes this long without emitting one; the remaining
# solutions are treated as "too difficult to find" and the search is stopped (already-found ones still run).
DEFAULT_SOLUTION_SEARCH_TIMEOUT_S = 900.0  # 15 minutes


# ─────────────────────────────── reused helpers (from v1) ────────────────────────────────


def _repo_root() -> Path:
    return Path(os.environ.get("TT_METAL_HOME", ".")).resolve()


def _raise_nproc_limit() -> Optional[Tuple[int, int]]:
    """Raise this process's max-processes/threads (nproc) soft limit to the hard limit.

    Child processes (the producer's mpirun and each workload) inherit it. Under a low soft nproc limit
    (e.g. 512) MPI/PMIx and the numeric libs fail to create threads once the user's baseline thread count
    exceeds it -- 'pmix_progress_thread_start failed' at MPI_Init, or 'OpenBLAS ... pthread_create failed'.
    Returns (old_soft, new_soft) if changed, else None."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
        if hard != resource.RLIM_INFINITY and soft >= hard:
            return None
        resource.setrlimit(resource.RLIMIT_NPROC, (hard, hard))
        return soft, hard
    except (ImportError, ValueError, OSError):
        return None


def _find_tt_run() -> str:
    """Locate the tt-run entrypoint (installed console script, or venv-adjacent script)."""
    exe = shutil.which("tt-run")
    if exe:
        return exe
    candidate = Path(sys.executable).parent / "tt-run"
    if candidate.exists():
        return str(candidate)
    raise click.ClickException("Could not find the `tt-run` executable on PATH or next to the Python interpreter.")


def _inject_solution_flags(cmd: List[str], extra: List[str]) -> List[str]:
    """Insert generate_rank_bindings sweep flags after every ``--output-dir <value>`` (real + mock layouts)."""
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
    """Build the per-solution ``tt-run`` legacy invocation, with ABSOLUTE paths to this solution's artifacts."""
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


# ─────────────────────────────── v2: producer (generation) ───────────────────────────────


def _build_generate_cmd(
    *,
    mesh_graph_descriptor: Path,
    hosts: Optional[List[str]],
    mock_cluster_rank_binding: Optional[Path],
    output_dir: Path,
    max_solutions: int,
    distinct_host_sets: bool,
    allow_shape_permutations: bool,
    mpi_args: Optional[List[str]],
) -> List[str]:
    """Build the background ``generate_rank_bindings --all-solutions`` command (producer)."""
    executable = find_generate_rank_bindings_executable()
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
        extra += ["--allow-shape-permutations"]  # hidden: turn OFF the always-on solver unique_shapes dedup
    return _inject_solution_flags(cmd, extra)


def _start_producer(cmd: List[str], cwd: Path, log_path: Path) -> subprocess.Popen:
    """Launch generation in the background; its stdout+stderr stream to generate.log.

    Runs in its own session (setsid) so the whole mpirun process tree can be reaped as a group if we
    have to stop the search early (e.g. the search timeout)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "w")
    return subprocess.Popen(cmd, cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)


def _stop_producer(producer: subprocess.Popen, *, grace_s: float = 10.0) -> None:
    """Terminate the producer and its whole process group (mpirun + mock ranks), escalating to SIGKILL."""
    if producer.poll() is not None:
        return
    try:
        pgid = os.getpgid(producer.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        try:
            producer.wait(timeout=grace_s)
            return
        except subprocess.TimeoutExpired:
            continue


# ─────────────────────────────── v2: index (the handoff) ─────────────────────────────────


def _read_index(index_path: Path) -> Optional[dict]:
    """Read solutions_index.yaml DEFENSIVELY: the producer rewrites it in place, so a mid-rewrite read may
    fail to parse -- return None then and let the caller retry on the next tick."""
    if not index_path.is_file():
        return None
    try:
        with open(index_path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except (yaml.YAMLError, OSError):
        return None


# ─────────────────────────────── v2: run.sh + workload ───────────────────────────────────


def _env_snapshot() -> Dict[str, str]:
    """Curated env allowlist to bake into each run.sh so a later re-run matches the sweep's environment."""
    keep_exact = {"TT_METAL_HOME", "ARCH_NAME", "LD_LIBRARY_PATH", "PYTHONPATH", "PATH"}
    snap: Dict[str, str] = {}
    for k, v in os.environ.items():
        if k in keep_exact or k.startswith("TT_"):
            snap[k] = v
    return snap


def _write_run_sh(sol_dir: Path, tt_run_cmd: List[str], env: Dict[str, str], solution_id: str) -> Path:
    """Write the self-contained reproducer AND the actual launcher for one solution."""
    run_sh = sol_dir / "run.sh"
    lines = [
        "#!/usr/bin/env bash",
        f"# Reproduce the sweep run for solution {solution_id} exactly. Generated by sweep_rank_binding_solutions.py.",
        "set -euo pipefail",
        "",
        "# --- Environment captured at sweep time ---",
    ]
    for k in sorted(env):
        lines.append(f"export {k}={shlex.quote(env[k])}")
    lines += [
        "",
        "# --- Raise the max processes/threads (nproc) soft limit to the hard limit ---",
        "# MPI/PMIx and the numeric libs each create threads at startup. Under a low soft nproc limit (e.g. 512)",
        "# pthread_create fails once the user's baseline thread count exceeds it, surfacing as",
        "# 'pmix_progress_thread_start failed' (MPI_Init abort) or 'OpenBLAS ... pthread_create failed'.",
        'ulimit -Su "$(ulimit -Hu)" 2>/dev/null || ulimit -u unlimited 2>/dev/null || true',
        "",
        "# --- BLAS/OpenMP thread caps (override by exporting these before running) ---",
        "# tt-run imports ttnn -> tools/tracy -> seaborn -> scipy, which pulls in OpenBLAS. OpenBLAS otherwise",
        "# spawns one thread per core; under the MPI-imposed RLIMIT_NPROC that can exhaust the process limit and",
        "# fail with 'pthread_create failed ... Resource temporarily unavailable' (surfacing as an ImportError or",
        "# a SIGSEGV). tt-run does no heavy BLAS, so 1 thread is plenty.",
        'export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"',
        'export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"',
        'export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"',
        'export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"',
    ]
    if "TT_METAL_HOME" in env:
        lines += ["", 'cd "$TT_METAL_HOME"']
    lines += [
        "",
        "# --- Exact tt-run launch (absolute paths to THIS solution's artifacts) ---",
        "exec " + " ".join(shlex.quote(c) for c in tt_run_cmd),
        "",
    ]
    run_sh.write_text("\n".join(lines))
    run_sh.chmod(0o755)
    return run_sh


def _as_host_list(value) -> List[str]:
    """Normalize a solution's ``host_set`` to a list of host strings.

    The generate_rank_bindings solutions index serializes ``host_set`` as a single comma-joined scalar, so PyYAML hands
    it back as a ``str``. Iterating/joining that directly would split it character-by-character; coerce to a real list
    here (accepting an already-list form too, for forward-compat)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [h for h in (p.strip() for p in value.split(",")) if h]
    return [str(h) for h in value]


def _workload_banner(sol: dict, sol_dir: Path, run_sh: Path, tt_run_cmd: List[str]) -> str:
    hosts = sol.get("host_set") or []
    sid = sol.get("id", sol["dir"])
    return "\n".join(
        [
            f"# ===== tt-sweep solution {sid} =====",
            f"# hosts ({sol.get('num_hosts', len(hosts))}): {' '.join(hosts)}",
            f"# rank_bindings: {(sol_dir / 'rank_bindings.yaml').resolve()}",
            f"# rankfile:      {(sol_dir / 'rankfile').resolve()}",
            f"# reproduce:     bash {run_sh.resolve()}",
            f"# command:       {' '.join(shlex.quote(c) for c in tt_run_cmd)}",
            f"# started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "# " + "=" * 46,
            "",
        ]
    )


# ─────────────────────────────── v2: report ──────────────────────────────────────────────


def _write_report(report_path: Path, meta: dict, results: List[dict], truncated: bool) -> None:
    """Rewrite the aggregate sweep_report.yaml (index order) after every completed workload."""
    passed = sum(r["status"] == "pass" for r in results)
    failed = sum(r["status"] == "fail" for r in results)
    timed_out = sum(r["status"] == "timeout" for r in results)
    report = {
        **meta,
        "summary": {
            "total": len(results),
            "passed": passed,
            "failed": failed,
            "timed_out": timed_out,
            "truncated": truncated,
        },
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        yaml.safe_dump(report, f, sort_keys=False, default_flow_style=False, width=float("inf"))


# ─────────────────────────────── v2: interleaved orchestrator ────────────────────────────


def _sweep_interleaved(
    *,
    producer: subprocess.Popen,
    solutions_dir: Path,
    tt_run: str,
    program: List[str],
    mock: bool,
    parsed_mpi_args: Optional[List[str]],
    passthrough: List[str],
    limit: Optional[int],
    stop_on_failure: bool,
    per_solution_timeout: Optional[int],
    solution_search_timeout: Optional[float],
    interleave: bool,
    dry_run: bool,
    report_meta: dict,
    report_path: Path,
) -> Tuple[List[dict], dict]:
    """The single control loop: one producer + one consumer, coordinated through solutions_index.yaml.

    Returns (results, outcome). ``outcome`` records why the search ended: search_timed_out / exhausted /
    capped, plus the number of solutions found -- used by the caller for the final message and exit code."""
    index_path = solutions_dir / "solutions_index.yaml"
    env = _env_snapshot()

    order: List[str] = []  # solution ids in index order (report order)
    sols: Dict[str, dict] = {}  # id -> index entry
    queue: List[str] = []  # ids waiting to launch (index order)
    dispatched = set()  # ids launched (or dry-run written)
    results: Dict[str, dict] = {}  # id -> result dict
    running: Optional[dict] = None  # currently-running workload
    stop = False
    t0 = time.time()
    last_heartbeat = t0
    last_solution_time = t0  # updated whenever a new solution appears; drives the search timeout
    search_timed_out = False  # set if the producer went too long without a new solution

    def rel() -> str:
        return f"+{int(time.time() - t0):03d}s"

    def producer_alive() -> bool:
        return producer.poll() is None

    def echo(stream: str, msg: str) -> None:
        click.echo(f"{PREFIX}[{stream} {rel()}] {msg}")

    def flush_report(trunc: bool) -> None:
        _write_report(report_path, report_meta, [results[i] for i in order if i in results], trunc)

    def ingest_index() -> None:
        nonlocal last_solution_time
        idx = _read_index(index_path)
        if not idx:
            return
        for sol in idx.get("solutions", []):
            sid = sol.get("id") or sol.get("dir")
            if not sid or sid in sols:
                continue
            if limit is not None and len(order) >= limit:
                break
            sol["host_set"] = _as_host_list(sol.get("host_set"))  # index stores it comma-joined; make it a real list
            sols[sid] = sol
            order.append(sid)
            queue.append(sid)
            last_solution_time = time.time()  # progress: reset the "search for next solution" timeout
            hs = sol.get("host_set") or []
            preview = ",".join(hs[:6]) + (f" …(+{len(hs) - 6})" if len(hs) > 6 else "")
            echo(
                "gen",
                f"found #{len(order)}  {sid}  hosts={sol.get('num_hosts', len(hs))}"
                + (f"   [{preview}]" if hs else "")
                + f"   (found {len(order)}, tested {len(results)}, queued {len(queue)})",
            )

    def launch_next() -> None:
        nonlocal running
        sid = queue.pop(0)
        sol = sols[sid]
        sol_dir = solutions_dir / sol["dir"]
        cmd = _build_tt_run_cmd(
            tt_run=tt_run,
            solutions_dir=solutions_dir,
            sol=sol,
            program=program,
            mock=mock,
            mpi_args=parsed_mpi_args,
            passthrough=passthrough,
        )
        run_sh = _write_run_sh(sol_dir, cmd, env, sid)
        log_path = sol_dir / "workload.log"
        dispatched.add(sid)

        if dry_run:
            echo("run", f"DRY-RUN #{order.index(sid) + 1}  {sid}  → wrote {run_sh}")
            _record(sid, sol, "dry-run", None, 0.0, run_sh, log_path, cmd, None, None)
            return

        log_fh = open(log_path, "w")
        log_fh.write(_workload_banner(sol, sol_dir, run_sh, cmd))
        log_fh.flush()
        proc = subprocess.Popen(["bash", str(run_sh)], cwd=_repo_root(), stdout=log_fh, stderr=subprocess.STDOUT)
        running = {
            "id": sid,
            "sol": sol,
            "proc": proc,
            "log_fh": log_fh,
            "log_path": log_path,
            "run_sh": run_sh,
            "cmd": cmd,
            "start": time.time(),
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        echo(
            "run",
            f"▶ START #{order.index(sid) + 1}  {sid}  ({sol.get('num_hosts', '?')} hosts)"
            f"  dir={sol['dir']}  started={running['started_at']}",
        )
        echo("run", f"           log {log_path}")

    def _record(sid, sol, status, rc, dur, run_sh, log_path, cmd, started_at, finished_at) -> None:
        hs = sol.get("host_set") or []
        results[sid] = {
            "solution_id": sid,
            "status": status,  # pass | fail | timeout | dry-run
            "returncode": rc,
            "duration_seconds": round(dur, 1),
            "num_hosts": sol.get("num_hosts", len(hs)),
            "host_set": hs,
            "run_script": str(run_sh.resolve()),
            "log_path": str(log_path.resolve()),
            "rank_binding_path": str((solutions_dir / sol["dir"] / "rank_bindings.yaml").resolve()),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        flush_report(trunc=(limit is not None and len(order) >= limit))

    def poll_running() -> None:
        nonlocal running, stop
        if running is None:
            return
        proc = running["proc"]
        rc = proc.poll()
        timed_out = (
            per_solution_timeout is not None and time.time() - running["start"] > per_solution_timeout and rc is None
        )
        if timed_out:
            proc.kill()
            proc.wait()
            rc, status = None, "timeout"
        elif rc is None:
            return  # still running
        else:
            status = "pass" if rc == 0 else "fail"
        dur = time.time() - running["start"]
        running["log_fh"].close()
        sid = running["id"]
        _record(
            sid,
            running["sol"],
            status,
            rc,
            dur,
            running["run_sh"],
            running["log_path"],
            running["cmd"],
            running["started_at"],
            datetime.now(timezone.utc).isoformat(),
        )
        mark = {"pass": "✔ END PASS", "fail": "✘ END FAIL", "timeout": "⏱ END TIMEOUT"}[status]
        line = (
            f"{mark}  #{order.index(sid) + 1}  {sid}   rc={rc}   {dur:.1f}s"
            f"   finished={datetime.now(timezone.utc).isoformat()}"
        )
        if status != "pass":
            line += f"   → {running['log_path']} ; reproduce: bash {running['run_sh']}"
        echo("run", line)
        running = None
        if status != "pass" and stop_on_failure:
            echo("run", f"--stop-on-failure: halting after {sid}; terminating producer.")
            stop = True

    def maybe_heartbeat() -> None:
        nonlocal last_heartbeat
        now = time.time()
        if now - last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        last_heartbeat = now
        if running is not None:
            elapsed = now - running["start"]
            echo(
                "run",
                f"… still running #{order.index(running['id']) + 1}  {running['id']}"
                f"   {elapsed:.0f}s elapsed   log={running['log_path']}",
            )
            return
        if producer_alive():
            waited = int(now - last_solution_time)
            budget = f", {waited}s/{int(solution_search_timeout)}s search budget" if solution_search_timeout else ""
            if order:
                echo(
                    "gen",
                    f"waiting for next solution   (found {len(order)}, tested {len(results)},"
                    f" queued {len(queue)}, producer running{budget})",
                )
            else:
                echo(
                    "gen",
                    "waiting for first solution from producer"
                    f"   (see {solutions_dir / 'sweep' / 'generate.log'}{budget})",
                )
        elif not order:
            echo("gen", "producer exited before any solution appeared in solutions_index.yaml")

    def search_timed_out_now() -> bool:
        """True once the producer has gone longer than the search timeout without emitting a new solution."""
        return (
            solution_search_timeout is not None
            and producer_alive()
            and (time.time() - last_solution_time) > solution_search_timeout
        )

    # ── main loop ───────────────────────────────────────────────────────────────────────
    while True:
        ingest_index()
        if stop:
            break
        # Give up searching for the next solution if the producer has stalled too long. Already-found
        # solutions still run: we only stop the producer, then let the queue drain below.
        if search_timed_out_now():
            search_timed_out = True
            mins = solution_search_timeout / 60.0
            echo(
                "gen",
                f"⏱ no new solution for {mins:.0f} min — stopping the search; remaining solutions were "
                f"too difficult to find (found {len(order)} so far). Draining {len(queue)} queued.",
            )
            _stop_producer(producer)
        # launch when idle; in --no-interleave, hold until the producer has fully finished
        if running is None and queue and (interleave or not producer_alive()):
            launch_next()
        poll_running()
        maybe_heartbeat()
        # terminate: producer done AND nothing running AND nothing queued (index re-read above caught stragglers)
        if not producer_alive() and running is None and not queue:
            break
        time.sleep(POLL_INTERVAL_S)

    # stop-on-failure / --limit / search-timeout ⇒ reap the still-running producer (whole group)
    if producer_alive():
        _stop_producer(producer)
    if running is not None:  # stop_on_failure left a workload mid-flight
        running["proc"].wait()
        running["log_fh"].close()

    limit_reached = limit is not None and len(order) >= limit
    capped = bool((_read_index(index_path) or {}).get("truncated"))
    outcome = {
        "found": len(order),
        "tested": len(results),
        "search_timed_out": search_timed_out,
        "stopped_on_failure": stop,
        "limit_reached": limit_reached,
        # "capped": producer hit --max-solutions; "exhausted": producer finished the whole space on its own.
        "capped": not search_timed_out and not stop and not limit_reached and capped,
        "exhausted": (
            not search_timed_out and not stop and not limit_reached and not capped and producer.poll() is not None
        ),
        "producer_rc": producer.poll(),
    }
    flush_report(trunc=limit_reached)
    return [results[i] for i in order if i in results], outcome


# ─────────────────────────────── CLI ─────────────────────────────────────────────────────


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True),
    help="Run <program> across every generate_rank_bindings --all-solutions solution (async, interleaved). "
    "Accepts the same arguments as tt-run, plus sweep extras.",
)
# ---- tt-run compatible options (forwarded to each per-solution launch) ----
@click.option(
    "--mesh-graph-descriptor",
    "-m",
    type=click.Path(path_type=Path),
    default=None,
    help="(tt-run) MGD to solve. Required. Requires --hosts or --mock-cluster-rank-binding.",
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
    help="(tt-run) Rankfile syntax; forwarded.",
)
@click.option("--tcp-interface", type=str, default=None, help="(tt-run) MPI TCP interface; forwarded.")
@click.option("--bare", is_flag=True, help="(tt-run) Disable tt-run defaults; forwarded.")
@click.option("--tracy", "tracy_args", type=str, default=None, help="(tt-run) Tracy profiling args; forwarded.")
@click.option("--debug-gdbserver", is_flag=True, help="(tt-run) Launch under gdbserver; forwarded.")
@click.option("--skip-executable-check", is_flag=True, help="(tt-run) forwarded.")
@click.option("--skip-mgd-check", is_flag=True, help="(tt-run) forwarded.")
@click.option("-v", "--verbose", is_flag=True, help="(tt-run) Verbose; forwarded.")
# ---- sweep extras ----
@click.option(
    "--solutions-output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where solutions + sweep artifacts are written (default generated/ttrun/sweep).",
)
@click.option(
    "--max-solutions",
    type=int,
    default=0,
    help="Cap solutions generated (0 = all). Forwarded to generate_rank_bindings.",
)
@click.option(
    "--distinct-host-sets",
    is_flag=True,
    help="Keep only one solution per unique host set. Forwarded to generate_rank_bindings.",
)
@click.option(
    "--allow-shape-permutations",
    is_flag=True,
    hidden=True,
    help="(hidden) Disable generate_rank_bindings' always-on solver unique_shapes dedup.",
)
@click.option("--limit", type=int, default=None, help="Sweep at most the first N solutions (index order).")
@click.option("--per-solution-timeout", type=int, default=None, help="Kill a launch after N seconds (=> timeout).")
@click.option(
    "--solution-search-timeout",
    type=float,
    default=DEFAULT_SOLUTION_SEARCH_TIMEOUT_S,
    help="Stop the search if the producer goes this many seconds without finding a new solution "
    "(default 900 = 15 min; 0 disables). Already-found solutions still run.",
)
@click.option(
    "--solve-timeout",
    type=float,
    default=None,
    help="Max seconds for a single solve, reset for every solution. If a solve exceeds this without "
    "finding the next solution, stop gracefully: report 'no more solutions within the timeout' and exit 0 "
    "(no error) even when zero solutions were found. Overrides --solution-search-timeout when set.",
)
@click.option(
    "--stop-on-failure/--continue-on-failure",
    default=False,
    help="Stop the sweep (and generation) on the first failing solution (default: continue).",
)
@click.option(
    "--interleave/--no-interleave",
    default=True,
    help="Overlap generation with testing (default). --no-interleave waits for all generation first.",
)
@click.option("--dry-run", is_flag=True, help="Generate + write each run.sh, but launch no workloads.")
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
    solutions_output_dir,
    max_solutions,
    distinct_host_sets,
    allow_shape_permutations,
    limit,
    per_solution_timeout,
    solution_search_timeout,
    solve_timeout,
    stop_on_failure,
    interleave,
    dry_run,
):
    program = list(ctx.args)
    if not program:
        raise click.ClickException("No <program> to run. Pass it after the options, e.g. `... -- ./my_app`.")
    if mesh_graph_descriptor is None:
        raise click.ClickException("Provide --mesh-graph-descriptor / -m (the sweep always generates).")

    parsed_hosts = [h for h in hosts.split(",") if h] if hosts else None
    parsed_mpi_args = shlex.split(mpi_args) if mpi_args else None
    mock = mock_cluster_rank_binding is not None
    if not mock and not parsed_hosts:
        raise click.ClickException("Provide --hosts (real cluster) or --mock-cluster-rank-binding (mock).")

    out = Path(solutions_output_dir).resolve() if solutions_output_dir else (_repo_root() / "generated/ttrun/sweep")
    out.mkdir(parents=True, exist_ok=True)
    sweep_dir = out / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    generate_log = sweep_dir / "generate.log"
    report_path = sweep_dir / "sweep_report.yaml"
    tt_run = _find_tt_run()

    # tt-run passthrough args forwarded verbatim to every per-solution launch.
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

    gen_cmd = _build_generate_cmd(
        mesh_graph_descriptor=Path(mesh_graph_descriptor),
        hosts=parsed_hosts,
        mock_cluster_rank_binding=Path(mock_cluster_rank_binding) if mock else None,
        output_dir=out,
        max_solutions=max_solutions,
        distinct_host_sets=distinct_host_sets,
        allow_shape_permutations=allow_shape_permutations,
        mpi_args=parsed_mpi_args,
    )

    # Sweep header.
    click.echo(f"{PREFIX} ┌ sweep start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    click.echo(f"{PREFIX} │ MGD          : {mesh_graph_descriptor}")
    click.echo(f"{PREFIX} │ cluster      : " + ("mock" if mock else f"real, hosts={','.join(parsed_hosts or [])}"))
    click.echo(
        f"{PREFIX} │ enumerate    : --all-solutions"
        + (f" --max-solutions {max_solutions}" if max_solutions else "")
        + (" --distinct-host-sets" if distinct_host_sets else "")
    )
    click.echo(f"{PREFIX} │ workload     : {' '.join(shlex.quote(p) for p in program)}")
    click.echo(f"{PREFIX} │ output dir   : {out}")
    click.echo(
        f"{PREFIX} │ concurrency  : 1 producer, 1 consumer (one workload at a time"
        + ("" if interleave else "; --no-interleave: generate all first")
        + ")"
    )

    raised = _raise_nproc_limit()
    if raised is not None:
        click.echo(f"{PREFIX} │ nproc limit  : raised soft {raised[0]} → {raised[1]} (MPI/PMIx + BLAS thread headroom)")

    producer = _start_producer(gen_cmd, cwd=_repo_root(), log_path=generate_log)
    click.echo(f"{PREFIX} └ producer pid {producer.pid}  →  log {generate_log}")

    report_meta = {
        "mesh_graph_desc_path": str(mesh_graph_descriptor),
        "workload_command": " ".join(shlex.quote(p) for p in program),
        "solutions_dir": str(out),
    }
    # --solve-timeout, when set, is the per-solution solve budget (reset for every solution) AND makes a
    # timeout that finds no (further) solutions a graceful exit-0 rather than an error. It takes precedence
    # over --solution-search-timeout. With neither set, behaviour is unchanged (900s cap, exit 1 on empty).
    graceful_solve_timeout = solve_timeout is not None and solve_timeout > 0
    effective_search_timeout = solve_timeout if graceful_solve_timeout else solution_search_timeout
    search_timeout = effective_search_timeout if effective_search_timeout and effective_search_timeout > 0 else None
    results, outcome = _sweep_interleaved(
        producer=producer,
        solutions_dir=out,
        tt_run=tt_run,
        program=program,
        mock=mock,
        parsed_mpi_args=parsed_mpi_args,
        passthrough=passthrough,
        limit=limit,
        stop_on_failure=stop_on_failure,
        per_solution_timeout=per_solution_timeout,
        solution_search_timeout=search_timeout,
        interleave=interleave,
        dry_run=dry_run,
        report_meta=report_meta,
        report_path=report_path,
    )

    gen_rc = producer.poll()
    passed = sum(r["status"] == "pass" for r in results)
    failed = sum(r["status"] == "fail" for r in results)
    timed_out = sum(r["status"] == "timeout" for r in results)
    click.echo(f"\n{PREFIX} ┌ SUMMARY  {passed}/{len(results)} passed · {failed} failed · {timed_out} timed out")
    # Why did the search end?
    if outcome.get("search_timed_out"):
        mins = search_timeout / 60.0 if search_timeout else 0
        click.echo(
            f"{PREFIX} │ SEARCH   stopped after {mins:.0f} min with no new solution — remaining solutions "
            f"were too difficult to find ({outcome['found']} found)."
        )
    elif outcome.get("exhausted"):
        click.echo(f"{PREFIX} │ SEARCH   search space exhausted — all {outcome['found']} solution(s) found and swept.")
    elif outcome.get("capped"):
        click.echo(
            f"{PREFIX} │ SEARCH   reached the --max-solutions cap ({outcome['found']} found; "
            f"more solutions may exist)."
        )
    elif outcome.get("limit_reached"):
        click.echo(f"{PREFIX} │ SEARCH   reached the --limit of {limit} ({outcome['found']} found).")
    for r in results:
        if r["status"] not in ("pass", "dry-run"):
            click.echo(
                f"{PREFIX} │ {r['status'].upper():7s} {r['solution_id']}  rc={r['returncode']}  "
                f"log={r['log_path']}  repro=bash {r['run_script']}"
            )
    click.echo(f"{PREFIX} └ report   {report_path}")
    if gen_rc not in (0, None) and not stop_on_failure:
        click.echo(f"{PREFIX} WARNING: producer (generation) exited rc={gen_rc}; see {generate_log}")

    if not results:
        if graceful_solve_timeout and outcome.get("search_timed_out"):
            click.echo(
                f"{PREFIX} No solution found within the {int(solve_timeout)}s solve timeout — "
                f"no more solutions. Returning 0."
            )
            return
        click.echo(f"{PREFIX} No solutions were produced.")
        sys.exit(1)
    if not dry_run and (failed or timed_out):
        sys.exit(1)


if __name__ == "__main__":
    main()
