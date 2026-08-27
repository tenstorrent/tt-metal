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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
# Sleep between every failed command and the next recover / tt-run retry.
RETRY_DELAY_S = 5
DEFAULT_RETRIES = 3
# Consumer poll cadence: how often to re-read the streaming solutions_index.yaml for newly generated
# solutions while the producer is still running, and the heartbeat cadence while waiting on it.
POLL_INTERVAL_S = 2.0
HEARTBEAT_INTERVAL_S = 30.0
# How long to wait for the producer (generate_rank_bindings) to finish on its own before a whole-cluster
# recover (which would otherwise disrupt the producer's live MPI job); if it overruns, it is stopped.
PRODUCER_SETTLE_BEFORE_RECOVER_S = 300.0
# After the producer exits: how long to keep re-polling solutions_index.yaml before trusting that
# the visible index is final. NFS close-to-open attribute-cache lag can hide the producer's last
# write from another client for up to ~60s (acregmax default); add 30s margin.
PRODUCER_EXIT_INDEX_SETTLE_S = 90.0
# Poll cadence inside the post-exit settle window.
PRODUCER_EXIT_INDEX_POLL_INTERVAL_S = 3.0


def _short_id(sid: str) -> str:
    """First 10 chars of a solution's content-hash id -- enough to eyeball, short enough to scan."""
    return (sid or "")[:10]


def _short_host(host_set) -> str:
    """Shorten a host_set to just the node tags, e.g. 'bh-glx-110-d07u08,…-d08u02' -> 'd07u08·d08u02'."""
    hs = host_set if isinstance(host_set, str) else ",".join(host_set or [])
    return "·".join(h.split("-")[-1] for h in hs.split(",") if h) or "?"


class SweepLog:
    """All console formatting for the sweep driver (which runs ONE combo).

    Every user-facing line goes through a semantic method here, so the box/rule/tree glyphs live in one
    place and the rest of the driver just says what happened (``log.solution_start(...)``,
    ``log.attempt(...)``). Output is this driver's own level: per-combo generation + per-solution trees +
    retries. The cross-combo framing (SWEEP banners, RUN SUMMARY) is the caller's level -- see blaze cli.py.
    """

    _STATUS = {"pass": "PASS", "timeout": "TIMEOUT", "fail": "FAIL", "dry-run": "DRY-RUN"}

    def line(self, s: str = "") -> None:
        click.echo(f"{PREFIX} {s}".rstrip())

    def blank(self) -> None:
        click.echo("")

    # -- combo framing (this driver = one combo / one solutions dir) --
    def combo_start(self, sol_dir: Path, recover_cmd: str, retries: int) -> None:
        self.line(f"sweeping → {sol_dir}")
        if recover_cmd and recover_cmd != "true":
            self.line(f"recover: {recover_cmd}   (retries {retries}, delay {RETRY_DELAY_S}s)")

    def initial_reset(self) -> None:
        self.line("▶ initial reset (reap all hosts + recover)")

    def generating(self, log_path: Path) -> None:
        self.line(f"▶ generating solutions (streaming) → {log_path}")

    def found(self, n: int) -> None:
        """Subtle progress note as the producer streams more solutions (lands between solution trees)."""
        self.line(f"·  {n} solution(s) generated so far")

    def combo_summary(self, name: str, total: int, passed: int, failed: int, timed_out: int) -> None:
        parts = [f"{total} solution(s)", f"{passed} passed"]
        if failed:
            parts.append(f"{failed} failed")
        if timed_out:
            parts.append(f"{timed_out} timed out")
        body = f"{name} · " + " · ".join(parts)
        pad = "═" * 2
        self.blank()
        self.line(f"{pad} {body} {pad}")

    def report(self, path: Path) -> None:
        self.line(f"report → {path}")

    # -- per-solution tree --
    def solution_start(self, position: str, sid: str, host: str, log_path: Path) -> None:
        self.blank()
        self.line(f"┌─ solution {position} · {_short_id(sid)} · {host}")
        # Show the per-solution log path up front (not just at close) so a long/hung tt-run can be tailed live.
        self.line(f"│  log: {log_path}")

    def solution_end(self) -> None:
        self.line("└─")

    def solution_dry_run(self, retries: int, recover_cmd: str) -> None:
        self.line(f"│    dry-run: would run tt-run (up to {retries}×, recover on fail) → {recover_cmd}")

    # -- attempts + recover; ``indent`` is the tree prefix ("│    " under a solution, "    " at top level) --
    def attempt_start(self, indent: str, label: str, n: int, m: int) -> None:
        """Emit BEFORE the attempt runs, so a long/hung tt-run shows live activity instead of a frozen log."""
        self.line(f"{indent}{label} {n}/{m} … running")

    def attempt(self, indent: str, label: str, n: int, m: int, status: str, seconds: Optional[float]) -> None:
        res = self._STATUS.get(status, status.upper())
        if label == "recover" and status == "pass":
            res = "OK"  # a recover "passing" reads better as OK
        dur = f"  {seconds:.1f}s" if seconds is not None else ""
        self.line(f"{indent}{label} {n}/{m} → {res}{dur}")

    def recover_start(self, indent: str, host_set: Optional[str]) -> None:
        where = f" {_short_host(host_set)}" if host_set else ""
        self.line(f"{indent}↻ reap{where} + recover → retry")

    def solution_failed(self, indent: str, attempts: int) -> None:
        self.line(f"{indent}✗ FAILED after {attempts} attempt(s)")

    def unrecoverable(self, label: str, retries: int, rc: Optional[int]) -> None:
        self.blank()
        self.line(f"✗ UNRECOVERABLE: recover failed after {retries} attempt(s) (rc={rc}); halting after {label}.")

    def stop_on_failure(self, label: str) -> None:
        self.line(f"■ --stop-on-failure: halting after {label}.")


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


def _build_producer_cmd(
    *,
    mesh_graph_descriptor: Path,
    hosts: Optional[List[str]],
    mock_cluster_rank_binding: Optional[Path],
    output_dir: Path,
    max_solutions: int,
    distinct_host_sets: bool,
    allow_shape_permutations: bool,
    mpi_args: Optional[List[str]],
    tcp_interface: Optional[str] = None,
) -> List[str]:
    """Build the ``generate_rank_bindings --all-solutions`` command (the streaming *producer*).

    The producer runs ONCE in the background and streams each solution to ``output_dir`` -- writing a
    per-solution subdir and rewriting ``solutions_index.yaml`` as the solver finds each one (see the
    streaming enumerator in generate_rank_bindings.cpp). The consumer loop below picks up solutions as
    they appear, so a tt-run can run on solution k while solution k+1 is still being searched for.
    """
    executable = find_generate_rank_bindings_executable()
    output_dir.mkdir(parents=True, exist_ok=True)

    mock_rank_to_desc: Optional[Dict[int, Path]] = None
    if mock_cluster_rank_binding is not None:
        mock_rank_to_desc = load_mock_rank_to_descriptors(mock_cluster_rank_binding.resolve())

    # The producer is its own mpirun job (independent of the per-solution tt-runs, which get
    # --tcp-interface via tt-run's flag synthesis). Without these flags the producer relies on
    # OMPI_MCA_* environment variables and multi-host runs fail interface selection
    # ("server accept cannot find guid"). Mirror the flags tt-run generates.
    if tcp_interface:
        # Prepend so explicitly-passed --mpi-args keep override precedence (later flags win in
        # Open MPI), matching tt-run's ordering of synthesized defaults vs user args.
        mpi_args = [
            "--mca",
            "btl",
            "self,tcp",
            "--mca",
            "btl_tcp_if_include",
            tcp_interface,
            "--prtemca",
            "oob_tcp_if_include",
            tcp_interface,
        ] + list(mpi_args or [])

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
    return _inject_solution_flags(cmd, extra)


class SolutionProducer:
    """Handle to the background solution-generation MPI job (the *producer*).

    Runs generate_rank_bindings in its own session so its whole process tree can be reaped as a group.
    The producer streams solutions to disk; the consumer polls the index for them. Because the producer
    is a live MPI job on the same hosts as the tt-runs, two consumer operations must be producer-aware:
      * per-tt-run reap -> pass spare_daemons=alive() so the blanket prted/mpirun kill is skipped;
      * whole-cluster recover -> settle_for_recover() first, so the reset does not disrupt a live producer.
    """

    def __init__(self, proc: Optional[subprocess.Popen]):
        self.proc = proc
        self.pgid: Optional[int] = None
        if proc is not None:
            try:
                self.pgid = os.getpgid(proc.pid)
            except (ProcessLookupError, OSError):
                self.pgid = None

    @classmethod
    def start(cls, cmd: List[str], *, cwd: Path, log_path: Path) -> "SolutionProducer":
        """Launch the producer in the background in its own session and return a handle to it.

        The full generate_rank_bindings command is written into the producer log's header (self-describing);
        the caller announces the phase on the console via SweepLog.generating()."""
        log_fh = open(log_path, "w")  # noqa: SIM115 (kept open for the life of the producer)
        log_fh.write("# producer: " + " ".join(shlex.quote(c) for c in cmd) + "\n")
        log_fh.flush()
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, start_new_session=True)
        return cls(proc)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def returncode(self) -> Optional[int]:
        return self.proc.poll() if self.proc is not None else None

    def stop(self) -> None:
        """SIGKILL the producer's whole process group (its mpirun + prted tree)."""
        if self.proc is None or self.proc.poll() is not None:
            return
        if self.pgid:
            try:
                os.killpg(self.pgid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        try:
            self.proc.wait(timeout=15)
        except Exception:  # noqa: BLE001
            pass

    def settle_for_recover(self) -> None:
        """Ensure the producer is not running before a whole-cluster recover (which would disrupt it).

        The producer's enumeration is a CPU SAT solve unaffected by the device state a recover fixes, so
        we simply wait for it to finish streaming its remaining solutions (bounded); if it overruns the
        bound we stop it (its already-streamed solutions on disk are kept and still get swept)."""
        if not self.alive():
            return
        click.echo(
            f"{PREFIX}   waiting up to {int(PRODUCER_SETTLE_BEFORE_RECOVER_S)}s for solution generation to "
            f"finish before recover (a cluster reset must not run alongside the live producer)..."
        )
        try:
            self.proc.wait(timeout=PRODUCER_SETTLE_BEFORE_RECOVER_S)
        except subprocess.TimeoutExpired:
            click.echo(f"{PREFIX}   producer still running after wait; stopping it so recover can proceed.")
            self.stop()


def _load_index(solutions_dir: Path) -> dict:
    index_path = solutions_dir / "solutions_index.yaml"
    if not index_path.is_file():
        raise click.ClickException(f"No solutions_index.yaml under {solutions_dir}. Run with --all-solutions first.")
    with open(index_path) as f:
        return yaml.safe_load(f)


def _read_index_safe(solutions_dir: Path) -> Optional[dict]:
    """Read solutions_index.yaml DEFENSIVELY for the streaming consumer: the producer rewrites it in
    place after every solution, so a mid-rewrite read may miss the file or hit a truncated document.
    Returns the parsed index, or None if it is not (yet) readable -- the caller just retries next poll."""
    index_path = solutions_dir / "solutions_index.yaml"
    try:
        if not index_path.is_file():
            return None
        with open(index_path) as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except (OSError, yaml.YAMLError):
        return None


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
    """Outcome of ``TtRunExecutor.run`` (one command's attempts, including any recover)."""

    status: str  # pass | fail | timeout
    returncode: Optional[int]
    attempts: int
    recover_returncode: Optional[int] = None
    recover_exhausted: bool = False


class LoopAction(Enum):
    """What the consumer loop should do after running one solution (see SolutionConsumer.run_solution)."""

    CONTINUE = auto()  # solution done (passed, or failed but keep going) -> run the next solution
    STOP = auto()  # solution failed and --stop-on-failure is set -> stop the sweep, report what ran
    UNRECOVERABLE = auto()  # recover exhausted its retries -> abort the whole sweep with an error


def _reap_pattern_for(program: List[str]) -> Optional[str]:
    """A distinctive substring of ``<program>`` used to pkill its worker ranks on the hosts.

    Prefers a pytest ``file.py::test`` target (unique to this workload), else the last program token.
    Returned as a ``pkill -f`` pattern. None -> reaping is local-only (no remote rank pattern known).
    """
    for tok in reversed(program):
        if "::" in tok:
            return tok
    return program[-1] if program else None


def _reap_command_processes(
    pgid: Optional[int], host_set: Optional[str], reap_pattern: Optional[str], spare_daemons: bool = False
) -> None:
    """Kill every process a launched command left behind — locally and on every host — then verify.

    A tt-run that fails/times out leaves worker ranks alive on the remote hosts; they keep holding the
    per-chip ``CHIP_IN_USE_*_PCIe`` locks, so the *next* attempt wedges on
    ``Waiting for lock 'CHIP_IN_USE_*'``. This SIGKILLs the local launcher process group
    (mpirun/prterun/ssh) and, on each host in ``host_set``, pkills the workload ranks + ``prted``,
    polling ``pgrep`` until they are actually gone so the next command starts clean. Runs after every
    command (timeout or normal exit). Best-effort; never raises.

    ``spare_daemons`` (set while the background solution *producer* -- an independent generate_rank_bindings
    MPI job -- is still running): skip the blanket by-NAME ``prted``/``mpirun`` kills, which would also kill
    the producer's daemons on these shared hosts. The tt-run's own process-GROUP kill (by pgid) and the
    remote kill of its worker ranks (by the distinctive pytest pattern) are producer-safe and still run --
    the ranks are what hold the CHIP_IN_USE locks, and killing the local launcher orphans this job's prted
    so they exit on their own. Once the producer has finished, the caller drops back to the full-strength reap.
    """
    # 1. Local: SIGKILL the command's whole process group (the launched parent -- tt-run/bash -- plus
    #    the ssh's it spawns), AND pkill the MPI launchers by name in case they setsid'd into their own
    #    session and escaped the process-group kill. Together this guarantees the parent process and all
    #    its child MPI processes on this host die. pkill -f on these names can't match the driver
    #    (python) or its own pkill, so it is self-match-safe.
    if pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if not spare_daemons:
        for launcher in ("mpirun-ulfm", "prterun", "prted"):
            try:
                subprocess.run(
                    ["pkill", "-9", "-f", launcher], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10
                )
            except (subprocess.TimeoutExpired, OSError):
                pass
    # 2. Remote worker ranks + prted on each host (these hold the CHIP_IN_USE PCIe locks).
    hosts = [h.strip() for h in (host_set or "").split(",") if h.strip()]
    if not hosts or not reap_pattern:
        return
    # Bracket the last char (the classic `[x]` trick) so the reap command's OWN shell -- whose cmdline
    # necessarily contains the pattern -- is NOT matched by pkill/pgrep -f, avoiding self-kill.
    bracketed = reap_pattern[:-1] + "[" + reap_pattern[-1] + "]" if reap_pattern else reap_pattern
    pat = shlex.quote(bracketed)
    # CRUCIAL: exclude THIS driver's own process group. host_set can include the login node this driver
    # runs on (the reap sshes there too), and the driver's cmdline carries the trailing pytest target, so
    # a blind `pkill -f <pattern>` would SIGKILL the driver itself. The worker ranks run in a *separate*
    # process group (tt-run uses start_new_session), so excluding the driver's pgid still reaps them. The
    # exclusion only ever matches on the login node; on remote hosts no process has this pgid.
    self_pgid = os.getpgid(0)
    # Kill workload ranks by cmdline, pid-by-pid, skipping anything in the driver's own process group.
    pattern_kill = (
        f"for pid in $(pgrep -f {pat} 2>/dev/null); do "
        f'pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d " "); '
        f'[ "$pg" = "{self_pgid}" ] && continue; '
        f'kill -9 "$pid" 2>/dev/null; done'
    )
    # Count survivors the same way (excluding the driver's group) so the login-node poll can reach 0.
    pattern_count = (
        f"c=0; for pid in $(pgrep -f {pat} 2>/dev/null); do "
        f'pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d " "); '
        f'[ "$pg" = "{self_pgid}" ] && continue; c=$((c+1)); done; echo "$c"'
    )
    # prted by exact name (-x) UNLESS sparing the producer's daemons: while the producer is alive, killing
    # prted would take the producer down too, so we rely on the pattern kill + the local process-group kill.
    prted_kill = "" if spare_daemons else "pkill -9 -x prted >/dev/null 2>&1; "
    kill_sh = f"{prted_kill}{pattern_kill}; true"
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=no"]
    for h in hosts:
        # Kill, then poll (bounded) until the ranks are gone, re-killing each round.
        for _ in range(8):
            try:
                subprocess.run(ssh + [h, kill_sh], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
                chk = subprocess.run(ssh + [h, pattern_count], capture_output=True, text=True, timeout=25)
            except subprocess.TimeoutExpired:
                break
            if (chk.stdout or "").strip() in ("", "0"):
                break
            time.sleep(1)


def _run_once(
    cmd: List[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: Optional[int],
    append: bool,
    header: Optional[str] = None,
    host_set: Optional[str] = None,
    reap_pattern: Optional[str] = None,
    spare_daemons_fn: Optional[Callable[[], bool]] = None,
) -> Tuple[str, Optional[int]]:
    """Run ``cmd`` once in its own process group, then reap everything it started (local group +
    remote ranks) so nothing lingers holding CHIP_IN_USE locks. Returns
    ``(status, returncode, workload_seconds)`` where workload_seconds times ONLY the workload wait
    (excludes the post-run reap/zombie-collect) so it is directly comparable to ``timeout``.

    ``spare_daemons_fn`` is evaluated at reap time: when it returns True (the background producer is
    still running), the reap skips the blanket prted/mpirun kill so the producer's MPI daemons survive."""
    mode = "a" if append else "w"
    status: str = "fail"
    rc: Optional[int] = None
    pgid: Optional[int] = None
    proc: Optional[subprocess.Popen] = None
    with open(log_path, mode) as log:
        if header:
            log.write(header)
            log.flush()
        # start_new_session=True -> the command is its own process-group/session leader, so we can
        # SIGKILL the whole tree (mpirun/prterun/ssh) on timeout, not just the direct child.
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            pgid = None
        # Time ONLY the workload wait, so the reported duration is directly comparable to `timeout`
        # (the post-run reap + zombie-collect below can add ~15s and would otherwise make a PASS look
        # like it ran past its timeout).
        t_wait0 = time.time()
        try:
            rc = proc.wait(timeout=timeout)
            status = "pass" if rc == 0 else "fail"
        except subprocess.TimeoutExpired:
            status = "timeout"
        workload_seconds = round(time.time() - t_wait0, 1)
    # Always reap this command's processes (on timeout AND on normal exit); cheap no-op on a clean pass.
    # While the producer is alive, spare its shared MPI daemons (evaluate now, not before the run).
    _reap_command_processes(pgid, host_set, reap_pattern, spare_daemons=bool(spare_daemons_fn and spare_daemons_fn()))
    if proc is not None:
        try:
            proc.wait(timeout=15)  # collect the (now-killed) child so it isn't left a zombie
        except Exception:  # noqa: BLE001
            pass
    return status, rc, workload_seconds


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
    host_set: Optional[str] = None,
    reap_pattern: Optional[str] = None,
    producer: Optional["SolutionProducer"] = None,
    log: Optional["SweepLog"] = None,
    indent: str = "    ",
) -> _RetryResult:
    """Run ``cmd`` up to ``retries`` times, recovering only after a failure.

    * Pass on any attempt is recorded as pass; recover is **not** run after success.
    * Fail/timeout: sleep ``RETRY_DELAY_S``, run ``recover_cmd`` (itself retried),
      sleep again, then retry ``cmd``. If ``recover_cmd`` is omitted, just sleep
      and retry ``cmd`` (used for recover itself).
    * Fail/timeout is returned only if every attempt fails. Recover exhausting
      its retries sets ``recover_exhausted`` and stops immediately.

    Progress is reported through ``log`` (a SweepLog) at ``indent`` (the tree prefix). The recover's own
    attempts are shown one level deeper.
    """
    log = log or SweepLog()
    attempts = max(1, retries)
    last_status = "fail"
    last_rc: Optional[int] = None
    recover_rc: Optional[int] = None

    for attempt in range(1, attempts + 1):
        append = append_log or attempt > 1
        # Header goes into the per-attempt LOG FILE (self-describing), not the console.
        display = cmd_display or " ".join(shlex.quote(c) for c in cmd)
        if attempt == 1 and not append_log:
            header = f"# {label}: {display}\n"
        else:
            header = f"\n# {label} attempt {attempt}/{attempts}\n"

        log.attempt_start(indent, label, attempt, attempts)  # live "… running" before the (maybe long) attempt
        last_status, last_rc, last_seconds = _run_once(
            cmd,
            cwd=cwd,
            log_path=log_path,
            timeout=timeout,
            append=append,
            header=header,
            host_set=host_set,
            reap_pattern=reap_pattern,
            spare_daemons_fn=(producer.alive if producer is not None else None),
        )
        # last_seconds is the workload-only wait (excludes reap/cleanup), so it's comparable to `timeout`.
        log.attempt(indent, label, attempt, attempts, last_status, last_seconds)
        if last_status == "pass":
            return _RetryResult(status=last_status, returncode=last_rc, attempts=attempt, recover_returncode=recover_rc)

        if recover_cmd and attempt < attempts:
            # Recover only when we'll actually retry -- no point resetting the cluster after the final
            # failed attempt (the per-attempt reap already freed its ranks; the next solution self-heals
            # by recovering if IT fails). A recover is a whole-cluster reset, so make sure the background
            # producer is not running alongside it (it would be disrupted); its streamed solutions are kept.
            if producer is not None:
                producer.settle_for_recover()
            log.recover_start(indent, host_set)
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
                log=log,
                indent=indent + "  ",
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
            time.sleep(RETRY_DELAY_S)

    return _RetryResult(
        status=last_status,
        returncode=last_rc,
        attempts=attempts,
        recover_returncode=recover_rc,
    )


class TtRunExecutor:
    """Runs one command with retries + reap + recover, holding the stable per-sweep config as fields.

    Groups the process-management primitives (run once in its own session, reap everything it started on
    every host, retry with a recover in between) behind one object so call sites pass only what varies --
    the command plus its log/timeout/host_set. ``producer`` is a field, not a threaded argument: while it
    is alive the reap spares its shared MPI daemons, and a recover settles it first (see SolutionProducer).
    The ``reap()``/``run()`` methods delegate to the module-private ``_reap_command_processes`` /
    ``_run_with_retries`` implementations, which own the (correctness-critical) kill/retry mechanics.
    """

    def __init__(
        self,
        *,
        cwd: Path,
        retries: int,
        log: "SweepLog",
        reap_pattern: Optional[str] = None,
        producer: Optional[SolutionProducer] = None,
    ):
        self.cwd = cwd
        self.retries = retries
        self.log = log
        self.reap_pattern = reap_pattern
        self.producer = producer  # set after the producer is started; drives producer-aware reap/recover

    def reap(self, pgid: Optional[int], host_set: Optional[str]) -> None:
        """Reap a command's processes locally + on ``host_set``; spares the producer's daemons if it's alive."""
        _reap_command_processes(
            pgid,
            host_set,
            self.reap_pattern,
            spare_daemons=bool(self.producer is not None and self.producer.alive()),
        )

    def run(
        self,
        cmd: List[str],
        *,
        log_path: Path,
        timeout: Optional[int],
        indent: str = "    ",
        recover_cmd: Optional[str] = None,
        label: str = "tt-run",
        append_log: bool = False,
        cmd_display: Optional[str] = None,
        host_set: Optional[str] = None,
    ) -> _RetryResult:
        """Run ``cmd`` up to ``retries`` times, reaping after each attempt and recovering between failures."""
        return _run_with_retries(
            cmd,
            cwd=self.cwd,
            log_path=log_path,
            timeout=timeout,
            retries=self.retries,
            recover_cmd=recover_cmd,
            label=label,
            append_log=append_log,
            cmd_display=cmd_display,
            host_set=host_set,
            reap_pattern=self.reap_pattern,
            producer=self.producer,
            log=self.log,
            indent=indent,
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
    # The console combo-summary + report line are emitted by the caller via SweepLog (this returns the data).
    return report_path, passed, failed, timed_out


def _sol_key(sol: dict) -> str:
    """Stable identity of a solution in the index (its content-hash id, or its dir name as a fallback)."""
    return sol.get("id") or sol.get("dir")


def _build_passthrough(
    *,
    rankfile_syntax: Optional[str],
    tcp_interface: Optional[str],
    bare: bool,
    tracy_args: Optional[str],
    debug_gdbserver: bool,
    skip_executable_check: bool,
    skip_mgd_check: bool,
    verbose: bool,
) -> List[str]:
    """The tt-run passthrough args forwarded verbatim to every per-solution launch."""
    out: List[str] = []
    if rankfile_syntax and rankfile_syntax != "auto":
        out += ["--rankfile-syntax", rankfile_syntax]
    if tcp_interface:
        out += ["--tcp-interface", tcp_interface]
    if bare:
        out += ["--bare"]
    if tracy_args is not None:
        out += ["--tracy", tracy_args]
    if debug_gdbserver:
        out += ["--debug-gdbserver"]
    if skip_executable_check:
        out += ["--skip-executable-check"]
    if skip_mgd_check:
        out += ["--skip-mgd-check"]
    if verbose:
        out += ["-v"]
    return out


@dataclass
class SweepConfig:
    """The parsed, stable configuration for one sweep -- everything the consumer needs but nothing that varies
    per solution. Built once in main() from the CLI, then passed to SolutionConsumer."""

    tt_run: str
    sol_dir: Path
    program: List[str]
    mock: bool
    mpi_args: Optional[List[str]]
    passthrough: List[str]
    dry_run: bool
    per_solution_timeout: Optional[int]
    recover_command: str
    retries: int
    stop_on_failure: bool
    logs_root: Path
    select: Optional[str]
    limit: Optional[int]
    sweep_timeout: Optional[int]


class SolutionConsumer:
    """Runs tt-runs ONE AT A TIME (serialized) across solutions, pulling them as they become available.

    The counterpart to SolutionProducer. Solutions come either from a fixed ``--solutions-dir`` list
    (``static_solutions``) or from the streaming index the producer writes as it enumerates -- the loop is
    identical either way. Owns the run's mutable state (``results``, ``stopped_early``, ``recover_exhausted``)
    so main() stays a thin wiring layer.
    """

    def __init__(
        self,
        cfg: SweepConfig,
        executor: TtRunExecutor,
        *,
        static_solutions: Optional[List[dict]],
        producer: Optional[SolutionProducer],
        sweep_start: float,
    ):
        self.cfg = cfg
        self.executor = executor
        self.log = executor.log
        self.static_solutions = static_solutions
        self.producer = producer
        self.sweep_start = sweep_start
        self.results: List[dict] = []
        self.stopped_early = False
        self.recover_exhausted = False

    def _available(self) -> List[dict]:
        """Solutions on offer right now: the fixed list, or the producer's streaming index (may be empty)."""
        if self.static_solutions is not None:
            return self.static_solutions
        idx = _read_index_safe(self.cfg.sol_dir)
        return _select_solutions(idx, self.cfg.select, self.cfg.limit) if idx else []

    def run_solution(self, sol: dict, position_label: str) -> LoopAction:
        """Run one solution's tt-run (retries/reap/recover) and record it. Returns the next loop action."""
        cfg = self.cfg
        cmd = _build_tt_run_cmd(
            tt_run=cfg.tt_run,
            solutions_dir=cfg.sol_dir,
            sol=sol,
            program=cfg.program,
            mock=cfg.mock,
            mpi_args=cfg.mpi_args,
            passthrough=cfg.passthrough,
        )
        label = sol.get("id", sol["dir"])
        cmd_str = " ".join(shlex.quote(c) for c in cmd)  # exact, copy-paste-reproducible tt-run command (-> log file)
        host = _short_host(sol.get("host_set"))
        log_path = cfg.logs_root / f"{label}.log"
        self.log.solution_start(position_label, label, host, log_path)

        if cfg.dry_run:
            self.log.solution_dry_run(cfg.retries, cfg.recover_command)
            self.results.append(
                _make_result(
                    sol=sol,
                    solutions_dir=cfg.sol_dir,
                    label=label,
                    cmd_str=cmd_str,
                    status="dry-run",
                    returncode=None,
                    duration_seconds=0.0,
                    tt_run_attempts=0,
                    recover_command=cfg.recover_command,
                )
            )
            return LoopAction.CONTINUE

        t0 = time.time()
        # After every attempt the executor reaps this solution's ranks on its hosts (sparing the producer's
        # daemons while it is alive), and recovers between failures -- see TtRunExecutor / SolutionProducer.
        # Its per-attempt tt-run/recover lines print under this solution's tree via the `│    ` indent.
        outcome = self.executor.run(
            cmd,
            log_path=log_path,
            timeout=cfg.per_solution_timeout,
            indent="│    ",
            recover_cmd=cfg.recover_command,
            label="tt-run",
            cmd_display=cmd_str,
            host_set=sol.get("host_set"),
        )
        dur = round(time.time() - t0, 1)
        if outcome.status != "pass" and not outcome.recover_exhausted:
            self.log.solution_failed("│    ", outcome.attempts)
        self.log.solution_end()
        self.results.append(
            _make_result(
                sol=sol,
                solutions_dir=cfg.sol_dir,
                label=label,
                cmd_str=cmd_str,
                status=outcome.status,
                returncode=outcome.returncode,
                duration_seconds=dur,
                tt_run_attempts=outcome.attempts,
                recover_command=cfg.recover_command if outcome.recover_returncode is not None else None,
                recover_returncode=outcome.recover_returncode,
                log_path=log_path,
            )
        )
        # Recover failure is unrecoverable: abort regardless of --stop-on-failure (that flag only governs a
        # *workload* fail/timeout after all tt-run retries).
        if outcome.recover_exhausted:
            self.log.unrecoverable(label, cfg.retries, outcome.recover_returncode)
            return LoopAction.UNRECOVERABLE
        if outcome.status != "pass" and cfg.stop_on_failure:
            self.log.stop_on_failure(label)
            return LoopAction.STOP
        return LoopAction.CONTINUE

    def consume(self) -> List[dict]:
        """The single control loop: pull the next unconsumed solution and run it, until exhausted/stopped.

        tt-runs run one at a time (serialized). In streaming mode, wait (with a heartbeat) while the producer
        is still generating and nothing new is available. Returns the per-solution results."""
        cfg = self.cfg
        consumed: set = set()
        found_count = 0  # streaming: how many solutions the producer has generated so far (for the "N found" note)
        producer_exited_at: Optional[float] = None  # set on first post-exit idle read; anchors the NFS settle window
        last_heartbeat = time.time()
        while True:
            # Total-budget check (before launching, so we never interrupt a running solve).
            if cfg.sweep_timeout is not None and (time.time() - self.sweep_start) >= cfg.sweep_timeout:
                self.log.line(
                    f"■ --sweep-timeout ({cfg.sweep_timeout}s) reached after "
                    f"{round(time.time() - self.sweep_start, 1)}s; stopping with {len(consumed)} solution(s) swept."
                )
                self.stopped_early = True
                break
            # --limit is a hard cap on how many solutions we run (the streaming index keeps growing otherwise).
            if cfg.limit is not None and len(consumed) >= cfg.limit:
                break

            avail = self._available()
            # Streaming: note each time the producer has generated more solutions. This only runs between
            # solution trees (the consumer polls the index only when idle), so it never splits a tree.
            if self.producer is not None and len(avail) > found_count:
                found_count = len(avail)
                self.log.found(found_count)
            new = [s for s in avail if _sol_key(s) not in consumed]
            if new:
                sol = new[0]
                consumed.add(_sol_key(sol))
                # Streaming: the total is unknown mid-run (the producer keeps finding more), so show just the
                # running index. Static --solutions-dir: the list is fixed, so show n/total.
                position = (
                    f"{len(consumed)}/{len(self.static_solutions)}"
                    if self.static_solutions is not None
                    else str(len(consumed))
                )
                action = self.run_solution(sol, position)
                if action is LoopAction.UNRECOVERABLE:
                    self.recover_exhausted = True
                    break
                if action is LoopAction.STOP:
                    break
                continue

            # Nothing new to consume yet.
            if self.producer is not None and self.producer.alive():
                if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL_S:
                    self.log.line(f"… generating: {len(consumed)} swept, {len(avail)} found so far")
                    last_heartbeat = time.time()
                time.sleep(POLL_INTERVAL_S)
                continue
            # Producer-exit settle: the producer may run on a different NFS client than this consumer,
            # and an index read right after its exit can be served from a stale attribute cache
            # (close-to-open consistency lag, up to ~60s) -- hiding trailing solutions, or wrongly
            # concluding 0. Keep polling (through the normal loop) until the settle window from
            # producer exit expires, clipped to --sweep-timeout. Fast path: if the freshest read
            # shows every enumerated solution consumed, the stream is complete -- no wait.
            if self.producer is not None:
                if consumed and len(avail) == len(consumed):
                    break  # index fully consumed as of the freshest read
                if producer_exited_at is None:
                    producer_exited_at = time.time()
                settle_deadline = producer_exited_at + PRODUCER_EXIT_INDEX_SETTLE_S
                if cfg.sweep_timeout is not None:
                    settle_deadline = min(settle_deadline, self.sweep_start + cfg.sweep_timeout)
                if time.time() < settle_deadline:
                    time.sleep(PRODUCER_EXIT_INDEX_POLL_INTERVAL_S)
                    continue
                if not consumed:
                    self.log.line(
                        f"⚠ producer exited and the solutions index stayed empty through the "
                        f"{PRODUCER_EXIT_INDEX_SETTLE_S:.0f}s settle window; reporting 0 solutions. "
                        f"If the producer log shows solutions were found, this is NFS index-visibility lag."
                    )
            break  # producer finished (or none) and nothing new left to consume
        return self.results


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

    # 1. Obtain solutions: either sweep an existing --solutions-dir (a fixed, already-generated list) or
    #    generate them with a background STREAMING producer that the consumer loop picks up incrementally.
    producer_cmd: Optional[List[str]] = None
    static_solutions: Optional[List[dict]] = None  # set only when a --solutions-dir was given
    if solutions_dir is not None:
        sol_dir = Path(solutions_dir).resolve()
        static_solutions = _select_solutions(_load_index(sol_dir), select, limit)
        if not static_solutions:
            raise click.ClickException(f"No solutions to sweep in {sol_dir} (after --select/--limit).")
        # Auto-detect mock mode when sweeping an existing solutions dir (per-solution phase2_mock_mapping.yaml).
        if not mock and (sol_dir / static_solutions[0]["dir"] / "phase2_mock_mapping.yaml").is_file():
            mock = True
    else:
        if mesh_graph_descriptor is None:
            raise click.ClickException("Provide --solutions-dir, or --mesh-graph-descriptor to generate solutions.")
        if not mock and not parsed_hosts:
            raise click.ClickException("New mode needs --hosts (real cluster) or --mock-cluster-rank-binding (mock).")
        sol_dir = (
            Path(solutions_output_dir).resolve() if solutions_output_dir else (_repo_root() / "generated/ttrun/sweep")
        )
        producer_cmd = _build_producer_cmd(
            mesh_graph_descriptor=Path(mesh_graph_descriptor),
            hosts=parsed_hosts,
            mock_cluster_rank_binding=Path(mock_cluster_rank_binding) if mock else None,
            output_dir=sol_dir,
            max_solutions=max_solutions,
            distinct_host_sets=distinct_host_sets,
            allow_shape_permutations=allow_shape_permutations,
            mpi_args=parsed_mpi_args,
            tcp_interface=tcp_interface,
        )
        if dry_run:
            click.echo(f"{PREFIX} Producer (stream solutions):\n  {' '.join(shlex.quote(c) for c in producer_cmd)}")
            click.echo(f"{PREFIX} --dry-run: would stream solutions and sweep each as it appears; nothing executed.")
            return

    tt_run = _find_tt_run()
    logs_root = Path(log_dir).resolve() if log_dir else (sol_dir / "sweep_logs")
    if not dry_run:
        logs_root.mkdir(parents=True, exist_ok=True)

    cfg = SweepConfig(
        tt_run=tt_run,
        sol_dir=sol_dir,
        program=program,
        mock=mock,
        mpi_args=parsed_mpi_args,
        passthrough=_build_passthrough(
            rankfile_syntax=rankfile_syntax,
            tcp_interface=tcp_interface,
            bare=bare,
            tracy_args=tracy_args,
            debug_gdbserver=debug_gdbserver,
            skip_executable_check=skip_executable_check,
            skip_mgd_check=skip_mgd_check,
            verbose=verbose,
        ),
        dry_run=dry_run,
        per_solution_timeout=per_solution_timeout,
        recover_command=recover_command,
        retries=retries,
        stop_on_failure=stop_on_failure,
        logs_root=logs_root,
        select=select,
        limit=limit,
        sweep_timeout=sweep_timeout,
    )

    # SweepLog owns all console formatting at this (per-combo) level; one executor owns the
    # process-management config (cwd/retries/reap-pattern) and, once set, the producer.
    log = SweepLog()
    log.combo_start(sol_dir, recover_command, retries)
    executor = TtRunExecutor(cwd=_repo_root(), retries=retries, log=log, reap_pattern=_reap_pattern_for(program))

    # PROACTIVE reset BEFORE the producer starts and BEFORE any tt-run: reap leftover ranks on every host
    # (they hold CHIP_IN_USE PCIe locks), then run the recover command once so both the producer's device
    # discovery and the first tt-run launch on a clean cluster. Same command used reactively after a failure.
    if not dry_run:
        executor.reap(None, ",".join(parsed_hosts) if parsed_hosts else None)
        if recover_command and recover_command != "true":
            log.initial_reset()
            reset = executor.run(
                ["/bin/bash", "-c", recover_command],
                log_path=logs_root / "_initial_recover.log",
                timeout=None,
                label="recover",
                cmd_display=recover_command,
            )
            # Don't sweep on hardware we've already classified as unrecoverable: if the initial reset failed
            # every retry, abort before starting the producer / any tt-run.
            if reset.status != "pass":
                raise click.ClickException(
                    f"Initial cluster recovery failed (recover rc={reset.returncode}) after {retries} attempt(s); "
                    f"refusing to sweep on unrecoverable hardware. See {logs_root / '_initial_recover.log'}."
                )

    # Start the streaming producer AFTER the reset (clean cluster). Clear any stale index first so the
    # consumer waits for THIS run's solutions. No producer in --solutions-dir mode (list already on disk).
    producer: Optional[SolutionProducer] = None
    if producer_cmd is not None and not dry_run:
        try:
            (sol_dir / "solutions_index.yaml").unlink()
        except FileNotFoundError:
            pass
        log.generating(logs_root / "_producer.log")
        producer = SolutionProducer.start(producer_cmd, cwd=_repo_root(), log_path=logs_root / "_producer.log")
        executor.producer = producer  # from here the reap spares it while alive; a recover settles it first

    # 2. Consume: run tt-runs one at a time as solutions become available (serialized).
    consumer = SolutionConsumer(
        cfg, executor, static_solutions=static_solutions, producer=producer, sweep_start=sweep_start
    )
    results = consumer.consume()

    # Stop the producer if it is still running (early stop via --limit / --stop-on-failure / --sweep-timeout).
    if producer is not None:
        producer_rc = producer.returncode()
        if producer.alive():
            log.line("■ stopping producer (sweep ending)")
            producer.stop()
        elif producer_rc not in (None, 0):
            # Producer exited non-zero on its OWN (a crash -- us stopping it leaves it alive, handled above).
            # Generation is therefore incomplete, so this is an error even if some solutions were already
            # swept -- UNLESS the user intentionally bounded the sweep (--limit / --stop-on-failure /
            # --sweep-timeout), in which case an incomplete generation is expected and we just warn.
            intentional = consumer.stopped_early or stop_on_failure or (limit is not None)
            msg = f"Solution generation failed (generate_rank_bindings exit {producer_rc}); sweep incomplete"
            if not results:
                raise click.ClickException(msg + " (nothing swept).")
            if not intentional:
                raise click.ClickException(msg + f" ({len(results)} swept before the failure).")
            log.line(f"■ WARNING: {msg}; kept {len(results)} swept (intentional early stop).")

    # 3. Report. Final solution set + index: the fixed list, or everything the producer streamed to disk.
    index = _read_index_safe(sol_dir) or {"solutions": []}
    solutions = static_solutions if static_solutions is not None else _select_solutions(index, select, limit)
    report_path, passed, failed, timed_out = _write_sweep_report(
        results=results,
        solutions=solutions,
        index=index,
        sol_dir=sol_dir,
        program=program,
        recover_command=recover_command,
        stopped_early=consumer.stopped_early,
        sweep_report=sweep_report,
        dry_run=dry_run,
    )
    log.combo_summary(sol_dir.name, len(results), passed, failed, timed_out)
    if not dry_run:
        log.report(report_path)
    # Stopping early on --sweep-timeout is NOT an error -- return what we swept. A sweep that found 0
    # solutions has already exited non-zero above (empty --solutions-dir, or a failed producer). Real
    # per-solution failures/timeouts still surface as a non-zero exit. Recover exhausting its retries
    # is a hard error: the machine could not be brought back, so later solutions would be untrustworthy.
    if consumer.recover_exhausted:
        raise click.ClickException(
            f"UNRECOVERABLE: hardware cannot recover with this command "
            f"(--recover-command failed after {retries} attempt(s)). Sweep aborted. See {report_path}."
        )
    # A sweep that ran NO workload must not report success (e.g. --select matched nothing, or streaming
    # produced nothing) -- automation would read exit 0 as "passed". An intentional --sweep-timeout stop is
    # exempt (it returns what it swept); a failed producer already raised above.
    if not dry_run and not results and not consumer.stopped_early:
        raise click.ClickException(
            "No solutions were swept (empty --select match, or none generated); refusing to report success."
        )
    if not dry_run and (failed or timed_out):
        sys.exit(1)


if __name__ == "__main__":
    main()
