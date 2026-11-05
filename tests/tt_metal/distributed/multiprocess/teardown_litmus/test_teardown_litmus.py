# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown litmus: does a distributed tt-metal job end, end cleanly, and report it, after one failure?

CPU only, on the mock dual galaxy (two ranks of 8x4, no device), launched by tt-run like every real job. Every
test is one fault with one assertion, taken from the teardown plan of tt-metal #56902 (tt-blaze
tools/teardown_probe, lane L4): the rank program is ``litmus_probe.py`` next to this file. The faults are the
two kinds a real job meets: the code failing on one rank (an exception, a shape the descriptor cannot hold, a
descriptor that disagrees, a collective with a wrong or dead peer) and the user (a signal at tt-run).

Requirements, as in the plan: a failure never becomes a hang (R1), a hang ends by itself and leaves nothing
(R2), nobody kills processes by hand (R3), the exit code reports the failure (R4).

Modes. A row that passes today is a gate. A row that fails today is marked ``xfail(strict=True)`` with the
plan's error id in the reason: it still runs and records, and when the fix lands the unexpected pass fails
the job, which is the signal to drop the marker. Each row has a deadline; a job that is still running at the
deadline is killed by its process group and counted as a hang.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_litmus -v   (from the repository root, with a
built tree and the python environment; the fabric CPU lane runs it as group ``teardown-litmus`` of
tests/scripts/multihost/run_fabric_cpu_only_unit_tests.sh).
"""

from __future__ import annotations

import glob
import os
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("TT_METAL_HOME", HERE.parents[4]))
PROBE = HERE / "litmus_probe.py"
BINDING = ROOT / "tests/tt_metal/distributed/config/dual_galaxy_rank_bindings.yaml"
MAPPING = (
    ROOT / "tt_metal/third_party/tt-cluster-descriptors/wormhole/6u_dual_host/6u_dual_host_cluster_desc_mapping.yaml"
)
T3K_DESC = "tt_metal/third_party/tt-cluster-descriptors/wormhole/t3k_cluster_desc/t3k_cluster_desc.yaml"
MGD = ROOT / "tt_metal/fabric/mesh_graph_descriptors/dual_galaxy_mesh_graph_descriptor.textproto"
PYTHON = os.environ.get("TT_LITMUS_PYTHON", "python3")
TTRUN = shutil.which("tt-run") or "tt-run"

# a row that must end gets this long; a row expected to hang today is given up on sooner, it is the whole budget
DEADLINE_END = float(os.environ.get("TT_LITMUS_DEADLINE", "90"))
DEADLINE_HANG = float(os.environ.get("TT_LITMUS_DEADLINE_HANG", "60"))
SETTLE_S = 3.0  # after the launcher returns, before the census

LAUNCHER_COMMS = {"prterun", "prted", "mpirun", "mpirun-ulfm", "tt-run"}
RANK_MARK = re.compile(r"\[probe r=(\d+) pid=\d+ t=[\d.]+\] (.*)")


# ---------------------------------------------------------------------------------------------------------------
# census: what of this job's kind is alive, and what it left in /dev/shm
# ---------------------------------------------------------------------------------------------------------------
def _procs() -> dict:
    """pid -> (comm, args) of this user's processes that belong to a launcher or run the probe or the solver."""
    out = {}
    uid = os.getuid()
    for p in glob.glob("/proc/[0-9]*"):
        try:
            if os.stat(p).st_uid != uid:
                continue
            comm = open(p + "/comm").read().strip()
            args = open(p + "/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if (
            comm in LAUNCHER_COMMS or "litmus_probe.py" in args or comm.startswith("generate_rank_b")
        ):  # comm is 15 chars
            out[int(p.rsplit("/", 1)[1])] = (comm, args[:120])
    return out


def _shm() -> set:
    uid = os.getuid()
    return {f for f in glob.glob("/dev/shm/sm_segment.*") if re.search(rf"\.{uid}\.", f)}


@dataclass
class Result:
    name: str
    cmd: str
    exit: int | None
    wall_s: float
    timed_out: bool
    signal_sent: str
    alive_at_deadline: list
    leftovers: list  # processes still alive SETTLE_S after the launcher returned (or was killed)
    shm_leaked: list
    reached: dict = field(default_factory=dict)  # rank -> last marker
    rcs: dict = field(default_factory=dict)  # rank -> [non-success rc lines]
    log: str = ""

    def __str__(self) -> str:
        head = "HANG (killed at %.0f s)" % self.wall_s if self.timed_out else f"exit {self.exit} in {self.wall_s:.1f} s"
        bits = [head]
        if self.signal_sent:
            bits.append(self.signal_sent)
        bits.append("reached " + ", ".join(f"r{r}:{m}" for r, m in sorted(self.reached.items())))
        for r, lines in sorted(self.rcs.items()):
            bits.append(f"r{r}: " + "; ".join(lines))
        bits.append("left: " + (", ".join(self.leftovers) if self.leftovers else "nothing"))
        if self.shm_leaked:
            bits.append("shm leaked: %d" % len(self.shm_leaked))
        return " | ".join(bits)


def _parse(out: str) -> tuple:
    reached, rcs = {}, {}
    for line in out.splitlines():
        m = RANK_MARK.search(line)
        if not m:
            continue
        r, text = int(m.group(1)), m.group(2).strip()
        reached[r] = text.split(" ", 1)[0]
        if "_rc=" in text and "SUCCESS" not in text:
            rcs.setdefault(r, []).append(text[:140])
    return reached, rcs


# ---------------------------------------------------------------------------------------------------------------
# launching
# ---------------------------------------------------------------------------------------------------------------
def legacy_cmd(mpi_extra: str = "", mapping: Path = MAPPING) -> list:
    """tt-run in legacy mode: the rank binding and the mock descriptor per rank, no solve."""
    return [
        TTRUN,
        "--rank-binding",
        str(BINDING),
        "--mock-cluster-rank-binding",
        str(mapping),
        "--mpi-args",
        ("--allow-run-as-root --oversubscribe " + mpi_extra).strip(),
        PYTHON,
        str(PROBE),
    ]


def solve_cmd(mpi_extra: str = "") -> list:
    """tt-run in new mode: phase 1 (generate_rank_bindings on the mock descriptors) then the ranks."""
    return [
        TTRUN,
        "--mesh-graph-descriptor",
        str(MGD),
        "--mock-cluster-rank-binding",
        str(MAPPING),
        "--mpi-args",
        ("--allow-run-as-root --oversubscribe " + mpi_extra).strip(),
        "--force-rediscovery",
        PYTHON,
        str(PROBE),
    ]


def conflict_mapping(tmp: Path) -> Path:
    """The dual-galaxy mapping with rank 1's descriptor replaced by a T3K: two ranks that disagree about the machine."""
    text = MAPPING.read_text()
    lines = [(f'  1: "{T3K_DESC}"' if line.strip().startswith('1: "') else line) for line in text.splitlines()]
    p = tmp / "conflict_mapping.yaml"
    p.write_text("\n".join(lines) + "\n")
    return p


def run_case(
    name: str,
    env_over: dict,
    deadline: float,
    cmd: list | None = None,
    launcher_signal: int | None = None,
    signal_after: float = 0.0,
    rank_signal: tuple | None = None,  # (rank, signal, seconds after the rank's pid file appears)
    phase1_kill: bool = False,  # SIGKILL the first generate_rank_bindings process that appears (new mode)
) -> Result:
    cmd = cmd or legacy_cmd()
    piddir = tempfile.mkdtemp(prefix="litmus_pids_")
    env = dict(os.environ, PROBE_PIDDIR=piddir, **env_over)
    before = _procs()
    before_shm = _shm()
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    sent = []

    def pid_of(rank: int) -> int:
        f = Path(piddir) / f"rank_{rank}.pid"
        end = time.time() + deadline
        while time.time() < end:
            if f.exists() and f.read_text().strip():
                return int(f.read_text().strip())
            time.sleep(0.2)
        raise RuntimeError(f"no pid file for rank {rank}")

    def worker():
        try:
            if launcher_signal is not None:
                time.sleep(signal_after)
                os.kill(proc.pid, launcher_signal)
                sent.append(f"{signal.Signals(launcher_signal).name} to tt-run at {signal_after:.0f} s")
            if rank_signal is not None:
                r, sig, after = rank_signal
                pid = pid_of(r)
                time.sleep(after)
                os.kill(pid, sig)
                sent.append(f"{signal.Signals(sig).name} to rank {r} (pid {pid}) {after:.0f} s after it started")
            if phase1_kill:
                end = time.time() + deadline
                while time.time() < end:
                    new = [p for p, (c, a) in _procs().items() if c.startswith("generate_rank_b") and p not in before]
                    if new:
                        os.kill(new[0], signal.SIGKILL)
                        sent.append(f"SIGKILL to phase-1 rank (pid {new[0]})")
                        return
                    time.sleep(0.2)
        except Exception as e:  # noqa: BLE001
            sent.append(f"signal worker failed: {e!r}")

    if launcher_signal is not None or rank_signal is not None or phase1_kill:
        threading.Thread(target=worker, daemon=True).start()
    timed_out, alive = False, []
    try:
        out, _ = proc.communicate(timeout=deadline)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        alive = [f"{c}({p})" for p, (c, a) in _procs().items() if p not in before]
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
        rc = proc.returncode
    wall = time.time() - t0
    time.sleep(SETTLE_S)
    after = _procs()
    left = [f"{c}({p})" for p, (c, a) in after.items() if p not in before]
    shm = sorted(_shm() - before_shm)
    # leave the host as we found it for the next row, whatever this one left
    for p in after:
        if p not in before:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass
    for f in shm:
        try:
            os.remove(f)
        except OSError:
            pass
    shutil.rmtree(piddir, ignore_errors=True)
    reached, rcs = _parse(out)
    res = Result(name, " ".join(cmd), rc, wall, timed_out, "; ".join(sent), alive, left, shm, reached, rcs, out)
    print(f"\n[litmus] {name}: {res}")
    return res


# ---------------------------------------------------------------------------------------------------------------
# the assertions a row can make
# ---------------------------------------------------------------------------------------------------------------
def ended(r: Result) -> None:
    assert (
        not r.timed_out
    ), f"R1/R2: the job did not end within the deadline; alive: {r.alive_at_deadline}\n{r.log[-3000:]}"


def clean(r: Result) -> None:
    assert not r.leftovers, f"R3: left running after the launcher returned: {r.leftovers}"
    assert not r.shm_leaked, f"R3: /dev/shm segments leaked: {r.shm_leaked}"


def nonzero(r: Result) -> None:
    assert r.exit not in (0, None), f"R4: the launcher returned {r.exit} although a rank failed\n{r.log[-3000:]}"


def rc_contains(r: Result, rank: int, needle: str) -> None:
    lines = r.rcs.get(rank, [])
    assert any(
        needle in x for x in lines
    ), f"rank {rank} did not report {needle!r}; it reported {lines}\n{r.log[-3000:]}"


# ---------------------------------------------------------------------------------------------------------------
# rows that pass today (gates)
# ---------------------------------------------------------------------------------------------------------------
def test_clean():
    """Control: both ranks open the mock galaxy, one barrier, close, exit 0, nothing left."""
    r = run_case("clean", {}, DEADLINE_END)
    ended(r)
    assert r.exit == 0, str(r)
    clean(r)


def test_dead_peer_barrier_ends():
    """Rank 1 dies before a barrier: the survivor's barrier raises MPI_ERR_PROC_FAILED and the job ends."""
    r = run_case("dead_peer_barrier", {"PROBE_COLL": "barrier_dead_peer"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "PROC_FAILED")
    clean(r)


def test_dead_peer_recv_raises():
    """Rank 1 dies before rank 0's recv: recv raises instead of waiting forever."""
    r = run_case("dead_peer_recv", {"PROBE_COLL": "recv_dead_peer"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "RuntimeError")
    clean(r)


def test_fabric_config_mismatch_ends_nonzero():
    """One rank sets FABRIC_2D before open and the other does not: TT_FATAL on one rank, the job ends non-zero."""
    r = run_case("fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    clean(r)


def test_prte_timeout_ends_deadlock():
    """A send nobody receives under PRRTE --timeout 60: the wall clock ends the job with 241 and nothing left.

    The timeout is longer than the ttnn import on a loaded runner, and the row checks that both ranks were past
    open when the clock fired, so it is the deadlock that was ended and not the start-up."""
    r = run_case(
        "send_no_recv+prte_timeout60", {"PROBE_COLL": "send_no_recv"}, DEADLINE_END + 60, cmd=legacy_cmd("--timeout 60")
    )
    ended(r)
    assert r.exit == 241, str(r)
    for rank in (0, 1):
        assert r.reached.get(rank, "") not in (
            "",
            "start",
            "start_delay",
            "import_done",
            "ctx_init",
        ), f"rank {rank} had not reached open when the wall clock fired: {r}"
    clean(r)


def test_exit_nofinalize_ends_nonzero():
    """Rank 1 _exits with the mesh open (the crash path, no finalize): the job ends non-zero on the production line."""
    r = run_case(
        "exit_nofinalize_after_open", {"PROBE_SCENARIO": "exit_nofinalize", "PROBE_PHASE": "after_open"}, DEADLINE_END
    )
    ended(r)
    nonzero(r)
    clean(r)


def test_invalid_rank_raises():
    """A receive from a rank that does not exist raises at once; the job goes on and ends."""
    r = run_case("recv_bad_rank", {"PROBE_COLL": "recv_bad_rank", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 0, "RuntimeError")
    clean(r)


def test_use_before_init_raises():
    """A collective before init_distributed_context raises 'not initialized' instead of crashing or hanging."""
    r = run_case("use_before_init", {"PROBE_SCENARIO": "use_before_init", "PROBE_ON_ERROR": "continue"}, DEADLINE_END)
    ended(r)
    rc_contains(r, 1, "not initialized")
    assert r.exit == 0, str(r)
    clean(r)


# ---------------------------------------------------------------------------------------------------------------
# rows that fail today: report mode. Each reason names the error of the plan; drop the marker with the fix.
# ---------------------------------------------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="P1: a rank that raises leaves the others in their barrier forever; fix: abort on an uncaught exception in the DistributedContext",
)
def test_raise_after_open_ends():
    r = run_case("raise_after_open", {"PROBE_SCENARIO": "raise", "PROBE_PHASE": "after_open"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P21: a TT_FATAL inside open on one rank (a shape the descriptor cannot hold) leaves its peer in the discovery barrier forever",
)
def test_bad_shape_on_one_rank_ends():
    r = run_case("bad_shape_rank1", {"PROBE_MESH_SHAPE_VICTIM": "9x9"}, DEADLINE_HANG)
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P21: a rank whose descriptor describes another machine makes the mapper TT_FATAL on one rank while the other waits forever",
)
def test_conflicting_descriptor_ends(tmp_path):
    r = run_case("conflicting_descriptor", {}, DEADLINE_HANG, cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)))
    ended(r)
    nonzero(r)
    clean(r)


@pytest.mark.xfail(
    strict=True, reason="P16: a collective on the distributed context has no timeout; a skipped barrier hangs forever"
)
def test_barrier_skipped_ends():
    r = run_case("barrier_skip", {"PROBE_COLL": "barrier_skip"}, DEADLINE_HANG)
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P16: a send nobody receives (the ssend/irecv deadlock of the topology mapper) hangs forever without PRRTE --timeout",
)
def test_send_without_recv_ends():
    r = run_case("send_no_recv", {"PROBE_COLL": "send_no_recv"}, DEADLINE_HANG)
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P13: a rank killed before MPI_Init is never noticed; the survivor waits in the context init forever",
)
def test_sigkill_before_init_ends():
    r = run_case("sigkill_before_init", {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "before_init"}, DEADLINE_HANG)
    ended(r)
    clean(r)


@pytest.mark.xfail(strict=True, reason="P17/P4: a rank killed inside the control plane's init ends the job with exit 0")
def test_dead_peer_in_open_exit_nonzero():
    r = run_case(
        "sigkill_in_open",
        {"PROBE_SCENARIO": "sigkill", "PROBE_PHASE": "in_open", "PROBE_DELAY_MS": "100"},
        DEADLINE_END,
    )
    ended(r)
    clean(r)
    nonzero(r)


def test_ctrl_c_in_deadlock_leaves_nothing():
    """SIGINT at tt-run while the ranks sit in the send/barrier deadlock: tt-run ends and nothing is left."""
    r = run_case(
        "ctrl_c_in_deadlock",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_END,
        launcher_signal=signal.SIGINT,
        signal_after=45.0,  # both ranks are past open and in the deadlock by then
    )
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True,
    reason="P6: Ctrl-C at tt-run while a rank waits in the context init SIGKILLs prterun and leaves that rank behind",
)
def test_ctrl_c_in_init_leaves_nothing():
    """SIGINT at tt-run while rank 0 waits in MPI_Init for a rank 1 that starts 75 s late: nothing may be left."""
    r = run_case(
        "ctrl_c_in_init",
        {"PROBE_START_DELAY_S": "75", "PROBE_START_DELAY_RANK": "1"},
        DEADLINE_END + 30,
        launcher_signal=signal.SIGINT,
        signal_after=50.0,
    )
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True, reason="P7: SIGTERM at tt-run is not forwarded; prterun and the ranks stay in the deadlock"
)
def test_sigterm_at_ttrun_ends():
    r = run_case(
        "sigterm_at_ttrun",
        {"PROBE_COLL": "send_no_recv"},
        DEADLINE_HANG,
        launcher_signal=signal.SIGTERM,
        signal_after=20.0,
    )
    ended(r)
    clean(r)


@pytest.mark.xfail(
    strict=True, reason="P18: a phase-1 (solve) rank killed leaves generate_rank_bindings and prterun forever"
)
def test_solve_rank_killed_ends():
    r = run_case("solve_rank_killed", {}, DEADLINE_HANG, cmd=solve_cmd(), phase1_kill=True)
    ended(r)
    nonzero(r)
    clean(r)
