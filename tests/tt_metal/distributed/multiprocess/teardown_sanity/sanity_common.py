# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared by the teardown sanity rows: paths, the census, the launch, the deadline, the assertions.

Two launchers: tt-run with tt-metal ranks (``legacy_cmd``, ``solve_cmd``, the Python probe) and plain mpirun with
the probe (``mpi_cmd``, ``sanity_mpi_probe.cpp``). The MPI install is ``TT_SANITY_MPI_PREFIX`` when set (its
bin/mpirun and bin/mpicxx, the probe recompiled against it, so another OpenMPI version or build can be tried),
else the mpirun on PATH and the probe binary the build made (``build/test/tt_metal/distributed/teardown_mpi_probe``).
"""

from __future__ import annotations

import functools
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
PROBE = HERE / "sanity_probe.py"
# A mock Blackhole dual galaxy: two ranks, no device (CPU only). The suite targets Blackhole, so it uses a
# blackhole mock everywhere; the same descriptor + mapping pairing is already validated by the BH dual-galaxy rows
# of the fabric CPU-only lane. This is a mock topology for the control plane to parse — the teardown behaviour it
# exercises (MPI collectives, errhandler, timeout) is independent of the chip.
BINDING = ROOT / "tests/tt_metal/distributed/config/dual_bh_galaxy_rank_binding.yaml"
MAPPING = (
    ROOT
    / "tt_metal/third_party/tt-cluster-descriptors/blackhole/dual_glx_2.5d_torus/dual_glx_2.5d_torus_cluster_desc_mapping.yaml"
)
# a different Blackhole machine, dropped into one rank's slot to make the two ranks disagree (test_bad_descriptor)
CONFLICT_DESC = (
    "tt_metal/third_party/tt-cluster-descriptors/blackhole/single_bh_galaxy_clus_desc/single_bh_galaxy_clus_desc.yaml"
)
MGD = ROOT / "tt_metal/fabric/mesh_graph_descriptors/dual_bh_galaxy_torus_xy_graph_descriptor.textproto"
PYTHON = os.environ.get("TT_SANITY_PYTHON", "python3")
TTRUN = shutil.which("tt-run") or "tt-run"

# a row that must end gets this long; a row expected to hang today is given up on sooner, it is the whole budget
DEADLINE_END = float(os.environ.get("TT_SANITY_DEADLINE", "90"))
DEADLINE_HANG = float(os.environ.get("TT_SANITY_DEADLINE_HANG", "60"))
SETTLE_S = 3.0  # after the launcher returns, before the census

RANK_MARK = re.compile(r"\[probe r=(\d+) pid=\d+ t=[\d.]+\] (.*)")


# ---------------------------------------------------------------------------------------------------------------
# census: what of THIS job is alive, and what it left in /dev/shm
# ---------------------------------------------------------------------------------------------------------------
def _procs(tag: str) -> dict:
    """pid -> (comm, args) of this user's processes that belong to THIS job: their environment carries the job's
    unique ``PROBE_PIDDIR`` tag. Every process in the launcher's tree inherits it (the launcher, its ranks, a
    prted, the solver), so the census and cleanup touch only this job — never a concurrent job on the same runner,
    even one PRRTE re-sessioned into its own session, and never this harness's own pytest process. Scoping by the
    tag (not by comm or by a before/after diff of all same-uid processes) is what makes the suite safe to run
    alongside other jobs and in ``--parallel``."""
    out = {}
    uid = os.getuid()
    needle = ("PROBE_PIDDIR=" + tag).encode()
    for p in glob.glob("/proc/[0-9]*"):
        try:
            if os.stat(p).st_uid != uid:
                continue
            if needle not in open(p + "/environ", "rb").read():
                continue
            comm = open(p + "/comm").read().strip()
            args = open(p + "/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        out[int(p.rsplit("/", 1)[1])] = (comm, args[:120])
    return out


def _shm() -> set:
    uid = os.getuid()
    return {f for f in glob.glob("/dev/shm/sm_segment.*") if re.search(rf"\.{uid}\.", f)}


def _shm_mapped() -> set:
    """Shared-memory segments that some live process still maps — never remove these: they belong to a running
    job (possibly a concurrent one), not to a job that has gone."""
    mapped = set()
    for m in glob.glob("/proc/[0-9]*/maps"):
        try:
            for line in open(m):
                if "sm_segment." in line:
                    mapped.add(line.rsplit(" ", 1)[1].strip())
        except OSError:
            continue
    return mapped


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
        if "_rc=" in text and "SUCCESS" not in text and "rc=MPI_SUCCESS" not in text:
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


MGD_QUAD = ROOT / "tt_metal/fabric/mesh_graph_descriptors/32x4_quad_bh_galaxy_torus_xy_graph_descriptor.textproto"


def solve_cmd(mpi_extra: str = "", mgd: Path = MGD) -> list:
    """tt-run in new mode: phase 1 (generate_rank_bindings on the mock descriptors) then the ranks."""
    return [
        TTRUN,
        "--mesh-graph-descriptor",
        str(mgd),
        "--mock-cluster-rank-binding",
        str(MAPPING),
        "--mpi-args",
        ("--allow-run-as-root --oversubscribe " + mpi_extra).strip(),
        "--force-rediscovery",
        PYTHON,
        str(PROBE),
    ]


MPI_PREFIX = os.environ.get("TT_SANITY_MPI_PREFIX", "")
MPI_PROBE_SRC = HERE / "sanity_mpi_probe.cpp"
MPI_PROBE_BUILT = ROOT / "build/test/tt_metal/distributed/teardown_mpi_probe"
_MPI_PROBE_CACHE: dict = {}


def mpirun_path() -> str:
    if MPI_PREFIX:
        return str(Path(MPI_PREFIX) / "bin" / "mpirun")
    return shutil.which("mpirun") or "mpirun"


def mpi_probe_path() -> str:
    """The probe: the build's binary, or one compiled now with the chosen install's mpicxx; skips if neither."""
    if "path" in _MPI_PROBE_CACHE:
        return _MPI_PROBE_CACHE["path"]
    if not MPI_PREFIX and MPI_PROBE_BUILT.exists():
        _MPI_PROBE_CACHE["path"] = str(MPI_PROBE_BUILT)
        return _MPI_PROBE_CACHE["path"]
    mpicxx = os.environ.get("TT_SANITY_MPICXX") or (
        str(Path(MPI_PREFIX) / "bin" / "mpicxx") if MPI_PREFIX else shutil.which("mpicxx")
    )
    if not mpicxx or not Path(mpicxx).exists():
        pytest.skip(
            "no teardown_mpi_probe binary in the build and no mpicxx to compile sanity_mpi_probe.cpp (set TT_SANITY_MPI_PREFIX or TT_SANITY_MPICXX)"
        )
    out = Path(tempfile.mkdtemp(prefix="sanity_mpi_")) / "teardown_mpi_probe"
    subprocess.run([mpicxx, "-O2", "-o", str(out), str(MPI_PROBE_SRC)], check=True)
    _MPI_PROBE_CACHE["path"] = str(out)
    return _MPI_PROBE_CACHE["path"]


def mpi_cmd(ulfm: bool = True, extra: str = "", btl: str = "self,tcp", np: int = 3) -> list:
    """Direct mpirun with the C probe: the production transport line (btl self,tcp) unless told otherwise."""
    cmd = [mpirun_path()]
    if ulfm:
        cmd += ["--with-ft", "ulfm"]
    cmd += ["-np", str(np), "--oversubscribe", "--bind-to", "none", "--allow-run-as-root", "--tag-output"]
    if btl:
        cmd += ["--mca", "btl", btl]
    cmd += extra.split()
    for v in ("PROBE_SCENARIO", "PROBE_VICTIM", "PROBE_DELAY_MS", "PROBE_SURVIVOR_ACTION", "PROBE_PIDDIR", "PROBE_NP"):
        cmd += ["-x", v]
    return cmd + [mpi_probe_path()]


MPI_BINDING = HERE / "sanity_mpi_binding.yaml"


def ttrun_mpi_cmd(mpi_extra: str = "", np: int = 3) -> list:
    """Launch the raw C++ probe THROUGH tt-run (legacy mode, np local ranks), so the row tests tt-run's own teardown:
    its exit code, signal handling and cleanup. tt-run picks the plain launcher and adds --mca btl self,tcp; extra
    group-E settings (a transport, a component, an interface) go through --mpi-args."""
    return [
        TTRUN,
        "--rank-binding",
        str(MPI_BINDING),
        "--mpi-args",
        ("--allow-run-as-root --oversubscribe " + mpi_extra).strip(),
        mpi_probe_path(),
    ]


def conflict_mapping(tmp: Path) -> Path:
    """The dual-galaxy mapping with rank 1's descriptor replaced by a different Blackhole machine: two ranks that disagree."""
    text = MAPPING.read_text()
    lines = [(f'  1: "{CONFLICT_DESC}"' if line.strip().startswith('1: "') else line) for line in text.splitlines()]
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
    launcher_ready: tuple | None = None,  # (rank, marker): wait for this rank's phase marker before the launcher signal
    rank_signal: tuple | None = None,  # (rank, signal, seconds after the rank is ready)
    signal_on_marker: str | None = None,  # gate rank_signal on this phase marker file (e.g. "open_enter") not the pid
    phase1_kill: bool = False,  # SIGKILL the first generate_rank_bindings process that appears (new mode)
) -> Result:
    cmd = cmd or legacy_cmd()
    piddir = tempfile.mkdtemp(prefix="sanity_pids_")
    env = dict(os.environ, PROBE_PIDDIR=piddir, **env_over)
    before_shm = _shm()
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True
    )
    sent = []

    def kill_job():
        """SIGKILL every process of THIS job (matched by the unique tag), its ranks included, whatever session or
        process group PRRTE put them in. Scoped by the tag, so it never reaches another job on the runner."""
        for p in _procs(piddir):
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass

    def pid_of(rank: int) -> int:
        f = Path(piddir) / f"rank_{rank}.pid"
        end = time.time() + deadline
        while time.time() < end:
            if f.exists() and f.read_text().strip():
                return int(f.read_text().strip())
            time.sleep(0.2)
        raise RuntimeError(f"no pid file for rank {rank}")

    def wait_marker(rank: int, name: str) -> None:
        """Block until the rank drops this phase marker (e.g. ``open_enter``), so a parent-sent signal lands at the
        tested point. If the marker never arrives within the deadline the fault was not delivered — a harness error."""
        f = Path(piddir) / f"rank_{rank}.{name}"
        end = time.time() + deadline
        while time.time() < end:
            if f.exists():
                return
            time.sleep(0.05)
        raise RuntimeError(f"rank {rank} never reached marker {name!r}")

    def worker():
        try:
            if launcher_signal is not None:
                where = "immediately"
                if launcher_ready is not None:
                    gr, gm = launcher_ready
                    wait_marker(gr, gm)  # the ranks are at the tested phase before we signal, not a wall-clock guess
                    where = f"{signal_after:.0f} s after rank {gr} reached {gm}"
                time.sleep(signal_after)
                os.kill(proc.pid, launcher_signal)
                sent.append(f"{signal.Signals(launcher_signal).name} to tt-run {where}")
            if rank_signal is not None:
                r, sig, after = rank_signal
                pid = pid_of(r)
                if signal_on_marker:
                    wait_marker(r, signal_on_marker)  # send it at the tested phase, not just "after it started"
                time.sleep(after)
                os.kill(pid, sig)
                where = f"at {signal_on_marker}" if signal_on_marker else f"{after:.0f} s after it started"
                sent.append(f"{signal.Signals(sig).name} to rank {r} (pid {pid}) {where}")
            if phase1_kill:
                end = time.time() + deadline
                while time.time() < end:
                    new = [p for p, (c, a) in _procs(piddir).items() if c.startswith("generate_rank_b")]
                    if new:
                        os.kill(new[0], signal.SIGKILL)
                        sent.append(f"SIGKILL to phase-1 rank (pid {new[0]})")
                        return
                    time.sleep(0.2)
                sent.append("FAULT NOT DELIVERED: no phase-1 (generate_rank_bindings) process appeared")
        except Exception as e:  # noqa: BLE001
            # a fault that could not be delivered is a harness error, not a teardown outcome: mark it loudly so a
            # row cannot quietly "pass" (or xfail) without the fault it was supposed to inject (review: P9/P4 ack).
            sent.append(f"FAULT NOT DELIVERED: {e!r}")

    if launcher_signal is not None or rank_signal is not None or phase1_kill:
        threading.Thread(target=worker, daemon=True).start()
    timed_out, alive, out, rc = False, [], "", None
    try:
        try:
            out, _ = proc.communicate(timeout=deadline)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            alive = [f"{c}({p})" for p, (c, a) in _procs(piddir).items()]
            kill_job()  # the whole job tree, not just the launcher's process group, so the stdout pipe is released
            try:
                out, _ = proc.communicate(timeout=SETTLE_S)  # bounded: the writers are dead, so this is EOF, not a wait
            except subprocess.TimeoutExpired:
                proc.kill()
                out, _ = proc.communicate(timeout=SETTLE_S)
            rc = proc.returncode
        wall = time.time() - t0
        time.sleep(SETTLE_S)
        left = [f"{c}({p})" for p, (c, a) in _procs(piddir).items()]
        mapped = _shm_mapped()
        shm = sorted(s for s in (_shm() - before_shm) if s not in mapped)  # our dead leftovers, not a live job's
    finally:
        # leave the host as we found it for the next row, whatever this one left — always, even on an assertion or
        # an error above. Scoped to this job (tag) and to unmapped segments (never a concurrent job's live memory).
        kill_job()
        for f in sorted(s for s in (_shm() - before_shm) if s not in _shm_mapped()):
            try:
                os.remove(f)
            except OSError:
                pass
        shutil.rmtree(piddir, ignore_errors=True)
    reached, rcs = _parse(out)
    res = Result(name, " ".join(cmd), rc, wall, timed_out, "; ".join(sent), alive, left, shm, reached, rcs, out)
    print(f"\n[sanity] {name}: {res}")
    return res


# ---------------------------------------------------------------------------------------------------------------
# the assertions a row can make
# ---------------------------------------------------------------------------------------------------------------
def known(error: str):
    """Mark a row that fails today with the error of the teardown plan it exposes. The row still fails (CI is red
    until the fix lands), and the failure message starts with the error id and the fix, so the job's annotation
    says what is broken and where. When the fix lands the row passes and the marker is simply dropped."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **k):
            try:
                return fn(*a, **k)
            except AssertionError as e:
                raise AssertionError(f"known error {error}\n{e}") from None

        return wrapper

    return deco


def ended(r: Result) -> None:
    assert (
        not r.timed_out
    ), f"R1/R2: the job did not end within the deadline; alive: {r.alive_at_deadline}\n{r.log[-3000:]}"


def clean(r: Result) -> None:
    assert not r.leftovers, f"R3: left running after the launcher returned: {r.leftovers}"
    assert not r.shm_leaked, f"R3: /dev/shm segments leaked: {r.shm_leaked}"


def nonzero(r: Result) -> None:
    assert r.exit not in (0, None), f"R4: the launcher returned {r.exit} although a rank failed\n{r.log[-3000:]}"


def fault_delivered(r: Result) -> None:
    """The fault this row injects from the harness (a signal to the launcher or a rank, a phase-1 kill) really
    happened. Without this a row can pass for the wrong reason — a job that ends non-zero on its own, or a signal
    that arrived during a clean early exit — so a missed injection must fail the row, not slip through."""
    assert "FAULT NOT DELIVERED" not in r.signal_sent, f"the fault was not injected: {r}"
    assert r.signal_sent, f"no fault was injected (nothing sent): {r}"


def rank_reached(r: Result, rank: int, marker: str) -> None:
    assert (
        r.reached.get(rank) == marker
    ), f"rank {rank} ended at {r.reached.get(rank)!r}, not {marker!r}\n{r.log[-3000:]}"


def rc_contains(r: Result, rank: int, needle: str) -> None:
    lines = r.rcs.get(rank, [])
    assert any(
        needle in x for x in lines
    ), f"rank {rank} did not report {needle!r}; it reported {lines}\n{r.log[-3000:]}"
