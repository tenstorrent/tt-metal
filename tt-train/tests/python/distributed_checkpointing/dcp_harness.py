# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""What the multi-process DCP test modules share: running a world of the cross-world and multi-host modules as a
child process (the driver, tt-run, the output filter), and the two-rank world the tt-run modules open."""

import contextlib
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import zlib

import pytest
import ttnn
from dcp_driver import MESH_OPEN_FAILED

import ttml

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = [sys.executable, os.path.join(HERE, "dcp_driver.py")]
TT_RUN = shutil.which("tt-run")
MGD_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "..", "configs", "mgd"))  # tt-train/configs/mgd
# The interpreter's exit prints a long nanobind leak report after any traceback; output_lines drops it.
NOISE = re.compile(
    r"nanobind|leaked (instance|type|function|keep_alive)|refleaks|skipped remainder|\| DEBUG |^Config\{"
)


def tt_run_command(mgd: str, hosts: str, command: list[str], mpi_args: str = "--tag-output") -> list[str]:
    """``command`` under tt-run on ``hosts`` (comma-separated), with mesh graph descriptor ``mgd``. A world whose ranks
    all run on this host goes ``--bare``: without tt-run's multi-host defaults (TCP only, on every interface but lo
    and docker0) Open MPI connects the ranks through shared memory, while over TCP they can hang on a host that also
    has a Kubernetes overlay network. As root, as in a CI container, Open MPI starts only with --allow-run-as-root."""
    options = ["--bare"] if set(hosts.split(",")) == {socket.gethostname()} else []
    if os.geteuid() == 0:
        mpi_args += " --allow-run-as-root"
    # "--" ends tt-run's own options: its -m would otherwise take a "python -m" module for a mesh graph descriptor.
    return [TT_RUN, "--mesh-graph-descriptor", mgd, "--hosts", hosts, *options, "--mpi-args", mpi_args, "--", *command]


def output_lines(proc: subprocess.CompletedProcess) -> list[str]:
    """A world's stdout and stderr lines, without the interpreter's exit noise."""
    return [line for line in (proc.stdout + proc.stderr).splitlines() if not NOISE.search(line)]


TIMED_OUT = -9  # the harness's own exit code for a world it had to terminate


def run_world(command: list[str], env: dict, timeout: int) -> subprocess.CompletedProcess:
    """Run a world to completion. On timeout the whole process group gets SIGTERM, so mpirun tears the ranks down on
    every host (a rank parked in a hung device call would otherwise keep its chips locked for the next world), then
    SIGKILL for whatever is left; what the world printed so far is kept."""
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, start_new_session=True
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            out, err = proc.communicate(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            out, err = proc.communicate()
        return subprocess.CompletedProcess(
            command, TIMED_OUT, out + f"\n[harness] the world ran for {timeout} s and was terminated\n", err
        )
    return subprocess.CompletedProcess(command, proc.returncode, out, err)


def run_checked(
    command: list[str], what: str, timeout: int, *, needs_tt_run: bool, save: bool, expect_ok: bool = True
) -> list[str]:
    """Run a world of the driver (``command``) and check how it ended; returns its output lines. Skips when the world
    needs tt-run and it is not installed, or when a host cannot open the mesh: the driver then reports
    MESH_OPEN_FAILED, in its exit status and in its output, since tt-run may not carry a rank's exit status. The exit
    status must be 0 (1 with ``expect_ok=False``), and a save must have printed its summary, which the driver prints
    only when it succeeded; a failure shows the output's tail."""
    if needs_tt_run and TT_RUN is None:
        pytest.skip(f"{what}: tt-run is not installed")
    env = dict(os.environ)
    env.pop("TT_MESH_GRAPH_DESC_PATH", None)
    proc = run_world(command, env, timeout)
    lines = output_lines(proc)
    reason = next((line for line in lines if "MESH_OPEN_FAILED" in line), None)
    if proc.returncode == MESH_OPEN_FAILED or reason is not None:
        pytest.skip(f"{what}: a host cannot open the mesh: {reason or 'no reason given'}")
    tail = "\n".join(lines[-80:])
    assert proc.returncode == (0 if expect_ok else 1), f"{what} failed with exit {proc.returncode}:\n{tail}"
    if save and expect_ok:
        assert '"saved"' in proc.stdout, f"{what} printed no summary:\n{tail}"
    return lines


def ranks_in_step(where: str) -> None:
    """Fail unless every rank of a tt-run module is at ``where`` (an allgather of its hash). A rank whose test failed
    early skips the collectives the other ranks are still in; without this check its next test's collectives would
    pair with theirs, and the module would report the wrong errors or hang."""
    mine = zlib.crc32(where.encode("utf-8")) & 0x7FFFFFFF
    values = [int(v) for v in ttnn.distributed_context_allgather_int(mine)]
    if any(v != mine for v in values):
        pytest.fail(f"the ranks are out of step at {where}: {values}")


@contextlib.contextmanager
def two_rank_world(mesh_shape: tuple[int, int], tag: str, what: str):
    """For a tt-run module: open this rank's view of ``mesh_shape`` with axes ``("dp", "tp")`` and TP, and a directory
    both ranks see (``DCP_SHARED_TMP``, else the system temporary directory), emptied before and removed after.
    Yields ``(rank, root)``; skips without two ranks."""
    if not ttnn.using_distributed_env():
        pytest.skip(f"needs {what}; launch with tt-run")
    ttml.open_device_mesh(ttml.Mesh(mesh_shape, ("dp", "tp")))
    try:
        size = int(ttnn.distributed_context_get_size())
        if size != 2:
            pytest.skip(f"needs exactly two ranks, launched with {size}")
        rank = int(ttnn.distributed_context_get_rank())
        ctx = ttml.autograd.AutoContext.get_instance()
        ctx.initialize_parallelism_context(ttml.autograd.DistributedConfig(enable_ddp=False, enable_tp=True))
        root = os.path.join(os.environ.get("DCP_SHARED_TMP", tempfile.gettempdir()), f"dcp-{tag}-{os.getuid()}")
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
            os.makedirs(root, exist_ok=True)
        ttnn.distributed_context_barrier()
        yield rank, root
        ttnn.distributed_context_barrier()
        if rank == 0:
            shutil.rmtree(root, ignore_errors=True)
    finally:
        ttml.close_device_mesh()
