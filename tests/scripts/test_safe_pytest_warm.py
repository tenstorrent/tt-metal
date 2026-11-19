# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tests for scripts/safe_pytest_warm.py, the warm fork server behind SAFE_PYTEST_WARM=1. No device needed.

  * concurrent forked children run numpy/torch compute correctly (fork safety of the preloaded modules);
  * a kill -9 of the client, or of the server, leaves no child behind and frees the caller's flock;
  * the server refuses to serve when a thread that does not survive fork is running.
"""

from __future__ import annotations

import fcntl
import importlib.util
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "safe_pytest_warm.py"


@pytest.fixture
def warm(tmp_path):
    """Run the warm client against a private server directory; stop its servers afterwards."""
    sock_dir = tmp_path / "sock"
    sock_dir.mkdir(mode=0o700)
    env = dict(os.environ)
    env.pop("SAFE_PYTEST_WARM_JOB", None)  # set when this suite itself runs under a warm server
    env["SAFE_PYTEST_WARM_DIR"] = str(sock_dir)
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env.get("PATH", "")
    work = tmp_path / "work"  # outside the repo, so the repo's conftest.py is not loaded by the children
    work.mkdir()

    def popen(pytest_args, **kw):
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "run", "--pytest", "-q", "-p", "no:cacheprovider", *pytest_args],
            cwd=work,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **kw,
        )

    yield work, popen
    subprocess.run([sys.executable, str(SCRIPT), "stop"], env=env, timeout=60)


def _wait_for(path: Path, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return path.read_text()
        time.sleep(0.1)
    raise TimeoutError(f"{path} did not appear within {timeout}s")


def _gone(pgid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_concurrent_children_compute_correctly(warm):
    """Forked children run threaded numpy/torch math correctly, all at once, from preloaded modules."""
    work, popen = warm
    (work / "test_compute.py").write_text(
        textwrap.dedent(
            """
            import gc
            import numpy as np
            import torch

            def test_compute():
                assert gc.get_freeze_count() > 1000  # forked from the warm server, preloaded objects frozen
                a = np.arange(512 * 512, dtype=np.float64).reshape(512, 512) % 7
                assert np.allclose(a @ a.T, (a @ a.T).T)  # OpenBLAS, multi-threaded
                torch.manual_seed(0)
                x = torch.randn(1024, 1024)
                y = x @ x.t()
                assert torch.allclose(y, y.t(), atol=1e-3)
                assert torch.get_num_threads() >= 1
            """
        )
    )
    first = popen(["test_compute.py"])  # starts the server
    out, err = first.communicate(timeout=300)
    assert first.returncode == 0, out + err
    procs = [popen(["test_compute.py"]) for _ in range(16)]
    for p in procs:
        out, err = p.communicate(timeout=300)
        assert p.returncode == 0, out + err
        assert "running cold" not in err, err


@pytest.mark.parametrize("victim", ["client", "server"])
def test_kill_leaves_no_child_and_frees_lock(warm, tmp_path, victim):
    """kill -9 of the client or the server kills the forked child's process group and frees the flock it held."""
    work, popen = warm
    info = tmp_path / "child.txt"
    (work / "test_sleep.py").write_text(
        textwrap.dedent(
            f"""
            import os, time

            def test_sleep():
                with open({str(info)!r}, "w") as f:
                    f.write(f"{{os.getpgrp()}} {{os.getppid()}}")
                time.sleep(120)
            """
        )
    )
    lock_path = tmp_path / "device.lock"
    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # like run_safe_pytest.sh: the run inherits the locked fd
    client = popen(["test_sleep.py"], pass_fds=(lock_fd,))
    os.close(lock_fd)  # now only the client and its forked child hold the lock
    try:
        pgid, server_pid = map(int, _wait_for(info, 120).split())
        # On a cold fallback the client execs pytest in place, so the "server" would be this test process.
        assert server_pid not in (os.getpid(), client.pid), "run went cold, not through the warm server"
        os.kill(client.pid if victim == "client" else server_pid, signal.SIGKILL)
        assert _gone(pgid, 15), f"child process group {pgid} still alive after kill -9 of the {victim}"
        client.wait(timeout=30)  # a surviving client notices the lost server, kills the run and returns
        assert client.returncode != 0
        probe = os.open(lock_path, os.O_RDWR)
        try:
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if anyone still holds it
        finally:
            os.close(probe)
    finally:
        if client.poll() is None:
            client.kill()


def test_fork_hazard_detects_running_thread():
    """A Python thread running in the server makes _fork_hazard refuse to fork."""
    spec = importlib.util.spec_from_file_location("safe_pytest_warm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    stop = threading.Event()
    t = threading.Thread(target=stop.wait, name="leaky-preload-thread")
    t.start()
    try:
        assert "leaky-preload-thread" in (mod._fork_hazard() or "")
    finally:
        stop.set()
        t.join()


def test_fork_hazard_accepts_openblas_pools():
    """OpenBLAS worker pools (numpy's and scipy's bundled libraries) are fork-safe and must not count as a hazard."""
    code = textwrap.dedent(
        f"""
        import importlib.util, numpy, scipy.linalg
        numpy.ones((64, 64)) @ numpy.ones((64, 64))       # start numpy's OpenBLAS pool
        scipy.linalg.blas.dgemm(1.0, numpy.ones((64, 64)), numpy.ones((64, 64)))  # and scipy's
        spec = importlib.util.spec_from_file_location("safe_pytest_warm", {str(SCRIPT)!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(mod._fork_hazard())
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "None", out.stdout


def test_unsafe_marker_expires(tmp_path):
    """A fresh "not fork-safe" marker sends clients cold; an expired one is removed so a new server re-checks."""
    spec = importlib.util.spec_from_file_location("safe_pytest_warm", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sock = str(tmp_path / "x.sock")
    assert mod._unsafe_reason(sock) is None
    Path(sock + ".unsafe").write_text("native thread(s) ['leaky'] started by a preloaded module\n")
    assert "leaky" in mod._unsafe_reason(sock)
    old = time.time() - mod.UNSAFE_TTL_S - 1
    os.utime(sock + ".unsafe", (old, old))
    assert mod._unsafe_reason(sock) is None
    assert not os.path.exists(sock + ".unsafe")
