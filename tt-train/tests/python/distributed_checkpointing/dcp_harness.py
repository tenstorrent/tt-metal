# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Running one world of a cross-world test as a child process, for the cross-world and the multi-host modules."""

import os
import signal
import subprocess
import zlib

import pytest
import ttnn

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


def ranks_in_step(where: str) -> None:
    """Fail unless every rank of a tt-run module is at ``where`` (an allgather of its hash). A rank whose test failed
    early skips the collectives the other ranks are still in; without this check its next test's collectives would
    pair with theirs, and the module would report the wrong errors or hang."""
    mine = zlib.crc32(where.encode("utf-8")) & 0x7FFFFFFF
    values = [int(v) for v in ttnn.distributed_context_allgather_int(mine)]
    if any(v != mine for v in values):
        pytest.fail(f"the ranks are out of step at {where}: {values}")
