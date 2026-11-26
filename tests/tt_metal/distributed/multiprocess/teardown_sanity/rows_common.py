# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""What the two Python suites of the teardown sanity share: the harness import (sanity_common, next to the
raw-MPI rows), the per-rank checks (the teardown line, the wrapper's stop line, the markers) and the probe
fixture that points the harness at the suite's rank program."""

from __future__ import annotations

import os
import re
import signal
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sanity_common, next to this file
sys.path.insert(
    0, str(Path(__file__).resolve().parents[5] / "tests/ttnn/distributed/teardown_sanity")
)  # sanity_rank_rows
import sanity_common  # noqa: E402

from sanity_common import (
    _session_dirs,
    DEADLINE_END,
    DEADLINE_HANG,
    clean,
    ended,
    fault_delivered,
    legacy_cmd,
    mpirun_path,
    nonzero,
    run_case,
    PYTHON,
    ROOT,
)
from sanity_rank_rows import CONFLICT_DESC, MGD_QUAD, conflict_mapping, rc_contains, solve_cmd  # noqa: E402


def probe_fixture(path: Path):
    """An autouse fixture that points the harness at this suite's rank program before every row (two suites may
    share one pytest session)."""

    @pytest.fixture(autouse=True)
    def _probe():
        sanity_common.PROBE = path
        yield

    return _probe


def reached(r, rank: int, marker: str) -> None:
    """The row got as far as it means to test: the rank reached the marker. Raised as a harness error, not an
    AssertionError, so a strict xfail (raises=AssertionError) fails loudly instead of passing as the known gap."""
    if not any(f"[probe r={rank} " in line and marker in line for line in r.log.splitlines()):
        raise RuntimeError(f"harness: rank {rank} never reached {marker!r}; the row tested nothing\n{r.log[-3000:]}")


def closed(r, ranks=(0, 1)) -> None:
    """Every listed rank closed its mesh on the way out (the close_done marker): a graceful end, not a kill."""
    for rank in ranks:
        assert any(
            f"[probe r={rank} " in line and "close_done" in line for line in r.log.splitlines()
        ), f"rank {rank} did not close its mesh; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"


def tore_down(r, ranks=(0, 1), reason: str = "SIGTERM") -> None:
    """Every listed rank went through teardown on the way out (the rank's own line): for now it says so and exits
    without closing its mesh; the close comes with the later teardown step."""
    for rank in ranks:
        m = re.search(rf"\[rank {rank}\] teardown initiated: (.*?); exiting with code (\d+)", r.log)
        assert m, f"rank {rank} did not go through teardown; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"
        assert reason in m.group(1), f"rank {rank} tore down for {m.group(1)!r}, expected {reason!r}"


def stopped_by(r, rank: int, reason: str, code: int) -> None:
    """Rank ``rank`` stopped the job for ``reason``: its wrapper printed the stop line, every other rank was sent SIGTERM
    and exited within the grace (nothing forced), and tt-run names the rank and returns its code."""
    m = re.search(rf"\[rank {rank}\] (.*?) \(on \S+, pid \d+\); stopping the job \(exit {code}\)", r.log)
    assert m, f"no stop line from rank {rank} with exit {code}\n{r.log[-3000:]}"
    assert reason in m.group(1), f"rank {rank} stopped the job for {m.group(1)!r}, expected {reason!r}"
    assert r.exit == code, f"tt-run returned {r.exit}, expected the stopping rank's {code}: {r}"
    assert "the other ranks were sent SIGTERM and exited" in r.log, r.log[-3000:]
    assert f"origin=rank {rank} on" in r.log and "outcome=failed" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, f"the stop was forced, not graceful\n{r.log[-3000:]}"


def stopped_at_ttrun(r, sig: int) -> None:
    """A stop at tt-run: every rank was sent SIGTERM, closed its mesh and exited within the grace; 128 + the signal."""
    assert r.exit == 128 + sig, f"expected {128 + sig}: {r}"
    assert "sent SIGTERM to all ranks" in r.log and f"outcome=stopped exit={128 + sig}" in r.log, r.log[-3000:]
    assert "job forced down" not in r.log, r.log[-3000:]
    tore_down(r)


def _both_ranks_were_in_the_collective(r) -> None:
    """Positive evidence that the wall clock ended the intended deadlock, not a slow start-up: both ranks entered the
    collective section (a ``coll_*`` marker), past open. The last marker is not the collective itself: the SIGTERM
    makes the collective return and the rank close its device (``close_done``) on the way out."""
    for rank in (0, 1):
        assert f"[probe r={rank} " in r.log and any(
            f"[probe r={rank} " in line and "] coll_" in line for line in r.log.splitlines()
        ), f"rank {rank} had not reached the collective when the wall clock fired: {r}"


def rank_saw(r, rank: int, marker: str) -> None:
    """The rank printed this marker at some point (not necessarily its last one)."""
    assert any(
        f"[probe r={rank} " in line and marker in line for line in r.log.splitlines()
    ), f"rank {rank} never printed {marker!r}; it ended at {r.reached.get(rank)!r}\n{r.log[-3000:]}"


def _singleton(name: str, code: str, tmp_path, deadline=DEADLINE_END, env=None):
    """Run one unlaunched tt-metal process with TMPDIR scoped to the row; return the result and what it left there."""
    sess = tmp_path / "mpi"
    sess.mkdir()
    r = run_case(name, {"TMPDIR": str(sess), **(env or {})}, deadline, cmd=[PYTHON, "-c", code])
    left = sorted(p.name for p in sess.iterdir() if p.name.startswith(("tt_mpi_singleton", "ompi.")))
    return r, left
