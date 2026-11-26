# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Teardown sanity, level 2, C++: a mock MeshDevice open on the dual Blackhole galaxy, from tt-metalium.

Two C++ ranks (``rank_probe.cpp`` built as ``teardown_mesh_probe``) under tt-run on the CPU, each opening a mock
mesh with ``MeshDevice::create`` (the control plane's distributed init: discovery, the fabric-config check, the
topology solve, the exchange; no kernels), one injected fault per row: a death or a stop inside open, a shape,
fabric or descriptor conflict, a failure or a death with the mesh open. The teardown must be initiated from inside
tt-metal, with no Python above it. The same rows run from ttnn in tests/ttnn/distributed/teardown_sanity/test_mesh_py.py.

Run: pytest tests/tt_metal/distributed/multiprocess/teardown_sanity/test_mesh_cpp.py -v (a tree built with
BUILD_TESTS, the python environment; the rows skip when the probe is not built).
"""

from __future__ import annotations

import re
import signal
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rows_common import *  # noqa: E402,F401,F403 - the harness, the checks and the helpers
from rows_common import probe_fixture  # noqa: E402
from sanity_rank_rows import conflict_mapping  # noqa: E402

PROBE_BIN = ROOT / "build/test/tt_metal/distributed/teardown_mesh_probe"
_probe = probe_fixture(PROBE_BIN)


def test_open_close_clean():
    """Both ranks open the mock mesh, meet in a barrier, close, exit 0: nothing stopped, nothing left."""
    r = run_case("cppmesh_clean", {}, DEADLINE_END)
    ended(r)
    assert r.exit == 0 and "outcome=clean exit=0" in r.log, str(r)
    closed(r)
    assert "teardown initiated" not in r.log, r.log[-3000:]
    clean(r)


def test_peer_dies_in_open():
    """Rank 1 is SIGKILLed inside MeshDevice::create (gated on its open_enter marker): rank 0 leaves the control
    plane's collective (the stop, or the ULFM error then the stop) and tears down; the job ends 137, nothing left."""
    r = run_case(
        "cppmesh_sigkill_in_open",
        {"PROBE_SCENARIO": "clean", "PROBE_VICTIM": "1"},
        DEADLINE_END,
        rank_signal=(1, signal.SIGKILL, 0.2),
        signal_on_marker="open_enter",
    )
    ended(r)
    assert "FAULT NOT DELIVERED" not in r.signal_sent, r
    nonzero(r)
    clean(r)


def test_stop_inside_open():
    """SIGTERM at tt-run while both ranks are inside open: the control-plane collective in flight throws and the
    rank leaves through teardown from inside tt-metal; 143."""
    r = run_case(
        "cppmesh_sigterm_in_open",
        {},
        DEADLINE_END,
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "open_enter"),
        signal_after=0.3,
    )
    fault_delivered(r)
    ended(r)
    assert r.exit == 143 and "sent SIGTERM to all ranks" in r.log, str(r)
    assert "job forced down" not in r.log, r.log[-3000:]
    tore_down(r)
    clean(r)


@pytest.mark.parametrize("shape", ["4x4", "9x9"], ids=["mismatch_4x4", "bad_9x9"])
def test_shape_conflict(shape):
    """Rank 1 asks a different mesh shape (a valid 4x4 against rank 0's 8x4, or a 9x9 nothing holds): the open
    fails on every rank with the "ranks disagree" TT_FATAL, no rank hangs; 1, a rank named."""
    r = run_case(f"cppmesh_shape_{shape}", {"PROBE_MESH_SHAPE_VICTIM": shape}, DEADLINE_END)
    ended(r)
    assert r.exit == 1 and "outcome=failed" in r.log, str(r)
    assert re.search(
        r"\[rank \d\] .*ranks disagree on the requested mesh shape.*stopping the job \(exit 1\)", r.log
    ), r.log[-3000:]
    clean(r)


def test_fabric_mismatch():
    """Rank 1 sets FABRIC_2D before its open, rank 0 none: the fabric-config check across ranks fails the open on
    every rank; 1, no hang."""
    r = run_case("cppmesh_fabric_mismatch", {"PROBE_FABRIC_RANK": "1"}, DEADLINE_END)
    ended(r)
    nonzero(r)
    assert "stopping the job" in r.log, r.log[-3000:]
    clean(r)


def test_descriptor_conflict(tmp_path):
    """Rank 1 gets a different Blackhole machine's cluster descriptor: the control plane cannot place the mesh,
    the open fails with a clear error on every rank; 1, no hang."""
    r = run_case(
        "cppmesh_descriptor_conflict",
        {"PROBE_SCENARIO": "mesh_open"},
        DEADLINE_HANG,
        cmd=legacy_cmd(mapping=conflict_mapping(tmp_path)),
    )
    ended(r)
    nonzero(r)
    assert "stopping the job" in r.log, r.log[-3000:]
    clean(r)


def test_throw_with_mesh_open():
    """Rank 1 throws after its open while rank 0 waits in a barrier: std::terminate, the failed word, teardown with
    the exception text; rank 0 tears down; 1 naming rank 1."""
    r = run_case("cppmesh_throw", {"PROBE_SCENARIO": "throw", "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "uncaught C++ exception: injected failure at after_open", 1)
    tore_down(r, ranks=(1,), reason="uncaught C++ exception")
    tore_down(r, ranks=(0,))
    clean(r)


def test_throw_while_peer_in_split():
    """Rank 1 throws while rank 0 is inside a communicator split with the mesh open: rank 0 leaves the split on the
    SIGTERM and tears down; 1 naming rank 1."""
    r = run_case("cppmesh_split_peer_throws", {"PROBE_COLL": "split_peer_throws"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, "uncaught C++ exception: injected failure while the peers are in split", 1)
    reached(r, 0, "coll_split_enter")
    tore_down(r, ranks=(0,))
    clean(r)


@pytest.mark.parametrize(
    "scenario, signum", [("sigkill", 9), ("segv", 11), ("abort", 6)], ids=["sigkill", "segv", "abort"]
)
def test_death_with_mesh_open(scenario, signum):
    """Rank 1 dies on a signal with its mesh open: its wrapper stops the job in its place, rank 0 tears down;
    128 + the signal."""
    r = run_case(f"cppmesh_{scenario}", {"PROBE_SCENARIO": scenario, "PROBE_PHASE": "after_open"}, DEADLINE_END)
    ended(r)
    stopped_by(r, 1, f"killed by signal {signum}", 128 + signum)
    tore_down(r, ranks=(0,))
    clean(r)


def test_stop_with_mesh_open():
    """SIGTERM at tt-run while both ranks sit in a slow barrier with their mesh open: every rank's wait throws and
    leaves through teardown; today without closing the mesh (the later teardown step closes it and this row then
    asserts close_done); 143."""
    r = run_case(
        "cppmesh_sigterm_at_ttrun",
        {"PROBE_COLL": "barrier_slow", "PROBE_SLOW_S": "8"},
        DEADLINE_END,
        launcher_signal=signal.SIGTERM,
        launcher_ready=(0, "coll_ready"),
        signal_after=1.0,
    )
    fault_delivered(r)
    stopped_at_ttrun(r, signal.SIGTERM)
    clean(r)


def test_survivor_tears_down_ulfm():
    """Rank 1 SIGKILLed with the mesh open, ULFM on: rank 0's wait sees the error, gives the SIGTERM its moment and
    leaves as a stopped rank, not as a second failure (its own close is the later teardown step)."""
    r = run_case(
        "cppmesh_survivor_ulfm",
        {
            "PROBE_SCENARIO": "sigkill",
            "PROBE_PHASE": "after_open",
            "PROBE_ON_ERROR": "continue",
            "TT_RUN_STOP_GRACE_S": "6",
        },
        DEADLINE_HANG,
        cmd=legacy_cmd("--with-ft ulfm"),
    )
    ended(r)
    tore_down(r, ranks=(0,), reason="SIGTERM")
    clean(r)
