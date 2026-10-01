# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""This suite must not be able to touch the device, and must say so loudly if it tries.

Several tests here drive the real gate, and the gate's failure paths end in a reset that begins by
SIGKILLing every process holding /dev/tenstorrent. A test that reaches it unpatched does not fail --
it passes, having killed whatever real work was on the board.

That happened twice. On 2026-09-29 at 20:36:05 this suite killed a live trace capture 8.6 minutes
into its run; the gate then reported "terminated by SIGKILL (rc=-9)" and the round was recorded as a
G6 failure of the model. The audits that missed it both times looked for DIRECT calls to the reaper,
and both times the route in was a production function that reaps several frames down
(_retry_after_wedge -> _device_reset -> recover -> reap_device_holders) -- which no scan of test
bodies can see.

So conftest replaces the primitives that touch hardware for the whole suite. These tests are the
guard's own guard: they fail if the block is removed, narrowed, or quietly made harmless.
"""

from __future__ import annotations

import importlib

import pytest

from scripts.tt_hw_planner.tests.conftest import _DEVICE_PRIMITIVES


def test_enumerating_device_holders_is_blocked():
    """The scan the reaper kills from. Blocked at the bottom, so everything above it is covered."""
    dr = importlib.import_module("models.experimental.perf_automation.agent.device_recovery")
    with pytest.raises(AssertionError, match="reached the real device"):
        dr.device_holders()


def test_the_shared_reset_is_blocked():
    pr = importlib.import_module("models.experimental.perf_automation.agent.probes")
    with pytest.raises(AssertionError, match="reached the real device"):
        pr._device_reset()


def test_the_recovery_primitive_is_blocked():
    dr = importlib.import_module("models.experimental.perf_automation.agent.device_recovery")
    with pytest.raises(AssertionError, match="reached the real device"):
        dr.recover()


def test_the_mesh_reclaim_is_blocked():
    tg = importlib.import_module("scripts.tt_hw_planner.trace_gate")
    with pytest.raises(AssertionError, match="reached the real device"):
        tg.reclaim_mesh()


def test_the_message_says_what_to_do_about_it():
    """A guard that only says "no" gets worked around; this one names the fix."""
    dr = importlib.import_module("models.experimental.perf_automation.agent.device_recovery")
    try:
        dr.device_holders()
    except AssertionError as exc:
        msg = str(exc)
    assert "Patch it" in msg and "SIGKILLs every process" in msg


def test_the_reaper_itself_is_NOT_blocked():
    """Deliberate: a test that patches `device_holders` and exercises the reaper must still work.

    The block is on the primitives, not the orchestration -- otherwise
    test_agent_leftover_wait's reaper test could not run at all."""
    dr = importlib.import_module("models.experimental.perf_automation.agent.device_recovery")
    assert callable(dr.reap_device_holders)
    dr.device_holders = lambda: set()  # what that test does, via monkeypatch
    try:
        assert dr.reap_device_holders() == []
    finally:
        importlib.reload(dr)


def test_every_primitive_the_guard_names_actually_exists():
    """A typo in the list is a silent hole: the attribute would simply never be replaced."""
    for mod_path, attr in _DEVICE_PRIMITIVES:
        mod = importlib.import_module(mod_path)
        assert hasattr(mod, attr), f"{mod_path}.{attr} does not exist -- the guard covers nothing"


def test_launching_a_run_on_the_board_is_blocked_too():
    """Not a kill, but the same accident: contention is how the kills started."""
    ptg = importlib.import_module("models.experimental.perf_automation.agent.perf_test_gen")
    with pytest.raises(AssertionError, match="reached the real device"):
        ptg._run_perf_node("some_node::t", {})
