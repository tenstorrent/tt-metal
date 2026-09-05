"""The gate must fire under the import shapes the TOOL actually uses, not the one a test picks.

Twice now the thermal gates were inert in production while every existing test passed, because the
tests imported the modules the convenient way and production does not:

  run.py    is loaded BY PATH (`spec_from_file_location("cc_optimize_run", ...)`, no package), so
            `from .perf_mcp import ...` raises "attempted relative import with no known parent
            package".
  probes.py is imported as `agent.probes` with perf_automation on sys.path, so
            `from ..cc_optimize.run import ...` raises "attempted relative import beyond top-level
            package".

Both sat inside `except: pass`. On 2026-08-29 the board held 99-103C for an hour with no gate
running at all and chips 2 and 3 stopped answering; the run after it failed the same way and was
caught only because the gate had learned to announce itself. These tests load the modules the way
the tool does and assert the gate actually speaks.
"""
from __future__ import annotations

import importlib.util
import io
import contextlib
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]


def _load_run_by_path():
    """Exactly what tt_hw_planner's _load_cc_runner does: no package, no sys.path entry."""
    spec = importlib.util.spec_from_file_location("cc_optimize_run", _PA / "cc_optimize" / "run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_py_reaches_perf_mcp_when_loaded_by_path():
    mod = _load_run_by_path()
    assert mod.__package__ in ("", None), "this test is meaningless unless the module has no package"
    assert mod._perf_mcp() is not None, "the gate cannot reach its own thermometer"


def test_the_gate_speaks_when_the_board_is_hot_and_run_py_was_loaded_by_path():
    mod = _load_run_by_path()
    mcp = mod._perf_mcp()
    # The suite conftest deliberately neuters thermal gating (PERF_MCP_MAX_START_TEMP_C=200, a fresh
    # board-state dir with no learned clamp history), so the THRESHOLD has to be pinned here too --
    # otherwise this asserts nothing and would have passed against both of the broken imports.
    saved_read, saved_wait = mcp._read_die_temp_c, mcp._THERMAL_WAIT_S
    saved_thresh, saved_ceiling = mcp._clamp_threshold_c, mcp._SAFETY_CEILING_C
    try:
        # The launch gate now holds work only at the SAFETY ceiling -- the 65C measurement gate fires
        # once at run start, not here. So a 99.5C board must be held by the ceiling, and the reading
        # has to FALL or the hold never ends (it has no deadline, on purpose).
        readings = iter([99.5, 99.5, 85.0, 64.0, 64.0, 64.0])
        mcp._read_die_temp_c = lambda: next(readings, 64.0)
        mcp._clamp_threshold_c = lambda: 65.0
        mcp._THERMAL_WAIT_S = 0.01
        mcp._COOLDOWN_POLL_S = 0.001
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            mod._wait_for_thermal_headroom_before_device_work("regression")
            # The watcher runs while work is IN FLIGHT, so it sees whatever the board is doing then,
            # not the cooled reading the ceiling just waited for. Put it back on a hot board.
            mcp._read_die_temp_c = lambda: 99.5
            state = mod._thermal_watch_new()
            mod._thermal_watch_sample(state, "regression")
        out = buf.getvalue()
    finally:
        mcp._read_die_temp_c, mcp._THERMAL_WAIT_S = saved_read, saved_wait
        mcp._clamp_threshold_c = saved_thresh
        mcp._SAFETY_CEILING_C = saved_ceiling
    assert "thermal-ceiling" in out, "the launch gate did not hold work on a 99.5C board"
    assert "thermal-watch" in out, "the mid-run watcher stayed silent on a 99.5C board"
    assert "WARNING" not in out, "the gate reported itself inert"


def test_probes_reaches_perf_mcp_when_imported_as_a_top_level_agent_package():
    if str(_PA) not in sys.path:
        sys.path.insert(0, str(_PA))
    for name in [n for n in list(sys.modules) if n == "agent" or n.startswith("agent.")]:
        del sys.modules[name]
    import agent.probes as probes

    assert probes.__package__ == "agent", "this test is meaningless unless probes is a top-level agent"
    assert probes._cc_optimize("perf_mcp") is not None, "thermal_yield cannot reach its thermometer"


def test_an_inert_gate_announces_itself_rather_than_failing_silently():
    if str(_PA) not in sys.path:
        sys.path.insert(0, str(_PA))
    import agent.probes as probes

    saved_resolver, saved_warned = probes._cc_optimize, probes._THERMAL_INERT_WARNED[0]

    def unreachable(_name):
        raise ImportError("simulated: the owner cannot be reached")

    try:
        probes._cc_optimize = unreachable
        probes._THERMAL_INERT_WARNED[0] = False
        probes._thermal_yield_last[0] = 0.0
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            probes.thermal_yield("regression")  # must NOT raise: a broken gate never blocks work
        out = buf.getvalue()
    finally:
        probes._cc_optimize = saved_resolver
        probes._THERMAL_INERT_WARNED[0] = saved_warned
    assert "WARNING" in out, "an inert gate said nothing -- the failure mode that cost two chips"


def test_the_safety_ceiling_holds_work_until_the_board_is_cool_again():
    """65C protects the READING; this protects the HARDWARE, and they are not the same number.

    The launch gate is satisfied once, before work starts, and says nothing afterwards -- so the
    board went 65C -> 99-103C twice in nine hours and chip 2 stopped answering at 98.8C and 98.7C.
    The ceiling is the second number: crossing it holds work until the board is back to the
    measurement threshold, and unlike the launch gate it does not give up and proceed hot.
    """
    mod = _load_run_by_path()
    mcp = mod._perf_mcp()
    saved_read, saved_poll = mcp._read_die_temp_c, mcp._COOLDOWN_POLL_S
    readings = iter([95.0, 95.0, 80.0, 64.0, 64.0, 64.0, 64.0])
    seen = []

    def falling():
        v = next(readings, 64.0)
        seen.append(v)
        return v

    try:
        mcp._read_die_temp_c = falling
        mcp._COOLDOWN_POLL_S = 0.001
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            fired = mcp.cool_if_over_safety_ceiling("regression")
        out = buf.getvalue()
    finally:
        mcp._read_die_temp_c, mcp._COOLDOWN_POLL_S = saved_read, saved_poll

    assert fired, "the board was above the ceiling and work was not held"
    assert "thermal-ceiling" in out, "the ceiling engaged without saying so"
    assert len(seen) >= 3, "it stopped reading before the board had cooled"


def test_the_safety_ceiling_ignores_an_ordinarily_warm_board():
    """A ceiling that fires at ordinary running temperature would pause the run constantly."""
    mod = _load_run_by_path()
    mcp = mod._perf_mcp()
    saved = mcp._read_die_temp_c
    try:
        mcp._read_die_temp_c = lambda: 80.0
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            fired = mcp.cool_if_over_safety_ceiling("regression")
    finally:
        mcp._read_die_temp_c = saved
    assert not fired and buf.getvalue() == "", "the ceiling fired on a normally warm board"
