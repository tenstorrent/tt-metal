# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""BUG 1: `wedged` is one bucket for three different events.

`_autorecord_wedge` writes `"wedged": True` UNCONDITIONALLY, even though classify_failure has
already worked out what actually happened. Observed in the gemma-3-12b-it run of 2026-07-31, where
one op collected three `· wedged` rows and not one of them was a wedged board:

    grid   -> "round killed (UNPRODUCTIVE 574s -- agent watchdog judged the round stuck)"
    dtype  -> "TT_FATAL: All input tensors must have dtype = bfloat16"
    shard  -> "TT_FATAL: MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: currently only
               support in0 tensor height of tile"

The two TT_FATALs are op VALIDATION rejecting an illegal config -- a clean assert, the device
untouched. The first is our own watchdog killing a round, so the lever was never measured at all.

Two costs, both real:
  * termination_check reads these rows to decide what is left. A `wedged` row reads as "tried and
    it broke", so a lever that was never measured looks explored and the ladder advances past it.
  * the supervisor treats a wedge as hardware and resets the board -- the `killed holders none`
    resets seen repeatedly, on healthy silicon.

The taxonomy already exists (FAULT_DEVICE / FAULT_MEASUREMENT / FAULT_UNKNOWN) and the record
throws it away. This keeps `wedged` for what the field means to existing readers -- "not a clean
measured result" -- and adds `fault_kind` so a reader can tell WHY, plus `retryable` so a lever
killed by our own watchdog is not counted as evidence against itself.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _mod():
    spec = importlib.util.spec_from_file_location("pmcp_wedge", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# Real strings from the gemma-3-12b-it run and from device_recovery's own dead-board test.
WATCHDOG = "round killed (UNPRODUCTIVE 574s — agent watchdog judged the round stuck (no real progress))"
TT_FATAL_DTYPE = "perf test crashed at runtime: TT_FATAL: All input tensors must have dtype = bfloat16 (assert.hpp:104)"
TT_FATAL_SHARD = (
    "perf test crashed at runtime: TT_FATAL: MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: "
    "currently only support in0 tensor height of tile"
)
DEAD_BOARD = "Read 0xffffffff over PCIe ID 3: the board should be reset"


def _classifier():
    m = _mod()
    fn = getattr(m, "fault_kind_for", None)
    if fn is None:
        pytest.fail(
            "perf_mcp has no fault_kind_for: _autorecord_wedge still writes wedged=True "
            "unconditionally, so a watchdog timeout and an op-validation TT_FATAL are recorded as "
            "identically as a dead board -- and each one resets healthy silicon."
        )
    return m, fn


def test_classifier_exists():
    assert _classifier()[1] is not None


# --------------------------------------------------------------------------- the three kinds
def test_watchdog_timeout_is_a_timeout():
    _m, fn = _classifier()
    assert fn(WATCHDOG, killed_by_watchdog=True) == "timeout"


def test_watchdog_flag_wins_over_text():
    """WE killed it -- that is known at the call site and needs no parsing. Even text that looks
    like a device fault must not override the fact that the workload never got to finish."""
    _m, fn = _classifier()
    assert fn(DEAD_BOARD, killed_by_watchdog=True) == "timeout"


@pytest.mark.parametrize("msg", [TT_FATAL_DTYPE, TT_FATAL_SHARD])
def test_op_validation_fatal_is_a_crash_not_a_wedge(msg):
    """The op rejected an illegal config. The board is fine; the lever is simply inapplicable."""
    _m, fn = _classifier()
    assert fn(msg) == "crashed"


def test_dead_board_is_a_wedge():
    _m, fn = _classifier()
    assert fn(DEAD_BOARD) == "wedged"


def test_unknown_stays_conservative():
    """No recognisable signature -> assume the board may be gone. Under-reacting to a real dead
    board is worse than an unnecessary reset."""
    _m, fn = _classifier()
    assert fn("something nobody has ever seen before") == "wedged"


# --------------------------------------------------------------------------- the record
def _rec(m, reason, **kw):
    captured = {}
    m._load_target = lambda: {"op": "Matmul 128x3840x15360", "rung": "knob:grid"}
    m._append_attempt = lambda r: captured.update(r)
    m._autorecord_wedge(reason, **kw)
    return captured


def test_record_carries_fault_kind():
    m, _fn = _classifier()
    assert _rec(m, TT_FATAL_DTYPE).get("fault_kind") == "crashed"
    assert _rec(m, DEAD_BOARD).get("fault_kind") == "wedged"
    assert _rec(m, WATCHDOG, killed_by_watchdog=True).get("fault_kind") == "timeout"


def test_timeout_is_retryable_and_not_evidence():
    """A lever killed by our own watchdog was never measured, so it must not count against itself
    in the ladder -- that is what made an unexplored rung look explored."""
    m, _fn = _classifier()
    r = _rec(m, WATCHDOG, killed_by_watchdog=True)
    assert r.get("retryable") is True
    assert r.get("measurement_failed") is True
    assert r.get("beat_baseline") is False


def test_crash_is_not_retryable_but_is_measured_evidence():
    """An illegal config is a real answer about the lever: it does not apply to this op. Retrying
    it verbatim would fail identically."""
    m, _fn = _classifier()
    r = _rec(m, TT_FATAL_DTYPE)
    assert r.get("retryable") is False


def test_wedged_field_is_preserved_for_existing_readers():
    """_rung_state, the report renderer and termination_check all read `wedged`. Keep it truthy for
    every non-clean outcome so nothing silently changes meaning; fault_kind carries the detail."""
    m, _fn = _classifier()
    for msg, kw in ((TT_FATAL_DTYPE, {}), (DEAD_BOARD, {}), (WATCHDOG, {"killed_by_watchdog": True})):
        assert _rec(m, msg, **kw).get("wedged") is True


def test_only_a_real_device_fault_asks_for_recovery():
    """The supervisor resets the board on a wedge. Only a genuine device fault should reach it."""
    m, _fn = _classifier()
    assert _rec(m, DEAD_BOARD).get("needs_device_recovery") is True
    assert _rec(m, TT_FATAL_DTYPE).get("needs_device_recovery") is False
    assert _rec(m, WATCHDOG, killed_by_watchdog=True).get("needs_device_recovery") is False


@pytest.mark.parametrize("bad", [None, "", 0, [], {}])
def test_hostile_reasons_never_raise(bad):
    m, fn = _classifier()
    assert fn(bad) in ("wedged", "crashed", "timeout")
    _rec(m, bad)
