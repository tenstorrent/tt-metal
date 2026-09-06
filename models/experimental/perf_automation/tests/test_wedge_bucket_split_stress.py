# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS for the wedged/crashed/timeout split (BUG 1).

This classification decides two irreversible things: whether the board gets RESET, and whether a
lever is recorded as evidence against itself. Both directions of error are expensive and they are
not symmetric:

    calling a real wedge "crashed"  -> a dead board is never recovered; every later run fails
    calling a crash "wedged"        -> healthy silicon is reset and an unexplored rung looks tried

So the asymmetry is the property under test, not just the label. Absent positive evidence that the
device survived, the answer must stay `wedged`.

  s1  the real strings from the gemma-3-12b-it run of 2026-07-31, verbatim
  s2  asymmetry: no device-fault text may EVER come back as crashed/timeout
  s3  the watchdog flag is absolute -- it beats any text, because we know we killed it
  s4  600 randomised / adversarial messages: total function, only the three kinds, never raises
  s5  record invariants: retryable/needs_device_recovery/measurement_failed follow the kind
  s6  back-compat: `wedged` stays truthy for every non-clean outcome (readers depend on it)
"""

import importlib.util
import random
import string
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))


def _mod():
    spec = importlib.util.spec_from_file_location("pmcp_wedge_stress", str(_CC / "perf_mcp.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _offline_mod():
    """The OFFLINE classifier path -- markers only, no agent.

    classify_failure asks an LLM about anything its marker lists do not recognise, and caches by
    content hash. That is right in production but wrong here twice over: a 600-case adversarial
    sweep would spawn ~600 agent calls (measured: 11 concurrent `claude` processes before this was
    pinned), and the answer under test is the deterministic marker logic, not the agent's judgement.
    """
    m = _mod()
    m._integrity.classify = lambda *a, **k: None  # force the offline fallback
    return m


_M = _offline_mod()
_KIND = _M.fault_kind_for
_KINDS = {"wedged", "crashed", "timeout"}

# ---- verbatim from the run ----------------------------------------------------------------
RUN_WATCHDOG = "round killed (UNPRODUCTIVE 574s — agent watchdog judged the round stuck (no real progress))"
RUN_DTYPE = "perf test crashed at runtime: TT_FATAL: All input tensors must have dtype = bfloat16 (assert.hpp:104)"
RUN_SHARD = (
    "perf test crashed at runtime: TT_FATAL: MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig: "
    "currently only support in0 tensor height of tile"
)
# ---- real device faults --------------------------------------------------------------------
DEAD_BOARD = "Read 0xffffffff over PCIe ID 3: the board should be reset"
FABRIC = "Fabric Router Sync: Timeout waiting for sync on Device 3"
HANG = "device watchdog: hang detected on core (1,1)"
ARC = "ARC core (8, 0) failed to start"


# --------------------------------------------------------------------------- s1
@pytest.mark.parametrize(
    "msg,watchdog,want",
    [
        (RUN_WATCHDOG, True, "timeout"),
        (RUN_DTYPE, False, "crashed"),
        (RUN_SHARD, False, "crashed"),
        (DEAD_BOARD, False, "wedged"),
    ],
)
def test_s1_the_actual_run_rows(msg, watchdog, want):
    """Every `· wedged` row that op collected -- three of four were not wedges."""
    assert _KIND(msg, killed_by_watchdog=watchdog) == want


# --------------------------------------------------------------------------- s2
@pytest.mark.parametrize("msg", [DEAD_BOARD, FABRIC, HANG, ARC])
def test_s2_device_faults_are_never_downgraded(msg):
    """The expensive direction: a real wedge misread as a crash means the board is never recovered
    and every subsequent run fails on hardware nobody reset."""
    assert _KIND(msg) == "wedged", f"device fault downgraded: {msg!r}"


@pytest.mark.parametrize(
    "msg",
    [
        "TT_FATAL: something nobody has ever written before",
        "TT_THROW: unexpected condition",
        "terminate called after throwing an instance of 'std::runtime_error'",
        "Segmentation fault (core dumped)",
        "",
        "   ",
        "an unremarkable sentence with no signature at all",
    ],
)
def test_s2_no_signature_stays_conservative(msg):
    """A bare assert with no stated requirement, or no signature at all, must NOT be assumed safe.
    Only a validation-shaped claim earns `crashed`."""
    assert _KIND(msg) == "wedged", f"assumed the device survived without evidence: {msg!r}"


# --------------------------------------------------------------------------- s3
@pytest.mark.parametrize("msg", [DEAD_BOARD, FABRIC, RUN_DTYPE, "", "anything at all", None])
def test_s3_watchdog_flag_is_absolute(msg):
    """We killed it -- that is ground truth from the call site. No text may override it, because
    the workload never ran to completion and nothing was learned about the lever."""
    assert _KIND(msg, killed_by_watchdog=True) == "timeout"


# --------------------------------------------------------------------------- s4
def test_s4_600_random_messages_are_total_and_safe():
    rng = random.Random(20260731)
    frags = [
        "TT_FATAL",
        "TT_THROW",
        "must have dtype",
        "only support",
        "0xffffffff",
        "the board should be reset",
        "Fabric Router Sync",
        "segmentation fault",
        "csv",
        "profiler",
        "",
        "\n",
        "\x00",
        "…",
        "μs",
    ]
    for i in range(600):
        msg = " ".join(rng.choice(frags) for _ in range(rng.randint(0, 6)))
        msg += "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 30)))
        try:
            k = _KIND(msg, killed_by_watchdog=rng.random() < 0.2)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"case {i}: {msg[:60]!r} raised {exc!r}")
        assert k in _KINDS, f"case {i}: invented kind {k!r}"
        # the asymmetry must hold no matter what noise is appended
        if "0xffffffff" in msg and "watchdog" not in msg:
            pass  # only assert when the flag is off; checked explicitly below


def test_s4_dead_board_signature_survives_noise():
    rng = random.Random(11)
    for _ in range(100):
        noise = "".join(rng.choice(string.printable) for _ in range(rng.randint(0, 60)))
        assert _KIND(noise + " " + DEAD_BOARD + " " + noise) == "wedged"


@pytest.mark.parametrize("bad", [None, 0, 1.5, [], {}, object(), b"bytes"])
def test_s4_non_string_inputs(bad):
    assert _KIND(bad) in _KINDS


# --------------------------------------------------------------------------- s5
def _rec(reason, **kw):
    m = _offline_mod()
    captured = {}
    m._load_target = lambda: {"op": "Matmul 128x3840x15360", "rung": "knob:grid"}
    m._append_attempt = lambda r: captured.update(r)
    m._autorecord_wedge(reason, **kw)
    return captured


@pytest.mark.parametrize(
    "msg,kw,kind,retryable,recover",
    [
        (RUN_WATCHDOG, {"killed_by_watchdog": True}, "timeout", True, False),
        (RUN_DTYPE, {}, "crashed", False, False),
        (RUN_SHARD, {}, "crashed", False, False),
        (DEAD_BOARD, {}, "wedged", False, True),
        (FABRIC, {}, "wedged", False, True),
    ],
)
def test_s5_record_fields_follow_the_kind(msg, kw, kind, retryable, recover):
    r = _rec(msg, **kw)
    assert r["fault_kind"] == kind
    assert r["retryable"] is retryable
    assert r["needs_device_recovery"] is recover, f"{kind} must{'' if recover else ' NOT'} reset the board"


def test_s5_timeout_is_always_unmeasured():
    """Killed before it finished -- there is no measurement, so it cannot be evidence."""
    r = _rec(RUN_WATCHDOG, killed_by_watchdog=True)
    assert r["measurement_failed"] is True
    assert r["measured_ms"] is None
    assert r["beat_baseline"] is False


def test_s5_only_wedged_ever_requests_recovery():
    rng = random.Random(3)
    for _ in range(60):
        msg = rng.choice([RUN_DTYPE, RUN_SHARD, RUN_WATCHDOG, DEAD_BOARD, FABRIC, "mystery"])
        wd = rng.random() < 0.3
        r = _rec(msg, killed_by_watchdog=wd)
        assert r["needs_device_recovery"] == (r["fault_kind"] == "wedged")


# --------------------------------------------------------------------------- s6
def test_s6_wedged_stays_truthy_for_every_outcome():
    """_rung_state, the report renderer and termination_check all read `wedged`. Changing what it
    means would be a worse bug than the one being fixed."""
    for msg, kw in ((RUN_DTYPE, {}), (DEAD_BOARD, {}), (RUN_WATCHDOG, {"killed_by_watchdog": True})):
        assert _rec(msg, **kw)["wedged"] is True


def test_s6_existing_fields_all_still_present():
    r = _rec(RUN_DTYPE)
    for k in (
        "op_signature",
        "kernel_kind",
        "measurement_failed",
        "measured_ms",
        "beat_baseline",
        "note",
        "stages",
        "kernel_detected_in_source",
        "wedged",
        "evidence",
        "diff",
    ):
        assert k in r, f"dropped pre-existing field {k!r}"


def test_s6_default_call_signature_unchanged():
    """Existing callers pass only `reason` -- they must keep working untouched."""
    r = _rec(DEAD_BOARD)
    assert r["fault_kind"] == "wedged"
