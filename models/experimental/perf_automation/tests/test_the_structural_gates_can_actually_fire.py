# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Three transforms that were in the source and absent from every run.

fold, order and conv-prep take their work queue from `open_ops` -- a list residual_report builds and
returns in ITS OWN dict, never a field on the profile. termination_check handed them the profile, so
all three read an empty list, bailed on their first applicability check, and had fired exactly zero
times since the day they landed. A 91-attempt run on a model with 68 conv ops and 13862 datamove ops
recorded no fold, order or conv-prep row at all.

What let it survive: each gate's tests hand it `{"device_ms": ..., "open_ops": [...]}` directly --
the shape the gate expects and NOT the shape its caller had -- and each gate's wiring test matched
the literal `_conv_gate(prof, attempts)`, asserting the call existed while asserting nothing about
the dict going into it. The logic was pinned to the millimetre and the join between the two ends was
never exercised. So these tests are about the JOIN, and the gate-logic tests stay where they are.

Firing is only half of clearing. A gate retires on a measured win or after N recorded attempts, and
the kinds these three name -- fold, order, conv-prep -- were not in the set record_kernel_attempt
takes without a custom-kernel marker, which a restructure does not leave. A gate that fires but
cannot be cleared is worse than one that never fires: it re-emits the same target forever. The host
bucket deadlocked in exactly that way once already.
"""

import importlib.util
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PA))

from agent import roofline as R  # noqa: E402

_SRC = (_PA / "cc_optimize" / "perf_mcp.py").read_text()
_GATES = ("_host_gate", "_decode_gate", "_conv_gate", "_fold_gate", "_order_gate")
HW = {"dram_bw_gbps": 512.0, "worker_cores": 110, "mesh_chips": 1, "peak_tflops_per_core": {"lofi": 4.0, "hifi4": 1.0}}


def _mcp():
    spec = importlib.util.spec_from_file_location("_pm_gates", _PA / "cc_optimize" / "perf_mcp.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _matmul(name, count, ms):
    return {
        "op_code": "MatmulDeviceOperation %s" % name,
        "shape": "512x3840 @ 3840x15360",
        "device_ms": ms,
        "count": count,
        "bytes": 1e9,
        "cores": 8,
        "fidelity": "lofi",
        "grid": "partial",
        "memory": "dram_interleaved",
    }


def _profile():
    """Three fingerprints at the per-layer mode and one running 3x within the layer.

    Deliberately built the way a capture is -- buckets of top_ops -- rather than as the open_ops the
    gate wants, because the step being tested is precisely the one that turns the first into the
    second.

    The cheap op earns its place. The dispatch floor is self-calibrated as the SMALLEST per-call
    time in the capture, so with only the four below the 3x op calibrates the floor off its own
    per-call time, lands exactly on it, and is dropped as at_floor -- removing the single op the
    fold gate exists to find. Something cheaper has to set that floor.
    """
    tops = [
        _matmul("cheap", 30, 3.0),
        _matmul("A", 30, 50.0),
        _matmul("B", 30, 48.0),
        _matmul("C", 30, 46.0),
        _matmul("D", 90, 44.0),
    ]
    return {"device_ms": sum(t["device_ms"] for t in tops), "buckets": [{"id": "matmul", "top_ops": tops}]}


def _gate_view(prof):
    """The dict termination_check assembles, built here the one way it is built there."""
    return {**prof, "open_ops": R.residual_report(prof, HW).get("open_ops") or []}


# ---------------------------------------------------------------- the join


def test_the_work_queue_does_not_reach_the_gate_on_the_profile_alone(monkeypatch):
    """The bug, stated as a fact about the two dicts: one carries open_ops and the other never does.

    Not a tautology about a missing key -- it is the whole reason three gates never ran. If some
    later change does start putting open_ops on the profile, this failing is the right outcome: the
    merge below would then be redundant and should go.
    """
    prof = _profile()
    assert "open_ops" not in prof

    # The fixture has to actually present a fold for the gate to find, or the gate returning None
    # below says nothing at all. Stated as the gate's own preconditions, so a change in how the
    # roofline models these ops fails HERE, naming the reason, instead of further down as a silent
    # False. That is not hypothetical: at_floor filtering removed this op the first time round.
    ops = [o for o in _gate_view(prof)["open_ops"] if int(o.get("count") or 0) > 0]
    counts = [int(o["count"]) for o in ops]
    assert len(ops) >= 3, "too few open ops for a per-layer mode to mean anything: %r" % (counts,)
    mode = max(set(counts), key=lambda c: (counts.count(c), -c))
    assert any(c >= 2 * mode and c % mode == 0 for c in counts), "no repeat within a layer to find: %r" % (counts,)

    m = _mcp()
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    assert m._fold_gate(prof, []) is None, "the profile alone cannot make this gate fire -- it never could"


def test_the_gate_fires_on_the_view_the_caller_now_builds(monkeypatch):
    """Same profile, same fold to find, reached through the dict termination_check assembles."""
    m = _mcp()
    monkeypatch.setattr(m, "_load_attempts", lambda: [])
    block = m._fold_gate(_gate_view(_profile()), [])
    assert block is not None, "the fold gate still cannot see a 3x repeat that the roofline reports"
    assert block["next_rung"] == "structural-fold"
    assert "D" in block["op"], block["op"]


def test_termination_check_assembles_that_view_from_the_roofline_report():
    assert '_gate_prof = {**prof, "open_ops": rep.get("open_ops") or []}' in _SRC


def test_no_gate_is_handed_the_dict_without_the_work_queue():
    """Including the two that read other fields today.

    A gate added later will be copied from the line above it, so the line above it has to be right.
    """
    for name in _GATES:
        assert "%s(prof," % name not in _SRC, "%s is back on the dict that has no open_ops" % name
        assert "%s(_gate_prof," % name in _SRC, "%s is not consulted at all" % name


# ---------------------------------------------------------------- and can then be cleared


def test_every_kind_a_gate_names_is_one_the_recorder_accepts():
    """The gate's instruction and the recorder's vocabulary are two ends of one contract.

    Each gate's reason text tells the agent verbatim which kind to record. A kind the recorder does
    not know is refused for want of a kernel marker that a restructure never leaves, and the gate
    goes on blocking on the strength of an attempt that could not be written down.
    """
    m = _mcp()
    assert "} | _GATE_KINDS" in _SRC, "record_kernel_attempt no longer accepts the gates' own levers"
    for kinds, _cap_env in m._GATE_LEVERS:
        assert set(kinds) <= m._GATE_KINDS


def test_each_gate_asks_for_a_kind_from_its_own_lever(monkeypatch):
    """Read off the reason the gate actually emits, not off a list restating what it should say."""
    m = _mcp()
    monkeypatch.setattr(m, "_load_attempts", lambda: [])

    reason = m._fold_gate(_gate_view(_profile()), [])["reason"]
    assert any("'%s'" % k in reason for k in m._FOLD_LEVER[0]), reason

    convs = [{"op_code": "Conv2d", "bucket": "conv_pool", "gap_ms": 20.0, "count": 4}]
    reason = m._conv_gate({"device_ms": 100.0, "open_ops": convs}, [])["reason"]
    assert any("'%s'" % k in reason for k in m._CONV_LEVER[0]), reason


def test_the_recorder_permits_as_many_tries_as_the_gate_will_ask_for():
    """THE DEADLOCK, pinned per lever.

    Every gate keeps re-emitting its target until a win or its cap. Every non-knob rung used to get
    a single attempt. So the agent recorded once, was refused as CLOSED on the next round, and had
    no move left that could clear a gate still blocking -- and `none: <evidence>`, which these gates
    invite by name, changes no code, so the profile is identical and the same target returns.
    """
    m = _mcp()
    for kinds, cap_env in m._GATE_LEVERS:
        cap = m._gate_cap(cap_env)
        for kind in kinds:
            _tries, allowed = m._rung_allowance("MatmulDeviceOperation D", kind, [])
            assert allowed >= cap, "%s: gate demands %d attempts, recorder permits %d" % (kind, cap, allowed)


def test_the_last_try_before_the_cap_is_still_accepted():
    """Counted the way record_kernel_attempt counts them, so an off-by-one shows up here."""
    m = _mcp()
    for kinds, cap_env in m._GATE_LEVERS:
        cap = m._gate_cap(cap_env)
        spent = [
            {"op_signature": "MatmulDeviceOperation D", "kernel_kind": kinds[0], "measured_ms": 1.0}
            for _ in range(cap - 1)
        ]
        tries, allowed = m._rung_allowance("MatmulDeviceOperation D", kinds[0], spent)
        assert tries < allowed, "the attempt that would retire the gate is refused"


def test_a_deep_rung_no_gate_re_asks_for_still_gets_exactly_one():
    """The allowance was widened for gate levers only. One measured kernel is still the whole budget
    for tt-lang and C++, because nothing is re-asking for those."""
    m = _mcp()
    for kind in ("tt-lang", "cpp", "tp-fracture"):
        _tries, allowed = m._rung_allowance("MatmulDeviceOperation D", kind, [])
        assert allowed == 1, "%s allowance changed" % kind


def test_the_lever_names_are_written_once():
    """Three places have to agree on them; three copies is how they stopped agreeing."""
    body = _SRC[_SRC.index("def _fold_gate") :]
    for literal in ('("fold", "structural-fold")', '("order", "structural-order")'):
        assert literal not in body, "%s is spelled out again inside a gate" % literal
    for cap_env in ("PERF_MCP_MAX_FOLD_ATTEMPTS", "PERF_MCP_MAX_ORDER_ATTEMPTS", "PERF_MCP_MAX_CONV_ATTEMPTS"):
        # Read through _gate_cap and nowhere else, so the gate's cap and the recorder's allowance
        # cannot come from two different readings of the same variable. Prose may still name it.
        assert 'os.environ.get("%s"' % cap_env not in _SRC, "%s is read outside _gate_cap" % cap_env
