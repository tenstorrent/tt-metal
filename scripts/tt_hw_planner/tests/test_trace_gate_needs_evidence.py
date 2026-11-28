"""The trace gate must not waive its requirement on no evidence.

`classify_trace_verdict` waives the trace requirement when ungraduated modules are present:
those modules are legitimately still eager, so the pipeline cannot be expected to trace. That
waiver is an argument FROM EVIDENCE -- "these named modules are still eager".

`read_graduation` returns an empty mapping for several distinct reasons: the demo dir has no
`bringup_status.json` (a composite keeps status per component, not in the e2e dir), the JSON
will not parse, or the import it needs is unavailable. `trace_policy` collapsed all of those
into the same shape as "every module is ungraduated", so the gate took the waiver branch with
no modules to name -- emitting a bare "?" where the justification belonged, and passing a
half-checked gate. Observed on a composite e2e run, which reported

    verdict=EAGER_WAIVED (0 graduated, 0 ungraduated): trace not engaged;
    eager permitted because ungraduated module(s) present: ?

Absence of evidence is not proof, so an unreadable graduation state now FAILS.
"""

from scripts.tt_hw_planner import trace_gate as tg

_NO_TRACE = {"trace_1cq": False}


def test_policy_reports_whether_graduation_is_known():
    assert tg.trace_policy({})["known"] is False
    assert tg.trace_policy({"a": "sharded"})["known"] is True
    assert tg.trace_policy({"a": None})["known"] is True


def test_unreadable_graduation_no_longer_waives():
    """The regression: no evidence used to waive, and now fails."""
    verdict, reason = tg.classify_trace_verdict(_NO_TRACE, tg.trace_policy({}))
    assert verdict == "FAIL"
    assert "could not be read" in reason
    assert "?" not in reason


def test_the_real_waiver_still_works_and_names_its_evidence():
    """A genuinely eager module still waives, and says which one."""
    verdict, reason = tg.classify_trace_verdict(_NO_TRACE, tg.trace_policy({"a": "sharded", "b": None}))
    assert verdict == "EAGER_WAIVED"
    assert "b" in reason


def test_all_graduated_still_requires_trace():
    verdict, reason = tg.classify_trace_verdict(_NO_TRACE, tg.trace_policy({"a": "sharded"}))
    assert verdict == "FAIL"
    assert "eager not permitted" in reason


def test_engaged_trace_passes_regardless_of_graduation_knowledge():
    """Trace actually engaging is proof on its own; it needs no graduation data."""
    for graduation in ({}, {"a": "sharded"}, {"a": None}):
        verdict, _ = tg.classify_trace_verdict({"trace_1cq": True}, tg.trace_policy(graduation))
        assert verdict == "PASS"


def test_overflow_proof_path_is_untouched():
    """A verified physical overflow still waives when every module graduated."""
    policy = tg.trace_policy({"a": "sharded"})
    verdict, reason = tg.classify_trace_verdict(
        _NO_TRACE, policy, allow_no_trace=True, overflow_proof={"required_bytes": 999, "budget_bytes": 10}
    )
    assert verdict == "EAGER_WAIVED"
    assert "verified physical overflow" in reason


def test_unreadable_graduation_is_not_rescued_by_the_proof_flag_alone():
    """`allow_no_trace` is not itself evidence: without a real overflow it must still fail."""
    verdict, _ = tg.classify_trace_verdict(
        _NO_TRACE, tg.trace_policy({}), allow_no_trace=True, overflow_proof={"required_bytes": 10, "budget_bytes": 999}
    )
    assert verdict == "FAIL"


def test_waiver_reason_never_degenerates_to_a_placeholder():
    """Any waiver must name at least one module; a placeholder means it had no evidence."""
    for graduation in ({"a": None}, {"a": "sharded", "b": None}, {"x": None, "y": None}):
        verdict, reason = tg.classify_trace_verdict(_NO_TRACE, tg.trace_policy(graduation))
        if verdict == "EAGER_WAIVED":
            named = reason.split(":")[-1].strip()
            assert named and named != "?", reason
