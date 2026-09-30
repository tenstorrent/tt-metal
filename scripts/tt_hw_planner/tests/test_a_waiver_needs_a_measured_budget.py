# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The trace requirement may be waived only by a MEASURED physical budget, never by giving up.

G6 exists so a pipeline cannot be called trace-ready without tracing. It has one escape hatch: a
proof that the trace physically cannot fit. `valid_overflow_proof` accepted any proof where
required > budget, and `overflow_fix_loop` filled `budget_bytes` with 0 when it ran out of rounds --
so anything exceeded it, every unfixed memory failure minted a valid proof, and the hatch was
permanently open.

It fired for the first time on a Qwen-Image-Edit T3K run, 2026-09-30. A VAE decode could not
allocate a 100663296 B DRAM buffer across 12 banks; the fix loop could not help; the gate recorded

    verdict: EAGER_WAIVED
    trace waived: verified physical overflow required=191102976 > budget=0

and returned can_stop=true. No trace was ever captured, there was no trace_caps sidecar, and the run
was one gate away from writing a PASS and a README claiming the pipeline traces.

It had never bitten before because the hatch only opens when a capture fails with an
out-of-memory-shaped error, and no earlier model's capture did.

The second half of the defect is that the remedy never fitted this failure. The loop's fix is to GROW
the trace region, which is right when the trace region overflowed and actively harmful otherwise: a
bigger trace region leaves LESS device memory, so on a plain buffer allocation the loop made the
fault worse on each of its three doublings before waiving the gate.
"""

from __future__ import annotations

from scripts.tt_hw_planner import trace_gate as tg

_NO_TRACE = {"trace_1cq": False}
# the literal numbers from the run that exposed this
_THE_BOGUS_PROOF = {"required_bytes": 191102976, "budget_bytes": 0, "rounds": 3}
_DRAM_OOM = (
    "invalid  TT_FATAL: Out of Memory: Not enough space to allocate 100663296 B DRAM buffer "
    "across 12 banks, where 8388608 B were requested"
)


# --- the proof ------------------------------------------------------------------------------------


def test_the_proof_that_waived_the_real_run_is_rejected():
    assert tg.valid_overflow_proof(_THE_BOGUS_PROOF) is False


def test_a_zero_or_negative_budget_is_not_a_budget():
    assert tg.valid_overflow_proof({"required_bytes": 10, "budget_bytes": 0}) is False
    assert tg.valid_overflow_proof({"required_bytes": 10, "budget_bytes": -1}) is False


def test_a_measured_budget_still_waives():
    """The hatch must stay usable: a genuinely non-traceable model is why it exists."""
    assert tg.valid_overflow_proof({"required_bytes": 191102976, "budget_bytes": 100663296}) is True


def test_a_requirement_that_fits_is_still_no_waiver():
    assert tg.valid_overflow_proof({"required_bytes": 10, "budget_bytes": 999}) is False


def test_malformed_proofs_are_still_rejected():
    for bad in (
        None,
        "nope",
        {},
        {"required_bytes": 10},
        {"budget_bytes": 10},
        {"required_bytes": "x", "budget_bytes": 1},
    ):
        assert tg.valid_overflow_proof(bad) is False, bad


# --- what the gate does with it -------------------------------------------------------------------


def test_the_gate_now_FAILS_where_it_waived():
    """End of the chain: the same inputs that produced EAGER_WAIVED must produce FAIL."""
    policy = tg.trace_policy({"a": "sharded", "b": "native"})  # all graduated -> trace required
    verdict, reason = tg.classify_trace_verdict(_NO_TRACE, policy, allow_no_trace=True, overflow_proof=_THE_BOGUS_PROOF)
    assert verdict == "FAIL", verdict
    assert "eager not permitted" in reason


def test_a_real_proof_still_reaches_the_waiver():
    policy = tg.trace_policy({"a": "sharded"})
    verdict, reason = tg.classify_trace_verdict(
        _NO_TRACE, policy, allow_no_trace=True, overflow_proof={"required_bytes": 200, "budget_bytes": 100}
    )
    assert verdict == "EAGER_WAIVED"
    assert "200" in reason and "100" in reason


# --- the remedy must fit the failure --------------------------------------------------------------


def test_a_dram_allocation_failure_does_not_get_the_region_remedy():
    """Growing the trace region takes memory AWAY from what just ran out of it."""
    calls = {"n": 0}

    def _cap(demo):
        calls["n"] += 1
        return _NO_TRACE, _DRAM_OOM

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=5, base_region=1000)
    assert calls["n"] == 1, "it grew the region on a failure growing cannot fix"
    assert res["resolved"] is False and res["proof"] is None
    assert "DRAM" in res["detail"], "the real fault must survive into the detail"


def test_a_trace_region_overflow_still_gets_it():
    calls = {"n": 0}

    def _cap(demo):
        calls["n"] += 1
        return _NO_TRACE, "trace region overflow persists"

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=3, base_region=1000)
    assert calls["n"] == 3, "a region overflow is exactly what growing is for"
    assert res["proof"] is None


def test_a_resolved_overflow_is_unchanged():
    seq = [(_NO_TRACE, "trace region overflow"), ({"trace_1cq": True}, "ok")]

    def _cap(demo):
        return seq.pop(0)

    res = tg.overflow_fix_loop("x", capture_fn=_cap, max_rounds=3, base_region=1000)
    assert res["resolved"] is True and res["proof"] is None
    assert "traced at region=2000" in res["detail"]


def test_the_two_marker_sets_say_different_things():
    assert tg._is_overflow(_DRAM_OOM) is True, "it is still an allocation failure"
    assert tg._is_region_overflow(_DRAM_OOM) is False, "but not one growing the region can fix"
    assert tg._is_region_overflow("trace_region too small") is True


def test_it_names_no_model_or_stage():
    import ast
    import inspect
    import textwrap

    for fn in (tg.valid_overflow_proof, tg._is_region_overflow, tg.overflow_fix_loop):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        low = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "vision", "vae", "demos/"):
            assert name not in low, f"{name!r} in {fn.__name__}"
