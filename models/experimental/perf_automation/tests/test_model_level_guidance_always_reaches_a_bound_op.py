"""Model-scoped guidance must reach the agent whatever op it happens to be standing on.

recall_knobs routes by op_class, and GUIDELINES 13's sections declare `op_class: matmul`. That is
correct for what they describe -- weight streaming is a matmul concern -- but it means the agent only
sees them while working a matmul. On gemma-3-12b-it run 22 that was 1 of 52 recall_knobs calls: the
run spent its first two hours on reduction, eltwise, datamove and RoPE, and the only guidance capable
of addressing the actual bottleneck was never handed over.

The bottleneck does not belong to an op. 34.4 ms of producer wait on a GeGLU mul is the decode weight
stream stalling; the agent reading that op's guidance learns about eltwise. Whichever op it is
standing on, if the profile is memory-bound the model-level levers are the relevant knowledge.

So they are APPENDED to every memory-bound recall, regardless of op_class, and marked as such. This
is not a rung and does not make them a target -- can_stop still does not wait for them. It only
guarantees the knowledge is present when it applies.

Compute-bound and unknown-bound recalls are untouched: prefetch hides DRAM latency, and on a
FLOP-bound op there is none to hide.
"""

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))


@pytest.fixture()
def mcp(tmp_path, monkeypatch):
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("PERF_MCP_LEDGER_DIR", str(tmp_path))
    import models.experimental.perf_automation.cc_optimize.perf_mcp as m

    importlib.reload(m)
    return m


def _ids(res):
    return [k.get("id") for k in (res or {}).get("known_knobs", []) if isinstance(k, dict)]


# ---------------------------------------------------------------- it always arrives when bound=memory


@pytest.mark.parametrize("op_class", ["reduction", "eltwise", "datamove", "attention", "other"])
def test_a_memory_bound_op_of_any_class_gets_the_model_levers(mcp, op_class):
    """The run-22 case: two hours on reduction/eltwise/datamove, never handed the one relevant
    section because it declares op_class: matmul."""
    got = _ids(mcp.recall_knobs(op_class, bound_by="memory"))
    assert "model-prefetch" in got, (op_class, got)


def test_a_matmul_still_gets_them(mcp):
    assert "model-prefetch" in _ids(mcp.recall_knobs("matmul", bound_by="memory"))


def test_the_wait_diagnostic_comes_too(mcp):
    """Reading FW vs kernel is what tells the agent the op is idle rather than slow -- without it the
    prefetch advice has no evidence behind it."""
    got = _ids(mcp.recall_knobs("eltwise", bound_by="memory"))
    assert "model-read-the-wait" in got, got


# ---------------------------------------------------------------- and stays away otherwise


def test_a_host_bound_matmul_gets_the_host_loop_section(mcp):
    """Symmetric gap: model-host-loop declares op_class host_fallback,other, so a host-bound MATMUL
    -- the shape the decode step actually presents -- never saw it either. The rule is bound-driven,
    so the right model-level section arrives for whichever bound was diagnosed."""
    got = _ids(mcp.recall_knobs("matmul", bound_by="host"))
    assert "model-host-loop" in got, got


def test_a_dram_bound_op_is_not_told_about_the_host_loop(mcp):
    """Bound-driven cuts both ways: host round-trips are not what a DRAM-bound op is waiting on."""
    assert "model-host-loop" not in _ids(mcp.recall_knobs("reduction", bound_by="memory"))


def test_a_compute_bound_op_does_not_get_them(mcp):
    """Prefetch hides DRAM latency; a FLOP-bound op has none to hide, and unsolicited advice on every
    call is how a catalogue stops being read."""
    assert "model-prefetch" not in _ids(mcp.recall_knobs("matmul", bound_by="compute"))


def test_an_unspecified_bound_does_not_force_them(mcp):
    """Only append on evidence. No bound stated is not evidence of a memory bound."""
    assert "model-prefetch" not in _ids(mcp.recall_knobs("reduction"))


# ---------------------------------------------------------------- it does not displace anything


def test_the_op_specific_guidance_is_still_first(mcp):
    """Appended, not promoted. The op's own levers are what the named rung needs."""
    got = _ids(mcp.recall_knobs("reduction", bound_by="memory"))
    assert got and got[0] != "model-prefetch", got


def test_nothing_is_duplicated_for_a_matmul(mcp):
    """A matmul already routes to them; appending must not list them twice."""
    got = _ids(mcp.recall_knobs("matmul", bound_by="memory"))
    assert got.count("model-prefetch") == 1, got


def test_a_model_section_is_marked_as_model_scoped(mcp):
    """It is not this op's rung. Unmarked, it gets filed under whichever op the agent was standing
    on -- run 20 recorded the prefetcher as `Matmul 32x3840x15360 / shard`."""
    got = mcp.recall_knobs("reduction", bound_by="memory")["known_knobs"]
    scopes = {k["id"]: k.get("scope") for k in got}
    assert scopes.get("model-prefetch") == "model", scopes


def test_the_op_sections_are_not_marked_model(mcp):
    got = mcp.recall_knobs("matmul", bound_by="memory")["known_knobs"]
    assert all(k.get("scope") == "op" for k in got if not str(k["id"]).startswith("model-"))


def test_the_call_never_raises(mcp):
    """It runs before every rung's edit; a bad op_class must degrade, not end the round."""
    for oc in ("", "nonsense", "matmul"):
        assert isinstance(mcp.recall_knobs(oc, bound_by="memory"), dict)
