"""What recall_knobs must hold true for, independent of any one guidance section.

REPLACES test_model_level_guidance_always_reaches_a_bound_op.py, whose premise was reverted in
4a476112eb along with GUIDELINES 13 and the `model-prefetch` knob it asserted. That revert stands on
measurement: the gate behind the section fired on `FW - kernel` read as producer wait, and FW
durations OVERLAP, so summing them across buckets counts the same wall-clock repeatedly --

    sum of FW durations : 634.4 ms
    sum of KERNEL       : 408.9 ms
    total device_ms     : 379.7 ms      <- FW sums to 1.67x the real device time

which is how a "207 ms of idle" figure appeared on gemma-3-12b-it when the step was busy computing.
The lever it pointed at also cannot run: tech_reports/SubDevices/SubDevices.md lists "programs can
only span one sub-device" and "programs cannot be rerun on a different sub-device-manager
configuration" as unimplemented, and a prefetcher-enabled model doing prefill and decode hits one or
the other in every configuration.

The old file outlived its subject. Ten of its twelve tests demanded a knob that had been deliberately
removed, and the other two passed only because it was absent -- a suite asserting the opposite of the
intended design. These two carried over because they constrain recall_knobs itself rather than any
section's content.
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


def test_op_sections_are_scoped_to_the_op(mcp):
    """Scope is what keeps a lever filed against the right thing. Unmarked, model-level guidance got
    recorded under whichever op the agent happened to be standing on -- run 20 filed a whole-model
    lever as `Matmul 32x3840x15360 / shard`."""
    got = mcp.recall_knobs("matmul", bound_by="memory")["known_knobs"]
    assert got, "a memory-bound matmul must recall something"
    assert all(k.get("scope") == "op" for k in got if not str(k["id"]).startswith("model-"))


@pytest.mark.parametrize("op_class", ["", "nonsense", "matmul", "reduction", "eltwise", "host_fallback"])
@pytest.mark.parametrize("bound", ["memory", "compute", "host", None])
def test_the_call_never_raises(mcp, op_class, bound):
    """It runs before every rung's edit; an unrecognised op_class or bound must degrade to an empty
    recall, not end the round."""
    assert isinstance(mcp.recall_knobs(op_class, bound_by=bound), dict)
