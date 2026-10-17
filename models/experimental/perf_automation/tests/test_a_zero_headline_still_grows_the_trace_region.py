"""The full-pipeline gate's trace-region grow step must not read a 0.0000 headline as a trace that ran.

The trace-replay harness prints TRACE_PER_TOKEN_MS even when every stage raised (as 0.0000). The grow
loop used to stop on the bare sentinel, so a region too small for every stage was never grown and the
gate reported a crash with no retry. These pin the corrected rule: stop on a POSITIVE reading; grow a
zero headline only on overflow evidence; leave every other outcome exactly as it was.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from agent import tracy_tool
from cc_optimize import perf_mcp as m

_ZERO = "TRACE_PER_TOKEN_MS=0.0000\nTRACE_HEADLINE_UNIT=inference\n"
_OK = "TRACE_STAGE_MS[s0]=12.5 path=trace+1cq\nTRACE_PER_TOKEN_MS=12.5000\n"
_START = 200 * 1024 * 1024


def _bytes_msg(need: int, have: int) -> str:
    return (
        "critical | Always | TT_FATAL: Creating trace buffers of size %dB on MeshDevice 0, but only %dB is "
        "allocated for trace region. (assert.hpp:104)\n" % (need, have)
    )


@pytest.fixture
def harness(monkeypatch):
    """Record every re-run and reset the loop asks for; re-runs return the queued outputs in order."""
    calls = SimpleNamespace(runs=[], resets=0, queue=[])

    def fake_run(cmd, repo, env, label):
        calls.runs.append(int(env["TT_PERF_TRACE_REGION"]))
        return SimpleNamespace(stdout=calls.queue.pop(0) if calls.queue else _OK, stderr="")

    def fake_reset(**_kw):
        calls.resets += 1

    import agent.probes as probes

    monkeypatch.setattr(m, "_adaptive_run", fake_run)
    monkeypatch.setattr(probes, "_device_reset", fake_reset)
    monkeypatch.setattr(m, "_TRACE_REGION_MAX", 4 * 1024 * 1024 * 1024)
    return calls


def _grow(first_out):
    env = {"TT_PERF_TRACE_REGION": str(_START)}
    out, _ = m._grow_trace_region_and_retry(["cmd"], ".", env, first_out, SimpleNamespace(stdout=first_out, stderr=""))
    return out, env


def test_per_token_readings_parses_every_sentinel_value():
    assert tracy_tool.per_token_readings(_ZERO + _OK) == [0.0, 12.5]
    assert tracy_tool.per_token_readings("") == []
    assert tracy_tool.per_token_readings(None) == []


def test_zero_headline_with_the_devices_byte_count_grows_to_what_it_asked_for(harness):
    need = 797966336
    out, env = _grow(_bytes_msg(need, _START) + _ZERO)
    assert len(harness.runs) == 1, "a zero headline with an explicit overflow must be re-run"
    assert harness.runs[0] >= need, "the re-run must get at least the bytes the device reported"
    assert harness.resets == 0, "the run finished; nothing is wedged, so nothing is reset"
    assert "TRACE_PER_TOKEN_MS=12.5000" in out


def test_zero_headline_with_the_bare_mesh_assertion_doubles_without_a_reset(harness):
    bare = "TT_FATAL @ mesh_trace.cpp:78: get_trace_buffers_size() <= trace_region_size\n"
    _grow(bare + _ZERO)
    assert harness.runs == [_START * 2]
    assert harness.resets == 0


def test_zero_headline_for_another_reason_is_not_retried(harness):
    l1 = "Statically allocated circular buffers grow to 3830000 B which is beyond max L1 size\n"
    out, env = _grow(l1 + _ZERO)
    assert harness.runs == [], "a run that measured nothing for a non-trace reason is reported, not re-run"
    assert env["TT_PERF_TRACE_REGION"] == str(_START)
    assert out.endswith(_ZERO)


def test_a_positive_headline_is_left_alone(harness):
    _grow(_OK)
    assert harness.runs == [] and harness.resets == 0


def test_an_untraceable_pipeline_is_left_alone(harness):
    _grow("TRACE_NOT_TRACE_CAPABLE reason=x\n" + _ZERO)
    assert harness.runs == []


def test_no_headline_at_all_is_still_the_silent_hang_path(harness):
    _grow("some log\n[perf_test_gen] killed after 600s of no progress\n")
    assert harness.runs == [_START * 2], "the unchanged silent path doubles"
    assert harness.resets == 1, "and resets the wedged mesh first, exactly as before"


def test_growth_stops_at_the_ceiling(harness, monkeypatch):
    monkeypatch.setattr(m, "_TRACE_REGION_MAX", _START)
    _grow(_bytes_msg(797966336, _START) + _ZERO)
    assert harness.runs == [], "already at the DRAM ceiling -> genuinely does not fit, no re-run"
