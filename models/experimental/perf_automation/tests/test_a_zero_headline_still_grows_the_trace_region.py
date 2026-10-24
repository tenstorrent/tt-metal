"""The full-pipeline gate's trace-region grow step must not read a 0.0000 headline as a trace that ran.

The trace-replay harness prints TRACE_PER_TOKEN_MS even when every stage raised (as 0.0000). The grow
loop used to stop on the bare sentinel, so a region too small for every stage was never grown and the
gate reported a crash with no retry. These pin the corrected rule: stop on a POSITIVE reading; grow a
zero headline only on overflow evidence; leave every other outcome exactly as it was.

AND PARTIAL SUCCESS IS NOT A STOPPING POINT EITHER. A region big enough for the small stages and too
small for the big ones produces a headline from the survivors, which stopped the loop and stranded
the overflowing stages at that size for the rest of the run. That is what left Qwen-Image-Edit with a
three-stage reading for nineteen days while the attempts went to the two encoders. A stage that got
no row for want of room says so on its own line (stage_seams.UNMEASURED_MARK) and the loop grows
while any stage is saying it -- and keeps the best attempt, so growing can only add rows.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from agent import stage_seams as seams, tracy_tool
from cc_optimize import perf_mcp as m

_ZERO = "TRACE_PER_TOKEN_MS=0.0000\nTRACE_HEADLINE_UNIT=inference\n"
_OK = "TRACE_STAGE_MS[s0]=12.5 path=trace+1cq\nTRACE_PER_TOKEN_MS=12.5000\n"
_START = 200 * 1024 * 1024
# The bare mesh assertion as a stage's own failure reaches it: trace_replay gives the stage no row and
# states why on the marker line, which is the only place a PARTIAL overflow is visible.
_OVERFLOW = "RuntimeError: TT_FATAL @ mesh_trace.cpp:82: get_trace_buffers_size() <= trace_region_size"


def _lost(stage: str = "s1", why: str = _OVERFLOW) -> str:
    return seams.unmeasured_line(stage, why) + "\n"


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


def test_a_stage_short_of_room_is_grown_even_though_the_others_measured(harness):
    """THE NINETEEN-DAY REGRESSION. A positive headline from the stages that fit used to end the loop."""
    _grow(_OK + _lost())
    assert harness.runs == [_START * 2], "a stage with no row for want of room must be re-run for"


def test_a_partial_overflow_that_states_its_size_jumps_straight_to_it(harness):
    """The marker line carries the device's own bytes (stage_seams.unmeasured_why), so one re-run does it."""
    need = 1 << 30
    _grow(
        _OK
        + _lost(
            why="RuntimeError: TT_FATAL @ mesh_trace.cpp:82 ... Creating trace buffers of size "
            "%dB on MeshDevice 0, but only %dB is allocated for trace region." % (need, _START)
        )
    )
    assert len(harness.runs) == 1 and harness.runs[0] >= need


def test_a_stage_lost_for_another_reason_is_not_grown_for(harness):
    """Only trace room. A stage that died of an L1 overflow is reported as it was."""
    _grow(_OK + _lost(why="RuntimeError: Statically allocated circular buffers grow to 3830000 B, beyond max L1"))
    assert harness.runs == [], "a stage lost to something other than trace room is not a region problem"


def test_every_stage_measured_still_stops_the_loop(harness):
    """The ordinary healthy run: a row for every stage, no marker line, nothing to grow."""
    _grow(_OK + "TRACE_STAGE_MS[s1]=4.0 path=trace+1cq\n")
    assert harness.runs == []


def test_it_does_not_re_run_at_a_region_the_run_already_held(harness, monkeypatch):
    """The ask is a FLOOR (perf_adapter.resolve_trace_region), so a model declaring more than the ask
    is already holding more -- and the overflow line says how much. Sized off the ask alone, the loop
    reads the ceiling as a step up from 200 MiB and spends a 25-minute re-run reserving a region the
    run already had."""
    held = 1 << 30  # what the model declares and the run therefore reserved
    monkeypatch.setattr(m, "_TRACE_REGION_MAX", held)  # ... which is also the DRAM ceiling here
    _grow(_bytes_msg(2 << 30, held) + _ZERO)
    assert harness.runs == [], "it already held the ceiling -- the trace genuinely does not fit"


def test_it_still_grows_past_a_region_bigger_than_the_ask(harness):
    """The other half of the same rule: holding more than the ask is not a reason to stop."""
    held = 1 << 30
    want = 2 << 30
    _grow(_bytes_msg(want, held) + _ZERO)
    assert harness.runs and harness.runs[0] >= want, harness.runs


def test_a_grow_never_returns_less_than_it_was_handed(harness):
    """A re-run that comes back worse (a bigger region, and now nothing measures) must not be what the
    caller gets: growing may add rows, never cost the ones an earlier attempt already had."""
    harness.queue = ["an unrelated crash with no headline\n"] * 8
    out, _env = _grow(_OK + _lost())
    assert harness.runs, "it did try to grow"
    assert "TRACE_STAGE_MS[s0]=12.5" in out, "the partial reading it started with is what comes back"
