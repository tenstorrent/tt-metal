"""Three samples, and the per-stage split kept the last one.

The gate takes _FULLPIPE_SAMPLES readings and reports the MEDIAN as the headline, because one
reading is noise. The per-stage numbers were parsed in the same loop with a plain overwrite --
`stage_ms[name] = value` -- so the file ended up holding whatever the FINAL sample said.

Measured on voxtral 2026-08-24: one report carried decode at 12.87 ms (headline, median of 3) and
20.49 ms (stage table, last sample) for the SAME quantity in the same run, a 1.59x spread. Since the
median of three is always one of the three, the samples provably contained both numbers.

Not cosmetic: `stage_win` is computed from these values and gates _record_fullpipe_candidate, so an
unfiltered sample could fabricate a stage win or bury a real one -- beside a headline that WAS
filtered. Every stage is affected; decode is only the one with a median twin to contradict it."""
import statistics

import pytest


def _sample(decode, prefill, encode):
    return (
        "TRACE_STAGE_MS[encode]=%.4f path=trace+1cq\n"
        "TRACE_STAGE_MS[prefill]=%.4f path=trace+1cq\n"
        "TRACE_STAGE_MS[decode]=%.4f path=trace+1cq\n"
        "TRACE_PER_TOKEN_MS=%.4f\n"
        "TRACE_HEADLINE_UNIT=token\n"
        "TRACE_REPLAY_PATH=trace+1cq batch=8\n"
    ) % (encode, prefill, decode, decode)


# The observed voxtral case: the last sample is the outlier, and it is 1.59x the median.
_SAMPLES = [(11.90, 480.0, 88.0), (12.871, 491.0781, 89.7369), (20.4854, 640.0, 140.0)]


@pytest.fixture()
def gate(monkeypatch, tmp_path):
    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "voxtral"))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_FULLPIPE_SAMPLES", "3")
    # The stage file is refused unless it carries THIS run's stamp (_read_stage_doc): an
    # unstamped doc is stale by definition. Give the harness a run id so the reader accepts
    # what the writer just wrote, rather than testing the freshness rule by accident.
    monkeypatch.setenv("PERF_MCP_RUN_ID", "test-run-1")
    monkeypatch.setattr(pm, "_FULLPIPE_SAMPLES", 3, raising=False)
    monkeypatch.setattr(pm, "_MODEL_ROOT", tmp_path / "voxtral", raising=False)
    monkeypatch.setattr(pm, "_MANIFEST", {"perf_test_resolved": {"path": "t.py", "case": None}}, raising=False)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    calls = {"n": 0}

    def _fake_run(cmd, cwd, env, label="device run", stall_s=None, backstop=None):
        d, p, e = _SAMPLES[min(calls["n"], len(_SAMPLES) - 1)]
        calls["n"] += 1
        return pm._AdaptiveResult(0, _sample(d, p, e))

    monkeypatch.setattr(pm, "_adaptive_run", _fake_run)
    return pm, calls


def test_the_stage_split_is_the_median_not_the_last_sample(gate):
    pm, calls = gate
    pm._run_full_pipeline_ms()
    assert calls["n"] == 3, "the gate did not take three samples"
    got = pm.read_stage_ms()
    for i, stage in enumerate(("decode", "prefill", "encode")):
        want = statistics.median([s[i] for s in _SAMPLES])
        assert got.get(stage) == pytest.approx(want, rel=1e-6), "%s kept %s; the median of the three samples is %s" % (
            stage,
            got.get(stage),
            want,
        )


def test_the_outlier_sample_does_not_reach_the_file(gate):
    """The specific regression: 20.4854 was the last reading and it must not be what is stored."""
    pm, _ = gate
    pm._run_full_pipeline_ms()
    assert pm.read_stage_ms().get("decode") != pytest.approx(20.4854), "the last sample won again"


def test_the_headline_and_the_stage_agree(gate):
    """They are the same quantity -- trace_replay prints one `ms` as both -- so one filter, one answer."""
    pm, _ = gate
    ms, _method, _err, _path = pm._run_full_pipeline_ms()
    assert ms == pytest.approx(
        pm.read_stage_ms()["decode"], rel=1e-6
    ), "headline %s disagrees with the decode stage %s" % (ms, pm.read_stage_ms().get("decode"))
