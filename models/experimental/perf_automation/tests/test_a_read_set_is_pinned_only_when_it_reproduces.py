"""A read set that does not reproduce is not a ceiling.

KIND_STAGE_BYTES is pinned WRITE-ONCE: the first value becomes the memory roof's divisor for the
rest of the model's life, and every later "% of ceiling" inherits it. Verified against the ledger --
a second anchor of the same (kind, depth) returns the first value and the write is dropped.

The gate already takes _FULLPIPE_SAMPLES readings, so agreement across them is free evidence. A
stage's working set is a property of the build and must come out the same every time; one that
varies between samples is instrumentation noise, and pinning noise freezes it permanently.

The parse also used last-write-wins for bytes, the same defect the per-stage timings had -- so the
value that got pinned was whatever the final sample happened to say."""
import statistics

import pytest



@pytest.fixture(autouse=True)
def _no_env_leak():
    """_run_full_pipeline_ms writes os.environ DIRECTLY (PERF_MCP_LAST_HEADLINE_UNIT,
    PERF_MCP_MODEL_NAME), which monkeypatch cannot revert because it never saw the assignment. Left
    behind, those reached later tests in the same process: test_decode_roofline_physics then resolved
    a real model and sized its weights from the HF cache (15.0 GB) instead of the 8 GB its own patch
    installed. The tool then REFUSED TO START -- "refusing to start against a tool whose own tests
    fail" -- so a leak from a unit test blocked a hardware run.

    Snapshot and restore, rather than deleting known names: the next variable the measurement path
    sets would otherwise reintroduce this silently."""
    import os

    before = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(before)

def _sample(**stage_bytes):
    out = ["TRACE_STAGE_MS[decode]=12.0 path=trace+1cq", "TRACE_PER_TOKEN_MS=12.0", "TRACE_HEADLINE_UNIT=token"]
    for k, v in stage_bytes.items():
        out.append("TRACE_STAGE_BYTES[%s]=%d ops=99" % (k, v))
    return "\n".join(out) + "\n"


@pytest.fixture()
def gate(monkeypatch, tmp_path):
    from cc_optimize import perf_mcp as pm

    monkeypatch.setenv("PERF_MCP_MODEL_ROOT", str(tmp_path / "voxtral"))
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("PERF_MCP_TASK", "main")
    monkeypatch.setenv("PERF_MCP_RUN_ID", "test-run-1")
    monkeypatch.setattr(pm, "_FULLPIPE_SAMPLES", 3, raising=False)
    monkeypatch.setattr(pm, "_MODEL_ROOT", tmp_path / "voxtral", raising=False)
    monkeypatch.setattr(pm, "_MANIFEST", {"perf_test_resolved": {"path": "t.py", "case": None}}, raising=False)
    (tmp_path / "state").mkdir(parents=True, exist_ok=True)

    def _drive(samples):
        seq = {"i": 0}

        def _fake(cmd, cwd, env, label="d", stall_s=None, backstop=None):
            out = samples[min(seq["i"], len(samples) - 1)]
            seq["i"] += 1
            return pm._AdaptiveResult(0, out)

        monkeypatch.setattr(pm, "_adaptive_run", _fake)
        pm._run_full_pipeline_ms()
        return pm

    return _drive


def _pinned(pm, stage):
    from cc_optimize import measurements as M

    return M.anchor_value(M.KIND_STAGE_BYTES, depth=stage, model=pm._MODEL_ROOT.name, task="main")


def test_agreeing_readings_are_pinned(gate):
    pm = gate([_sample(decode=2_000_000)] * 3)
    assert _pinned(pm, "decode") == pytest.approx(2.0), "a reproducible read set was refused"


def test_a_reading_that_moves_between_samples_is_not_pinned(gate):
    """1.9 / 2.0 / 3.4 GB is not one measurement of one quantity."""
    pm = gate([_sample(decode=1_900_000), _sample(decode=2_000_000), _sample(decode=3_400_000)])
    assert _pinned(pm, "decode") is None, "instrumentation noise was frozen as a permanent ceiling"


def test_the_recorded_value_is_the_median_not_the_last_sample(gate):
    """Same last-write-wins defect the timings had: the final sample must not simply win."""
    vals = [1_990_000, 2_000_000, 2_010_000]
    pm = gate([_sample(decode=v) for v in vals])
    assert pm.read_stage_bytes().get("decode") == int(statistics.median(vals))


def test_one_reading_alone_is_not_agreement(gate):
    pm = gate([_sample(decode=2_000_000), _sample(), _sample()])
    assert _pinned(pm, "decode") is None, "a single reading was pinned as if corroborated"
