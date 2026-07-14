"""perf_test_gen: ALWAYS create the pipeline perf test from scratch, NEVER reuse an existing one.

Reuse makes no sense — the demo is the source of truth for the pipeline's forward, and a reused
test can be stale or cover only a slice (e.g. prefill-only). generate_perf_test(force=True) must
regenerate every call, retry transient bad drafts, and gate generative demos on decode coverage.
"""


import json

from agent.perf_test_gen import generate_perf_test

_VALID = "import ttnn\ndef test_text_generation_perf(device):\n    import os\n    os.environ.get('TT_PERF_MAX_NEW_TOKENS')\n    pass\n"
_PREFILL_ONLY = "import ttnn\ndef test_text_generation_perf(device):\n    pass\n"  # no decode loop env


def _demo(tmp_path, generative=True):
    d = tmp_path / "demo"
    d.mkdir(parents=True, exist_ok=True)
    src = "for _ in range(max_new_tokens):\n    step()\n" if generative else "model(x)\n"
    (d / "demo_text_generation.py").write_text(src)
    return "demo/demo_text_generation.py"


def test_force_overwrites_existing_never_reuses(tmp_path):
    """An existing perf test on disk must be OVERWRITTEN, not returned as-is."""
    demo_rel = _demo(tmp_path)
    out = tmp_path / "tests" / "e2e" / "test_text_generation_perf.py"
    out.parent.mkdir(parents=True)
    out.write_text("# STALE prior test — must be overwritten\n")
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True)
    assert node == "tests/e2e/test_text_generation_perf.py::test_text_generation_perf"
    assert "STALE" not in out.read_text()  # regenerated from scratch, the old content is gone


def test_retries_transient_bad_draft(tmp_path):
    """A junk draft must be retried (not dropped to None/reuse) until a well-formed test appears."""
    demo_rel = _demo(tmp_path)
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        return "garbage, no test here" if calls["n"] < 3 else _VALID

    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=flaky, force=True)
    assert node is not None and calls["n"] == 3


def test_generative_demo_requires_decode_coverage(tmp_path):
    """A generative demo whose generated test omits the decode-loop cap is rejected (-> None),
    never accepted as a prefill-only slice and never silently reused."""
    demo_rel = _demo(tmp_path, generative=True)
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _PREFILL_ONLY, force=True)
    assert node is None
    assert not (tmp_path / "tests" / "e2e" / "test_text_generation_perf.py").exists()


def test_no_demo_to_lift_from_returns_none(tmp_path):
    """No demo file -> nothing to generate from -> None (caller gates; still no reuse)."""
    node = generate_perf_test(
        tmp_path, "text_generation", "demo/demo_text_generation.py", runner=lambda p: _VALID, force=True
    )
    assert node is None


def _caps(tmp_path):
    return json.loads((tmp_path / "tests" / "e2e" / "test_text_generation_perf.py.trace_caps.json").read_text())


def test_validation_rejects_test_that_does_not_run_pipeline(tmp_path, monkeypatch):
    """A draft that RUNS but emits no full-pipeline marker (no-op / broke mid-forward) is rejected and
    regenerated — static checks alone used to accept it. All attempts no-op -> None, nothing on disk."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr("agent.perf_test_gen._run_perf_node", lambda node, env, timeout_s=2400: (0, "no marker\n"))
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is None


def test_validation_accepts_and_records_genuine_2cq(tmp_path, monkeypatch):
    """A draft that produces a real per-token marker is accepted; when the 2-CQ probe genuinely engages
    (path==trace+2cq) it is recorded trace_2cq=True in the capability sidecar."""
    demo_rel = _demo(tmp_path)

    def fake_run(node, env, timeout_s=2400):
        if env.get("TT_PERF_NUM_CQ") == "2":
            return 0, "TRACE_PER_TOKEN_MS=5.0\nTRACE_REPLAY_PATH=trace+2cq batch=1\n"
        return 0, "TRACE_PER_TOKEN_MS=6.0\nTRACE_REPLAY_PATH=trace+1cq batch=1\n"

    monkeypatch.setattr("agent.perf_test_gen._run_perf_node", fake_run)
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is not None
    caps = _caps(tmp_path)
    assert caps["trace_1cq"] is True and caps["trace_2cq"] is True and caps["trace_2cq_path"] == "trace+2cq"


def test_validation_records_2cq_degrade_as_not_capable(tmp_path, monkeypatch):
    """The core fix: a test that silently DEGRADES to 1CQ under a 2-CQ device is recorded trace_2cq=False
    — it is NEVER trusted/labelled as a 2CQ measurement downstream."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr(
        "agent.perf_test_gen._run_perf_node",
        lambda node, env, timeout_s=2400: (0, "TRACE_PER_TOKEN_MS=6.0\nTRACE_REPLAY_PATH=trace+1cq\n"),
    )
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is not None
    caps = _caps(tmp_path)
    assert caps["trace_1cq"] is True and caps["trace_2cq"] is False


def test_validation_skips_when_device_unavailable(tmp_path, monkeypatch):
    """No device / no ttnn at generation time -> soft-accept (skip), never a false rejection."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr(
        "agent.perf_test_gen._run_perf_node",
        lambda node, env, timeout_s=2400: (1, "ImportError: No module named 'ttnn'\n"),
    )
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is not None


def test_injected_runner_defaults_to_no_execution(tmp_path, monkeypatch):
    """validate defaults to (runner is None): an injected runner (unit/test path) must NOT execute pytest."""
    demo_rel = _demo(tmp_path)

    def boom(node, env, timeout_s=2400):
        raise AssertionError("execution-validation must not run when a runner is injected")

    monkeypatch.setattr("agent.perf_test_gen._run_perf_node", boom)
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True)
    assert node is not None
