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


def test_persistent_1cq_is_corrected_not_shipped(tmp_path, monkeypatch):
    """STRICT: a trace-capable pipeline that only ever degrades to 1CQ is NOT shipped — the loop keeps
    feeding the failure back to correct it and returns None rather than accept a test that fails 2CQ
    (it would silently downgrade at the optimize bookend). The degrade is still recorded along the way."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr(
        "agent.perf_test_gen._run_perf_node",
        lambda node, env, timeout_s=2400: (0, "TRACE_PER_TOKEN_MS=6.0\nTRACE_REPLAY_PATH=trace+1cq\n"),
    )
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is None  # never reached trace+2cq -> not accepted
    caps = _caps(tmp_path)
    assert caps["trace_1cq"] is True and caps["trace_2cq"] is False


def test_eager_terminal_pipeline_is_accepted(tmp_path, monkeypatch):
    """A pipeline that GENUINELY cannot trace (repeat-prefill / no decode_step) is the one legitimate
    eager terminal — the authoritative TRACE_NOT_TRACE_CAPABLE=1 marker makes it accepted, not looped
    forever chasing a trace+2cq it can never produce."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr(
        "agent.perf_test_gen._run_perf_node",
        lambda node, env, timeout_s=2400: (
            0,
            "FORWARD_WALL_MS=10.0\nTRACE_NOT_TRACE_CAPABLE=1\nTRACE_REPLAY_SKIPPED=NotTraceCapable('repeat-prefill')\n",
        ),
    )
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is not None  # eager is legitimate here (authoritative TRACE_NOT_TRACE_CAPABLE marker)


def test_validation_skips_when_device_unavailable(tmp_path, monkeypatch):
    """No device / no ttnn at generation time -> soft-accept (skip), never a false rejection."""
    demo_rel = _demo(tmp_path)
    monkeypatch.setattr(
        "agent.perf_test_gen._run_perf_node",
        lambda node, env, timeout_s=2400: (1, "ImportError: No module named 'ttnn'\n"),
    )
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True, validate=True)
    assert node is not None


def test_needed_trace_region_grows_to_device_reported_max():
    """Model/hardware-agnostic trace region: parse the device-reported required bytes (max over a
    multi-stage trace's cumulative captures), never a fixed guess; None when nothing is over-allocated."""
    from agent.perf_test_gen import _needed_trace_region

    assert _needed_trace_region("nothing here") is None
    one = "Creating trace buffers of size 100B on MeshDevice 1, but only 50B is allocated for trace region."
    assert _needed_trace_region(one) == 125  # 100 * 1.25
    multi = one + "\nCreating trace buffers of size 200B on MeshDevice 1, but only 125B is allocated for trace region."
    assert _needed_trace_region(multi) == 250  # MAX(100,200) * 1.25 — covers the biggest stage
    assert _needed_trace_region("Creating trace buffers of size 40B ... only 50B is allocated") is None  # fits


def test_run_perf_node_grows_trace_region_until_it_fits(monkeypatch):
    """_run_perf_node re-runs with the device-reported size (doubling) until no capture is too small.
    The perf test now runs through the wedge-safe probes._execute (own process group, tree-kill on
    stall), so the grow-loop is driven by patching that seam, not subprocess.run."""
    import agent.perf_test_gen as m
    from agent import probes

    sizes_used = []

    def fake_execute(cmd, cwd, env, timeout_s, log_path, stall_timeout_s=300):
        region = int(env.get("TT_PERF_TRACE_REGION", "23887872"))
        sizes_used.append(region)
        # cumulative multi-stage need: fits only once region >= 40_000_000 (forces the grow-loop)
        if region < 40_000_000:
            need = 30_000_000 if region < 25_000_000 else 40_000_000
            log_path.write_text(
                f"Creating trace buffers of size {need}B on MeshDevice 1, but only {region}B is allocated for trace region."
            )
            return 1
        log_path.write_text("TRACE_STAGE_MS[vocode]=1.0 path=trace+2cq")
        return 0

    monkeypatch.setattr(probes, "_execute", fake_execute)
    monkeypatch.delenv("TT_PERF_TRACE_REGION", raising=False)
    rc, out = m._run_perf_node("some_node::t", {"TT_PERF_NUM_CQ": "2"})
    assert rc == 0 and "path=trace+2cq" in out
    assert sizes_used[0] == 23887872 and sizes_used[-1] >= 40_000_000  # started at default, grew until it fit


def test_run_perf_node_wedge_is_caught_reset_and_reported(monkeypatch):
    """A wedging test (trace-capture fatal -> device teardown hang) must NOT hang or orphan: the tree is
    killed, the board is reset, and the partial output (with the fatal) is returned so the loop can
    correct — rc=124, never a silent hang."""
    import agent.perf_test_gen as m
    from agent import probes

    reset_calls = []

    def hang_execute(cmd, cwd, env, timeout_s, log_path, stall_timeout_s=300):
        log_path.write_text("TT_FATAL: Event Synchronization is not supported during trace capture.\n")
        raise probes.TracyHangError("made no forward progress; process group killed")

    monkeypatch.setattr(probes, "_execute", hang_execute)
    monkeypatch.setattr(probes, "_device_reset", lambda: reset_calls.append(True) or True)
    rc, out = m._run_perf_node("some_node::t", {"TT_PERF_NUM_CQ": "2"})
    assert rc == 124  # wedge signalled as a failure, not a hang
    assert "Event Synchronization" in out and "tt-smi -r" in out  # fatal fed back + board reset noted
    assert reset_calls == [True]  # device was reset for the next correction attempt


def test_injected_runner_defaults_to_no_execution(tmp_path, monkeypatch):
    """validate defaults to (runner is None): an injected runner (unit/test path) must NOT execute pytest."""
    demo_rel = _demo(tmp_path)

    def boom(node, env, timeout_s=2400):
        raise AssertionError("execution-validation must not run when a runner is injected")

    monkeypatch.setattr("agent.perf_test_gen._run_perf_node", boom)
    node = generate_perf_test(tmp_path, "text_generation", demo_rel, runner=lambda p: _VALID, force=True)
    assert node is not None


def test_self_recording_pipeline_detected_and_rerecording_rejected(tmp_path):
    """MODEL-AGNOSTIC: a pipeline whose own function self-captures a trace is detected from source, and a
    generated test that re-records it (measure_adapter/begin_trace_capture) is rejected; a time-it-directly
    draft is accepted. No per-model names."""
    from agent.perf_test_gen import _self_tracing_fns, generate_perf_test

    (tmp_path / "tt").mkdir(parents=True)
    (tmp_path / "tt" / "pipeline.py").write_text(
        "import ttnn\n"
        "def run_fast(p):\n"
        "    tid = ttnn.begin_trace_capture(dev, cq_id=0)\n"
        "    ttnn.end_trace_capture(dev, tid, cq_id=0)\n"
        "def run_plain(p):\n"
        "    return p()\n"
    )
    assert _self_tracing_fns(tmp_path) == {"run_fast"}  # only the self-recording one, derived from source

    d = tmp_path / "demo"
    d.mkdir()
    (d / "demo_fast.py").write_text("from tt.pipeline import run_fast\nout = run_fast(p)\n")

    # re-recording a self-recording function (measure_adapter AROUND run_fast) nests two captures -> rejected
    rerecord = "import ttnn\ndef test_fast_perf(device):\n" "    measure_adapter(lambda: run_fast(p), device)\n"
    assert generate_perf_test(tmp_path, "fast", "demo/demo_fast.py", runner=lambda p: rerecord, force=True) is None

    # timing the self-recording function directly (no external capture) is accepted
    time_it = (
        "import ttnn\ndef test_fast_perf(device):\n"
        "    run_fast(p)\n"
        "    print('TRACE_PER_TOKEN_MS=1.0'); print('TRACE_REPLAY_PATH=trace+2cq')\n"
    )
    assert generate_perf_test(tmp_path, "fast", "demo/demo_fast.py", runner=lambda p: time_it, force=True) is not None


def test_eager_mislabelled_as_trace_is_rejected(tmp_path):
    """AGNOSTIC GUARD (the kokoro `tts` failure): a task whose pipeline function does NOT self-record must
    NOT ship a test that times it directly and stamps TRACE_REPLAY_PATH=trace+2cq with no measure_adapter —
    that times the EAGER path and lies. Validated against the generated test, so no demo/launcher shape can
    smuggle it through. A test that actually captures via measure_adapter is accepted."""
    from agent.perf_test_gen import generate_perf_test

    (tmp_path / "tt").mkdir(parents=True)
    (tmp_path / "tt" / "pipeline.py").write_text(
        "import ttnn\n"
        "def run_slow(p):\n"  # NO begin_trace_capture -> not self-recording
        "    return p()\n"
    )
    d = tmp_path / "demo"
    d.mkdir()
    # demo runs the non-tracing op as a pipeline call AND ends with a bare launcher (the contamination trap)
    (d / "demo_slow.py").write_text(
        "from tt.pipeline import run_slow\n"
        "def main():\n    out = run_slow(p)\n"
        "if __name__ == '__main__':\n    main()\n"
    )

    eager_lie = (
        "import ttnn\ndef test_slow_perf(device):\n"
        "    run_slow(p)\n"
        "    print('TRACE_PER_TOKEN_MS=1.0'); print('TRACE_REPLAY_PATH=trace+2cq')\n"
    )
    assert generate_perf_test(tmp_path, "slow", "demo/demo_slow.py", runner=lambda p: eager_lie, force=True) is None

    proper = (
        "import ttnn\ndef test_slow_perf(device):\n"
        "    measure_adapter(lambda: run_slow(p), device)\n"
        "    print('TRACE_PER_TOKEN_MS=1.0'); print('TRACE_REPLAY_PATH=trace+2cq')\n"
    )
    assert generate_perf_test(tmp_path, "slow", "demo/demo_slow.py", runner=lambda p: proper, force=True) is not None


def test_self_recording_detects_all_shapes(tmp_path):
    """MODEL-AGNOSTIC: self-recording is detected whether it lives in a top-level function OR a class
    method, with nested private helpers rolling up to the public callable the demo invokes; a callable
    with no begin_trace_capture is not flagged."""
    from agent.perf_test_gen import _self_tracing_fns

    (tmp_path / "top_level.py").write_text(
        "def run_fast(p, x):\n"
        "    def _frame():\n"
        "        return ttnn.begin_trace_capture(dev, cq_id=0)\n"
        "    return _frame()\n"
    )
    (tmp_path / "as_method.py").write_text(
        "class Pipeline:\n"
        "    def generate(self, x):\n"
        "        def _inner():\n"
        "            ttnn.begin_trace_capture(self.dev, cq_id=0)\n"
        "        _inner()\n"
        "    def unrelated(self):\n"
        "        return 1\n"
    )
    got = _self_tracing_fns(tmp_path)
    assert "run_fast" in got  # top-level function shape
    assert "generate" in got  # class-method shape
    assert "unrelated" not in got  # a method that does not self-record
    assert "_frame" not in got and "_inner" not in got  # nested helpers roll up to the public callable


def test_self_traced_ignores_bare_launcher():
    """REGRESSION: a self-recording `main()` launcher must NOT flag a task whose real pipeline function
    does not self-record. Only a PIPELINE-OP call (P.fn(...) or fn(pipe, ...)) counts; a bare `main()`
    that every demo ends with does not. (kokoro: demo_tts.py runs the non-tracing run_tts but ends with
    main(); it must get the standard template, not the time-it-directly one.)"""
    from agent.perf_test_gen import _invoked_as_pipeline_op

    # a demo that ONLY runs the non-tracing op and then launches main() -> self-recording main is ignored
    tts_demo = "wav = P.run_tts(pipe, ids, ref)\nif __name__ == '__main__':\n    main()\n"
    assert _invoked_as_pipeline_op("main", tts_demo) is False
    assert _invoked_as_pipeline_op("run_tts_fast", tts_demo) is False

    # a demo that actually invokes the self-recording pipeline op -> flagged
    fast_demo = "wav = P.run_tts_fast(pipe, ids, ref)\nif __name__ == '__main__':\n    main()\n"
    assert _invoked_as_pipeline_op("run_tts_fast", fast_demo) is True
    assert _invoked_as_pipeline_op("main", fast_demo) is False  # bare launcher still ignored

    # direct import call WITH args also counts (no attribute access)
    assert _invoked_as_pipeline_op("run_tts_fast", "run_tts_fast(pipe, ids)\n") is True
    assert _invoked_as_pipeline_op("run_tts_fast", "run_tts_fast()\n") is False  # no args = not a pipeline op
