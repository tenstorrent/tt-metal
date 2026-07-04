"""perf_test_gen: ALWAYS create the pipeline perf test from scratch, NEVER reuse an existing one.

Reuse makes no sense — the demo is the source of truth for the pipeline's forward, and a reused
test can be stale or cover only a slice (e.g. prefill-only). generate_perf_test(force=True) must
regenerate every call, retry transient bad drafts, and gate generative demos on decode coverage.
"""


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
