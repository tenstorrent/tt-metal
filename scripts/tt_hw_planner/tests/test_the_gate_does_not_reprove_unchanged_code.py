# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""A correctness pass is reused while the code that earned it is unchanged.

`termination_check` runs the full correctness suite on every call and only then the trace check. The
order is right, but it re-ran UNCONDITIONALLY: no way to tell "the pipeline was rewritten" from
"nothing that matters was touched". On a Qwen-Image-Edit bring-up that meant ~3.5 h of 50-step device
work per round to re-prove a byte-identical pipeline before reaching the ~10-minute trace check that
was the real failure -- six rounds, ~33 h, one attempt at the blocker per four hours.

The verdict is now keyed by content, the way reference/golden.py already keys its goldens. The risk
being designed against is a STALE PASS, so the key covers the demo's whole import closure (a stub in
another package counts), only a pass is cached, and the run stamp is part of the key so a new run
re-verifies once.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.tt_hw_planner.commands import emit_e2e as E

_RUN_ENV = "PERF_MCP_RUN_ID"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A miniature checkout: a demo that imports a module from another package."""
    monkeypatch.setenv(_RUN_ENV, "run-1")
    monkeypatch.delenv(E._GATE_CACHE_OFF_ENV, raising=False)
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    other = tmp_path / "models" / "elsewhere"
    other.mkdir(parents=True)
    (other / "__init__.py").write_text("")
    (other / "stub.py").write_text("W = 1\n")
    (demo / "tt" / "__init__.py").write_text("")
    (demo / "tt" / "pipeline.py").write_text("from models.elsewhere.stub import W\n")
    return demo, other


def _key(demo):
    return E._correctness_key(demo, 0.99, 32)


# --- the key ------------------------------------------------------------------------------------


def test_identical_code_gives_the_same_key(repo):
    demo, _ = repo
    assert _key(demo) == _key(demo)


def test_editing_the_demo_changes_the_key(repo):
    demo, _ = repo
    before = _key(demo)
    (demo / "tt" / "pipeline.py").write_text("from models.elsewhere.stub import W\nX = 2\n")
    assert _key(demo) != before


def test_editing_a_stub_in_ANOTHER_package_changes_the_key(repo):
    """THE STALE-PASS CASE: the graduated stubs this model composes live outside the demo dir."""
    demo, other = repo
    before = _key(demo)
    (other / "stub.py").write_text("W = 2\n")
    assert _key(demo) != before, "an edit outside demo_dir must invalidate the pass"


def test_the_pcc_bar_and_the_batch_are_part_of_the_key(repo):
    demo, _ = repo
    assert E._correctness_key(demo, 0.99, 32) != E._correctness_key(demo, 0.95, 32)
    assert E._correctness_key(demo, 0.99, 32) != E._correctness_key(demo, 0.99, 4)


def test_a_different_run_gets_a_different_key(repo, monkeypatch):
    """A pass is about this code on THIS board; a new run re-verifies once."""
    demo, _ = repo
    before = _key(demo)
    monkeypatch.setenv(_RUN_ENV, "run-2")
    assert _key(demo) != before


def test_no_run_identity_means_no_caching(repo, monkeypatch):
    demo, _ = repo
    monkeypatch.setenv(_RUN_ENV, "")
    assert _key(demo) is None
    assert not E._cached_correctness_pass(demo, None)


def test_the_key_is_content_not_mtime(repo):
    """A touched-but-identical file must not invalidate a good answer."""
    import os
    import time

    demo, _ = repo
    before = _key(demo)
    f = demo / "tt" / "pipeline.py"
    os.utime(f, (time.time() + 500, time.time() + 500))
    assert _key(demo) == before


# --- what is cached -----------------------------------------------------------------------------


def test_a_pass_is_reused_and_a_failure_is_not(repo):
    demo, _ = repo
    k = _key(demo)
    assert E._cached_correctness_pass(demo, k) is None  # nothing recorded yet
    E._record_correctness_pass(demo, k, ["PCC=0.995"])
    assert E._cached_correctness_pass(demo, k) == ["PCC=0.995"]
    # a later edit invalidates it
    (demo / "tt" / "pipeline.py").write_text("X = 9\n")
    assert E._cached_correctness_pass(demo, _key(demo)) is None


def test_only_a_pass_is_ever_recorded():
    """The gate records only on `not reasons` -- a failing gate must re-run every time."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    assert "if not reasons:\n        _record_correctness_pass" in src


def test_the_cache_can_be_switched_off(repo, monkeypatch):
    demo, _ = repo
    k = _key(demo)
    E._record_correctness_pass(demo, k, [])
    monkeypatch.setenv(E._GATE_CACHE_OFF_ENV, "1")
    assert E._cached_correctness_pass(demo, k) is None


def test_a_corrupt_cache_file_is_ignored_not_raised(repo):
    demo, _ = repo
    (demo / E._GATE_CACHE_FILE).write_text("{not json")
    assert E._cached_correctness_pass(demo, _key(demo)) is None


def test_an_unwritable_demo_dir_does_not_raise(repo):
    demo, _ = repo
    E._record_correctness_pass(demo / "nonexistent-subdir", _key(demo), [])  # must not raise


# --- the gate still gates -----------------------------------------------------------------------


def test_a_demo_with_no_tests_still_fails_before_any_caching(tmp_path, monkeypatch):
    monkeypatch.setenv(_RUN_ENV, "run-1")
    demo = tmp_path / "models" / "demos" / "m"
    demo.mkdir(parents=True)
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60)
    assert ok is False and "no tests/e2e" in reasons[0]
    assert not (demo / E._GATE_CACHE_FILE).exists()


def test_the_reuse_is_announced(repo, capsys):
    """A skipped 3.5 h gate must say so, or the run looks like it cheated."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    assert "correctness unchanged since it passed this run" in src


def test_it_names_no_model_or_stage():
    """Constraint: nothing about WHERE code lives may be typed in.

    The EXECUTABLE body is scanned, not the prose -- a docstring may cite a package as an example of
    why the closure walk is needed, exactly as test_axis_choice_uses_no_model_or_stage_names does for
    the batch-axis block."""
    import ast
    import inspect
    import textwrap

    for fn in (E._import_closure, E._correctness_key, E._repo_root_of, E._source_fingerprint):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "encoder", "vae", "tt_dit", "demos/"):
            assert name not in lowered, f"{name!r} in {fn.__name__} would assume where code lives"


# --- the two ways it managed to never fire ------------------------------------------------------


def test_a_failing_trace_gate_does_not_throw_away_a_clean_correctness_pass():
    """BUG 1: the record sat at the END, under a `reasons` list that also carries trace and stack.

    On the bring-up this was written for, the trace gate was the ONLY thing failing -- so a clean
    3.5 h correctness pass was discarded every round because a ten-minute gate after it failed, and
    the cache never recorded once in a 25-hour run."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    record_at = src.index("_record_correctness_pass(demo_dir, _key")
    assert "trace_gate import" not in src[:record_at], "the record must happen BEFORE the trace gate runs"
    assert "_block_stack_gate" not in src[:record_at], "the record must happen BEFORE the stack gate runs"
    assert src.count("_record_correctness_pass(demo_dir, _key") == 1, "one record, at the end of correctness"


def test_a_cache_hit_does_not_waive_the_trace_and_stack_gates():
    """BUG 2: the hit returned (True, []) from the top, ahead of the very gate that was failing."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    hit_at = src.index("_cached_correctness_pass(demo_dir, _key)")
    after = src[hit_at:]
    assert "return True, []" not in after, "a hit must not return past the remaining gates"
    assert "_block_stack_gate" in after and "trace_gate import" in after


def test_a_hit_replays_the_evidence_the_later_checks_read():
    """A check with no input must never read as a pass: the batch report and PCC come back too."""
    from models.experimental.perf_automation.agent.perf_adapter import batch_report_line

    out = "\n".join(["noise", batch_report_line(32), "some_stage PCC = 0.9987", "another line", "1 passed"])
    kept = E._gate_pass_evidence(out)
    assert batch_report_line(32) in kept
    assert any("PCC" in k for k in kept)
    assert "noise" not in kept and "another line" not in kept
    replayed = "\n".join(kept + ["1 passed"])
    assert E._batch_gate_reason(32, replayed) is None  # the batch check still sees what it drove
    assert E._batch_gate_reason(4, replayed) is not None  # and still catches the wrong batch


def test_a_replayed_pass_cannot_hide_a_failing_pcc():
    """The stored lines are the run's own, so the PCC check reaches the same verdict it did live."""
    import re

    kept = E._gate_pass_evidence("stage PCC = 0.9600\n1 passed")
    replayed = "\n".join(kept)
    vals = [float(v) for v in re.findall(r"PCC[^=\n]*=\s*(-?\d+(?:\.\d+)?)", replayed)]
    assert vals == [0.96] and min(vals) < 0.99


def test_the_evidence_survives_a_round_trip(repo):
    demo, _ = repo
    k = _key(demo)
    E._record_correctness_pass(demo, k, ["PERF_BATCH=32", "PCC = 0.999"])
    assert E._cached_correctness_pass(demo, k) == ["PERF_BATCH=32", "PCC = 0.999"]


def test_an_old_record_without_evidence_is_not_trusted(repo):
    """A file written by the previous shape has no evidence to replay, so it must not count."""
    demo, _ = repo
    k = _key(demo)
    (demo / E._GATE_CACHE_FILE).write_text(json.dumps({"key": k, "version": E._GATE_KEY_VERSION}))
    assert E._cached_correctness_pass(demo, k) is None


# --- end to end: record on a real round, reuse on the next, waive nothing -----------------------


@pytest.fixture
def full_gate(monkeypatch, tmp_path):
    """The WHOLE gate, twice, with only the device work replaced.

    The two bugs were both about WHERE things sat relative to each other, so neither showed up in a
    unit test of the cache: one needed the trace gate to fail while correctness passed, the other
    needed a hit to happen at all. This drives the real `_run_deterministic_gates`."""
    import subprocess

    from models.experimental.perf_automation.agent import perf_adapter as PA
    from models.experimental.perf_automation.agent import probes as _PR

    monkeypatch.setenv(_RUN_ENV, "run-1")
    monkeypatch.delenv(E._GATE_CACHE_OFF_ENV, raising=False)
    monkeypatch.setenv("E2E_REQUIRE_ON_DEVICE", "0")

    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "demo").mkdir()
    (demo / "tt").mkdir()
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("def test_e2e():\n    pass\n")
    (demo / "demo" / "demo_m.py").write_text("if __name__ == '__main__':\n    pass\n")
    (demo / "README.md").write_text("# m\n")
    (demo / "tt" / "pipeline.py").write_text("X = 1\n")

    # A PASSING run, carrying exactly what the checks after it read.
    passing = "\n".join([PA.batch_report_line(32), "image PCC = 0.9987", "1 passed"])
    runs = {"pytest": 0}

    def _exec(cmd, cwd, env, timeout_s, log_path, **k):
        runs["pytest"] += 1
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        Path(log_path).write_text(passing)
        return 0

    class _Proc:
        """The G6 capture probe, producing no verdict: a deterministic non-device failure."""

        returncode = 1

        def __init__(self, *a, **k):
            self.pid = os.getpid()

        def communicate(self, timeout=None):
            return "", "no probe"

        def poll(self):
            return 1

    monkeypatch.setattr(_PR, "_execute", _exec)
    monkeypatch.setattr(E.subprocess, "run", lambda cmd, **k: subprocess.CompletedProcess(cmd, 1, "", ""))
    monkeypatch.setattr(E.subprocess, "Popen", _Proc)
    return demo, runs


def test_the_whole_gate_records_a_pass_even_though_the_trace_gate_fails(full_gate):
    """BUG 1, end to end: correctness clean, trace gate failing -- the pass must still be kept."""
    demo, runs = full_gate
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60, batch=32)

    assert runs["pytest"] == 1, "the expensive run must actually have happened this round"
    assert ok is False, "the gate must still fail: the trace gate has not passed"
    assert not [r for r in reasons if r.startswith("G2/G3") or r.startswith("G3")], reasons
    assert [r for r in reasons if "G6" in r], "the trace gate is the thing that failed"
    assert (demo / E._GATE_CACHE_FILE).is_file(), "THE BUG: a clean correctness pass was discarded"
    assert E._cached_correctness_pass(demo, E._correctness_key(demo, 0.99, 32)), "no evidence recorded"


def test_the_next_round_skips_the_run_and_still_fails_the_trace_gate(full_gate):
    """BUG 2, end to end: the hit must buy back the 3.5 h and waive NOTHING."""
    demo, runs = full_gate
    first_ok, first_reasons = E._run_deterministic_gates(demo, 0.99, 60, batch=32)
    second_ok, second_reasons = E._run_deterministic_gates(demo, 0.99, 60, batch=32)

    assert runs["pytest"] == 1, "the second round re-ran the device work it already had a verdict for"
    assert second_ok is False, "THE BUG: a hit reported the pipeline done, waiving the failing gate"
    assert [r for r in second_reasons if "G6" in r], "the trace gate must run again every round"
    assert not [r for r in second_reasons if r.startswith("G2/G3") or r.startswith("G3")], second_reasons


def test_a_hit_still_enforces_the_batch_and_the_pcc_it_replays(full_gate):
    """A check with no input must never read as a pass: the replayed lines are still judged."""
    demo, runs = full_gate
    E._run_deterministic_gates(demo, 0.99, 60, batch=32)
    # asked for a batch the recorded run did not drive -> the key changes, so it re-runs and fails
    ok, reasons = E._run_deterministic_gates(demo, 0.99, 60, batch=4)
    assert runs["pytest"] == 2, "a different batch is a different question and must be re-asked"
    assert [r for r in reasons if "drove 32" in r or "batch" in r.lower()], reasons


def test_an_edit_after_the_pass_re_runs_the_whole_thing(full_gate):
    demo, runs = full_gate
    E._run_deterministic_gates(demo, 0.99, 60, batch=32)
    (demo / "tt" / "pipeline.py").write_text("X = 2\n")
    E._run_deterministic_gates(demo, 0.99, 60, batch=32)
    assert runs["pytest"] == 2, "an edited pipeline must be re-proved"


# --- the third reason it never hit: the key hashed code the gate never runs ----------------------


@pytest.fixture
def demo_with_a_perf_test(tmp_path, monkeypatch):
    """A demo shaped like the real one: a gate test, a perf test, and a module only perf reaches."""
    monkeypatch.setenv(_RUN_ENV, "run-1")
    monkeypatch.delenv(E._GATE_CACHE_OFF_ENV, raising=False)
    (tmp_path / "scripts").mkdir()
    demo = tmp_path / "models" / "demos" / "m"
    (demo / "tt").mkdir(parents=True)
    (demo / "tests" / "e2e").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    measured = tmp_path / "models" / "harness"
    measured.mkdir(parents=True)
    (measured / "__init__.py").write_text("")
    (measured / "measurer.py").write_text("ITERS = 16\n")
    (measured / "shared.py").write_text("BATCH_ENV = 'B'\n")

    (demo / "tt" / "pipeline.py").write_text("from models.harness.shared import BATCH_ENV\n")
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("from models.demos.m.tt import pipeline\n")
    # the perf test -- the one file the gate does not run -- is the only route to measurer.py
    (demo / "tests" / "e2e" / "test_m_perf.py").write_text("from models.harness.measurer import ITERS\n")
    (demo / "tests" / "pcc" / "test_component.py").write_text("X = 1\n")
    return demo, measured


def test_the_key_ignores_code_only_the_perf_test_reaches(demo_with_a_perf_test):
    """THE FAILURE, measured live: editing the replay code moved the key 8e9c4eb -> fe8e4b5 and
    discarded a 3.5 h correctness pass it cannot possibly affect."""
    demo, measured = demo_with_a_perf_test
    before = _key(demo)
    (measured / "measurer.py").write_text("ITERS = 2\n")
    assert _key(demo) == before, "an edit to code the gate never runs must not invalidate the pass"


def test_the_key_still_follows_what_the_pipeline_really_uses(demo_with_a_perf_test):
    """The pipeline's own import of that package is real: the batch it names decides correctness."""
    demo, measured = demo_with_a_perf_test
    before = _key(demo)
    (measured / "shared.py").write_text("BATCH_ENV = 'OTHER'\n")
    assert _key(demo) != before


def test_the_key_still_follows_the_demo_and_the_gate_test(demo_with_a_perf_test):
    demo, _ = demo_with_a_perf_test
    before = _key(demo)
    (demo / "tt" / "pipeline.py").write_text("from models.harness.shared import BATCH_ENV\nX = 2\n")
    assert _key(demo) != before
    mid = _key(demo)
    (demo / "tests" / "e2e" / "test_e2e_m.py").write_text("from models.demos.m.tt import pipeline\n# edit\n")
    assert _key(demo) != mid, "the gate runs this file, so it decides the verdict"


def test_the_seeds_are_the_tests_the_gate_actually_runs(demo_with_a_perf_test):
    demo, _ = demo_with_a_perf_test
    names = [f.name for f in E._gate_test_files(demo)]
    assert names == ["test_e2e_m.py"], names


def test_the_gate_and_the_fingerprint_ask_the_same_question():
    """They disagreed, in the expensive direction; the selection now has one owner."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    assert "_gate_test_files(demo_dir)" in src
    assert '"perf" not in f.name' not in src, "the rule must not be re-spelled at the call site"


def test_the_no_edit_check_keeps_the_broad_fingerprint(demo_with_a_perf_test):
    """Different question: there, ANY edit is the thing being looked for, including the perf test."""
    import inspect

    import scripts.tt_hw_planner.e2e_mcp as M

    demo, measured = demo_with_a_perf_test
    before = E._source_fingerprint(demo)
    (measured / "measurer.py").write_text("ITERS = 2\n")
    assert E._source_fingerprint(demo) != before
    assert "seeds" not in inspect.getsource(M._count_unchanged_round)


def test_existing_closure_callers_are_unaffected(demo_with_a_perf_test):
    """`seeds` is additive: omitted, the walk is exactly what it was."""
    demo, _ = demo_with_a_perf_test
    assert len(E._import_closure(demo)) > len(E._import_closure(demo, seeds=E._correctness_seeds(demo)))
