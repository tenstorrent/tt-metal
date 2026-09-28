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
    assert E._cached_correctness_pass(demo, None) is False


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
    assert E._cached_correctness_pass(demo, k) is False  # nothing recorded yet
    E._record_correctness_pass(demo, k)
    assert E._cached_correctness_pass(demo, k) is True
    # a later edit invalidates it
    (demo / "tt" / "pipeline.py").write_text("X = 9\n")
    assert E._cached_correctness_pass(demo, _key(demo)) is False


def test_only_a_pass_is_ever_recorded():
    """The gate records only on `not reasons` -- a failing gate must re-run every time."""
    import inspect

    src = inspect.getsource(E._run_deterministic_gates)
    assert "if not reasons:\n        _record_correctness_pass" in src


def test_the_cache_can_be_switched_off(repo, monkeypatch):
    demo, _ = repo
    k = _key(demo)
    E._record_correctness_pass(demo, k)
    monkeypatch.setenv(E._GATE_CACHE_OFF_ENV, "1")
    assert E._cached_correctness_pass(demo, k) is False


def test_a_corrupt_cache_file_is_ignored_not_raised(repo):
    demo, _ = repo
    (demo / E._GATE_CACHE_FILE).write_text("{not json")
    assert E._cached_correctness_pass(demo, _key(demo)) is False


def test_an_unwritable_demo_dir_does_not_raise(repo):
    demo, _ = repo
    E._record_correctness_pass(demo / "nonexistent-subdir", _key(demo))  # must not raise


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

    for fn in (E._import_closure, E._correctness_key, E._repo_root_of):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        node = tree.body[0]
        if ast.get_docstring(node) is not None:
            node.body = node.body[1:]
        lowered = ast.unparse(node).lower()
        for name in ("qwen", "denoise", "prefill", "encoder", "vae", "tt_dit", "demos/"):
            assert name not in lowered, f"{name!r} in {fn.__name__} would assume where code lives"
