# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The facts file must describe the model that is running, and a refresh must never make it worse.

Run 51 optimized gemma-3-12b against a facts file reading `weight_bytes: 0`, 4B params and 34
layers -- gemma-3-4b's shape. Two independent faults had to line up, and each was harmless alone:

  THE RESOLVER PICKED THE WRONG MODEL. `_resolve_model_id` returned the first cached HF id found in
  the first .py that rglob happened to yield. gemma3's tree names three: conftest, the perf test, the
  PCC test and the host-split test all pin google/gemma-3-12b-it, while test_ci_dispatch.py lists the
  4b and the 27b as a CI matrix. Four files against one -- and it returned the 4b.

  THE REFRESH LET IT LAND. perf_target_inputs.json used to be write-once, so the correct file
  survived any later bad derivation. That rule was replaced with "refresh the file if I wrote it",
  which checks who WROTE the old file and never whether the new one is any good. So a wrong
  derivation overwrote 24.37 GB / 11.18B params / 48 layers with zero.

Write-once was crude but SAFE. The lesson is not to go back to it -- the geometry keys prefill needs
could not reach a model that already had a file -- but that a refresh has to be judged on the
CONTENT, not on the authorship.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import tempfile
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA.parents[2]))

_GOOD = {
    "weight_bytes": 24374793024,
    "total_params": 11180446320,
    "layers": 48,
    "source": "checkpoint bytes + HF config",
}
_WORSE = {"weight_bytes": 0, "total_params": 4000000000, "layers": 34, "source": "checkpoint bytes + HF config"}


def _run():
    spec = _ilu.spec_from_file_location("cc_run_facts_guard", str(_PA / "cc_optimize" / "run.py"))
    m = _ilu.module_from_spec(spec)
    sys.modules["cc_run_facts_guard"] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- the refresh


def test_a_refresh_never_drops_the_divisor(monkeypatch):
    """THE EXACT RUN-51 REGRESSION. With no byte count and no param count there is nothing to divide
    by, so a file that HAS one must not be replaced by one that does not."""
    m = _run()
    d = Path(tempfile.mkdtemp())
    (d / "perf_target_inputs.json").write_text(json.dumps(_GOOD))
    monkeypatch.setattr(m, "_perf_target_inputs", lambda *a, **k: dict(_WORSE), raising=False)
    m._emit_perf_target_inputs(d, d, None, {})
    got = json.loads((d / "perf_target_inputs.json").read_text())
    assert got["weight_bytes"] == 24374793024, "a regeneration that lost the divisor overwrote a good file"
    assert got["total_params"] == 11180446320
    assert got["layers"] == 48


def test_a_refresh_that_ADDS_facts_still_lands(monkeypatch):
    """The refusal must not resurrect write-once: the geometry keys prefill's byte model needs could
    not reach a model that already had a file, which is the problem the refresh exists to solve."""
    m = _run()
    d = Path(tempfile.mkdtemp())
    (d / "perf_target_inputs.json").write_text(json.dumps(_GOOD))
    better = dict(_GOOD, hidden_size=3840, intermediate_size=15360, kv_heads=8, dominant_dtype="bfloat16")
    monkeypatch.setattr(m, "_perf_target_inputs", lambda *a, **k: dict(better), raising=False)
    m._emit_perf_target_inputs(d, d, None, {})
    got = json.loads((d / "perf_target_inputs.json").read_text())
    assert got.get("hidden_size") == 3840 and got.get("intermediate_size") == 15360
    assert got["weight_bytes"] == 24374793024, "the refresh dropped what it was not replacing"


def test_a_hand_tuned_file_is_still_never_touched(monkeypatch):
    m = _run()
    d = Path(tempfile.mkdtemp())
    hand = {"total_params": 1, "source": "hand-tuned per-tensor dtypes"}
    (d / "perf_target_inputs.json").write_text(json.dumps(hand))
    monkeypatch.setattr(m, "_perf_target_inputs", lambda *a, **k: dict(_GOOD), raising=False)
    m._emit_perf_target_inputs(d, d, None, {})
    assert json.loads((d / "perf_target_inputs.json").read_text()) == hand


# ---------------------------------------------------------------- the resolver


class _Tree:
    """A model tree that names several cached variants, as gemma3's does."""

    def __init__(self, tmp, mapping):
        self.root = Path(tmp)
        for fname, ids in mapping.items():
            p = self.root / fname
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("\n".join('MODEL = "%s"' % i for i in ids))


_GEMMA_TREE = {
    "tests/e2e/conftest.py": ["google/gemma-3-12b-it"],
    "tests/e2e/test_main_perf.py": ["google/gemma-3-12b-it"],
    "tests/e2e/test_pcc_hf.py": ["google/gemma-3-12b-it"],
    "tests/e2e/test_prefill_host_split.py": ["google/gemma-3-12b-it"],
    "tests/test_ci_dispatch.py": ["google/gemma-3-4b-it", "google/gemma-3-27b-it"],
}


def test_the_most_attested_model_wins_not_the_first_file_scanned(monkeypatch):
    """Four files against one. It returned the 4b because rglob reached that file first."""
    m = _run()
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(m, "_is_cached_model_id", lambda c: bool(c) and "gemma-3" in str(c), raising=False)
    t = _Tree(tempfile.mkdtemp(), _GEMMA_TREE)
    assert m._resolve_model_id(t.root, None) == "google/gemma-3-12b-it"


def test_the_run_s_own_test_files_decide_it(monkeypatch, tmp_path):
    """THE FACT, NOT A VOTE. Counting which id appeared in the most files was the first attempt and
    was wrong: a vote has no reason to be right on a tree nobody has seen, and the stress suite over
    400 random trees said so immediately. The run already knows which files it EXECUTES -- its PCC
    test and its perf test -- and the model those pin is the model being run."""
    m = _run()
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(m, "_is_cached_model_id", lambda c: bool(c) and "gemma-3" in str(c), raising=False)
    t = _Tree(tempfile.mkdtemp(), _GEMMA_TREE)
    pcc = t.root / "tests/e2e/test_pcc_hf.py"
    assert m._resolve_model_id(t.root, None, (str(pcc) + "::test_e2e_pcc_hf",)) == "google/gemma-3-12b-it"


def test_without_that_fact_the_tree_scan_still_answers(monkeypatch):
    """The scan is the last resort and must never return None: with nothing stating which model this
    is, one answer from the family beats no answer. Refusing on ambiguity was tried and broke the
    400-tree stress contract."""
    m = _run()
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(m, "_is_cached_model_id", lambda c: bool(c) and "gemma-3" in str(c), raising=False)
    t = _Tree(tempfile.mkdtemp(), {"a.py": ["google/gemma-3-4b-it"]})
    assert m._resolve_model_id(t.root, None, ()) == "google/gemma-3-4b-it"


def test_an_explicit_hint_and_HF_MODEL_still_outrank_the_scan(monkeypatch):
    """The scan is the last resort; a stated model id is evidence, not a count."""
    m = _run()
    monkeypatch.setattr(m, "_is_cached_model_id", lambda c: bool(c) and "gemma-3" in str(c), raising=False)
    t = _Tree(tempfile.mkdtemp(), _GEMMA_TREE)
    assert m._resolve_model_id(t.root, "google/gemma-3-27b-it") == "google/gemma-3-27b-it"
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-1b-it")
    assert m._resolve_model_id(t.root, None) == "google/gemma-3-1b-it"


def test_no_caller_leaves_the_resolver_to_guess():
    """PRODUCTION SHAPE FIRST, TESTS SECOND. The previous version derived the run's test files INSIDE
    _resolve_model_id, and the reason written in the comment was that the callers are monkeypatched
    in tests with two-argument lambdas -- test scaffolding dictating the shape of the code under
    test. A function that goes hunting for its own inputs is one nobody can reason about from the
    call site.

    Every caller now states what it knows; the stubs were widened to the real signature instead."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    calls = [ln for ln in src.splitlines() if "_resolve_model_id(demo_dir" in ln and not ln.lstrip().startswith("def ")]
    assert calls, "no call sites found"
    for ln in calls:
        assert "_run_test_files(" in ln, "a caller still leaves the resolver to guess: %s" % ln.strip()
    body = src[src.index("def _run_test_files") : src.index("def _resolve_model_id")]
    assert "_latest_manifest" not in body, "the helper hunts for its own manifest instead of being given one"
