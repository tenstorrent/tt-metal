# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Depth-knob discovery must search the model, not a guessed subfolder -- and must not trust a knob
that does nothing.

_env_reads globbed ``model_root/tt/*.py``. llama's reader lives at tt/pipeline.py:70, so the knob was
found; gemma3's ONLY reader is the generated tests/e2e/test_main_perf.py, which that glob never
opens. Discovery therefore reported "no knob after 8 attempts" for TT_PERF_LAYERS -- a variable this
tool injects itself -- and coverage fell to the unverified floor of 2.

Widening the search alone would make gemma3 WORSE. Its perf test reads the variable but hands
build_pipeline no depth argument (build_pipeline has no such parameter and reads no env), so the cap
is dropped. Every ladder rung would build the identical 48-layer model, return the full op set,
satisfy coverage on the first rung, and report "measured" -- a confident wrong answer replacing an
honest fallback. The two changes only make sense together, so both are pinned here.

  s1  SCOPE     the reader is found wherever it lives, across layouts no folder list anticipates
  s2  REAL      the real llama and gemma3 trees in this checkout
  s3  INERT     a knob that does not shrink the model is refused, not laundered into "measured"
  s4  LIVE      a knob that does shrink the model still measures
  s5  HOSTILE   junk trees never raise
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_knob_scope", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run()
READ = 'num = os.environ.get("TT_PERF_LAYERS")\n'


def _tree(root: Path, rel: str, body: str = READ):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("import os\n" + body)
    return p


# --------------------------------------------------------------------------- s1 SCOPE
@pytest.mark.parametrize(
    "rel",
    [
        "tt/pipeline.py",
        "tests/e2e/test_main_perf.py",
        "tests/pcc/test_mod.py",
        "demo/text_demo.py",
        "perf.py",
        "src/deep/nested/again/builder.py",
        "a/b/c/d/e/f/g.py",
    ],
)
def test_s1_reader_found_wherever_it_lives(tmp_path, rel):
    _tree(tmp_path, rel)
    assert _M._known_depth_env(tmp_path), f"missed the reader at {rel}"


def test_s1_the_exact_gemma3_layout(tmp_path):
    """The real shape: a demo whose tt/ has no reader and whose perf test does."""
    _tree(tmp_path, "tt/pipeline.py", 'x = os.environ.get("HF_MODEL")\n')
    _tree(tmp_path, "tt/model_config.py", 'y = os.environ.get("TT_CACHE_PATH")\n')
    _tree(tmp_path, "tests/e2e/test_main_perf.py")
    assert _M._known_depth_env(tmp_path) == {"TT_PERF_LAYERS": "1"}


def test_s1_no_reader_anywhere_is_still_no_knob(tmp_path):
    """Widening must not invent a knob: absence stays absence."""
    _tree(tmp_path, "tt/pipeline.py", 'x = os.environ.get("HF_MODEL")\n')
    _tree(tmp_path, "tests/e2e/test_main_perf.py", 'y = os.environ.get("MESH_DEVICE")\n')
    assert _M._known_depth_env(tmp_path) == {}


@pytest.mark.parametrize(
    "form", ['os.environ.get("TT_PERF_LAYERS")', 'os.getenv("TT_PERF_LAYERS")', 'os.environ["TT_PERF_LAYERS"]']
)
def test_s1_every_read_form(tmp_path, form):
    _tree(tmp_path, "tt/x.py", f"v = {form}\n")
    assert _M._known_depth_env(tmp_path)


def test_s1_skips_caches_and_build_dirs(tmp_path):
    """A stale __pycache__ or build copy must not be the thing that answers."""
    _tree(tmp_path, "__pycache__/old.py")
    _tree(tmp_path, "build/copy.py")
    _tree(tmp_path, ".git/hook.py")
    assert _M._known_depth_env(tmp_path) == {}


def test_s1_returns_lines_not_files(tmp_path):
    _tree(tmp_path, "tests/e2e/test_main_perf.py")
    rows = _M._env_reads(tmp_path)
    assert rows and all(len(r) == 3 for r in rows)
    assert any(r[0] == "TT_PERF_LAYERS" and "environ" in r[2] for r in rows)


def test_s1_stays_small_on_a_big_tree(tmp_path):
    for i in range(300):
        _tree(tmp_path, f"pkg{i // 20}/mod{i}.py", "z = 1\n")
    _tree(tmp_path, "tests/e2e/test_main_perf.py")
    rows = _M._env_reads(tmp_path)
    assert len(rows) < 50 and sum(len(str(r)) for r in rows) < 20000


# --------------------------------------------------------------------------- s2 REAL TREES
_REPO = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "rel,expect",
    [
        ("models/demos/llama3_1_8b_p150", True),
        ("models/demos/multimodal/gemma3", None),
    ],
)
def test_s2_real_model_dirs(rel, expect):
    """llama declares the knob in tt/. gemma3 declares it only in its GENERATED perf test, which is
    absent from a clean tree -- so the expectation there is 'whatever the tree says', asserted
    against the file's presence rather than a hardcoded answer."""
    root = _REPO / rel
    if not root.is_dir():
        pytest.skip(f"{rel} not in this checkout")
    found = bool(_M._known_depth_env(root))
    reads_it = any(v == "TT_PERF_LAYERS" for v, _f, _l in _M._env_reads(root))
    assert found == reads_it, "discovery disagrees with the tree it just scanned"
    if expect is True:
        assert found, "llama's tt/pipeline.py reads TT_PERF_LAYERS; it must be found"


def test_s2_gemma3_generated_perf_test_would_be_found(tmp_path):
    """Reproduces the exact miss: the generated perf test, verbatim in shape, under tests/e2e."""
    _tree(
        tmp_path,
        "tests/e2e/test_main_perf.py",
        '_pl = (os.environ.get("TT_PERF_LAYERS") or "").strip()\n'
        "PERF_LAYERS = int(_pl) if (_pl.isdigit() and int(_pl) > 0) else None\n",
    )
    assert _M._known_depth_env(tmp_path) == {"TT_PERF_LAYERS": "1"}


# --------------------------------------------------------------------------- s3 INERT
def test_s3_identical_work_signal_means_inert(tmp_path):
    """gemma3's case: cap requested, dropped by the builder, so the probe does the same work."""
    seq = ["op"] * 5000
    assert _M._knob_is_inert(seq, _M._work_signal(seq), 2, tmp_path) is True


def test_s3_a_real_cap_is_not_inert(tmp_path):
    full = _M._work_signal(["op"] * 5000)
    sliced = ["op"] * 210
    assert _M._knob_is_inert(sliced, full, 2, tmp_path) is False


def test_s3_full_depth_rung_is_never_called_inert(tmp_path, monkeypatch):
    """At the model's own depth the signals SHOULD match -- that is not a broken knob."""
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 48)
    seq = ["op"] * 5000
    assert _M._knob_is_inert(seq, _M._work_signal(seq), 48, tmp_path) is False
    assert _M._knob_is_inert(seq, _M._work_signal(seq), 64, tmp_path) is False


def test_s3_the_real_measured_gemma3_numbers(tmp_path):
    """MEASURED ON DEVICE, not a fixture. gemma-3-12b built twice through the real pipeline:

        TT_PERF_LAYERS unset -> built_layers=48  work_signal=12461
        TT_PERF_LAYERS=2     -> built_layers=48  work_signal=12461

    Identical, because build_pipeline takes no depth argument and reads no env, so the cap is
    dropped. These are those two numbers.
    """
    assert _M._knob_is_inert(["op"] * 12461, 12461, 2, tmp_path) is True
    # what the same model would have produced had the cap reached the builder (2 of 48 layers)
    assert _M._knob_is_inert(["op"] * 519, 12461, 2, tmp_path) is False


def test_s3_undeclared_depth_only_judges_the_shallowest_rung(tmp_path):
    """_declared_depth returns None for gemma3, so "the cap was ignored" and "this rung IS the whole
    model" are indistinguishable at depth. A 4-layer model probed at rung 4 legitimately does
    full-model work; calling that an inert knob would be a false positive."""
    seq = ["op"] * 12461
    assert _M._knob_is_inert(seq, 12461, 2, tmp_path) is True
    for deep_rung in (4, 8, 16):
        assert _M._knob_is_inert(seq, 12461, deep_rung, tmp_path) is False, f"judged rung {deep_rung} blind"


def test_s3_declared_depth_judges_every_rung_below_it(tmp_path, monkeypatch):
    """With the depth known there is no ambiguity: any rung below it must shrink the work."""
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 48)
    seq = ["op"] * 12461
    for rung in (2, 4, 8, 16):
        assert _M._knob_is_inert(seq, 12461, rung, tmp_path) is True
    assert _M._knob_is_inert(seq, 12461, 48, tmp_path) is False


def test_s3_unknown_signal_is_not_a_verdict(tmp_path):
    assert _M._knob_is_inert(["op"] * 10, None, 2, tmp_path) is False
    assert _M._knob_is_inert([], 5000, 2, tmp_path) is False


def test_s3_measure_cov_refuses_the_inert_knob(tmp_path, monkeypatch, capsys):
    """END TO END: the false 'measured' that widening the search would otherwise create."""
    full_seq = ["op%d" % (i % 156) for i in range(5000)]
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 48)
    monkeypatch.setattr(_M, "_cov_ladder", lambda *a, **k: [2, 4, 8, 16])
    monkeypatch.setattr(_M, "_run_op_sigs", lambda *a, **k: (set("op%d" % (i % 156) for i in range(156)), "", full_seq))
    out = _M._measure_cov(
        tmp_path,
        {},
        "0",
        "n.py::t",
        None,
        ["op%d" % (i % 156) for i in range(156)],
        tmp_path,
        base_knob={"TT_PERF_LAYERS": "2"},
        full_signal=_M._work_signal(full_seq),
    )
    assert out is None, "an inert knob was laundered into a coverage window"
    assert "INERT" in capsys.readouterr().out


def test_s3_without_the_guard_it_would_have_said_measured(tmp_path, monkeypatch):
    """Control: same inputs, guard disabled -- shows the guard is what changes the outcome, not the
    fixture. Without it the first rung satisfies coverage and returns 'measured'."""
    full_seq = ["op%d" % (i % 156) for i in range(5000)]
    want = ["op%d" % (i % 156) for i in range(156)]
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 48)
    monkeypatch.setattr(_M, "_cov_ladder", lambda *a, **k: [2, 4, 8, 16])
    monkeypatch.setattr(_M, "_run_op_sigs", lambda *a, **k: (set(want), "", full_seq))
    monkeypatch.setattr(_M, "_knob_is_inert", lambda *a, **k: False)
    out = _M._measure_cov(
        tmp_path, {}, "0", "n.py::t", None, want, tmp_path, base_knob={"TT_PERF_LAYERS": "2"}, full_signal=5000
    )
    assert out == (2, [], "measured")


# --------------------------------------------------------------------------- s4 LIVE KNOB
def test_s4_a_working_knob_still_measures(tmp_path, monkeypatch):
    want = ["a", "b", "c"]
    monkeypatch.setattr(_M, "_declared_depth", lambda *a, **k: 32)
    monkeypatch.setattr(_M, "_cov_ladder", lambda *a, **k: [2, 4, 8, 16])
    monkeypatch.setattr(_M, "_run_op_sigs", lambda *a, **k: ({"a", "b", "c"}, "", ["op"] * 200))
    out = _M._measure_cov(
        tmp_path, {}, "0", "n.py::t", None, want, tmp_path, base_knob={"TT_PERF_LAYERS": "2"}, full_signal=3200
    )
    assert out == (2, [], "measured")


def test_s4_no_knob_still_skips(tmp_path, capsys):
    assert _M._measure_cov(tmp_path, {}, "0", "n.py::t", None, ["a"], tmp_path, base_knob=None) is None
    assert "no depth knob" in capsys.readouterr().out


# --------------------------------------------------------------------------- s5 HOSTILE
@pytest.mark.parametrize("bad", ["missing", "empty", "file-not-dir", "none"])
def test_s5_hostile_roots_never_raise(tmp_path, bad):
    if bad == "missing":
        root = tmp_path / "nope"
    elif bad == "empty":
        root = tmp_path
    elif bad == "file-not-dir":
        root = tmp_path / "f.py"
        root.write_text("import os\n")
    else:
        root = None
    assert _M._env_reads(root) == []
    assert _M._known_depth_env(root) == {}


def test_s5_unreadable_file_is_skipped(tmp_path):
    (tmp_path / "bin.py").write_bytes(b"\xff\xfe\x00broken")
    _tree(tmp_path, "tt/ok.py")
    assert _M._known_depth_env(tmp_path)
