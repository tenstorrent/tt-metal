# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""HARD STRESS on model-id resolution: hint -> HF_MODEL -> directory scan.

This value is a DIVISOR. Get it wrong and nothing errors -- the run measures the right model and
reports the wrong roofline, which is exactly what happened: a 4B id on a 12B run turned 84%
utilisation into 28%, and 20.5-27.3 into a 61.4-81.9 band the model could never reach. A wrong id is
silent, so the tests have to be the thing that is loud.

The failure to defend against is not "no answer" but "a PLAUSIBLE WRONG answer": a decoy id in a CI
file, a sibling variant of the same family, a stale env var from a previous run, a name that looks
downloaded but is not.

  s1  PRECEDENCE is total and stable across 400 randomised trees
  s2  the env tier is VALIDATED, never trusted blindly
  s3  DECOYS: sibling variants, many candidates, deep nesting, unicode/comments
  s4  HOSTILE trees never raise and never fabricate
  s5  PURITY: no writes, no env mutation, deterministic across repeats
  s6  the DIVISOR that follows -- the thing that actually broke
"""

import importlib.util
import os
import random
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_mid_stress", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run()
_FAMILY = ["google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it"]


def _cached(*known):
    """Stand-in for _is_cached_model_id: only these ids exist locally."""
    s = set(known)
    return lambda v: bool(v) and str(v) in s


def _tree(root: Path, files: dict):
    for rel, body in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


# --------------------------------------------------------------------------- s1 PRECEDENCE
def test_s1_precedence_holds_over_400_random_trees(tmp_path, monkeypatch):
    """hint > HF_MODEL > scan, whatever the tree looks like."""
    rng = random.Random(20260801)
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    for i in range(400):
        d = tmp_path / f"m{i}"
        n = rng.randint(1, 4)
        files = {}
        for j in range(n):
            depth = "/".join("d%d" % k for k in range(rng.randint(0, 3)))
            name = rng.choice(["conftest.py", "test_ci.py", "demo.py", "x%d.py" % j])
            files[f"{depth}/{name}".lstrip("/")] = 'ID = "%s"\n' % rng.choice(_FAMILY)
        _tree(d, files)
        hint = rng.choice([None, "google/gemma-3-27b-it"])
        env = rng.choice([None, "google/gemma-3-12b-it"])
        monkeypatch.delenv("HF_MODEL", raising=False)
        if env:
            monkeypatch.setenv("HF_MODEL", env)
        got = _M._resolve_model_id(d, hint)
        if hint:
            assert got == hint, f"case {i}: hint ignored"
        elif env:
            assert got == env, f"case {i}: HF_MODEL ignored, got {got}"
        else:
            assert got in _FAMILY, f"case {i}: scan produced {got}"


def test_s1_hint_wins_even_when_env_and_files_agree_on_another(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(tmp_path, {"conftest.py": 'X = "google/gemma-3-12b-it"'})
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-12b-it")
    assert _M._resolve_model_id(d, "google/gemma-3-4b-it") == "google/gemma-3-4b-it"


def test_s1_env_wins_even_when_files_shout_otherwise(tmp_path, monkeypatch):
    """Ten files naming the 4b must not outvote one HF_MODEL naming the 12b."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(tmp_path, {f"f{i}.py": 'X = "google/gemma-3-4b-it"' for i in range(10)})
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-12b-it")
    assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"


# --------------------------------------------------------------------------- s2 ENV VALIDATED
@pytest.mark.parametrize(
    "env",
    ["", "  ", "gemma3", "not/downloaded", "google/gemma-3-99b-it", "../../etc/passwd", "google/gemma-3-12b-it "],
)
def test_s2_unusable_env_never_returned(tmp_path, monkeypatch, env):
    """A junk or un-downloaded HF_MODEL must fall through, not become the divisor."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-4b-it"))
    d = _tree(tmp_path, {"conftest.py": 'X = "google/gemma-3-4b-it"'})
    monkeypatch.setenv("HF_MODEL", env)
    got = _M._resolve_model_id(d, None)
    assert got == "google/gemma-3-4b-it", f"env {env!r} leaked or blocked the scan (got {got})"


def test_s2_trailing_whitespace_is_stripped_then_validated(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    d = _tree(tmp_path, {"a.py": "# nothing"})
    monkeypatch.setenv("HF_MODEL", "  google/gemma-3-12b-it \n")
    assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"


def test_s2_unset_env_is_not_the_string_none(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-4b-it"))
    d = _tree(tmp_path, {"a.py": 'X = "google/gemma-3-4b-it"'})
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(d, None) == "google/gemma-3-4b-it"


# --------------------------------------------------------------------------- s3 DECOYS
def test_s3_the_real_gemma3_tree_shape(tmp_path, monkeypatch):
    """conftest pins the 12b; a CI dispatch file names the 4b and 27b. THE case."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(
        tmp_path,
        {
            "tests/e2e/conftest.py": 'os.environ["HF_MODEL"] = "google/gemma-3-12b-it"\n',
            "tests/test_ci_dispatch.py": 'IDS = ["google/gemma-3-4b-it", "google/gemma-3-27b-it"]\n',
        },
    )
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-12b-it")
    assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"


def test_s3_env_absent_is_honest_about_guessing(tmp_path, monkeypatch):
    """With nothing to ask, the answer is A candidate -- but it must be one that EXISTS."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(
        tmp_path,
        {
            "tests/e2e/conftest.py": 'X = "google/gemma-3-12b-it"\n',
            "tests/test_ci_dispatch.py": 'X = "google/gemma-3-4b-it"\n',
        },
    )
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(d, None) in _FAMILY


def test_s3_uncached_ids_in_files_are_skipped(tmp_path, monkeypatch):
    """A file may name a model nobody downloaded; the scan must keep looking."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    d = _tree(
        tmp_path,
        {"a.py": 'X = "org/never-seen"\n', "b.py": 'X = "another/ghost"\n', "c.py": 'X = "google/gemma-3-12b-it"\n'},
    )
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"


def test_s3_deeply_nested_and_many_files(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    files = {"/".join(f"d{i}" for i in range(8)) + "/deep.py": 'X = "google/gemma-3-12b-it"'}
    files.update({f"pad{i}.py": "# nothing here\n" for i in range(200)})
    d = _tree(tmp_path, files)
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"


# --------------------------------------------------------------------------- s4 HOSTILE
@pytest.mark.parametrize("shape", ["missing", "empty", "file-not-dir", "no-python", "binary"])
def test_s4_hostile_trees_never_raise(tmp_path, monkeypatch, shape):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    monkeypatch.delenv("HF_MODEL", raising=False)
    if shape == "missing":
        d = tmp_path / "gone"
    elif shape == "empty":
        d = tmp_path / "e"
        d.mkdir()
    elif shape == "file-not-dir":
        d = tmp_path / "f.py"
        d.write_text("x")
    elif shape == "no-python":
        d = tmp_path / "np"
        (d).mkdir()
        (d / "README.md").write_text("google/gemma-3-12b-it")
    else:
        d = tmp_path / "b"
        d.mkdir()
        (d / "x.py").write_bytes(b"\xff\xfe\x00\x00nonsense")
    assert _M._resolve_model_id(d, None) is None


def test_s4_unreadable_file_is_skipped_not_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    monkeypatch.delenv("HF_MODEL", raising=False)
    d = tmp_path / "u"
    d.mkdir()
    bad = d / "bad.py"
    bad.write_text('X = "google/gemma-3-12b-it"')
    bad.chmod(0o000)
    (d / "good.py").write_text('X = "google/gemma-3-12b-it"')
    try:
        assert _M._resolve_model_id(d, None) == "google/gemma-3-12b-it"
    finally:
        bad.chmod(0o644)


def test_s4_env_set_but_tree_missing_still_answers(tmp_path, monkeypatch):
    """The env tier must not depend on the directory existing at all."""
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-12b-it")
    assert _M._resolve_model_id(tmp_path / "does-not-exist", None) == "google/gemma-3-12b-it"


# --------------------------------------------------------------------------- s5 PURITY
def test_s5_deterministic_across_repeats(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(tmp_path, {"a.py": 'X = "google/gemma-3-4b-it"', "b.py": 'X = "google/gemma-3-27b-it"'})
    monkeypatch.delenv("HF_MODEL", raising=False)
    first = _M._resolve_model_id(d, None)
    assert all(_M._resolve_model_id(d, None) == first for _ in range(30))


def test_s5_does_not_mutate_the_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached("google/gemma-3-12b-it"))
    monkeypatch.setenv("HF_MODEL", "google/gemma-3-12b-it")
    before = dict(os.environ)
    _M._resolve_model_id(_tree(tmp_path, {"a.py": "#"}), None)
    assert dict(os.environ) == before


def test_s5_does_not_write_to_the_model_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(_M, "_is_cached_model_id", _cached(*_FAMILY))
    d = _tree(tmp_path, {"a.py": 'X = "google/gemma-3-4b-it"'})
    monkeypatch.delenv("HF_MODEL", raising=False)
    before = {p: p.stat().st_mtime_ns for p in d.rglob("*")}
    _M._resolve_model_id(d, None)
    assert {p: p.stat().st_mtime_ns for p in d.rglob("*")} == before


# --------------------------------------------------------------------------- s6 THE DIVISOR
@pytest.mark.parametrize(
    "model_id,params_b,ceiling",
    [
        ("google/gemma-3-12b-it", 12, 42.7),
        ("google/gemma-3-4b-it", 4, 128.0),
        ("google/gemma-3-27b-it", 27, 19.0),
        ("meta-llama/Llama-3.1-8B-Instruct", 8, 64.0),
    ],
)
def test_s6_the_id_determines_the_ceiling(model_id, params_b, ceiling):
    """Why any of this matters: the id sets the divisor, the divisor sets the roofline."""
    from agent import perf_target as pt

    total, _active = _M._params_from_model_id(model_id)
    assert total == pytest.approx(params_b * 1e9), f"{model_id} parsed as {total/1e9}B"
    # width DECLARED, not assumed -- see test_the_ceiling_uses_a_measured_width_not_one_byte
    facts = {"total_params": total, "dominant_dtype": "int8"}
    c, band = pt.rate_and_band(pt.simple_active_bytes(facts), 512e9, frac=pt.bw_fraction(facts))
    assert c == pytest.approx(ceiling, abs=0.15)
    assert band[0] == pytest.approx(c * 0.60, abs=0.05) and band[1] == pytest.approx(c * 0.80, abs=0.05)


def test_s6_the_gemma3_error_quantified():
    """The exact damage: 4B instead of 12B turned 67% utilisation into 22%."""
    from agent import perf_target as pt

    measured = 1000 / 34.82
    out = {}
    for label, p in (("wrong_4b", 4e9), ("right_12b", 12e9)):
        c, _ = pt.rate_and_band(pt.simple_active_bytes({"total_params": p, "dominant_dtype": "int8"}), 512e9, frac=0.80)
        out[label] = round(measured / c * 100)
    assert out["wrong_4b"] == 22 and out["right_12b"] == 67, out
