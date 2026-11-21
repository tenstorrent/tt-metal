# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the model id as: hint -> HF_MODEL -> directory scan.

The middle tier did not exist in the cc engine. optimize.py passes

    model_id_hint=(None if model_dir else args.target)

so pointing at a DEMO DIRECTORY -- how every brought-up model is optimized -- nulls the hint and
drops straight to the scan. The scan takes the first cached id found in any .py under the model dir,
and gemma3's tree names three: conftest.py pins google/gemma-3-12b-it, test_ci_dispatch.py mentions
the 4b and the 27b. It returned the 4b.

Nothing was mismeasured -- the model resolves its own identity from HF_MODEL and the 12B ran
throughout -- but every roofline figure divided by 4 GB instead of 12:

    ceiling      128.0 tok/s/u   should be 42.7   (spec 512/GB; band top 25.6-34.1)
    band         61.4-81.9       should be 20.5-27.3
    "measured" BW  115 GB/s      should be 345
    utilisation  28%             actually 84%, ABOVE the band

HF_MODEL was in the tool's own environment the whole time, unread. before_loop.py:262 already reads
it; this gives run.py the same tier.

  h1  precedence: hint > HF_MODEL > scan
  h2  the gemma3 case end to end
  h3  a junk HF_MODEL does not displace the scan
  h4  the scan still works when nothing is set
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_mid", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_M = _run()
_REAL = "google/gemma-3-12b-it"
_OTHER = "google/gemma-3-4b-it"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """A demo dir shaped like gemma3's: the real id in conftest, a decoy in a CI file."""
    (tmp_path / "tests" / "e2e").mkdir(parents=True)
    (tmp_path / "tests" / "e2e" / "conftest.py").write_text(f'os.environ["HF_MODEL"] = "{_REAL}"\n')
    (tmp_path / "tests" / "test_ci_dispatch.py").write_text(f'IDS = ["{_OTHER}", "google/gemma-3-27b-it"]\n')
    monkeypatch.delenv("HF_MODEL", raising=False)
    return tmp_path


# --------------------------------------------------------------------------- h1 PRECEDENCE
def test_h1_hint_wins_over_everything(tree, monkeypatch):
    """An explicit CLI target is the strongest statement of intent."""
    monkeypatch.setenv("HF_MODEL", _REAL)
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: bool(s) and "/" in str(s))
    assert _M._resolve_model_id(tree, "google/gemma-3-27b-it") == "google/gemma-3-27b-it"


def test_h1_hf_model_wins_over_the_scan(tree, monkeypatch):
    """THE FIX: with no hint, the env the model itself sets decides -- not a file scan."""
    monkeypatch.setenv("HF_MODEL", _REAL)
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: bool(s) and "/" in str(s))
    assert _M._resolve_model_id(tree, None) == _REAL


def test_h1_scan_only_when_both_are_absent(tree, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: bool(s) and "/" in str(s))
    got = _M._resolve_model_id(tree, None)
    assert got in (_REAL, _OTHER, "google/gemma-3-27b-it"), got


# --------------------------------------------------------------------------- h2 THE CASE
def test_h2_the_gemma3_run(tree, monkeypatch):
    """Directory target -> hint is None -> HF_MODEL must supply the 12b, not the scan's 4b."""
    monkeypatch.setenv("HF_MODEL", _REAL)
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: bool(s) and "/" in str(s))
    assert _M._resolve_model_id(tree, None) == _REAL, "still guessing from files with HF_MODEL set"


def test_h2_the_bytes_that_follow_from_it(monkeypatch):
    """12B -> 12 GB -> ceiling 42.7, not 128.0. The whole point of getting the id right."""
    sys.path.insert(0, str(_PA))
    from agent import perf_target as pt

    # A width is DECLARED now: the ceiling no longer assumes one byte per parameter, because that is
    # right only for a 1-byte format and a bf16 model was being handed a ceiling above what the
    # hardware permits. int8 keeps the arithmetic of this test exactly as written (12 GB -> 42.7) while
    # stating the width instead of assuming it.
    for params, want_bytes, want_ceiling in ((12e9, 12e9, 42.7), (4e9, 4e9, 128.0)):
        facts = {"total_params": params, "dominant_dtype": "int8"}
        b = pt.simple_active_bytes(facts)
        c, _band = pt.rate_and_band(b, 512e9, frac=pt.bw_fraction(facts))
        assert b == pytest.approx(want_bytes)
        assert c == pytest.approx(want_ceiling, abs=0.1)


# --------------------------------------------------------------------------- h3/h4 GUARDS
@pytest.mark.parametrize("junk", ["", "   ", "not-a-model", "gemma3"])
def test_h3_junk_env_falls_through_to_the_scan(tree, monkeypatch, junk):
    """An unusable HF_MODEL must not block the fallback, nor be returned."""
    monkeypatch.setenv("HF_MODEL", junk)
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: bool(s) and "/" in str(s))
    got = _M._resolve_model_id(tree, None)
    assert got != junk
    assert got is None or "/" in got


def test_h4_missing_dir_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("HF_MODEL", raising=False)
    assert _M._resolve_model_id(tmp_path / "nope", None) is None


def test_h4_env_id_must_still_be_a_known_model(tree, monkeypatch):
    """HF_MODEL is trusted only if it names a model that actually exists locally."""
    monkeypatch.setenv("HF_MODEL", "org/never-downloaded-model")
    monkeypatch.setattr(_M, "_is_cached_model_id", lambda s: s == _OTHER)
    assert _M._resolve_model_id(tree, None) == _OTHER


def test_h4_source_guard_env_tier_present():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    i = src.index("def _resolve_model_id")
    body = src[i : src.index("\ndef ", i + 10)]
    assert "HF_MODEL" in body, "the cc engine still never reads HF_MODEL"
