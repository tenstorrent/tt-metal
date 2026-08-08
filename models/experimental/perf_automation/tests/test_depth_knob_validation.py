# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 3: a discovered depth knob was cached without ever being validated.

``_llm_depth_env`` took ANY non-empty dict the agent returned and immediately handed it to
``_knob_cache_put``. It never checked that the named variable is read anywhere in the model, so a
fabricated variable was accepted, persisted to the knob cache, and reused by later runs -- and
because the cache is consulted first, the bad answer outlived the run that produced it.

That is what closed the llama3_1_8b_p150 failure chain: the agent had (correctly) answered ``{}``
four times, and on attempt five an invented ``TT_PERF_LAYERS``-shaped answer was taken at face
value. The coverage ladder then set a variable the demo never reads, so every rung profiled the
identical full model while the run believed it was slicing.

The model's own source is the ground truth, and checking it is a grep.
"""

import importlib.util
import sys
from pathlib import Path


_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"


def _mod():
    sys.path.insert(0, str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_val", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _model(tmp_path, body):
    tt = tmp_path / "tt"
    tt.mkdir(parents=True, exist_ok=True)
    (tt / "p.py").write_text(body)
    return tmp_path


def _wire(monkeypatch, answer):
    m = _mod()
    cached = {}
    monkeypatch.setattr(m, "_claude_text", lambda _p: answer)
    monkeypatch.setattr(m, "_knob_cache_get", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_knob_cache_put", lambda _r, env: cached.setdefault("env", env))
    monkeypatch.setenv("PERF_MCP_KNOB_RETRIES", "2")
    return m, cached


def test_fabricated_variable_is_rejected(tmp_path, monkeypatch):
    """The model reads REAL_CAP. The agent claims INVENTED_CAP. That must not be believed."""
    m, cached = _wire(monkeypatch, '{"INVENTED_CAP": "4"}')
    root = _model(tmp_path, 'import os\nn = os.environ.get("REAL_CAP")\n')
    got = m._llm_depth_env(root, 4)
    assert got == {}, f"a variable the model never reads was accepted as the depth knob: {got}"
    assert "env" not in cached, "the unvalidated answer was written to the knob cache"


def test_real_variable_is_accepted_and_cached(tmp_path, monkeypatch):
    m, cached = _wire(monkeypatch, '{"REAL_CAP": "4"}')
    root = _model(tmp_path, 'import os\nn = os.environ.get("REAL_CAP")\n')
    assert m._llm_depth_env(root, 4) == {"REAL_CAP": "4"}
    assert cached.get("env") == {"REAL_CAP": "4"}


def test_partly_fabricated_answer_is_rejected_whole(tmp_path, monkeypatch):
    """A knob is a SET of variables that must be set together (cap + permit-partial flag). If one
    half is invented, applying the other half alone is not a partial success -- it is a run that
    believes it is slicing and is not."""
    m, cached = _wire(monkeypatch, '{"REAL_CAP": "4", "INVENTED_FLAG": "1"}')
    root = _model(tmp_path, 'import os\nn = os.environ.get("REAL_CAP")\n')
    got = m._llm_depth_env(root, 4)
    assert got == {}, f"an answer containing an invented variable was partly accepted: {got}"
    assert "env" not in cached


def test_multi_variable_answer_all_real_is_accepted(tmp_path, monkeypatch):
    m, cached = _wire(monkeypatch, '{"REAL_CAP": "4", "ALLOW_PARTIAL": "1"}')
    root = _model(
        tmp_path,
        'import os\nn = os.environ.get("REAL_CAP")\nf = os.getenv("ALLOW_PARTIAL")\n',
    )
    got = m._llm_depth_env(root, 4)
    assert got == {"REAL_CAP": "4", "ALLOW_PARTIAL": "1"}
    assert cached.get("env")


def test_rejection_is_retried_then_gives_up_cleanly(tmp_path, monkeypatch, capsys):
    m, cached = _wire(monkeypatch, '{"INVENTED_CAP": "4"}')
    root = _model(tmp_path, 'import os\nn = os.environ.get("REAL_CAP")\n')
    assert m._llm_depth_env(root, 4) == {}
    out = capsys.readouterr().out
    assert "REAL_CAP" not in out or "INVENTED_CAP" in out  # message may name either; must not crash
    assert "env" not in cached


def test_validation_names_the_rejected_variable(tmp_path, monkeypatch, capsys):
    """An operator reading the log must be able to tell a rejection from an empty answer."""
    m, _cached = _wire(monkeypatch, '{"INVENTED_CAP": "4"}')
    root = _model(tmp_path, 'import os\nn = os.environ.get("REAL_CAP")\n')
    m._llm_depth_env(root, 4)
    out = capsys.readouterr().out
    assert "INVENTED_CAP" in out, f"the rejection did not say what was rejected:\n{out}"
