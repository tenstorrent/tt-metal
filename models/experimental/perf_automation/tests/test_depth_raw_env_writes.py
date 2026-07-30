# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Issue 5: three sites wrote the depth variable RAW, bypassing set_depth().

``set_depth()`` exists because expressing a depth is two coupled facts, not one:

    positive cap -> write the variable AND clear PERF_MCP_FORCE_ALL_LAYERS
    all layers   -> DELETE the variable AND arm PERF_MCP_FORCE_ALL_LAYERS

Its docstring is explicit that the flag is managed inside the helper "so no caller can express
'all layers' and forget to defend it". But ``_knob_at`` (run.py:788), ``_bridge_depth_env``
(run.py:982) and ``_measure_cov`` (run.py:1048) all did a bare ``env[numkey] = str(n)``, so:

  * a stale ``PERF_MCP_FORCE_ALL_LAYERS=1`` inherited from an earlier all-layers step stayed armed
    while a numeric cap was requested -- the depth guard then strips the very cap that was just
    written, and the "2-layer" rung silently profiles the whole model, which is exactly the
    "every rung profiled the SAME full model" symptom; and
  * a non-positive depth was written as the literal string "0", which every builder reads as a
    truthy string meaning "build zero layers" -- the sentinel set_depth was written to abolish.

set_depth() hardcoded TT_PERF_LAYERS, so these sites could not use it for a model-specific knob;
it now takes an optional key.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_PA = Path(__file__).resolve().parent.parent
_CC = _PA / "cc_optimize"
sys.path.insert(0, str(_PA))

from agent.layer_depth import ENV, FORCE_ALL, set_depth  # noqa: E402


def _mod():
    spec = importlib.util.spec_from_file_location("cc_run_raw", str(_CC / "run.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --------------------------------------------------------------------------- set_depth(key=)
def test_set_depth_accepts_a_custom_key():
    env = set_depth({}, 4, key="MY_MODEL_LAYERS")
    assert env == {"MY_MODEL_LAYERS": "4"}


def test_set_depth_custom_key_all_layers_deletes_and_arms():
    env = set_depth({"MY_MODEL_LAYERS": "4"}, None, key="MY_MODEL_LAYERS")
    assert "MY_MODEL_LAYERS" not in env
    assert env[FORCE_ALL] == "1"


def test_set_depth_default_key_unchanged():
    assert set_depth({}, 4) == {ENV: "4"}
    env = set_depth({ENV: "4"}, 0)
    assert ENV not in env and env[FORCE_ALL] == "1"


# --------------------------------------------------------------------------- _knob_at
def test_knob_at_clears_a_stale_force_all_flag():
    """The regression: a numeric cap requested while FORCE_ALL is armed from an earlier step. The
    guard would strip the cap and the rung would profile the whole model."""
    got = _mod()._knob_at({"TT_PERF_LAYERS": "1", FORCE_ALL: "1"}, 4)
    assert got["TT_PERF_LAYERS"] == "4"
    assert FORCE_ALL not in got, (
        "a positive cap was requested but PERF_MCP_FORCE_ALL_LAYERS stayed armed, so the depth "
        "guard will strip the cap and this rung will profile every layer"
    )


def test_knob_at_non_positive_removes_the_var_instead_of_writing_zero():
    got = _mod()._knob_at({"TT_PERF_LAYERS": "4"}, 0)
    assert got.get("TT_PERF_LAYERS") != "0", (
        '"0" is a truthy string; a builder reads it as "build zero layers". All-layers must be '
        "expressed by REMOVING the variable."
    )
    assert "TT_PERF_LAYERS" not in got
    assert got[FORCE_ALL] == "1"


def test_knob_at_honours_a_model_specific_key():
    got = _mod()._knob_at({"MY_MODEL_LAYERS": "1", FORCE_ALL: "1"}, 8)
    assert got["MY_MODEL_LAYERS"] == "8"
    assert FORCE_ALL not in got


def test_knob_at_keeps_companion_flags():
    """A knob may be several variables; only the numeric one is the depth."""
    got = _mod()._knob_at({"CAP": "1", "ALLOW_PARTIAL": "yes"}, 4)
    assert got["CAP"] == "4" and got["ALLOW_PARTIAL"] == "yes"


def test_knob_at_does_not_mutate_its_input():
    src = {"CAP": "1", FORCE_ALL: "1"}
    _mod()._knob_at(src, 4)
    assert src == {"CAP": "1", FORCE_ALL: "1"}


# --------------------------------------------------------------------------- wiring
@pytest.mark.parametrize("fn", ["_knob_at", "_bridge_depth_env", "_measure_cov"])
def test_no_raw_depth_writes_remain(fn):
    import inspect
    import re

    src = inspect.getsource(getattr(_mod(), fn))
    # CODE only: the fix's own comments quote the offending line to explain it.
    code = "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())
    raw = re.findall(r"env\[_?numkey\]\s*=\s*str\(", code)
    assert not raw, (
        f"{fn} still writes the depth variable raw ({len(raw)} site(s)), bypassing set_depth() and "
        "leaving PERF_MCP_FORCE_ALL_LAYERS in whatever state it was already in"
    )
