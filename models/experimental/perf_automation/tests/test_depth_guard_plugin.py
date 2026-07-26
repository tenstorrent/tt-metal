# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The depth guard must beat a perf test that fills in its own TT_PERF_LAYERS default.

Run as pytest-in-pytest against a module carrying the EXACT line xtts_v2's perf test has, so the
import-time-vs-test-body timing is real rather than simulated.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PLUGIN = "models.experimental.perf_automation.agent.depth_guard_plugin"
_REPO = Path(__file__).resolve().parents[4]

# The victim: setdefault at import (as xtts_v2 does), then report what the TEST BODY sees.
_VICTIM = """
import os
os.environ.setdefault("TT_PERF_LAYERS", "2")   # models/demos/xtts_v2/tests/e2e/test_tts_perf.py

def test_what_the_builder_would_see():
    seen = os.environ.get("TT_PERF_LAYERS")
    print("BUILDER_WOULD_SEE=%r" % (seen,))
"""


def _run(tmp_path, *, with_plugin: bool, force_all: str | None, preset: str | None = None):
    t = tmp_path / "test_victim.py"
    t.write_text(_VICTIM)
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "-s", "-q", str(t)]
    if with_plugin:
        cmd += ["-p", _PLUGIN]
    env = {k: v for k, v in __import__("os").environ.items() if k != "TT_PERF_LAYERS"}
    env["PYTHONPATH"] = str(_REPO)
    if force_all is not None:
        env["PERF_MCP_FORCE_ALL_LAYERS"] = force_all
    if preset is not None:
        env["TT_PERF_LAYERS"] = preset
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_REPO), timeout=300)
    for line in ((out.stdout or "") + (out.stderr or "")).splitlines():
        if line.startswith("BUILDER_WOULD_SEE="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("victim test did not report; output:\n%s" % ((out.stdout or "") + (out.stderr or ""))[-2000:])


def test_without_the_plugin_the_module_default_wins(tmp_path):
    """The bug, reproduced: absence means all layers, but the module turns it into 2."""
    assert _run(tmp_path, with_plugin=False, force_all=None) == "'2'"


def test_plugin_restores_absence_so_the_builder_builds_everything(tmp_path):
    """The fix: the module still sets 2 at import, but the test body sees the variable gone."""
    assert _run(tmp_path, with_plugin=True, force_all="1") == "None"


def test_plugin_is_inert_unless_all_layers_was_requested(tmp_path):
    """The tracy run loads no guard / does not set the flag: its capped window must survive."""
    assert _run(tmp_path, with_plugin=True, force_all=None) == "'2'"
    assert _run(tmp_path, with_plugin=True, force_all="0") == "'2'"


def test_an_explicit_caller_cap_is_not_clobbered_when_not_forcing(tmp_path):
    """A positive cap the CALLER set (the tracy slice) is honoured, not deleted."""
    assert _run(tmp_path, with_plugin=True, force_all=None, preset="16") == "'16'"


# --- the xtts_v2 case, and models that read a DIFFERENT variable name -------------------------

_VICTIM_CUSTOM = """
import os
os.environ.setdefault("MAX_LAYERS", "2")        # an existing demo with its own variable name

def test_what_the_builder_would_see():
    print("BUILDER_WOULD_SEE=%r" % (os.environ.get("MAX_LAYERS"),))
"""


def _run_custom(tmp_path, *, depth_vars: str | None):
    import os as _os

    t = tmp_path / "test_victim_custom.py"
    t.write_text(_VICTIM_CUSTOM)
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "-s", "-q", "-p", _PLUGIN, str(t)]
    env = {k: v for k, v in _os.environ.items() if k not in ("TT_PERF_LAYERS", "MAX_LAYERS")}
    env["PYTHONPATH"] = str(_REPO)
    env["PERF_MCP_FORCE_ALL_LAYERS"] = "1"
    if depth_vars is not None:
        env["PERF_MCP_DEPTH_VARS"] = depth_vars
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_REPO), timeout=300)
    for line in ((out.stdout or "") + (out.stderr or "")).splitlines():
        if line.startswith("BUILDER_WOULD_SEE="):
            return line.split("=", 1)[1].strip()
    raise AssertionError((out.stdout or "") + (out.stderr or ""))


def test_a_model_with_its_own_variable_name_needs_discovery(tmp_path):
    """Without the discovered name the guard cannot know what to drop -- this is the gap that
    hoisting _llm_depth_env above the first probe closes."""
    assert _run_custom(tmp_path, depth_vars=None) == "'2'"


def test_discovered_variable_name_is_guarded(tmp_path):
    """With the name discovered from the model's source, the guard drops the right key."""
    assert _run_custom(tmp_path, depth_vars="MAX_LAYERS") == "None"


def test_set_depth_arms_and_disarms_the_guard():
    """set_depth is the single place that decides: asking for all layers arms the guard, asking for a
    positive cap disarms it, so the tracy slice can never be stripped."""
    from models.experimental.perf_automation.agent.layer_depth import ENV, FORCE_ALL, set_depth

    e = {}
    set_depth(e, None)
    assert ENV not in e and e[FORCE_ALL] == "1"

    set_depth(e, 16)
    assert e[ENV] == "16" and FORCE_ALL not in e
