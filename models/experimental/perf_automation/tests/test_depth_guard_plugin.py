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
    # Without an explicit --rootdir, pytest computes one as the common ancestor of cwd (_REPO) and
    # the given test path (t, under pytest's own tmp_path fixture -- always somewhere under /tmp).
    # _REPO and tmp_path are unrelated locations that can both happen to sit directly under /tmp
    # (e.g. _REPO staged there by the tool's own preflight isolation), making their nearest common
    # ancestor /tmp itself -- which pytest then scans, tripping over whatever unrelated file
    # (another user's session, a broken symlink) happens to be sitting in a shared /tmp. Pinning
    # --rootdir to tmp_path (already an ancestor of t) sidesteps that computation entirely.
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "--rootdir", str(tmp_path), "-s", "-q", str(t)]
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
    # See _run's comment: pins rootdir so pytest never computes /tmp as the common ancestor.
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-o",
        "addopts=",
        "--rootdir",
        str(tmp_path),
        "-s",
        "-q",
        "-p",
        _PLUGIN,
        str(t),
    ]
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


# --- the hook must not bill its own recursion to the model it measures -------------------------

_VICTIM_IMPORT_COST = """
import builtins, time

def test_a_runtime_import_is_a_dict_lookup():
    # One runtime import statement, like `import torch` inside ttnn.from_torch's body, executed on
    # every decode token of the model under measurement.
    calls = [0]
    hooked = builtins.__import__
    def counting(*a, **k):
        calls[0] += 1
        return hooked(*a, **k)
    builtins.__import__ = counting
    try:
        import os as _o  # noqa: F401
    finally:
        builtins.__import__ = hooked
    t = time.perf_counter()
    for _ in range(1000):
        import os as _o2  # noqa: F401
    per_us = (time.perf_counter() - t) / 1000 * 1e6
    print("IMPORT_HOOK_CALLS=%d IMPORT_US=%.1f" % (calls[0], per_us))
"""


def test_the_hook_does_not_recurse_into_itself(tmp_path):
    """The plugin wraps builtins.__import__ and its body used to contain `import sys` -- an import
    statement, i.e. a call of the wrapper itself -- so every runtime import recursed ~1000 frames deep
    to a swallowed RecursionError and cost ~1 ms. With the plugin loaded, one `import os` must reach
    __import__ a handful of times (the wrapper and the nested _orig_import call chain), not hundreds,
    and must cost microseconds, not a millisecond."""
    t = tmp_path / "test_import_cost.py"
    t.write_text(_VICTIM_IMPORT_COST)
    cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", "--rootdir", str(tmp_path), "-s", "-q", "-p", _PLUGIN, str(t)]
    env = dict(__import__("os").environ)
    env["PYTHONPATH"] = str(_REPO)
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_REPO), timeout=300)
    text = (out.stdout or "") + (out.stderr or "")
    line = next((ln for ln in text.splitlines() if ln.startswith("IMPORT_HOOK_CALLS=")), None)
    assert line, text[-2000:]
    calls = int(line.split()[0].split("=")[1])
    per_us = float(line.split()[1].split("=")[1])
    assert calls < 10, "one `import os` re-entered __import__ %d times: the hook is importing inside itself" % calls
    assert per_us < 100.0, "a runtime import costs %.1f us under the plugin; it must stay a dict lookup" % per_us


def test_set_depth_arms_and_disarms_the_guard():
    """set_depth is the single place that decides: asking for all layers arms the guard, asking for a
    positive cap disarms it, so the tracy slice can never be stripped."""
    from models.experimental.perf_automation.agent.layer_depth import ENV, FORCE_ALL, set_depth

    e = {}
    set_depth(e, None)
    assert ENV not in e and e[FORCE_ALL] == "1"

    set_depth(e, 16)
    assert e[ENV] == "16" and FORCE_ALL not in e
