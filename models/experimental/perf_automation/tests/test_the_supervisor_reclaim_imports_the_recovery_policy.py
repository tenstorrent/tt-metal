"""The supervisor's reclaim reaches the recovery policy as a package module, not a by-path orphan.

commands/optimize.py imports run.py by its dotted name with only the repo root on sys.path, so
`from agent import device_recovery` failed and the by-path fallback gave the module no package: its own
relative imports raised "attempted relative import with no known parent package", and every reclaim
fell back to a bare tt-smi -r. Imports only -- nothing here touches a device.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parents[1]
_REPO = _PA.parent.parent.parent


def _in_a_bare_process(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run([sys.executable, "-c", code], cwd=str(_REPO), env=env, capture_output=True, text=True)


def test_the_dotted_import_reaches_both_modules_as_package_members():
    r = _in_a_bare_process(
        "import sys\n"
        "assert not any(p.endswith('perf_automation') for p in sys.path)\n"
        "import models.experimental.perf_automation.cc_optimize.run as r\n"
        "print(r._dr().__name__); print(r._ap().__name__)\n"
    )
    assert r.returncode == 0, r.stderr[-800:]
    names = r.stdout.split()
    assert names == [
        "models.experimental.perf_automation.agent.device_recovery",
        "models.experimental.perf_automation.agent.agent_provider",
    ]


def test_a_run_py_loaded_by_path_still_finds_them(monkeypatch):
    monkeypatch.syspath_prepend(str(_PA))
    spec = importlib.util.spec_from_file_location("cc_run_reclaim_ut", str(_PA / "cc_optimize" / "run.py"))
    run = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run)
    assert callable(run._dr().recover) and callable(run._ap().get)


def test_one_helper_serves_both_lookups():
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert src.count("def _agent_module(") == 1
    assert '_agent_module("device_recovery", "tt_device_recovery")' in src
    assert '_agent_module("agent_provider", "tt_agent_provider")' in src
