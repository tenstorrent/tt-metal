# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Device-free coverage for the weight-load / JIT-compile observability primitives:
the wall-time ledger, the stall Watchdog, and the per-kernel-compile env opt-in."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from loguru import logger

from ...utils import walltime
from ...utils.progress import Watchdog

_REPO_ROOT = Path(__file__).resolve().parents[4]  # .../<repo>/models/tt_dit/tests/unit/ -> <repo>


@pytest.fixture(autouse=True)
def _fresh_ledger():
    """The ledger is process-global; isolate each test."""
    walltime._ledger = walltime._Ledger()
    yield


# Fixed source; the module name travels as argv (data), never interpolated into the -c program.
_PROBE = "import importlib, os, sys; importlib.import_module(sys.argv[1]); print(os.environ.get('TT_METAL_LOG_KERNEL_COMPILE', '<unset>'))"


def _env_after_import(module: str, preset: str | None = None) -> str:
    """Import `module` in a clean subprocess and return TT_METAL_LOG_KERNEL_COMPILE ('<unset>' if not
    set). `preset` sets the var before the import (to test that an explicit value survives)."""
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    env.pop("TT_METAL_LOG_KERNEL_COMPILE", None)
    if preset is not None:
        env["TT_METAL_LOG_KERNEL_COMPILE"] = preset
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, module], cwd=_REPO_ROOT, env=env, capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.strip()


def test_ledger_reconciles_to_wall():
    walltime.record("weight_load", "vae", 2.0, cached=True)
    walltime.record("gen", "denoise", 3.0)
    out = walltime.render("t", wall=10.0)
    # tracked (5.0) + untracked (5.0) must reconcile to the 10.0 wall we passed.
    assert "TOTAL (tracked)" in out and "5.0" in out
    assert "untracked" in out and "TOTAL (wall)" in out and "10.0" in out
    assert "weight_load" in out and "gen" in out


def test_anomalies_flag_only_cache_misses():
    walltime.record("weight_load", "hit_tensor", 1.0, cached=True)
    assert "ANOMALIES: none" in walltime.render("t")
    walltime.record("weight_load", "miss_tensor", 4.0, cached=False, detail="TT_DIT_CACHE_DIR unset")
    with_miss = walltime.render("t")
    assert "miss_tensor" in with_miss and "TT_DIT_CACHE_DIR unset" in with_miss


def test_atexit_emits_under_pytest(capsys):
    """Regression guard: the ledger must surface at teardown even though pytest is imported (the bug
    was an unconditional `"pytest" in sys.modules` suppression that dropped it on the only LTX path)."""
    assert "pytest" in sys.modules  # we ARE under pytest — the exact suppressed condition
    walltime.record("gen", "denoise", 1.0)
    walltime._atexit()
    assert "WALL-TIME LEDGER" in capsys.readouterr().out


def test_disabled_is_a_noop(monkeypatch):
    monkeypatch.setattr(walltime, "_ENABLED", False)
    walltime.record("gen", "denoise", 9.9)
    assert not walltime._ledger.cats  # nothing recorded when disabled


def test_watchdog_heartbeats_and_records_phase():
    msgs: list[str] = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        with Watchdog("unit phase", interval=0.05):
            time.sleep(0.14)  # long enough for >=1 heartbeat tick
    finally:
        logger.remove(sink)
    assert any("still working" in m for m in msgs), "watchdog emitted no heartbeat"
    assert any("done in" in m for m in msgs), "watchdog logged no completion"
    assert "phase" in walltime._ledger.cats  # non-cache-load label is recorded on exit


def test_dit_model_import_opts_in():
    """Importing a DiT model package sets the per-kernel-compile flag (models/tt_dit/models/__init__)."""
    assert _env_after_import("models.tt_dit.models") == "1"


def test_bare_util_import_stays_quiet():
    """The gate must NOT flip on for non-DiT workloads that only borrow a tt_dit util (e.g. Qwen36
    imports tt_dit.utils.tensor) — otherwise it defeats the build.cpp default-off gate."""
    assert _env_after_import("models.tt_dit.utils.walltime") == "<unset>"


def test_explicit_opt_out_wins():
    """An explicit =0 survives the model-import setdefault (a DiT run can still be silenced)."""
    assert _env_after_import("models.tt_dit.models", preset="0") == "0"
