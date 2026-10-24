# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--trace-replays: the operator's say over how many times the trace speed test replays a stage.

trace_replay defaults to 16 back-to-back replays, cut only when the TIME budget runs short. On a 4-chip
p300c, 2026-10-12, Devstral's encode stage cost 34.3 s a replay: 16 of them took the board from 84C to
the 95C abort limit four and a half minutes in, on every attempt, so emit-e2e's trace gate could never
finish. The count was settable only through TT_TRACE_REPLAY_ITERS, and the gate server's env is built
from scratch, so even that never reached it.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.tt_hw_planner import cli
from scripts.tt_hw_planner.commands import emit_e2e as E
from models.experimental.perf_automation.agent.trace_replay import REPLAY_ITERS_ENV

REPO = Path(cli.__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _no_inherited_count(monkeypatch):
    # set-then-delete so the undo always restores the original: cli.main writes os.environ directly
    monkeypatch.setenv(REPLAY_ITERS_ENV, "")
    monkeypatch.delenv(REPLAY_ITERS_ENV)
    monkeypatch.setenv("_TT_TTNN_PREFLIGHT_DONE", "1")  # parsing only: never probe the device


def _parse(monkeypatch, argv, command):
    seen = {}
    # rc 1: the stage wrapper auto-commits only a clean (rc 0) run, so a parse test can never commit
    monkeypatch.setattr(cli, command, lambda args: (seen.setdefault("args", args), 1)[1])
    try:
        cli.main(argv)
    except SystemExit as exc:  # argparse's error path
        seen["exit"] = exc.code
    return seen


@pytest.mark.parametrize("sub, command", [("emit-e2e", "cmd_emit_e2e"), ("optimize", "cmd_optimize")])
def test_the_flag_reaches_the_environment_every_device_child_inherits(monkeypatch, sub, command):
    seen = _parse(monkeypatch, [sub, "m", "--trace-replays", "2"], command)
    assert seen["args"].trace_replays == 2
    assert os.environ[REPLAY_ITERS_ENV] == "2"


@pytest.mark.parametrize("sub, command", [("emit-e2e", "cmd_emit_e2e"), ("optimize", "cmd_optimize")])
def test_without_the_flag_nothing_changes(monkeypatch, sub, command):
    seen = _parse(monkeypatch, [sub, "m"], command)
    assert seen["args"].trace_replays is None
    assert REPLAY_ITERS_ENV not in os.environ


@pytest.mark.parametrize("bad", ["0", "-3", "two"])
def test_a_count_below_one_is_refused(monkeypatch, bad):
    seen = _parse(monkeypatch, ["emit-e2e", "m", "--trace-replays", bad], "cmd_emit_e2e")
    assert seen.get("exit") == 2 and "args" not in seen


def test_trace_replay_replays_the_count_asked_for():
    """The real module, in a fresh interpreter, reads the name the flag writes."""
    code = "from models.experimental.perf_automation.agent import trace_replay as t; print(t._REPLAY_ITERS)"
    env = {**os.environ, REPLAY_ITERS_ENV: "3", "PYTHONPATH": str(REPO)}
    out = subprocess.run([sys.executable, "-c", code], env=env, cwd=str(REPO), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "3"


class _Stop(Exception):
    pass


def _gate_server_env(monkeypatch, tmp_path, expect_error):
    """The env emit-e2e hands its gate server, captured where the server's config is built."""
    from scripts.tt_hw_planner import cc_harness

    seen = {}

    def _capture(pybin, server_path, env, name):
        seen["env"] = dict(env)
        raise _Stop

    monkeypatch.setattr(cc_harness, "build_mcp_config", _capture)
    with expect_error(_Stop, ""):
        E._run_emit_e2e_cc(model_id="m", demo_dir=tmp_path, pcc=0.99, timeout_s=60, agent_bin="claude", max_rounds=1)
    return seen["env"]


def test_the_gate_server_gets_the_count(monkeypatch, tmp_path, expect_error):
    monkeypatch.setenv(REPLAY_ITERS_ENV, "2")
    assert _gate_server_env(monkeypatch, tmp_path, expect_error)[REPLAY_ITERS_ENV] == "2"


def test_the_gate_server_env_is_unchanged_without_it(monkeypatch, tmp_path, expect_error):
    assert REPLAY_ITERS_ENV not in _gate_server_env(monkeypatch, tmp_path, expect_error)
