# SPDX-License-Identifier: Apache-2.0
"""HITL gate: file handshake between the agent (perf-mcp subprocess) and the orchestrator (run.py, which
owns the terminal), plus the pure pause-screen render. Same long-lived agent: it posts a proposal and
blocks on a decision; the orchestrator reads the proposal, answers, and the agent continues."""

import importlib.util
import threading
import time
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cc_hitl", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "hitl.py")
)
hitl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hitl)


def _proposal():
    return {
        "model": "kokoro_82m",
        "step": 3,
        "stages": [
            {"name": "encode", "ms": 157.8},
            {"name": "prosody", "ms": 118.2},
            {"name": "vocode", "ms": 207.1, "dominant": "conv"},
        ],
        "tried": {"op": "MoE.gate", "lever": "dtype bf16->bf8_b", "why": "DRAM-bw bound"},
        "result": {"win": True, "before_ms": 89.2, "after_ms": 86.4, "checks": "PCC 0.987 ok"},
        "next": {"target": "grid on attn.qkv", "why": "core-underutilized"},
    }


def test_handshake_roundtrip_agent_blocks_until_orchestrator_answers(tmp_path):
    got = {}

    def agent():
        hitl.post_proposal(tmp_path, _proposal())
        got["decision"] = hitl.await_decision(tmp_path, poll=0.02, timeout=5)

    t = threading.Thread(target=agent)
    t.start()
    # orchestrator: wait for the proposal, then answer
    prop = None
    for _ in range(200):
        prop = hitl.read_proposal(tmp_path)
        if prop:
            break
        time.sleep(0.02)
    assert prop is not None and prop["step"] == 3 and prop["tried"]["op"] == "MoE.gate"
    hitl.post_decision(tmp_path, "commit", note="looks good")
    t.join(timeout=5)
    assert got["decision"]["action"] == "commit" and got["decision"]["note"] == "looks good"


def test_await_decision_times_out_to_revert(tmp_path):
    hitl.post_proposal(tmp_path, _proposal())
    dec = hitl.await_decision(tmp_path, poll=0.02, timeout=0.1)
    assert dec["action"] == "revert"  # unattended -> never bank an unreviewed edit


def test_render_pause_screen_shows_hotspot_reason_and_options(tmp_path):
    screen = hitl.render_pause_screen(_proposal())
    assert "BLOCK-LEVEL TIMING" in screen
    assert "vocode" in screen and "<- hottest" in screen  # highest-ms stage flagged
    assert "dtype bf16->bf8_b" in screen and "DRAM-bw bound" in screen  # what + why
    assert "WIN" in screen and "89.2" in screen and "86.4" in screen  # result + delta
    assert "grid on attn.qkv" in screen and "core-underutilized" in screen  # next + why
    assert "[c] commit" in screen and "[r] revert" in screen and "[t] try" in screen


def test_render_handles_no_win_with_reason():
    p = _proposal()
    p["result"] = {"win": False, "why_not": "PCC 0.71 < 0.95"}
    screen = hitl.render_pause_screen(p)
    assert "NO WIN" in screen and "PCC 0.71 < 0.95" in screen


def _load_run():
    """run.py is stdlib-only, so it loads by path without ttnn — gives us the REAL _hitl_watch + _git."""
    spec = importlib.util.spec_from_file_location(
        "cc_run", str(Path(__file__).resolve().parents[1] / "cc_optimize" / "run.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _git_repo(tmp_path):
    import subprocess

    r = tmp_path / "repo"
    r.mkdir()
    for c in (["git", "init", "-q"], ["git", "config", "user.email", "t@t"], ["git", "config", "user.name", "t"]):
        subprocess.run(c, cwd=r, check=True)
    (r / "model.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=r, check=True)
    return r


def _run_gate_once(tmp_path, monkeypatch, answer, knob=""):
    """Drive the REAL orchestrator watcher ↔ handshake end-to-end with a scripted operator."""
    import subprocess

    run = _load_run()
    repo = _git_repo(tmp_path)
    hdir = tmp_path / "h"
    hdir.mkdir()
    (repo / "model.py").write_text("x = 2  # a lever edit\n")  # the agent's uncommitted lever

    inputs = iter([answer] + ([knob] if answer == "t" else []))
    monkeypatch.setattr("builtins.input", lambda *a: next(inputs))

    stop = __import__("threading").Event()
    wt = __import__("threading").Thread(target=run._hitl_watch, args=(repo, str(hdir), stop), daemon=True)
    wt.start()
    got = {}

    def agent():
        hitl.post_proposal(hdir, _proposal())
        got["dec"] = hitl.await_decision(hdir, poll=0.02, timeout=8)

    a = __import__("threading").Thread(target=agent)
    a.start()
    a.join(timeout=8)
    stop.set()
    head = subprocess.run(["git", "log", "--oneline", "-1"], cwd=repo, capture_output=True, text=True).stdout
    body = (repo / "model.py").read_text()
    return got.get("dec"), head, body


def test_gate_commit_actually_commits(tmp_path, monkeypatch):
    dec, head, body = _run_gate_once(tmp_path, monkeypatch, "c")
    assert dec and dec["action"] == "commit"
    assert "hitl:" in head and "x = 2" in body  # lever committed, edit kept


def test_gate_revert_discards_edit(tmp_path, monkeypatch):
    dec, head, body = _run_gate_once(tmp_path, monkeypatch, "r")
    assert dec and dec["action"] == "revert"
    assert "base" in head and body == "x = 1\n"  # edit discarded, no new commit


def test_gate_try_returns_knob(tmp_path, monkeypatch):
    dec, head, body = _run_gate_once(tmp_path, monkeypatch, "t", knob="grid on attn.qkv")
    assert dec and dec["action"] == "try" and dec["knob"] == "grid on attn.qkv"
