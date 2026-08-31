"""`runs/latest` must name the run that is happening, and only its own model's report.

Two failures of the same pointer, both observed on one box:

* An existing demo is optimized in a throwaway worktree, so the run directory and the `latest`
  beside it are created THERE. The `latest` in the checkout the operator has open was never
  touched, so it went on naming a run from three days earlier while a current one was in flight.
  Every reader who followed it read a stale report and concluded the live one was never written.

* `report_path` returned that run directory to ANY caller, ignoring the model_root it was handed.
  The tool's own suite passes a temporary model directory and expects the report there, so a test
  run overwrote the real report -- a finished 6.2 KB report replaced by a 57-byte fixture.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
if str(PERF) not in sys.path:
    sys.path.insert(0, str(PERF))

from agent.before_loop import _publish_latest_to_main_worktree  # noqa: E402
from cc_optimize import summary as S  # noqa: E402

_RUNS_REL = Path("models") / "experimental" / "perf_automation" / "runs"


def _repo(at: Path) -> Path:
    at.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(at)], check=True)
    (at / "seed").write_text("x")
    subprocess.run(["git", "-C", str(at), "add", "-A"], check=True)
    subprocess.run(
        ["git", "-C", str(at), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "seed"],
        check=True,
    )
    return at


def _worktree_run(tmp_path):
    main = _repo(tmp_path / "main")
    wt = tmp_path / "wt"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", "--detach", str(wt)], check=True)
    (main / _RUNS_REL).mkdir(parents=True)
    run_dir = wt / _RUNS_REL / "2026-01-01T00-00-00-m"
    run_dir.mkdir(parents=True)
    (wt / _RUNS_REL / "latest").symlink_to(run_dir.name)
    return main, wt, run_dir


def test_a_worktree_run_is_findable_from_the_main_checkout(tmp_path, monkeypatch):
    """Without a durable home there is nothing local to point at, so name the worktree directly."""
    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    main, wt, run_dir = _worktree_run(tmp_path)

    _publish_latest_to_main_worktree(run_dir, wt / _RUNS_REL)

    pointer = main / _RUNS_REL / "latest"
    assert pointer.is_symlink(), "the main checkout still has no pointer to the running run"
    assert pointer.resolve() == run_dir.resolve()


def test_persist_gives_the_main_checkout_its_own_run_directory(tmp_path, monkeypatch):
    """--persist must behave in the main checkout exactly as the worktree behaves in itself:
    `latest` names a real local directory, and the report is inside it. Pointing across at /tmp
    made the pointer a trap -- it died with the worktree it named."""
    state = tmp_path / "state"
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(state))
    main, wt, run_dir = _worktree_run(tmp_path)

    _publish_latest_to_main_worktree(run_dir, wt / _RUNS_REL)

    pointer = main / _RUNS_REL / "latest"
    assert Path(os.readlink(pointer)) == Path(run_dir.name), "pointer should name the run, not a path"
    assert pointer.resolve().is_relative_to(main), "the pointer must stay inside the main checkout"

    monkeypatch.setattr(S, "_runs_root", lambda: wt / _RUNS_REL)
    S._mirror_report("live text")
    assert (state / "RUN_REPORT.md").read_text() == "live text"
    assert (pointer / "RUN_REPORT.md").read_text() == "live text", "following latest found no report"


def test_an_in_place_run_is_left_alone(tmp_path):
    """No worktree means the pointer Run.create already wrote is correct -- do not touch it."""
    main = _repo(tmp_path / "main")
    runs = main / _RUNS_REL
    run_dir = runs / "2026-01-01T00-00-00-m"
    run_dir.mkdir(parents=True)
    pointer = runs / "latest"
    pointer.symlink_to(run_dir.name)

    _publish_latest_to_main_worktree(run_dir, runs)

    assert pointer.is_symlink() and pointer.resolve() == run_dir.resolve()


def test_a_run_directory_is_not_handed_to_another_model(tmp_path, monkeypatch):
    """The report of a model that owns no run must not land in another model's run directory."""
    runs = tmp_path / "runs"
    latest = runs / "r1"
    latest.mkdir(parents=True)
    (latest / "manifest.json").write_text(json.dumps({"config": {"model_root": str(tmp_path / "owner")}}))
    (runs / "latest").symlink_to("r1")
    monkeypatch.setattr(S, "_runs_root", lambda: runs)

    stranger = tmp_path / "stranger"
    stranger.mkdir()
    assert S.report_path(stranger) == stranger / "RUN_REPORT.md"
    assert S.report_path(tmp_path / "owner") == runs / "latest" / "RUN_REPORT.md"


def test_a_run_that_has_not_declared_its_model_keeps_its_directory(tmp_path, monkeypatch):
    """Conservative fallback: a manifest that is missing or silent must not move the report."""
    runs = tmp_path / "runs"
    latest = runs / "r1"
    latest.mkdir(parents=True)
    (runs / "latest").symlink_to("r1")
    monkeypatch.setattr(S, "_runs_root", lambda: runs)
    assert S.report_path(tmp_path / "anything") == runs / "latest" / "RUN_REPORT.md"

    (latest / "manifest.json").write_text(json.dumps({"config": {}}))
    assert S.report_path(tmp_path / "anything") == runs / "latest" / "RUN_REPORT.md"

    (latest / "manifest.json").write_text("{ not json")
    assert S.report_path(tmp_path / "anything") == runs / "latest" / "RUN_REPORT.md"


def test_the_mcp_config_carries_a_silence_ceiling():
    """The client aborts a tool call that goes quiet, and the device steps go quiet.

    "sent no response or progress for 1800s; aborting" is a SILENCE limit, and a profile holds the
    device far longer than that emitting nothing. The client's own abort message names the per-server
    `timeout` field as the lever, and that field overrides the MCP_TOOL_TIMEOUT environment variable --
    setting only the variable left the abort in place, which is how this was first missed.
    """
    from cc_optimize.run import _mcp_config, _mcp_silence_timeout_ms

    cfg = _mcp_config(
        Path("/nonexistent-repo"),
        "manifest.json",
        {"perf_test": "t.py::t", "pcc_test": "p.py::p", "case": ""},
        "0",
        "kernel.json",
    )
    server = cfg["mcpServers"]["perf-mcp"]
    assert set(("command", "args", "env")) <= set(server), "the server entry lost a required key"
    assert server["timeout"] == _mcp_silence_timeout_ms()
    assert server["timeout"] >= 1000, "the client ignores values below 1000ms"
