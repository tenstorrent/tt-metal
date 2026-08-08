# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The copy that RUNS is not the copy that was edited, and nothing checked the difference.

The tool is developed in one checkout and synced into the repo the run executes from. Three
distinct failures came out of that gap, each costing a run:

  * an edit that applied to the WRONG PLACE -- DEFAULT_ISL_TOKENS landed inside a template STRING
    instead of at module scope. The module imported cleanly and the symbol did not exist;
  * a module reachable by package name but not by PATH -- the report loader uses
    spec_from_file_location, which gives no package context and no sys.path entry for the module's
    own directory, so both relative and absolute imports raise. The report rendered with three
    blank sections and every failure was silent;
  * a `git stash` during a debugging detour that never popped, so two committed fixes were absent
    from the tree that then ran.

Each was found hours in, from an unrelated symptom, and each would have been caught by running the
tool's own suite against the tree about to be used -- ~90 seconds against a run measured in hours.

The rule this encodes is the one that kept being broken by hand: a preflight that could not RUN has
not cleared anything. A timeout, a missing suite, a crashed pytest -- none of those are a pass.

  r1  a red suite stops the run
  r2  a preflight that could not run is UNKNOWN, not OK
  r3  the skip is explicit and says so
  r4  it runs against the RUN'S repo, not the developer's
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
sys.path.insert(0, str(_PA.parent.parent.parent))


def _run():
    spec = importlib.util.spec_from_file_location("cc_run_preflight", str(_PA / "cc_optimize" / "run.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["cc_run_preflight"] = m
    spec.loader.exec_module(m)
    return m


def _repo(tmp_path):
    (tmp_path / "models/experimental/perf_automation/tests").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _result(rc, stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


# --------------------------------------------------------------------------- r1
def test_r1_a_red_suite_stops_the_run(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _result(1, "FAILED tests/test_x.py::test_y\n1 failed\n"))
    assert m._preflight_tool(_repo(tmp_path)) is False
    out = capsys.readouterr().out
    assert "preflight FAILED" in out and "test_x.py::test_y" in out


def test_r1_a_green_suite_proceeds(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: _result(0, "2504 passed, 5 skipped in 87s\n"))
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert "preflight OK" in capsys.readouterr().out


# --------------------------------------------------------------------------- r2
def test_r2_a_preflight_that_could_not_run_is_not_a_pass(monkeypatch, tmp_path, capsys):
    """A guard initialised to the passing value and wrapped in `except` is defect shape 2 in
    agent/integrity.py. It proceeds by default -- a broken preflight must not brick every run --
    but it SAYS it is unknown, and PERF_MCP_REQUIRE_PREFLIGHT=1 makes it a stop."""
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)

    def _boom(*a, **k):
        raise subprocess.TimeoutExpired(cmd="pytest", timeout=900)

    monkeypatch.setattr(m.subprocess, "run", _boom)
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert "treating as UNKNOWN, not as passed" in capsys.readouterr().out

    monkeypatch.setenv("PERF_MCP_REQUIRE_PREFLIGHT", "1")
    assert m._preflight_tool(_repo(tmp_path)) is False


def test_r2_a_missing_suite_is_unknown_too(monkeypatch, tmp_path, capsys):
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    monkeypatch.delenv("PERF_MCP_REQUIRE_PREFLIGHT", raising=False)
    assert m._preflight_tool(tmp_path) is True  # no tests dir at all
    assert "cannot verify the tool" in capsys.readouterr().out


# --------------------------------------------------------------------------- r3
def test_r3_the_skip_is_explicit_and_announced(monkeypatch, tmp_path, capsys):
    """A silent skip is how the check stops existing."""
    m = _run()
    monkeypatch.setenv("PERF_MCP_SKIP_PREFLIGHT", "1")
    called = []
    monkeypatch.setattr(m.subprocess, "run", lambda *a, **k: called.append(1) or _result(0))
    assert m._preflight_tool(_repo(tmp_path)) is True
    assert not called, "the suite ran despite the skip"
    assert "preflight SKIPPED" in capsys.readouterr().out


# --------------------------------------------------------------------------- r4
def test_r4_it_tests_the_repo_the_run_will_use(monkeypatch, tmp_path):
    """Against the DEVELOPER's checkout it proves nothing: the sync is the step that fails."""
    m = _run()
    monkeypatch.delenv("PERF_MCP_SKIP_PREFLIGHT", raising=False)
    seen = {}

    def _cap(cmd, **k):
        seen["cmd"], seen["cwd"] = cmd, k.get("cwd")
        return _result(0, "ok")

    monkeypatch.setattr(m.subprocess, "run", _cap)
    repo = _repo(tmp_path)
    m._preflight_tool(repo)
    assert seen["cwd"] == str(repo)
    assert str(repo / "models/experimental/perf_automation/tests") in seen["cmd"]


# --------------------------------------------------------------------------- r5 A REFUSAL IS NOT A CRASH
def test_r5_a_refusal_exits_with_its_own_code():
    """The auto-restart supervisor exists for a native tt-metal SIGSEGV, and read ANY non-zero exit
    as that case. The first real preflight refusal was therefore reported as "likely native crash /
    device wedge", the board was reset, and the same decision was re-derived from the same evidence
    -- three times, ten minutes, for a verdict available at once."""
    m = _run()
    assert m.EXIT_REFUSED != 0 and m.EXIT_REFUSED != 1, m.EXIT_REFUSED


def test_r5_the_supervisor_does_not_restart_a_refusal():
    sup = (_PA.parent.parent.parent / "scripts/tt_hw_planner/commands/optimize.py").read_text()
    i = sup.index("_EXIT_REFUSED")
    body = sup[i : i + 2000]
    assert "return _rc" in body
    assert "not a crash" in body.lower() or "Not restarting" in body


def test_r5_the_exit_code_has_one_definition():
    """A second literal in the supervisor that drifted from run.py's would turn every refusal back
    into three device resets, silently."""
    m = _run()
    sup = (_PA.parent.parent.parent / "scripts/tt_hw_planner/commands/optimize.py").read_text()
    assert "from models.experimental.perf_automation.cc_optimize.run import EXIT_REFUSED" in sup
    assert "_EXIT_REFUSED = %d" % m.EXIT_REFUSED in sup, "the supervisor fallback disagrees with run.py"


def test_r5_both_deliberate_refusals_use_it():
    """The dirty-tree refusal is the same kind of decision and was the same rc=1."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    assert src.count("raise SystemExit(EXIT_REFUSED)") >= 2, "a deliberate refusal still exits 1"


def test_r4_the_run_calls_it_before_touching_the_device():
    """Ordered ahead of discovery, which spends an agent call, and ahead of any device work."""
    src = (_PA / "cc_optimize" / "run.py").read_text()
    body = src[src.index("def run_cc_optimize") :]
    _end = body.find("\ndef ", 1)
    body = body if _end < 0 else body[:_end]
    assert body.index("_preflight_tool") < body.index("discover("), "preflight runs after discovery"
