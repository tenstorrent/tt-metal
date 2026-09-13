# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""When the perf-test agent loses its one device tool, the run must say WHY.

WHAT IT COST, 2026-08-15. A fresh venv installed mcp 2.0.0, which relocated FastMCP out of
`mcp.server.fastmcp`. Every in-process MCP server here imports that path, so the perf-test server
died at import and never registered its one tool, `run_perf_test`.

The agent noticed immediately and said so in its own transcript:

    compiles cleanly ... but I cannot reach PASS_TRACE because the `run_perf_test` tool is not
    registered in this session and I won't run pytest myself; please expose it so I can iterate.

The operator saw none of that. The launcher returned a bare False, the caller printed

    · agentic builder did not converge; falling back to one-shot generator

and the one-shot generator produced files that failed to collect. That reads as "this model is hard
to generate a perf test for", so three runs were spent looking at the model, the branch, the board
and the build -- for a pinned dependency. The moment mcp<2 was installed, the very next attempt
returned a real verdict instead.

WHAT THE PIN DID NOT FIX, 2026-08-27. The identical failure recurred on a box whose venv was built
from a tree that predated the pin: uv resolved mcp 2.1.1 from claude-agent-sdk's uncapped
`mcp>=1.23.0` floor, and the file saying `<2` arrived hours later by merge -- a merge changes the
file, not the venv. A ceiling only helps machines built after it lands, and it holds the tool one
major behind forever. So the servers were made to run on EITHER major instead, and the requirement
became a floor.

THREE RULES:

    1. Every in-process MCP server imports FastMCP under mcp 1.x AND MCPServer under 2.x. This is
       what replaced the upper bound: compatibility travels with the source, a pin does not.
    2. The probe checks the SAME PAIR the servers import. Hard-coding one spelling reintroduces the
       silent fallback with the check as its cause rather than the dependency.
    3. "Unavailable" must carry its reason. An agent with no way to RUN what it writes cannot
       converge, and reporting that as non-convergence sends the reader after the wrong thing.
"""

import re
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


_SERVERS = (
    _PA / "cc_optimize" / "perf_mcp.py",
    _PA / "cc_optimize" / "perf_test_mcp.py",
    _PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "bringup_mcp.py",
    _PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "e2e_mcp.py",
)


def test_every_in_process_mcp_server_survives_the_fastmcp_relocation():
    """The compatibility that replaced the pin -- and it has to be in EVERY server, not most.

    perf_mcp.py carried this fallback from the start; the other three did not, which is exactly why
    a 2.x venv killed the perf-test server while the optimize server kept running and the failure
    looked model-shaped.
    """
    for path in _SERVERS:
        src = path.read_text()
        assert "from mcp.server.fastmcp import FastMCP" in src, f"{path.name} lost the mcp 1.x import"
        assert (
            "from mcp.server.mcpserver import MCPServer as FastMCP" in src
        ), f"{path.name} dies at import on mcp >= 2.0"


def test_the_requirement_is_a_floor_not_a_ceiling():
    """An upper bound puts the tool one major behind and only helps machines built after it lands."""
    req = (_PA / "requirements-agent.txt").read_text()
    assert re.search(r"^mcp>=[\d.]+\s*$", req, re.M), "the mcp requirement is not a plain floor"
    assert not re.search(r"^mcp[^\n]*<2", req, re.M), "the <2 ceiling is back; make the servers compatible instead"


def test_the_requirement_explains_itself():
    """Without the history, the next dependency sweep re-pins it or drops the fallback."""
    req = (_PA / "requirements-agent.txt").read_text()
    i = req.index("mcp>=")
    window = req[max(0, i - 900) : i]
    assert "FastMCP" in window, "nothing records WHAT broke"
    assert "did not converge" in window, "the symptom is not recorded, so the fallback looks arbitrary"


def _launcher():
    src = (_PA / "agent" / "perf_test_agent.py").read_text()
    i = src.index("server = repo_root / _PERF_TEST_MCP_REL")
    return src[i : i + 2200]


def _probe_source():
    """The literal the launcher hands to `python -c`."""
    import ast

    tree = ast.parse((_PA / "agent" / "perf_test_agent.py").read_text())
    return next(
        ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "_MCP_IMPORT_PROBE"
    )


def test_an_unstartable_server_is_reported_not_swallowed():
    body = _launcher()
    assert "_MCP_IMPORT_PROBE" in body, "nothing checks that the MCP server can start"
    assert "cannot start" in body, "the failure is still reported as plain non-convergence"


def test_the_probe_accepts_either_mcp_major():
    """A probe narrower than the servers reports a working server as unavailable."""
    probe = _probe_source()
    assert "from mcp.server.fastmcp import FastMCP" in probe
    assert "from mcp.server.mcpserver import MCPServer as FastMCP" in probe


def test_the_message_names_the_fix():
    """The reader should not have to find the dependency themselves."""
    body = _launcher()
    assert "requirements-agent.txt" in body


def test_a_missing_server_file_is_reported_too():
    """The other silent `return False` on the same path."""
    body = _launcher()
    i = body.index("if not server.is_file():")
    assert "agentic builder unavailable" in body[i : i + 300], "a missing server file still returns quietly"


def test_the_probe_cannot_hang_the_run():
    """It runs a subprocess on the device-work path; an unbounded one would be a new way to stall."""
    body = _launcher()
    i = body.index("_MCP_IMPORT_PROBE")
    assert "timeout=" in body[i : i + 300], "the import probe has no timeout"


def test_it_still_returns_false_so_the_fallback_path_is_unchanged():
    """This adds a REASON, not a new failure mode: the one-shot generator must still get its turn."""
    body = _launcher()
    i = body.index("cannot start")
    assert "return False" in body[i : i + 700], "the guard raises instead of falling back"
