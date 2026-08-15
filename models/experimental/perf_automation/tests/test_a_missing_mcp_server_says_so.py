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

TWO RULES:

    1. The version that works is PINNED where people install from, not just in one venv. An upper
       bound is the whole point -- 2.0.0 was a breaking relocation, and a floor would not have helped.
    2. "Unavailable" must carry its reason. An agent with no way to RUN what it writes cannot
       converge, and reporting that as non-convergence sends the reader after the wrong thing.
"""

import re
import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def test_mcp_is_pinned_below_the_release_that_moved_fastmcp():
    req = (_PA / "requirements-agent.txt").read_text()
    m = re.search(r"^mcp>=[\d.]+,<2\s*$", req, re.M)
    assert m, "mcp is not pinned below 2 in requirements-agent.txt"


def test_the_pin_explains_itself():
    """A bare `<2` invites someone to relax it on the next dependency sweep."""
    req = (_PA / "requirements-agent.txt").read_text()
    i = req.index("mcp>=")
    window = req[max(0, i - 700) : i]
    assert "FastMCP" in window, "nothing records WHY the upper bound exists"
    assert "did not converge" in window, "the symptom is not recorded, so the pin looks arbitrary"


def _launcher():
    src = (_PA / "agent" / "perf_test_agent.py").read_text()
    i = src.index("server = repo_root / _PERF_TEST_MCP_REL")
    return src[i : i + 2200]


def test_an_unstartable_server_is_reported_not_swallowed():
    body = _launcher()
    assert "import mcp.server.fastmcp" in body, "nothing checks that the MCP server can start"
    assert "cannot start" in body, "the failure is still reported as plain non-convergence"


def test_the_message_names_the_fix():
    """The reader should not have to find the pin themselves."""
    body = _launcher()
    assert "mcp>=1.0,<2" in body
    assert "requirements-agent.txt" in body


def test_a_missing_server_file_is_reported_too():
    """The other silent `return False` on the same path."""
    body = _launcher()
    i = body.index("if not server.is_file():")
    assert "agentic builder unavailable" in body[i : i + 300], "a missing server file still returns quietly"


def test_the_probe_cannot_hang_the_run():
    """It runs a subprocess on the device-work path; an unbounded one would be a new way to stall."""
    body = _launcher()
    i = body.index("import mcp.server.fastmcp")
    assert "timeout=" in body[i : i + 300], "the import probe has no timeout"


def test_it_still_returns_false_so_the_fallback_path_is_unchanged():
    """This adds a REASON, not a new failure mode: the one-shot generator must still get its turn."""
    body = _launcher()
    i = body.index("cannot start")
    assert "return False" in body[i : i + 700], "the guard raises instead of falling back"
