"""A loop selects an agent by name; the CLI's flags belong to that agent's entry, not to the loop.

Both loops used to spell one CLI's command line inline -- optimize in cc_optimize/run.py, bring-up in
scripts/tt_hw_planner/cc_harness.py -- so a second agent could not be driven without editing the
loops. What a loop actually requires of an agent is narrow: optimize pipes the agent's stdout to a
log file and never reads it back (the round's outcome comes from _gate_status, which reads the MCP
server's own state), and bring-up parses the stream only to print a heartbeat. Neither needs a
particular transcript shape -- only a process that can reach the server and edit files.

So the four things that genuinely differ per agent are described once, in agent_provider: where the
CLI lives, which flag carries the prompt, how the MCP server is declared, and whether tools can be
named individually.

The default has to keep producing exactly the command line it produced before, or this refactor
silently changes every existing run. That is the first test here.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from models.experimental.perf_automation.agent import agent_provider as ap  # noqa: E402


@pytest.fixture(autouse=True)
def _no_ambient_override(monkeypatch):
    """A shared override in the environment would decide these answers instead of the table."""
    for var in ("TT_PLANNER_AGENT_BIN", "TT_PLANNER_AGENT_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    for p in ap.PROVIDERS.values():
        monkeypatch.delenv(p.bin_env, raising=False)


# ---------------------------------------------------------------- the default may not move


def test_the_default_provider_is_unchanged():
    """The exact argv both loops spelled before this registry existed."""
    launch = ap.launch(
        None,
        prompt="THE PROMPT",
        mcp_config="/cfg/.mcp_config_m_main.json",
        tools=["mcp__perf-mcp__git_head", "mcp__perf-mcp__check_pcc"],
        bin_path="/bin/claude",
    )

    assert launch.argv == [
        "/bin/claude",
        "-p",
        "THE PROMPT",
        "--mcp-config",
        "/cfg/.mcp_config_m_main.json",
        "--strict-mcp-config",
        "--allowedTools",
        "mcp__perf-mcp__git_head",
        "mcp__perf-mcp__check_pcc",
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    assert launch.env == {}, "the default needed no environment of its own before, and must not now"


def test_the_default_writes_the_declaration_where_it_always_did(tmp_path):
    """A provider read on the command line keeps the caller's own filename, so no path moves."""
    stem = ".mcp_config_m_main.json"

    path = ap.mcp_config_path(None, tmp_path, stem)
    ap.write_mcp_config(None, path, {"perf-mcp": {"command": "/py", "args": ["/s.py"], "env": {"A": "1"}}})

    assert path == tmp_path / stem
    import json

    assert json.loads(path.read_text()) == {
        "mcpServers": {"perf-mcp": {"command": "/py", "args": ["/s.py"], "env": {"A": "1"}}}
    }


# ---------------------------------------------------------------- what a second agent needs


def test_an_agent_that_cannot_name_tools_is_not_handed_a_list():
    """Naming tools it cannot accept would put unknown words on its command line.

    The set such an agent can reach is the set the declared server exposes, which is the same set
    the list would have named -- so dropping it removes nothing.
    """
    for p in ap.PROVIDERS.values():
        if p.supports_tool_allowlist:
            continue
        argv = ap.launch(p.name, prompt="P", mcp_config="/c/config.toml", tools=["a", "b"], bin_path="/bin/x").argv
        assert "a" not in argv and "b" not in argv, p.name


def test_an_agent_that_reads_a_config_directory_is_told_where_it_is(tmp_path):
    """Its declaration cannot ride on the command line, so it rides in the environment instead."""
    for p in ap.PROVIDERS.values():
        path = ap.mcp_config_path(p.name, tmp_path, ".mcp_config_m_main.json")
        launch = ap.launch(p.name, prompt="P", mcp_config=path, tools=(), bin_path="/bin/x")
        on_argv = str(path) in launch.argv
        in_env = str(path.parent) in launch.env.values()
        assert on_argv or in_env, "%s is never told where its server declaration is" % p.name


def test_every_provider_can_declare_the_same_server(tmp_path):
    """One server dict, whatever format the agent reads it in -- callers build it once."""
    servers = {"perf-mcp": {"command": "/py", "args": ["/s.py"], "env": {"A": "1", "B": "2"}}}

    for p in ap.PROVIDERS.values():
        path = ap.mcp_config_path(p.name, tmp_path / p.name, ".mcp_config_m_main.json")
        written = ap.write_mcp_config(p.name, path, servers)
        text = written.read_text()
        assert written.exists() and text.strip(), p.name
        for token in ("/py", "/s.py", "A", "B"):
            assert token in text, (p.name, token)


# ---------------------------------------------------------------- the table itself


def test_a_typo_is_refused_rather_than_quietly_run_as_the_default():
    with pytest.raises(ValueError) as exc:  # allow-pytest.raises: no expect_error fixture
        ap.get("cluade")
    assert "cluade" in str(exc.value)
    for name in ap.provider_names():
        assert name in str(exc.value), "the refusal should say what IS available"


def test_the_shared_override_reaches_whichever_agent_is_selected(monkeypatch):
    """One variable to point every loop at a binary, without naming the provider twice."""
    monkeypatch.setenv("TT_PLANNER_AGENT_BIN", "/somewhere/else/agent")
    for name in ap.provider_names():
        assert ap.resolve_bin(name) == "/somewhere/else/agent", name


def test_a_providers_own_variable_wins_over_the_search_path(monkeypatch):
    for p in ap.PROVIDERS.values():
        monkeypatch.setenv(p.bin_env, "/pinned/" + p.name)
        assert ap.resolve_bin(p.name) == "/pinned/" + p.name


def test_an_unvalidated_provider_says_so():
    """Nothing here has been run against every CLI. The ones written from documentation are marked,
    so a loop can warn rather than present a guessed command line as a tested one."""
    assert ap.get(ap.DEFAULT_PROVIDER).validated is True
    assert any(p.validated is False for p in ap.PROVIDERS.values())


def test_no_install_path_is_typed_into_the_table():
    """A home directory belongs to whoever is running, not to the source."""
    src = Path(ap.__file__).read_text()
    for typed in ("/home/", "/Users/", "ttuser"):
        assert typed not in src, typed
