"""M0 smoke test: the package imports and the model-invocation path is resolvable."""


def test_agent_imports():
    import agent  # noqa: F401
    from agent import config  # noqa: F401


def test_the_claude_cli_is_the_model_invocation_path():
    """This test used to assert the claude_agent_sdk building blocks (tool, create_sdk_mcp_server,
    ClaudeAgentOptions, query) were importable, because the workflow was built on them. The SDK is
    gone -- every model call is now a headless `claude` CLI subprocess -- so the thing to check is
    that the CLI is resolvable to a path, which is the one dependency that would break every
    generator and every optimize round if it were missing."""
    from agent.agent_bin import resolve_claude_bin

    path = resolve_claude_bin()
    assert isinstance(path, str) and path
