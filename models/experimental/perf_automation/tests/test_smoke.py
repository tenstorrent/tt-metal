"""M0 smoke test: package imports and the SDK building blocks are available."""


def test_agent_imports():
    import agent  # noqa: F401
    from agent import config  # noqa: F401


def test_sdk_building_blocks_importable():
    # PLAN section 3: the workflow is built on these SDK pieces.
    from claude_agent_sdk import (  # noqa: F401
        tool,
        create_sdk_mcp_server,
        ClaudeAgentOptions,
        query,
    )
