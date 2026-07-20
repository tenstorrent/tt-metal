# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""cc_harness clean-output rendering: the bring-up cc loop must render `claude -p` stream-json into
fsm-style clean lines (tool/text summaries, framework frames dropped) instead of dumping raw JSON,
and fire the pre_round hook with the gate state so the caller can print fsm-style banners."""
import os
import stat

from scripts.tt_hw_planner.cc_harness import _fmt_tool, _render_cc_event, run_cc_loop


def test_fmt_tool_shapes():
    assert _fmt_tool("Bash", {"command": "ls -la"}) == "Bash: ls -la"
    assert _fmt_tool("Read", {"file_path": "a/b.py"}) == "Read a/b.py"
    assert _fmt_tool("Edit", {"file_path": "x.py"}) == "Edit x.py"
    assert _fmt_tool("Grep", {"pattern": "foo", "path": "src"}) == "Grep foo src"


def test_render_drops_framework_frames():
    assert _render_cc_event('{"type":"system","subtype":"init"}') == (None, 0)
    assert _render_cc_event('{"type":"result","result":"done"}') == (None, 0)
    assert _render_cc_event('{"type":"user","message":{"content":[]}}') == (None, 0)
    assert _render_cc_event("") == (None, 0)


def test_render_assistant_tool_and_text():
    r, n = _render_cc_event(
        '{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"pytest x"}}]}}'
    )
    assert r == "  → Bash: pytest x" and n == 1
    r2, n2 = _render_cc_event('{"type":"assistant","message":{"content":[{"type":"text","text":"hello"}]}}')
    assert r2 == "  hello" and n2 == 0


def _fake_claude(tmp_path):
    p = tmp_path / "claude_stream.sh"
    p.write_text(
        "#!/bin/sh\n"
        'echo \'{"type":"system","subtype":"init"}\'\n'
        'echo \'{"type":"assistant","message":{"content":[{"type":"tool_use","name":"Bash","input":{"command":"ls"}}]}}\'\n'
        'echo \'{"type":"result","result":"ok"}\'\n'
    )
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(p)


def test_loop_fires_pre_round_and_renders(tmp_path):
    fake = _fake_claude(tmp_path)
    calls = {"n": 0}
    seen = []

    def gate():
        calls["n"] += 1
        if calls["n"] > 1:
            return {"can_stop": True, "graduated": ["a", "b"], "shard_graduated": []}
        return {"graduated": [], "shard_graduated": [], "next_op": "attn", "next_rung": "emit"}

    res = run_cc_loop(
        prompt="x",
        mcp_config_path=str(tmp_path / "cfg.json"),
        allowed_tools=["Read"],
        cwd=str(tmp_path),
        env=dict(os.environ),
        gate_fn=gate,
        claude_bin=fake,
        agent_timeout_s=30,
        pre_round=lambda r, st: seen.append((r, st.get("next_op"), st.get("next_rung"))),
    )
    assert res["can_stop"] is True
    assert res["rounds"] == 1
    assert seen == [(1, "attn", "emit")]
