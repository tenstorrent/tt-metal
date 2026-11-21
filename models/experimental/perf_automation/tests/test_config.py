# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Agent auth config (native Anthropic, no LiteLLM proxy, no required creds file).

ANTHROPIC_API_KEY is used if exported, else the claude-agent-SDK falls back to `claude` login.
An optional .env.agent supplies model/effort OVERRIDES only — never credentials."""
import pytest

from agent.config import (
    EDIT_LADDER_DEFAULTS,
    INHERITED_ENV_TO_CLEAR,
    MODEL_DEFAULTS,
    STATIC_SDK_ENV,
    agent_effort,
    apply_agent_env,
    get_edit_model,
    get_model,
    load_agent_env,
)


def _write(path, text):
    path.write_text(text)
    return path


def test_no_creds_file_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resolved = load_agent_env(tmp_path / ".env.agent")
    assert resolved == {}


def test_exported_api_key_is_picked_up(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-123")
    resolved = load_agent_env(tmp_path / ".env.agent")
    assert resolved["ANTHROPIC_API_KEY"] == "sk-secret-123"


def test_env_overrides_win_over_file(tmp_path, monkeypatch):
    p = _write(tmp_path / ".env.agent", "AGENT_MODEL_LEAD=from-file\n")
    monkeypatch.setenv("AGENT_MODEL_LEAD", "from-env")
    resolved = load_agent_env(p)
    assert resolved["AGENT_MODEL_LEAD"] == "from-env"


def test_file_supplies_overrides_when_env_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MODEL_SUB", raising=False)
    p = _write(tmp_path / ".env.agent", "AGENT_MODEL_SUB=custom-sub\n# a comment\nNOT_AN_OVERRIDE=x\n")
    resolved = load_agent_env(p)
    assert resolved["AGENT_MODEL_SUB"] == "custom-sub"
    assert "NOT_AN_OVERRIDE" not in resolved


def test_get_model_defaults_and_overrides():
    assert get_model("sub", {}) == MODEL_DEFAULTS["sub"]
    assert get_model("lead", {}) == MODEL_DEFAULTS["lead"]
    cfg = {"AGENT_MODEL_SUB": "sub-x", "AGENT_MODEL_LEAD": "lead-y"}
    assert get_model("sub", cfg) == "sub-x"
    assert get_model("lead", cfg) == "lead-y"


def test_get_model_edit_falls_back_to_sub_then_default():
    assert get_model("edit", {}) == MODEL_DEFAULTS["edit"]
    assert get_model("edit", {"AGENT_MODEL_SUB": "sub-x"}) == "sub-x"
    assert get_model("edit", {"AGENT_MODEL_EDIT": "edit-x"}) == "edit-x"


def test_get_model_rejects_unknown_role():
    with pytest.raises(ValueError):  # allow-pytest.raises: no expect_error fixture
        get_model("bogus", {})


def test_edit_ladder_climbs_and_caps():
    assert get_edit_model(0, {}) == EDIT_LADDER_DEFAULTS[0]
    assert get_edit_model(1, {}) == EDIT_LADDER_DEFAULTS[1]
    assert get_edit_model(2, {}) == EDIT_LADDER_DEFAULTS[2]
    assert get_edit_model(99, {}) == EDIT_LADDER_DEFAULTS[-1]
    assert get_edit_model(0, {"AGENT_MODEL_EDIT_1": "rung0-override"}) == "rung0-override"


def test_agent_effort_default_and_override():
    assert agent_effort({}) == "low"
    assert agent_effort({"AGENT_EFFORT": "high"}) == "high"


def test_apply_sets_small_fast_model_and_static_optouts(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MODEL_SUB", raising=False)
    env: dict[str, str] = {}
    apply_agent_env(tmp_path / ".env.agent", env)
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == MODEL_DEFAULTS["sub"]
    for k, v in STATIC_SDK_ENV.items():
        assert env[k] == v


def test_apply_strips_inherited_proxy_and_session_vars(tmp_path):
    env = {k: "stale" for k in INHERITED_ENV_TO_CLEAR}
    apply_agent_env(tmp_path / ".env.agent", env)
    for k in INHERITED_ENV_TO_CLEAR:
        assert k not in env


def test_apply_small_fast_model_honors_override(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENT_MODEL_SUB", raising=False)
    p = _write(tmp_path / ".env.agent", "AGENT_MODEL_SUB=custom-sub\n")
    env: dict[str, str] = {}
    apply_agent_env(p, env)
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "custom-sub"
