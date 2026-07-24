"""M0 tests for the .env.agent credential loader (PLAN section 3.1).

.env.agent is the ONLY credential source: no shell-env fallback, fail fast
with an actionable message, and map LiteLLM creds to ANTHROPIC_* for the SDK.
"""

import pytest

from agent.config import (
    ConfigError,
    MISSING_ENV_MESSAGE,
    MODEL_DEFAULTS,
    SDK_ENV_KEYS,
    STATIC_SDK_ENV,
    apply_agent_env,
    get_model,
    load_agent_env,
)


def _write(path, text):
    path.write_text(text)
    return path


def test_env_agent_is_sole_source(tmp_path, monkeypatch):
    # Key present in the shell env, but NO .env.agent file -> still fails.
    monkeypatch.setenv("LITELLM_API_KEY", "sk-from-shell")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://from-shell")
    missing = tmp_path / ".env.agent"
    with pytest.raises(ConfigError) as exc:
        load_agent_env(missing)
    assert str(exc.value) == MISSING_ENV_MESSAGE


def test_env_agent_missing_or_incomplete_prompts(tmp_path):
    # Absent file.
    with pytest.raises(ConfigError) as exc:
        load_agent_env(tmp_path / ".env.agent")
    assert str(exc.value) == MISSING_ENV_MESSAGE

    # Missing key.
    p = _write(tmp_path / "only_base", "LITELLM_BASE_URL=https://x\n")
    with pytest.raises(ConfigError):
        load_agent_env(p)

    # Empty value.
    p2 = _write(tmp_path / "empty_val", "LITELLM_BASE_URL=https://x\nLITELLM_API_KEY=\n")
    with pytest.raises(ConfigError):
        load_agent_env(p2)


def test_env_agent_loads_and_maps(tmp_path):
    p = _write(
        tmp_path / ".env.agent",
        "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=sk-secret-123\n",
    )
    resolved = load_agent_env(p)
    assert resolved["ANTHROPIC_BASE_URL"] == "https://proxy.example"
    assert resolved["ANTHROPIC_AUTH_TOKEN"] == "sk-secret-123"
    assert resolved["ANTHROPIC_API_KEY"] == "sk-secret-123"


def test_apply_injects_only_sdk_keys(tmp_path):
    p = _write(
        tmp_path / ".env.agent",
        "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=sk-secret-123\n",
    )
    env: dict[str, str] = {}
    apply_agent_env(p, env)
    for k in SDK_ENV_KEYS:
        assert env[k]
    assert env["ANTHROPIC_API_KEY"] == "sk-secret-123"


def test_get_model_defaults_and_overrides():
    assert get_model("sub", {}) == MODEL_DEFAULTS["sub"]
    assert get_model("lead", {}) == MODEL_DEFAULTS["lead"]
    cfg = {"AGENT_MODEL_SUB": "anthropic/sub-x", "AGENT_MODEL_LEAD": "anthropic/lead-y"}
    assert get_model("sub", cfg) == "anthropic/sub-x"
    assert get_model("lead", cfg) == "anthropic/lead-y"


def test_get_model_rejects_unknown_role():
    with pytest.raises(ValueError):
        get_model("bogus", {})


def test_apply_sets_small_fast_model_and_optouts(tmp_path):
    p = _write(
        tmp_path / ".env.agent",
        "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=sk-secret-123\n",
    )
    env = {}
    apply_agent_env(p, env)
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == MODEL_DEFAULTS["sub"]
    for k, v in STATIC_SDK_ENV.items():
        assert env[k] == v


def test_apply_small_fast_model_honors_override(tmp_path):
    p = _write(
        tmp_path / ".env.agent",
        "LITELLM_BASE_URL=https://proxy.example\nLITELLM_API_KEY=sk-secret-123\n"
        "AGENT_MODEL_SUB=anthropic/custom-sub\n",
    )
    env = {}
    apply_agent_env(p, env)
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == "anthropic/custom-sub"
