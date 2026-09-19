"""Agent auth — native Anthropic (no LiteLLM proxy, no required creds file).

Auth precedence:
  - Use ANTHROPIC_API_KEY from the shell env if exported; otherwise the claude-agent-SDK falls back
    to the `claude` login credentials. No .env.agent / LiteLLM proxy is required.
  - Optional model/effort overrides (AGENT_MODEL_<ROLE>, AGENT_MODEL_EDIT_{1,2,3}, AGENT_EFFORT,
    ANTHROPIC_SMALL_FAST_MODEL) may be set in the shell env, or in an optional perf_automation/.env.agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

MODEL_ENV_KEYS = {
    "lead": "AGENT_MODEL_LEAD",
    "sub": "AGENT_MODEL_SUB",
    "edit": "AGENT_MODEL_EDIT",
    "structural": "AGENT_MODEL_STRUCTURAL",
}
MODEL_DEFAULTS = {
    "sub": "claude-sonnet-4-6",
    "lead": "claude-sonnet-4-6",
    "edit": "claude-haiku-4-5-20251001",
    "structural": "claude-sonnet-4-6",
}

EDIT_LADDER_ENV_KEYS = ("AGENT_MODEL_EDIT_1", "AGENT_MODEL_EDIT_2", "AGENT_MODEL_EDIT_3")
EDIT_LADDER_DEFAULTS = (
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "claude-opus-4-8",
)

DEFAULT_AGENT_EFFORT = "low"

STATIC_SDK_ENV = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
}

INHERITED_ENV_TO_CLEAR = (
    "CLAUDE_EFFORT",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SSE_PORT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
)

_OVERRIDE_KEYS = (*MODEL_ENV_KEYS.values(), *EDIT_LADDER_ENV_KEYS, "AGENT_EFFORT", "ANTHROPIC_SMALL_FAST_MODEL")


class ConfigError(Exception):
    """Retained for API compatibility; native auth no longer requires a creds file."""


def _read_overrides(env_path: str | os.PathLike[str] | None) -> dict[str, str]:
    """Optional model/effort overrides ONLY (no creds, no LiteLLM): shell env wins; an optional
    .env.agent file is also read if present."""
    out: dict[str, str] = {}
    if env_path:
        path = Path(env_path)
        if path.is_file():
            for raw in path.read_text(errors="ignore").splitlines():
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if k in _OVERRIDE_KEYS and v:
                        out[k] = v
    for k in _OVERRIDE_KEYS:
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


def load_agent_env(env_path: str | os.PathLike[str] | None = None) -> dict[str, str]:
    """Resolve agent config for native Anthropic auth. Never requires a creds file: ANTHROPIC_API_KEY
    (if exported) is used; otherwise the SDK falls back to `claude` login. Returns model/effort
    overrides plus ANTHROPIC_API_KEY when present."""
    resolved = _read_overrides(env_path)
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        resolved["ANTHROPIC_API_KEY"] = key
    return resolved


def get_model(role: str, config: dict[str, str] | None = None) -> str:
    """Resolve the model id for a role. Precedence: AGENT_MODEL_<ROLE> override else the bare default."""
    if role not in MODEL_ENV_KEYS:
        raise ValueError(f"unknown model role: {role!r}")
    config = config or {}
    override = config.get(MODEL_ENV_KEYS[role])
    if override:
        return override
    if role == "edit":
        return config.get(MODEL_ENV_KEYS["edit"]) or config.get(MODEL_ENV_KEYS["sub"]) or MODEL_DEFAULTS["edit"]
    return MODEL_DEFAULTS[role]


def get_edit_model(attempt: int, config: dict[str, str] | None = None) -> str:
    """Model for the Nth edit attempt: rung 0 -> haiku, 1 -> sonnet, 2+ -> opus. Override per rung
    via AGENT_MODEL_EDIT_{1,2,3}; capped at the top rung once exhausted."""
    config = config or {}
    rung = max(0, min(int(attempt), len(EDIT_LADDER_DEFAULTS) - 1))
    return config.get(EDIT_LADDER_ENV_KEYS[rung]) or EDIT_LADDER_DEFAULTS[rung]


def agent_effort(config: dict[str, str] | None = None) -> str:
    """Reasoning effort for agent SDK calls. AGENT_EFFORT override else the default."""
    return (config or {}).get("AGENT_EFFORT") or DEFAULT_AGENT_EFFORT


def apply_agent_env(
    env_path: str | os.PathLike[str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set up native Anthropic auth in `environ` and return the resolved overrides. Uses
    ANTHROPIC_API_KEY if present, else `claude` login; strips any LiteLLM proxy vars + nested-session
    markers; sets the small-fast model + telemetry opt-outs. No creds file required."""
    if environ is None:
        environ = os.environ
    resolved = load_agent_env(env_path)
    environ.setdefault("PERF_NATIVE_ANTHROPIC_API_KEY", environ.get("ANTHROPIC_API_KEY", ""))
    environ["ANTHROPIC_SMALL_FAST_MODEL"] = resolved.get("ANTHROPIC_SMALL_FAST_MODEL") or get_model("sub", resolved)
    for key, value in STATIC_SDK_ENV.items():
        environ[key] = value
    for key in INHERITED_ENV_TO_CLEAR:
        environ.pop(key, None)
    return resolved
