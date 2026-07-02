"""Credential loading — `.env.agent` is the ONLY credential source (PLAN section 3.1).

Rules enforced here (not by convention):
  1. Load credentials ONLY from `.env.agent`; never fall back to the shell env.
  2. Fail fast with an actionable message when the file is missing/incomplete.
  3. Map LiteLLM creds to the ANTHROPIC_* vars the SDK process consumes
     (the POC wiring: base url, auth token, api key, small-fast model, and
     telemetry/autoupdater opt-outs).
  4. The key never leaves the process env (never logged/printed/persisted).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping

from dotenv import dotenv_values

REQUIRED_KEYS = ("LITELLM_BASE_URL", "LITELLM_API_KEY")

# Exact, actionable prompt (PLAN section 3.1). Carries no secret values.
MISSING_ENV_MESSAGE = (
    "Missing .env.agent — create perf_automation/.env.agent with "
    "LITELLM_BASE_URL=... and LITELLM_API_KEY=... then re-run."
)

# Model roles (PLAN section 3.1). Sub-agents use sonnet; lead model is TBD(model-lead)
# (likely Opus 4.8) — default to sonnet until resolved. Both overridable via .env.agent.
MODEL_ENV_KEYS = {"lead": "AGENT_MODEL_LEAD", "sub": "AGENT_MODEL_SUB"}
MODEL_DEFAULTS = {
    "sub": "anthropic/claude-sonnet-4-6",
    "lead": "anthropic/claude-sonnet-4-6",  # TBD(model-lead)
}


class ConfigError(Exception):
    """Raised when `.env.agent` is absent or incomplete."""


def load_agent_env(env_path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse `.env.agent` (and ONLY that file) and return the resolved config.

    Returns a dict of the file's own keys plus the mapped ANTHROPIC_* vars.
    Never reads the ambient shell environment for the required keys.
    Raises ConfigError (with MISSING_ENV_MESSAGE) when missing/incomplete.
    """
    path = Path(env_path)
    if not path.is_file():
        raise ConfigError(MISSING_ENV_MESSAGE)

    # dotenv_values reads ONLY this file — no shell-env fallback.
    values = {k: v for k, v in dotenv_values(path).items() if v is not None}

    for key in REQUIRED_KEYS:
        if not values.get(key):
            raise ConfigError(MISSING_ENV_MESSAGE)

    api_key = values["LITELLM_API_KEY"]
    resolved: dict[str, str] = dict(values)
    resolved.update(
        {
            "ANTHROPIC_BASE_URL": values["LITELLM_BASE_URL"],
            "ANTHROPIC_AUTH_TOKEN": api_key,
            "ANTHROPIC_API_KEY": api_key,
        }
    )
    return resolved


def get_model(role: str, config: dict[str, str] | None = None) -> str:
    """Resolve the model id for a role ('lead' | 'sub').

    Precedence: AGENT_MODEL_<ROLE> in the resolved `.env.agent` config, else the
    documented default (PLAN section 3.1). Centralized so M3+ call sites never
    invent their own fallback.
    """
    if role not in MODEL_ENV_KEYS:
        raise ValueError(f"unknown model role: {role!r} (expected 'lead' or 'sub')")
    config = config or {}
    override = config.get(MODEL_ENV_KEYS[role])
    return override or MODEL_DEFAULTS[role]


# Vars injected into the SDK process env. The ANTHROPIC_* creds plus the POC
# wiring: small-fast model (haiku-class internal calls must hit a model the
# proxy serves) and telemetry/autoupdater opt-outs.
SDK_ENV_KEYS = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_SMALL_FAST_MODEL",
)
STATIC_SDK_ENV = {
    "DISABLE_TELEMETRY": "1",
    "DISABLE_AUTOUPDATER": "1",
}


def apply_agent_env(
    env_path: str | os.PathLike[str],
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Load `.env.agent` and inject the SDK env vars into `environ`.

    Injects the ANTHROPIC_* creds, ANTHROPIC_SMALL_FAST_MODEL (= the sub-agent
    model), and the static telemetry/autoupdater opt-outs. Defaults to
    os.environ (the live SDK process env). Returns the resolved config.
    """
    if environ is None:
        environ = os.environ
    resolved = load_agent_env(env_path)
    # Small-fast model = sub-agent model so SDK internal calls hit a served model.
    resolved.setdefault("ANTHROPIC_SMALL_FAST_MODEL", get_model("sub", resolved))
    for key in SDK_ENV_KEYS:
        environ[key] = resolved[key]
    for key, value in STATIC_SDK_ENV.items():
        environ[key] = value
    return resolved
