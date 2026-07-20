"""Self-heal the claude-agent-sdk <-> claude CLI version drift.

The `claude` CLI auto-updates itself; the pinned python `claude-agent-sdk` can fall out of
sync, after which the SDK raises "Claude Code returned an error result: success" on a result
the CLI actually completed successfully — so EVERY agent call fails and a sweep is dead on
arrival. This module detects that at preflight via a trivial SDK call in a clean SUBPROCESS
(so no SDK module is imported into the long-lived process), and by default auto-upgrades
claude-agent-sdk and re-tests. Because the heal runs before the first in-process SDK import,
the real run picks up the upgraded version. Disable the auto-pip with AGENT_SDK_AUTOSYNC=0.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .pkgtools import installer_hint as _hint

# A trivial, no-tools agent call. Prints SDK_SMOKE_OK on a clean result, or SMOKE_ERR with the
# REAL error the endpoint reported (status + result text) — so the caller can tell a model/auth
# problem from a genuine SDK/CLI version issue instead of acting on the SDK's masked
# "error result: success". Uses the configured model verbatim (native Anthropic bare ids).
_SMOKE_SNIPPET = (
    "import asyncio, os\n"
    "from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query\n"
    "m = os.environ.get('AGENT_MODEL_LEAD', 'claude-sonnet-4-6')\n"
    "async def go():\n"
    "    o = ClaudeAgentOptions(model=m, system_prompt='reply READY', allowed_tools=[],\n"
    "        permission_mode='bypassPermissions', setting_sources=[], max_turns=2,\n"
    "        max_buffer_size=8*1024*1024)\n"
    "    ok = False\n"
    "    async for msg in query(prompt='Reply with the single word READY.', options=o):\n"
    "        if isinstance(msg, ResultMessage) and getattr(msg, 'is_error', False):\n"
    "            print('SMOKE_ERR status=' + str(getattr(msg, 'api_error_status', None)) +\n"
    "                  ' result=' + str(getattr(msg, 'result', ''))[:200]); return\n"
    "        if isinstance(msg, AssistantMessage): ok = True\n"
    "    print('SDK_SMOKE_OK' if ok else 'SMOKE_ERR status=None result=no assistant message')\n"
    "asyncio.run(asyncio.wait_for(go(), 90))\n"
)

# A bad/inaccessible model NAME for this endpoint (wrong/unknown model id) — fix AGENT_MODEL_*,
# do NOT pip-upgrade.
_MODEL_ERROR_MARKERS = (
    "status=404",
    "may not exist",
    "not have access",
    "model not found",
    "unknown model",
    "invalid model",
    "view available models",
)
# A genuine SDK<->CLI version/control-protocol problem — pip-upgrading the SDK may help.
_MISMATCH_MARKERS = (
    "returned an error result",
    "type': 'error",
    '"type": "error"',
    "control protocol",
    "unsupported protocol",
)


def _env_agent_path() -> Path:
    return Path(__file__).resolve().parent.parent / ".env.agent"


def _creds_env() -> dict:
    """The env the smoke test runs under — native Anthropic auth, the SAME path the real run uses:
    ANTHROPIC_API_KEY if exported, else `claude` login. Any stale LiteLLM proxy vars are stripped so
    the smoke hits Anthropic directly (the real endpoint), and optional AGENT_MODEL_* overrides from
    an optional .env.agent are honored."""
    env = dict(os.environ)
    p = _env_agent_path()
    if p.is_file():
        for raw in p.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("AGENT_MODEL") or k in ("AGENT_EFFORT", "ANTHROPIC_SMALL_FAST_MODEL"):
                    env[k] = v.strip()
    env.pop("ANTHROPIC_BASE_URL", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)
    return env


def package_version() -> str:
    try:
        import importlib.metadata as m

        return m.version("claude-agent-sdk")
    except Exception:  # noqa: BLE001
        return "?"


def cli_version() -> str:
    """Version of the `claude` CLI the SDK drives. The CLI AUTO-UPDATES, so it can run ahead of
    the newest published SDK — in which case upgrading the SDK can't help and the CLI is the knob."""
    try:
        r = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr or "?").strip().split()[0]
    except Exception:  # noqa: BLE001
        return "?"


def is_model_error(detail: str) -> bool:
    d = (detail or "").lower()
    return any(m in d for m in _MODEL_ERROR_MARKERS)


def is_mismatch(detail: str) -> bool:
    d = (detail or "").lower()
    # a model/auth 404 is NOT a version mismatch — never pip-upgrade over it
    if is_model_error(d):
        return False
    return any(m in d for m in _MISMATCH_MARKERS)


def smoke_test(timeout: int = 150) -> tuple[bool, str]:
    """Run the trivial agent call in a clean subprocess. Returns (ok, last-400-chars-of-output)."""
    try:
        r = subprocess.run(
            [sys.executable, "-c", _SMOKE_SNIPPET],
            env=_creds_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, "smoke test timed out"
    out = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
    return ("SDK_SMOKE_OK" in (r.stdout or "")), out[-400:]


def _pip_upgrade() -> None:
    from .pkgtools import run_pip

    run_pip(["install", "-U", "claude-agent-sdk"], check=True)


def ensure_compatible(autosync: bool | None = None, log=print) -> dict:
    """Preflight self-heal. Smoke-test the agent SDK; on the version-drift signature, optionally
    pip-upgrade claude-agent-sdk and re-test. Returns a status dict (ok / healed / version / reason).
    Never raises — the caller decides whether a not-ok result is fatal."""
    if autosync is None:
        autosync = os.environ.get("AGENT_SDK_AUTOSYNC", "1").lower() not in ("0", "false", "no")
    before = package_version()

    ok, detail = smoke_test()
    if ok:
        return {"ok": True, "healed": False, "version": before}
    if is_model_error(detail):
        # bad/inaccessible model name (e.g. a leftover 'anthropic/' prefix the CLI now 404s).
        # This is a CONFIG problem — fix AGENT_MODEL_* / the defaults; pip-upgrading won't help.
        return {
            "ok": False,
            "healed": False,
            "version": before,
            "detail": detail,
            "reason": "model name invalid/inaccessible for this endpoint (check AGENT_MODEL_*; native Anthropic wants bare ids like 'claude-sonnet-4-6')",
        }
    if not is_mismatch(detail):
        # a different failure (no network / bad creds / rate limit) — do NOT pip-install over it
        return {
            "ok": False,
            "healed": False,
            "version": before,
            "detail": detail,
            "reason": "agent call failed, but not the version-drift signature",
        }

    log(f"      claude-agent-sdk {before} is out of sync with the claude CLI (agent calls fail)")
    if not autosync:
        return {
            "ok": False,
            "healed": False,
            "version": before,
            "detail": detail,
            "reason": f"AGENT_SDK_AUTOSYNC disabled — run: {_hint()} -U claude-agent-sdk",
        }

    log(f"      auto-syncing: {_hint()} -U claude-agent-sdk ...")
    try:
        _pip_upgrade()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "healed": False, "version": before, "reason": f"pip upgrade failed: {exc}"}

    after = package_version()
    ok2, detail2 = smoke_test()
    if ok2:
        log(f"      auto-synced claude-agent-sdk {before} -> {after}; agent calls healthy")
        return {"ok": True, "healed": True, "version_before": before, "version": after}
    # Even the newest SDK fails -> the auto-updating claude CLI has run AHEAD of any published SDK.
    # Upgrading the SDK cannot help; the actionable knob is the CLI (pin/downgrade it, or wait for
    # a newer claude-agent-sdk that supports this CLI's control protocol).
    cli = cli_version()
    return {
        "ok": False,
        "healed": True,
        "version_before": before,
        "version": after,
        "cli_version": cli,
        "detail": detail2,
        "reason": (
            f"latest claude-agent-sdk {after} still incompatible with claude CLI {cli} "
            "(CLI auto-updated ahead of the SDK); pin/downgrade the CLI or wait for a newer SDK"
        ),
    }
