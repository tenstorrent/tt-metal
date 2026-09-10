# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One description per coding agent a loop can drive, so the loops name a provider, not a CLI.

WHAT THE LOOP ACTUALLY NEEDS FROM AN AGENT. `run.py` pipes the agent's stdout straight to a log
file and never reads it back: the round's outcome comes from `_gate_status`, which reads the MCP
server's own state. So the contract is not the agent's output schema -- it is only that the process
can talk to the MCP server and edit files. That is what makes a second provider tractable at all,
and it is why nothing here describes how to parse a transcript.

Four things do differ per agent, and they are the whole of this module:

  bins        where the CLI is found, and which env var overrides it
  prompt      the flag the single instruction string rides on
  servers     how the MCP server is declared -- a JSON file for some, a TOML file for others
  allowlist   whether the agent can be restricted to named tools, or only to a sandbox mode

`resolve_claude_bin` in agent_bin stays as it is: this module is additive, and the default
provider reproduces today's argv exactly (asserted by test_the_default_provider_is_unchanged).

The scripts-side tree imports this one; the dependency never runs the other way, so this file
stays stdlib-only.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_PROVIDER = "claude"

# Read before any provider-specific variable, so one override reaches whichever agent is selected.
_ANY_BIN_ENV = "TT_PLANNER_AGENT_BIN"
_PROVIDER_ENV = "TT_PLANNER_AGENT_PROVIDER"


@dataclass(frozen=True)
class Launch:
    """A spawnable agent: argv, plus any environment it needs beyond the caller's own."""

    argv: List[str]
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Provider:
    name: str
    label: str
    bin_env: str
    bin_names: Tuple[str, ...]
    api_key_env: str
    # Whether the agent can be handed a list of tool names. When it cannot, the MCP server is the
    # only tool surface it is given and the allowlist is carried by the server, not the argv.
    supports_tool_allowlist: bool
    build: Callable[..., Launch]
    # False where the shape below is written from documentation rather than from a CLI that ran.
    validated: bool = True


def _mcp_json(path: Path, servers: dict) -> Path:
    path.write_text(json.dumps({"mcpServers": servers}, indent=2))
    return path


def _toml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[%s]" % ", ".join(_toml_scalar(v) for v in value)
    return json.dumps(str(value))


def _mcp_toml(path: Path, servers: dict) -> Path:
    """The same server declaration, in the config file a TOML-configured agent reads.

    Written by hand rather than with a TOML library because this tree is stdlib-only, and the
    shape is small: one table per server, with scalar/list values and a nested env table.
    """
    lines: List[str] = []
    for name, spec in sorted(servers.items()):
        lines.append("[mcp_servers.%s]" % name.replace("-", "_"))
        for key in ("command", "args"):
            if spec.get(key) is not None:
                lines.append("%s = %s" % (key, _toml_scalar(spec[key])))
        env = spec.get("env") or {}
        if env:
            lines.append("")
            lines.append("[mcp_servers.%s.env]" % name.replace("-", "_"))
            for k in sorted(env):
                lines.append("%s = %s" % (k, _toml_scalar(env[k])))
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def _claude_launch(*, bin_path, prompt, mcp_config, tools, verbose=True, **_) -> Launch:
    argv = [bin_path, "-p", prompt]
    if mcp_config:
        argv += ["--mcp-config", str(mcp_config), "--strict-mcp-config"]
    if tools:
        argv += ["--allowedTools", *list(tools)]
    argv += ["--output-format", "stream-json"]
    if verbose:
        argv.append("--verbose")
    return Launch(argv=argv)


def _cursor_launch(*, bin_path, prompt, mcp_config, tools, verbose=True, **_) -> Launch:
    argv = [bin_path, "-p", prompt, "--output-format", "stream-json"]
    if mcp_config:
        argv += ["--mcp-config", str(mcp_config)]
    return Launch(argv=argv)


def _codex_launch(*, bin_path, prompt, mcp_config, tools, verbose=True, model=None, **_) -> Launch:
    """Non-interactive Codex, with the MCP server declared through its own config directory.

    UNVALIDATED against a CLI that ran -- see `validated=False` below. Codex reads config.toml
    from CODEX_HOME, so the server is written there and the directory is handed over in the
    environment rather than on the command line. It has no per-tool allowlist: the tools it can
    reach are the ones the declared server exposes, which is the same set the allowlist names.
    """
    argv = [bin_path, "exec", "--skip-git-repo-check"]
    if model:
        argv += ["--model", str(model)]
    if verbose:
        argv.append("--json")
    argv.append(prompt)
    env: Dict[str, str] = {}
    if mcp_config:
        env["CODEX_HOME"] = str(Path(mcp_config).parent)
    return Launch(argv=argv, env=env)


PROVIDERS: Dict[str, Provider] = {
    "claude": Provider(
        name="claude",
        label="Anthropic (Claude Code)",
        bin_env="CLAUDE_BIN",
        bin_names=("claude",),
        api_key_env="ANTHROPIC_API_KEY",
        supports_tool_allowlist=True,
        build=_claude_launch,
    ),
    "cursor": Provider(
        name="cursor",
        label="Cursor",
        bin_env="CURSOR_BIN",
        bin_names=("cursor-agent", "agent"),
        api_key_env="CURSOR_API_KEY",
        supports_tool_allowlist=False,
        build=_cursor_launch,
    ),
    "codex": Provider(
        name="codex",
        label="OpenAI (Codex CLI)",
        bin_env="CODEX_BIN",
        bin_names=("codex",),
        api_key_env="OPENAI_API_KEY",
        supports_tool_allowlist=False,
        build=_codex_launch,
        validated=False,
    ),
}

# The config filename each provider's declaration has to be written under, keyed by how it is read.
_MCP_WRITERS = {"claude": _mcp_json, "cursor": _mcp_json, "codex": _mcp_toml}
_MCP_FILENAMES = {"claude": None, "cursor": None, "codex": "config.toml"}


def provider_names() -> List[str]:
    return sorted(PROVIDERS)


def get(name: Optional[str] = None) -> Provider:
    """The named provider, or the one this environment selects, or the default.

    An unknown name is an error rather than a silent fall back to the default: a typo that
    quietly ran a different agent than the one asked for is worse than a refusal.
    """
    key = (name or os.environ.get(_PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
    if key not in PROVIDERS:
        raise ValueError("unknown agent provider %r (have: %s)" % (key, ", ".join(provider_names())))
    return PROVIDERS[key]


def resolve_bin(provider: Optional[str] = None) -> Optional[str]:
    """Absolute path to the agent's CLI: shared override -> its own env -> PATH -> ~/.local/bin.

    None when nothing resolves, so a caller can fail once with a clear message instead of a late
    FileNotFoundError from a bare spawn. No install path is written here -- ~/.local/bin is the
    documented install target of every CLI in the table and is expanded, not typed.
    """
    p = get(provider)
    shared = os.environ.get(_ANY_BIN_ENV)
    if shared:
        return shared
    own = os.environ.get(p.bin_env)
    if own:
        return own
    for name in p.bin_names:
        found = shutil.which(name)
        if found:
            return found
    for name in p.bin_names:
        local = os.path.expanduser(os.path.join("~", ".local", "bin", name))
        if os.path.exists(local):
            return local
    return None


def mcp_config_path(provider: Optional[str], directory, stem: str) -> Path:
    """Where this provider's server declaration belongs.

    A provider that reads a config DIRECTORY gets the filename it looks for; one that is handed a
    file on the command line keeps the caller's own stem, so existing paths do not move.
    """
    fixed = _MCP_FILENAMES[get(provider).name]
    return Path(directory) / (fixed or stem)


def write_mcp_config(provider: Optional[str], path, servers: dict) -> Path:
    """Declare the MCP servers in whichever format this provider reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _MCP_WRITERS[get(provider).name](path, servers)


def launch(
    provider: Optional[str],
    *,
    prompt: str,
    mcp_config=None,
    tools: Sequence[str] = (),
    verbose: bool = True,
    model: Optional[str] = None,
    bin_path: Optional[str] = None,
) -> Launch:
    """The argv and extra environment for one agent run.

    `tools` is dropped for a provider that cannot express an allowlist; the server declaration is
    then the only tool surface, which is the same set the list would have named.
    """
    p = get(provider)
    resolved = bin_path or resolve_bin(p.name)
    if not resolved:
        raise RuntimeError("%s CLI not found -- install it, or set %s / %s" % (p.label, p.bin_env, _ANY_BIN_ENV))
    return p.build(
        bin_path=resolved,
        prompt=prompt,
        mcp_config=mcp_config,
        tools=tuple(tools) if p.supports_tool_allowlist else (),
        verbose=verbose,
        model=model,
    )
