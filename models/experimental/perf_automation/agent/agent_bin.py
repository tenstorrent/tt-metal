# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the `claude` CLI to an absolute path for perf_automation spawns.

Self-contained (stdlib only) twin of scripts.tt_hw_planner._cli_helpers.agent.
resolve_claude_bin — the perf_automation tree loads standalone, so it can't
import the scripts-side helper. Makes every `["claude", ...]` spawn here
PATH-independent (fixes-plan Point 9): env override -> PATH -> ~/.local/bin.
Always returns a str (falls back to bare "claude") so a spawn never gets None.
"""


def resolve_claude_bin() -> str:
    """The default provider's CLI. Kept as a name because every spawn here already calls it.

    The search itself moved to agent_provider, which does it for whichever agent is selected --
    same order (shared override, own variable, PATH, ~/.local/bin) and the same bare-name fallback,
    so a spawn still never gets None.
    """
    from .agent_provider import DEFAULT_PROVIDER, resolve_bin

    return resolve_bin(DEFAULT_PROVIDER) or DEFAULT_PROVIDER
