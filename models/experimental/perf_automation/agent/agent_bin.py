# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resolve the `claude` CLI to an absolute path for perf_automation spawns.

Self-contained (stdlib only) twin of scripts.tt_hw_planner._cli_helpers.agent.
resolve_claude_bin — the perf_automation tree loads standalone, so it can't
import the scripts-side helper. Makes every `["claude", ...]` spawn here
PATH-independent (fixes-plan Point 9): env override -> PATH -> ~/.local/bin.
Always returns a str (falls back to bare "claude") so a spawn never gets None.
"""

import os
import shutil


def resolve_claude_bin() -> str:
    local = os.path.expanduser("~/.local/bin/claude")
    return (
        os.environ.get("TT_PLANNER_AGENT_BIN")
        or os.environ.get("CLAUDE_BIN")
        or shutil.which("claude")
        or (local if os.path.exists(local) else None)
        or "claude"
    )
