# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / ".claude"
    / "scripts"
    / "sync_cursor_skills.py"
)


def _load_sync():
    spec = importlib.util.spec_from_file_location("sync_cursor_skills", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cursor_skills_match_claude_skills():
    problems = _load_sync().drift()
    assert not problems, (
        "\n".join(problems)
        + "\nRun tt_metal/tt-llk/.claude/scripts/sync_cursor_skills.py to update the mirror."
    )
