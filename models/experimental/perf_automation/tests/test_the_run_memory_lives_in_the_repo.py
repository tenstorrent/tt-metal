# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""--persist keeps the run's memory in the REPO, gitignored, and never in the worktree.

Three properties, and each of them was a bug that already happened:

  DURABLE      the default is /tmp, and /tmp is cleared on boot. Run 10 lost fourteen hours of
               report to a reboot at 12:56 on 2026-08-19.
  UNTRACKED    the model directory's RUN_REPORT.md is a tracked file, so writing run state beside
               it leaves the working tree dirty -- and optimize REFUSES to isolate a dirty tree,
               so one run's leftovers block the next one's.
  NOT THE WORKTREE   run_root becomes /tmp/tt_hw_planner_... once isolation is set up. Putting the
               memory there is precisely the bug --persist exists to fix.

`.state/` beside `runs/` satisfies all three: in the repo where everything else a run produces
lives, ignored by the same .gitignore that already covers `runs/`.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))
_OPT = _PA.parent.parent.parent / "scripts" / "tt_hw_planner" / "commands" / "optimize.py"


def _persist_block() -> str:
    src = _OPT.read_text()
    i = src.index('if getattr(args, "persist", False):')
    return src[i : i + 2200]


def test_the_state_dir_is_in_the_repo_not_the_home_directory():
    block = _persist_block()
    assert 'repo_root / "models" / "experimental" / "perf_automation" / ".state"' in block
    assert 'Path.home() / ".perf_mcp" / _slug\n            _persist_dir' not in block


def test_it_uses_the_real_repo_and_never_the_worktree():
    """run_root is /tmp/tt_hw_planner_... after isolation; writing memory there is the original bug."""
    block = _persist_block()
    assigns = [ln for ln in block.splitlines() if "_persist_dir =" in ln]
    assert assigns, block[:200]
    for ln in assigns:
        assert "run_root" not in ln, ln


def test_the_state_dir_is_gitignored():
    ig = (_PA / ".gitignore").read_text().splitlines()
    entries = [ln.strip() for ln in ig if ln.strip() and not ln.strip().startswith("#")]
    assert ".state/" in entries, "run memory would dirty the tree and block the next run's isolation"
    assert "runs/" in entries


def test_existing_memory_is_carried_over_not_abandoned():
    """A model's ledger and write-once ceiling anchor are expensive; moving the directory must not
    silently restart it from nothing."""
    block = _persist_block()
    assert ".perf_mcp" in block, "the old location is not consulted at all"
    assert "copytree" in block or "copy2" in block, "nothing carries the old state forward"
    assert "shutil.move" not in block and "_shutil.move" not in block, "the old copy must survive"


def test_the_ledger_follows_the_state_dir():
    """measurements.py resolves the ledger relative to the state dir; setting one without the other
    splits them and the report then finds no anchors."""
    block = _persist_block() + _OPT.read_text().split('if getattr(args, "persist", False):')[1][:1200]
    assert "PERF_MCP_STATE_DIR" in block and "PERF_MCP_LEDGER_DIR" in block
