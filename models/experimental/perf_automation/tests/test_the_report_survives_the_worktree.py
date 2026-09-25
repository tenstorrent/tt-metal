# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The live report had no durable home, and a reboot took fourteen hours of it.

RUN 10, 2026-08-19. The box rebooted at 12:56 after a fourteen-hour run. The report was being
written live -- 18805 bytes at 11:50, 18935 at 12:13 -- into

    /tmp/tt_hw_planner_voxtral_mini_3b_2507_1787091512/.../runs/2026-08-18T22-22-41/RUN_REPORT.md

and /tmp is cleared on boot. Every measurement survived, because --persist keeps those in
~/.perf_mcp; only the rendered report was lost, and only because it was the one artifact still
living in the disposable copy.

BOTH OF ITS PREVIOUS HOMES WERE WRONG, for opposite reasons:

  the model directory   git-visible, so a git_revert after a no-gain attempt restored the committed
                        blob and rewound a live report from 30 attempts back to 7
  the run directory     gitignored, which fixes that -- but it sits inside the optimize worktree,
                        which is created under /tmp

The mirror is the third place: outside git, so no revert reaches it, and outside the worktree, so no
reboot does either.
"""

import sys
from pathlib import Path

_PA = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PA))


def test_the_report_is_mirrored_to_the_durable_state_dir(tmp_path, monkeypatch):
    import cc_optimize.summary as S

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "durable"))
    root = tmp_path / "model"
    root.mkdir()

    S.upsert_report_section(root, "optimize", "# Optimize\n32 attempts")

    mirror = tmp_path / "durable" / "RUN_REPORT.md"
    assert mirror.is_file(), "the live report has no copy outside the worktree"
    assert "32 attempts" in mirror.read_text()


def test_the_mirror_tracks_every_update(tmp_path, monkeypatch):
    """It is written on EVERY upsert, not once at the end -- an unfinished run is exactly the case
    that loses everything."""
    import cc_optimize.summary as S

    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(tmp_path / "durable"))
    root = tmp_path / "model"
    root.mkdir()
    mirror = tmp_path / "durable" / "RUN_REPORT.md"

    S.upsert_report_section(root, "optimize", "attempt 1")
    assert "attempt 1" in mirror.read_text()
    S.upsert_report_section(root, "optimize", "attempt 2")
    assert "attempt 2" in mirror.read_text()


def test_no_durable_dir_configured_writes_no_mirror(tmp_path, monkeypatch):
    """Without --persist, state_dir() IS the system temp dir -- the place this exists to escape."""
    import tempfile

    import cc_optimize.summary as S

    monkeypatch.delenv("PERF_MCP_STATE_DIR", raising=False)
    root = tmp_path / "model"
    root.mkdir()
    stray = Path(tempfile.gettempdir()) / "RUN_REPORT.md"
    before = stray.exists()

    S.upsert_report_section(root, "optimize", "no mirror please")

    assert stray.exists() == before, "a mirror was written into the system temp dir"


def test_a_broken_mirror_never_costs_the_report(tmp_path, monkeypatch):
    """Best-effort: the copy that cannot be written must not take the one that can."""
    import cc_optimize.summary as S

    # a FILE where the directory needs to be, so mkdir raises
    blocker = tmp_path / "blocked"
    blocker.write_text("i am not a directory")
    monkeypatch.setenv("PERF_MCP_STATE_DIR", str(blocker))
    root = tmp_path / "model"
    root.mkdir()

    written = S.upsert_report_section(root, "optimize", "still here")
    assert written is not None and "still here" in Path(written).read_text()
