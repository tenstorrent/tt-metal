"""Pin: on a successful isolation-worktree run, the FULL graduated demo dir is
persisted back to the main tree — not just RUN_REPORT.md.

Regression guard for the 31/32 emit-e2e undercount: a module that graduated via
recompose in the worktree has its `.last_good_*` snapshot only in the overlay;
without persisting the demo dir back, the in-place emit-e2e/promote read a stale
main tree and miss it.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.cli import _persist_graduated_demos  # noqa: E402


def _make_worktree_demo(root: Path) -> Path:
    demo = root / "models" / "demos" / "xtts_v2"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    (demo / "bringup_status.json").write_text('{"components": []}')
    (demo / "RUN_REPORT.md").write_text("# 32/32\n")
    (demo / "_stubs" / "g_p_t.py").write_text("# native ttnn body\n")
    (demo / "_stubs" / "g_p_t.py.last_good_sharded").write_text("# graduation snapshot\n")
    return demo


def test_persists_full_demo_including_snapshots(tmp_path: Path) -> None:
    wt = tmp_path / "worktree"
    main = tmp_path / "maintree"
    _make_worktree_demo(wt)
    (main).mkdir()

    synced = _persist_graduated_demos(wt, main)
    assert synced == ["models/demos/xtts_v2"]

    md = main / "models" / "demos" / "xtts_v2"
    assert (md / "_stubs" / "g_p_t.py.last_good_sharded").is_file()
    assert (md / "_stubs" / "g_p_t.py").is_file()
    assert (md / "bringup_status.json").is_file()
    assert (md / "RUN_REPORT.md").read_text() == "# 32/32\n"


def test_merges_over_existing_stale_main(tmp_path: Path) -> None:
    """The snapshot must land even when the main tree already has a stale copy
    of the demo WITHOUT the recompose snapshot."""
    wt = tmp_path / "worktree"
    main = tmp_path / "maintree"
    _make_worktree_demo(wt)
    stale = main / "models" / "demos" / "xtts_v2" / "_stubs"
    stale.mkdir(parents=True)
    (stale / "g_p_t.py").write_text("# OLD body\n")
    (main / "models" / "demos" / "xtts_v2" / "RUN_REPORT.md").write_text("# 31/32 STALE\n")

    _persist_graduated_demos(wt, main)
    md = main / "models" / "demos" / "xtts_v2"
    assert (md / "_stubs" / "g_p_t.py.last_good_sharded").is_file()  # snapshot now present
    assert (md / "RUN_REPORT.md").read_text() == "# 32/32\n"        # report refreshed
    assert (md / "_stubs" / "g_p_t.py").read_text() == "# native ttnn body\n"  # stub refreshed


def test_no_demo_is_safe(tmp_path: Path) -> None:
    wt = tmp_path / "worktree"
    (wt / "models").mkdir(parents=True)
    assert _persist_graduated_demos(wt, tmp_path / "maintree") == []
