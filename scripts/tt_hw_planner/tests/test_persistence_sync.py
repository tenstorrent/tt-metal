"""Tests for the brain's worktree → main-tree sync primitive.

This primitive solves the persistence-layer gap observed in SAM2's
2026-05-30 run: the brain's iterated stubs lived in the worktree,
the overlay-apply mechanism restored scaffold-stage stubs to the
main tree, so the main tree pytest failed even though the worktree
pytest passed.

The brain now explicitly syncs graduated stubs from worktree to main
tree at end-of-run, atomically, bypassing the overlay system."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.tt_hw_planner.agentic.persistence import (
    SyncResult,
    sync_demo_to_main_tree,
    sync_graduated_to_main_tree,
)


def _safe_id_stub(name: str) -> str:
    """Local minimal copy of bringup_loop._safe_id to keep the test
    self-contained."""
    import re

    return re.sub(r"[^A-Za-z0-9_]+", "_", (name or "").strip()).strip("_")


def _make_worktree_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Build a fake worktree + main-tree pair. Returns (worktree, main_tree)."""
    main_tree = tmp_path / "main"
    worktree = tmp_path / "worktree"
    for root in (main_tree, worktree):
        (root / "models" / "demos" / "X" / "_stubs").mkdir(parents=True)
    # Session metadata in worktree points back to main
    (worktree / ".tt_hw_planner_session.json").write_text(json.dumps({"model_id": "X", "source_repo": str(main_tree)}))
    return worktree, main_tree


def test_sync_copies_graduated_stub_to_main_tree(tmp_path: Path) -> None:
    """The happy path: worktree has the brain's iterated stub; main
    tree has nothing (or stale); sync writes the worktree version to
    main."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    src = worktree / "models" / "demos" / "X" / "_stubs" / "foo.py"
    src.write_text("# brain iterated implementation\nclass Foo: pass\n")

    result = sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    assert result.synced == ["foo"]
    assert result.skipped == []
    dst = main_tree / "models" / "demos" / "X" / "_stubs" / "foo.py"
    assert dst.is_file()
    assert dst.read_text() == "# brain iterated implementation\nclass Foo: pass\n"


def test_sync_also_copies_graduation_snapshot(tmp_path: Path) -> None:
    """The .last_good_native snapshot is what marks a component as
    'already graduated' on subsequent runs. Without copying it, the
    next run would re-iterate the component."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    src_stubs = worktree / "models" / "demos" / "X" / "_stubs"
    (src_stubs / "foo.py").write_text("class Foo: pass\n")
    (src_stubs / "foo.py.last_good_native").write_text("class Foo: pass\n")

    sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    dst_stubs = main_tree / "models" / "demos" / "X" / "_stubs"
    assert (dst_stubs / "foo.py").is_file()
    assert (dst_stubs / "foo.py.last_good_native").is_file()


def test_sync_overwrites_stale_main_tree_stub(tmp_path: Path) -> None:
    """The exact SAM2 case: main tree has a scaffold-stage stub;
    worktree has the brain's iterated version. Sync must overwrite."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    src = worktree / "models" / "demos" / "X" / "_stubs" / "foo.py"
    src.write_text("# brain's iterated version\n")
    dst = main_tree / "models" / "demos" / "X" / "_stubs" / "foo.py"
    dst.write_text("# scaffold-stage (older)\n")

    sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    assert dst.read_text() == "# brain's iterated version\n"


def test_sync_only_touches_graduated_components(tmp_path: Path) -> None:
    """Components NOT in graduated_this_run must not be synced — we
    don't trust un-graduated stubs."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    src_stubs = worktree / "models" / "demos" / "X" / "_stubs"
    (src_stubs / "foo.py").write_text("# graduated\n")
    (src_stubs / "bar.py").write_text("# NOT graduated — broken\n")

    result = sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],  # only foo
        safe_id_fn=_safe_id_stub,
    )

    dst_stubs = main_tree / "models" / "demos" / "X" / "_stubs"
    assert (dst_stubs / "foo.py").is_file()
    assert not (dst_stubs / "bar.py").exists(), "un-graduated component must not be synced — its stub is untrusted"
    assert result.synced == ["foo"]


def test_sync_noop_when_not_in_worktree(tmp_path: Path) -> None:
    """If there's no session.json, we're not in an isolated worktree
    — sync is a clean no-op."""
    fake_main = tmp_path / "main"
    fake_main.mkdir()

    result = sync_graduated_to_main_tree(
        worktree_root=fake_main,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    assert result.synced == []
    assert result.main_tree_path is None
    assert any("session" in n.lower() or "not in a worktree" in n.lower() for n in result.notes)


def test_sync_noop_when_worktree_is_main_tree(tmp_path: Path) -> None:
    """If session.json's source_repo points at the worktree itself
    (e.g. --isolation none), no copy is needed."""
    main_tree = tmp_path / "main"
    (main_tree / "models" / "demos" / "X" / "_stubs").mkdir(parents=True)
    (main_tree / ".tt_hw_planner_session.json").write_text(json.dumps({"model_id": "X", "source_repo": str(main_tree)}))

    result = sync_graduated_to_main_tree(
        worktree_root=main_tree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    assert result.synced == []
    assert any("no sync needed" in n.lower() or "==" in n for n in result.notes)


def test_sync_handles_missing_source_stub_gracefully(tmp_path: Path) -> None:
    """If a graduated component's stub file is unexpectedly missing
    in the worktree, sync logs and continues — doesn't crash."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    # No stub file created for "foo"

    result = sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=["foo"],
        safe_id_fn=_safe_id_stub,
    )

    assert result.synced == []
    assert result.skipped == ["foo"]
    assert any("worktree stub missing" in n.lower() for n in result.notes)


def test_sync_handles_no_graduations_gracefully(tmp_path: Path) -> None:
    """Empty graduated_this_run → no-op, clean notes."""
    worktree, main_tree = _make_worktree_setup(tmp_path)

    result = sync_graduated_to_main_tree(
        worktree_root=worktree,
        demo_subpath="models/demos/X",
        graduated_components=[],
        safe_id_fn=_safe_id_stub,
    )

    assert result.synced == []
    assert any("no graduated" in n.lower() for n in result.notes)


def test_sync_result_shape() -> None:
    """Verify the dataclass interface auto_iterate consumes."""
    r = SyncResult()
    assert r.synced == []
    assert r.skipped == []
    assert r.notes == []
    assert r.main_tree_path is None


def test_sync_demo_copies_recovered_demo_to_main_tree(tmp_path: Path) -> None:
    """When brain demo-recovery modifies the demo (e.g. disables a
    broken wired component), the modified file must land in main tree
    so the user can run it. Sync walks up to find session.json then
    copies."""
    worktree, main_tree = _make_worktree_setup(tmp_path)
    demo_dir = worktree / "models" / "demos" / "X" / "demo"
    demo_dir.mkdir(parents=True)
    demo = demo_dir / "demo.py"
    demo.write_text("# recovered demo (brain disabled video_memory_fuser)\n")

    synced = sync_demo_to_main_tree(worktree_demo_path=demo)

    assert synced is not None
    main_demo = main_tree / "models" / "demos" / "X" / "demo" / "demo.py"
    assert main_demo.is_file()
    assert main_demo.read_text() == "# recovered demo (brain disabled video_memory_fuser)\n"


def test_sync_demo_noop_when_not_in_worktree(tmp_path: Path) -> None:
    """No session.json → no worktree → no-op. Returns DemoSyncResult
    with status `noop_not_in_worktree`."""
    demo_dir = tmp_path / "models" / "demos" / "X" / "demo"
    demo_dir.mkdir(parents=True)
    demo = demo_dir / "demo.py"
    demo.write_text("# demo\n")

    result = sync_demo_to_main_tree(worktree_demo_path=demo)
    assert result.status == "noop_not_in_worktree"
    assert result.synced_path is None


def test_sync_demo_noop_when_demo_missing(tmp_path: Path) -> None:
    """Demo file missing → returns DemoSyncResult with status
    `source_missing`. Distinct from no-op (different status)."""
    fake = tmp_path / "nonexistent" / "demo.py"
    result = sync_demo_to_main_tree(worktree_demo_path=fake)
    assert result.status == "source_missing"
    assert result.synced_path is None
