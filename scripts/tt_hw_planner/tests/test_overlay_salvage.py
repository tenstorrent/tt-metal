"""overlay-apply salvages the graduation state from a skipped directory patch.

A captured demo's graduation state (.bringup_cc_state.json / bringup_status.json /
.py.last_good_native) is bundled in the directory-level patch, which git skips
whole when its target dir has drifted ("already exists"). Without salvage, an
overlay-materialized demo has stubs + tests but NO graduation markers, so
emit-e2e / optimize see zero graduated modules. _salvage_graduation_state applies
just those new-file state additions, which don't conflict, restoring the
graduated set.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _om():
    return importlib.import_module("scripts.tt_hw_planner.overlay_manager")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False).stdout


def _make_patch_and_conflict(repo: Path) -> str:
    """Build a patch adding a stub + state files, then leave ONLY the stub on disk
    (so the stub hunk conflicts and the whole patch would be skipped)."""
    subprocess.run(["git", "init", "-q", str(repo)], check=False)
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    d = repo / "models" / "demos" / "m"
    (d / "_stubs").mkdir(parents=True)
    (d / "_stubs" / "a.py").write_text("import ttnn\n")
    (d / "_stubs" / "a.py.last_good_native").write_text("import ttnn\n")
    (d / ".bringup_cc_state.json").write_text('{"best_pcc": {"a": 0.999}}\n')
    (d / "bringup_status.json").write_text('{"components": [{"name": "a", "status": "NEW"}]}\n')
    _git(repo, "add", "-A")
    patch = subprocess.run(
        ["git", "-C", str(repo), "diff", "--cached"], capture_output=True, text=True, check=False
    ).stdout
    # reset to a clean tree, then leave ONLY the stub (conflicts with the patch)
    _git(repo, "reset", "-q")
    for p in (d / "_stubs" / "a.py.last_good_native", d / ".bringup_cc_state.json", d / "bringup_status.json"):
        p.unlink()
    return patch


def test_salvage_restores_state_when_whole_patch_conflicts(tmp_path):
    om = _om()
    repo = tmp_path / "repo"
    patch = _make_patch_and_conflict(repo)

    with om.using_repo(repo):
        # the whole patch cannot apply (a.py already exists)
        rc_check, _ = om._git_apply(patch, check_only=True)
        assert rc_check != 0, "precondition: full patch should conflict on the existing stub"

        written = om._salvage_graduation_state(patch)

    d = repo / "models" / "demos" / "m"
    assert (d / ".bringup_cc_state.json").is_file(), "cc-state must be salvaged"
    assert (d / "_stubs" / "a.py.last_good_native").is_file(), "graduation marker must be salvaged"
    assert (d / "bringup_status.json").is_file(), "status must be salvaged"
    assert (d / "_stubs" / "a.py").read_text() == "import ttnn\n"  # existing stub untouched
    assert any("bringup_cc_state.json" in w for w in written)
    assert any("last_good_native" in w for w in written)
    assert not any(w.endswith("_stubs/a.py") for w in written)  # stub is NOT a salvaged state file


def test_salvage_wired_into_apply_for():
    src = (_REPO_ROOT / "scripts/tt_hw_planner/overlay_manager.py").read_text()
    idx = src.find("def apply_for(")
    assert idx != -1
    assert "_salvage_graduation_state(patch_text)" in src[idx:], "apply_for does not salvage on skip"
