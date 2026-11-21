"""Pin that tt_hw_planner emits demos at the tt-metal standard path:
<demo_dir>/demo/demo.py, matching existing demos like qwen3_vl/demo/demo.py,
bert/demo/demo.py, etc.

Before this alignment, the auto-emitted demo lived at <demo_dir>/demo.py
(no `demo/` subdir), which deviated from the repo convention.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_emit_runnable_demo_writes_to_demo_subdir() -> None:
    """Source-grep: emit_runnable_demo must write to <demo_dir>/demo/demo.py,
    not <demo_dir>/demo.py."""
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    src = (Path(bl.__file__)).read_text()
    # The destination must include the `demo/` subdir
    assert 'demo_subdir / "demo.py"' in src, (
        "emit_runnable_demo must emit at <demo_dir>/demo/demo.py to align "
        "with the tt-metal repo convention (qwen3_vl/demo/demo.py, etc.)"
    )
    # And the pytest_path string must reflect the demo/ subdir
    assert 'demo_subdir / "demo.py"' in src


def test_llm_synth_response_path_uses_demo_subdir() -> None:
    """The LLM-synth response path for demos also writes to the standard
    location."""
    ls = importlib.import_module("scripts.tt_hw_planner.llm_synth")
    src = (Path(ls.__file__)).read_text()
    assert 'demo_subdir = demo_dir / "demo"' in src
    assert 'destination = demo_subdir / "demo.py"' in src


def test_scaffold_demo_folder_creates_standard_path() -> None:
    """scaffold_demo_folder's initial demo creation also uses the standard
    <model>/demo/demo.py path."""
    sdf = importlib.import_module("scripts.tt_hw_planner.scaffold_demo_folder")
    src = (Path(sdf.__file__)).read_text()
    # The creates list now writes to <new_dir>/demo/demo.py and
    # <new_dir>/demo/__init__.py
    assert 'new_dir_rel / "demo" / "demo.py"' in src, "scaffold_demo_folder must place demo.py at <model>/demo/demo.py"
    assert (
        'new_dir_rel / "demo" / "__init__.py"' in src
    ), "scaffold_demo_folder must add demo/__init__.py for package import"


def test_scaffold_demo_folder_respects_legacy_and_standard_paths() -> None:
    """If a demo.py already exists at EITHER the legacy or standard path,
    scaffold-demo-folder must skip emission (don't clobber)."""
    sdf = importlib.import_module("scripts.tt_hw_planner.scaffold_demo_folder")
    src = (Path(sdf.__file__)).read_text()
    assert 'legacy_demo = target_dir / "demo.py"' in src
    assert 'standard_demo = target_dir / "demo" / "demo.py"' in src


def test_overlay_capture_scans_demo_subdir() -> None:
    """The overlay-capture secondary scan (Bug 1 fix) must include
    `demo/` so the auto-emitted demo.py gets preserved across runs."""
    cli_mod = importlib.import_module("scripts.tt_hw_planner.cli")
    src = (Path(cli_mod.__file__)).read_text()
    # The scan_paths loop iterates over a tuple that includes "demo"
    assert '("_stubs", "tests/pcc", "demo")' in src, (
        "overlay capture must scan the demo/ subdir so the auto-emitted "
        "demo.py is preserved across worktree destruction"
    )
