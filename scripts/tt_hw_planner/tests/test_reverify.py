"""--reverify clears restored graduation markers so a rerun re-earns graduation.

Guards against trusting a stale .last_good_native carried over by the overlay's
directory-level patch (which the per-file session-local exclusion does not cover):
with --reverify the gate re-runs each component's PCC instead of short-circuiting.
The stub bodies are kept and the overlay/capture contract is untouched, so
promote/emit-e2e still see freshly-written markers afterward.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bcc():
    return importlib.import_module("scripts.tt_hw_planner._cli_helpers.bringup_cc")


def test_clear_graduation_snapshots_removes_markers_keeps_stubs(tmp_path):
    bcc = _bcc()
    stubs = tmp_path / "_stubs"
    stubs.mkdir()
    (stubs / "a.py").write_text("import ttnn\n")
    (stubs / "a.py.last_good_native").write_text("import ttnn\n")
    (stubs / "b.py").write_text("import ttnn\n")
    (stubs / "b.py.last_good_sharded").write_text("import ttnn\n")

    n = bcc._clear_graduation_snapshots(tmp_path)

    assert n == 2
    assert not (stubs / "a.py.last_good_native").exists()
    assert not (stubs / "b.py.last_good_sharded").exists()
    assert (stubs / "a.py").read_text() == "import ttnn\n"
    assert (stubs / "b.py").read_text() == "import ttnn\n"


def test_clear_is_noop_when_no_markers(tmp_path):
    bcc = _bcc()
    (tmp_path / "_stubs").mkdir()
    (tmp_path / "_stubs" / "a.py").write_text("x\n")
    assert bcc._clear_graduation_snapshots(tmp_path) == 0
    assert (tmp_path / "_stubs" / "a.py").exists()


def test_reverify_flag_wired_on_all_three_commands_and_driver():
    cli = (_REPO_ROOT / "scripts/tt_hw_planner/cli.py").read_text()
    prom = (_REPO_ROOT / "scripts/tt_hw_planner/commands/promote.py").read_text()
    bcc = (_REPO_ROOT / "scripts/tt_hw_planner/_cli_helpers/bringup_cc.py").read_text()

    for parser in ("pup", "paut", "pprom"):
        idx = cli.find(f'{parser}.add_argument(\n        "--reverify"')
        assert idx != -1, f"--reverify not added to {parser} (up/auto-up/promote)"

    assert "reverify: bool = False," in bcc, "run_bringup_cc missing reverify param"
    assert "_clear_graduation_snapshots(demo_dir)" in bcc, "driver does not clear snapshots on reverify"
    assert 'reverify=bool(getattr(args, "reverify", False))' in cli, "up call site does not pass reverify"
    assert 'reverify=bool(getattr(args, "reverify", False))' in prom, "promote call site does not pass reverify"
