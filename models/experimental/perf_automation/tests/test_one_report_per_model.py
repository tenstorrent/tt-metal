"""When optimize finishes, its section lands beside the model's other sections.

bringup, trace-gate and emit-e2e all upsert into <model_dir>/RUN_REPORT.md. optimize is the only one
that does not: report_path redirects it to the git-ignored run directory. That redirect is right
DURING the run -- a report in the model directory is swept into the run's first commit and every
later git_revert restores the stale blob, which rewound a live report from 30 attempts back to 7 --
but it left voxtral holding two reports that each knew half the story, and the optimize half was the
one nobody could find beside the others.

Published at the END, when no revert can follow, and into the MAIN tree, because the worktree's copy
of the model directory is deleted with the worktree.
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[1]
for _p in (PERF, PERF / "cc_optimize"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_rspec = _ilu.spec_from_file_location("_cc_run_onereport", PERF / "cc_optimize" / "run.py")
_run = _ilu.module_from_spec(_rspec)
_rspec.loader.exec_module(_run)

_sspec = _ilu.spec_from_file_location("_cc_sum_onereport", PERF / "cc_optimize" / "summary.py")
_sm = _ilu.module_from_spec(_sspec)
_sspec.loader.exec_module(_sm)


def _tree(tmp_path, *, isolated: bool):
    main = tmp_path / "main"
    (main / "models" / "demo").mkdir(parents=True)
    if not isolated:
        return main, main / "models" / "demo"
    wt = tmp_path / "wt"
    (wt / "models" / "demo").mkdir(parents=True)
    (wt / ".tt_hw_planner_session.json").write_text(json.dumps({"source_repo": str(main)}))
    return main, wt / "models" / "demo"


def test_a_worktree_model_dir_resolves_to_the_main_tree(tmp_path):
    main, demo = _tree(tmp_path, isolated=True)
    assert _run._main_tree_demo_dir(demo) == main / "models" / "demo"


def test_running_in_place_resolves_to_nothing_to_redirect(tmp_path):
    _main, demo = _tree(tmp_path, isolated=False)
    assert _run._main_tree_demo_dir(demo) is None


def test_a_session_naming_its_own_root_is_not_a_redirect(tmp_path):
    """source_repo == the worktree root means this IS the main tree."""
    main, _ = _tree(tmp_path, isolated=False)
    (main / ".tt_hw_planner_session.json").write_text(json.dumps({"source_repo": str(main)}))
    assert _run._main_tree_demo_dir(main / "models" / "demo") is None


def test_the_section_lands_beside_the_others_without_disturbing_them(tmp_path):
    main, demo = _tree(tmp_path, isolated=True)
    prior = "<!-- BEGIN bringup -->\n22/24 native\n<!-- END bringup -->\n"
    (main / "models" / "demo" / "RUN_REPORT.md").write_text(prior)
    _run._publish_optimize_section(_sm, demo, "optimize", "53.25 -> 10.99 ms")
    txt = (main / "models" / "demo" / "RUN_REPORT.md").read_text()
    assert "22/24 native" in txt, "the earlier section must survive"
    assert "53.25 -> 10.99 ms" in txt
    assert txt.count("<!-- BEGIN optimize -->") == 1, "the writer wraps the block; it must not nest"


def test_publishing_twice_replaces_rather_than_appends(tmp_path):
    main, demo = _tree(tmp_path, isolated=True)
    _run._publish_optimize_section(_sm, demo, "optimize", "first")
    _run._publish_optimize_section(_sm, demo, "optimize", "second")
    txt = (main / "models" / "demo" / "RUN_REPORT.md").read_text()
    assert txt.count("<!-- BEGIN optimize -->") == 1
    assert "second" in txt and "first" not in txt


def test_a_missing_main_tree_costs_nothing(tmp_path, capsys):
    """Best-effort: the report that matters is already written before this runs."""
    wt = tmp_path / "wt" / "models" / "demo"
    wt.mkdir(parents=True)
    (tmp_path / "wt" / ".tt_hw_planner_session.json").write_text(json.dumps({"source_repo": str(tmp_path / "gone")}))
    _run._publish_optimize_section(_sm, wt, "optimize", "x")  # must not raise


def test_a_failure_is_reported_rather_than_swallowed():
    """Silent here means the model quietly keeps two half-reports again."""
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    i = src.index("def _publish_optimize_section(")
    seg = src[i : i + 1600]
    assert "could not publish" in seg, "a failed publish must say so"


def test_only_the_final_render_publishes():
    """The per-round render must keep going to the run directory alone -- that is what the redirect
    protects. _emit_summary is the final render and the only caller."""
    src = (PERF / "cc_optimize" / "run.py").read_text(encoding="utf-8")
    calls = [l for l in src.splitlines() if "_publish_optimize_section(" in l and not l.lstrip().startswith("def ")]
    assert len(calls) == 1, calls
