"""runs/latest is handed out only to the model whose run it is.

report_path asks _latest_belongs_to before writing into the run directory. That check read the
manifest, and returned True whenever it could not -- so a run whose manifest never reached this
checkout handed its directory to ANY caller. An isolated run writes its manifest inside the /tmp
worktree and mirrors back only the report, which is exactly that case.

Twice on 2026-09-03 the tool's own suite replaced a finished 32 KB report with a fixture, 57 bytes
and then 48, and two tests in test_one_report_per_model failed for the same reason in the checkout
where a real runs/latest exists.
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

_spec = _ilu.spec_from_file_location("_cc_sum_belongs", PERF / "cc_optimize" / "summary.py")
_sm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_sm)


def _latest(tmp_path, run_name, *, manifest_model=None):
    d = tmp_path / "runs" / run_name
    d.mkdir(parents=True)
    if manifest_model is not None:
        (d / "manifest.json").write_text(json.dumps({"config": {"model_root": str(manifest_model)}}))
    link = tmp_path / "runs" / "latest"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(run_name)
    return link


def test_a_manifest_that_names_this_model_still_wins(tmp_path):
    demo = tmp_path / "models" / "voxtral_mini_3b_2507"
    demo.mkdir(parents=True)
    latest = _latest(tmp_path, "2026-09-01T23-00-22-voxtral_mini_3b_2507", manifest_model=demo)
    assert _sm._latest_belongs_to(latest, demo) is True


def test_a_manifest_naming_another_model_is_refused(tmp_path):
    demo = tmp_path / "models" / "mine"
    demo.mkdir(parents=True)
    other = tmp_path / "models" / "theirs"
    other.mkdir(parents=True)
    latest = _latest(tmp_path, "2026-09-01T23-00-22-theirs", manifest_model=other)
    assert _sm._latest_belongs_to(latest, demo) is False


def test_no_manifest_falls_back_to_the_directory_name(tmp_path):
    """THE DEFECT. Run.create names the directory "<timestamp>-<model dir name>"."""
    demo = tmp_path / "models" / "voxtral_mini_3b_2507"
    demo.mkdir(parents=True)
    mine = _latest(tmp_path, "2026-09-01T23-00-22-voxtral_mini_3b_2507")
    assert _sm._latest_belongs_to(mine, demo) is True

    stranger = tmp_path / "models" / "some_other_model"
    stranger.mkdir(parents=True)
    assert _sm._latest_belongs_to(mine, stranger) is False, "a run must not be handed to another model"


def test_a_directory_that_names_nothing_still_gets_its_own_run(tmp_path):
    """The early-run case this must not regress: nothing to read is not a refusal."""
    demo = tmp_path / "models" / "m"
    demo.mkdir(parents=True)
    latest = _latest(tmp_path, "m")  # name IS the model leaf
    assert _sm._latest_belongs_to(latest, demo) is True


def test_the_report_of_one_model_is_never_written_by_another(tmp_path):
    """End to end through report_path, which is what actually did the damage."""
    demo = tmp_path / "models" / "voxtral_mini_3b_2507"
    demo.mkdir(parents=True)
    stranger = tmp_path / "models" / "throwaway"
    stranger.mkdir(parents=True)
    _latest(tmp_path, "2026-09-01T23-00-22-voxtral_mini_3b_2507")
    orig = _sm._runs_root
    _sm._runs_root = lambda: tmp_path / "runs"
    try:
        # report_path returns the `latest` SYMLINK; resolve it to see which run it points at.
        assert _sm.report_path(demo).parent.resolve().name.endswith("voxtral_mini_3b_2507")
        assert _sm.report_path(stranger) == stranger / "RUN_REPORT.md", "a stranger gets its own file"
    finally:
        _sm._runs_root = orig
