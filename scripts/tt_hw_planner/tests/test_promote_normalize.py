"""Promote must graduate every not-on-device component regardless of tag.

`_promote_normalize_ungraduated` folds any unwired REUSE/ADAPT into the NEW
work-path (keeping the reuse target as a wrap-hint) so the existing capture /
scaffold / gate / can_stop graduate it like anything else; a genuinely-wired
reuse is left alone. Scoped to promote — up/auto-up untouched.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _promote():
    return importlib.import_module("scripts.tt_hw_planner.commands.promote")


def test_unwired_reuse_adapt_normalized_wired_untouched(tmp_path) -> None:
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tt").mkdir()
    (demo / "tt" / "rmsnorm.py").write_text("# wired sibling copied into this demo\n")
    comps = [
        {"name": "new_a", "status": "NEW"},
        {"name": "reuse_wired", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"},
        {"name": "reuse_bare", "status": "REUSE", "tt_reuse_target": "models/common/rope_absent.py", "notes": "orig"},
        {"name": "adapt_bare", "status": "ADAPT", "tt_reuse_target": "models/tt_transformers/tt/attn_absent.py"},
    ]
    (demo / "bringup_status.json").write_text(json.dumps({"components": comps}))

    _promote()._promote_normalize_ungraduated(demo)

    out = {c["name"]: c for c in json.loads((demo / "bringup_status.json").read_text())["components"]}
    assert out["new_a"]["status"] == "NEW"
    assert out["reuse_wired"]["status"] == "REUSE"  # genuinely on device -> untouched
    assert out["reuse_bare"]["status"] == "NEW"  # unwired -> normalized to graduate path
    assert out["adapt_bare"]["status"] == "NEW"
    assert out["reuse_bare"]["tt_reuse_target"] == "models/common/rope_absent.py"  # wrap-hint kept
    assert "not wired" in out["reuse_bare"]["notes"] and "orig" in out["reuse_bare"]["notes"]


def test_normalize_noop_when_all_verified_or_new(tmp_path) -> None:
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m2"
    (demo / "_stubs").mkdir(parents=True)
    comps = [{"name": "a", "status": "NEW"}, {"name": "b", "status": "NEW"}]
    (demo / "bringup_status.json").write_text(json.dumps({"components": comps}))
    _promote()._promote_normalize_ungraduated(demo)
    out = {c["name"]: c["status"] for c in json.loads((demo / "bringup_status.json").read_text())["components"]}
    assert out == {"a": "NEW", "b": "NEW"}
