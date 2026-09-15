"""Op-level compute-split must not count an unwired REUSE/ADAPT tag on-device.

Mirror of the component-level verification fix: `_compute_op_split` credits a
REUSE/ADAPT component's op on-device only when its reuse target is genuinely
wired into the demo; an unwired tag runs on CPU via the eager runner.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_op_split_unwired_reuse_counts_on_cpu(tmp_path, monkeypatch) -> None:
    cli = importlib.import_module("scripts.tt_hw_planner.cli")
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")

    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tt").mkdir()
    (demo / "tt" / "rmsnorm.py").write_text("# wired sibling copied into this demo\n")
    status = {
        "components": [
            {"name": "reuse_wired", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"},
            {"name": "reuse_bare", "status": "REUSE", "tt_reuse_target": "models/common/rope_absent.py"},
            {"name": "adapt_bare", "status": "ADAPT", "tt_reuse_target": "models/tt_transformers/tt/attn_absent.py"},
        ]
    }
    (demo / "bringup_status.json").write_text(json.dumps(status))
    monkeypatch.setattr(bl, "find_demo_dir", lambda *a, **k: demo)

    s = cli._compute_op_split("fake/m")
    assert s["on_device"] == 1, s  # only the wired reuse
    assert s["on_cpu"] == 2, s  # the two unwired tags run on CPU
    where = {r["name"]: r["where"] for r in s["components"]}
    assert where == {"reuse_wired": "device", "reuse_bare": "cpu", "adapt_bare": "cpu"}, where
