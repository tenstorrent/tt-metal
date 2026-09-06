"""REUSE must be scaffolded like ADAPT — a stub + PCC test — so it enters the
per-component graduation gate instead of being silently trusted on CPU.

The gate builds its work-list from PCC test files on disk and judges placement
by the native-body check; a REUSE that never got a test was invisible and never
worked. Including REUSE in autofill_stubs' filter makes it create both, so it is
iterated and only counts on-device when its stub is genuinely native.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def test_reuse_component_gets_stub_and_pcc_test(tmp_path, monkeypatch) -> None:
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")

    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "tests" / "pcc").mkdir(parents=True)
    status = {
        "new_model_id": "org/m",
        "components": [
            {
                "name": "reuse_norm",
                "status": "REUSE",
                "kind": "RMSNorm",
                "submodule_path": "model.norm",
                "tt_reuse_target": "models/common/rmsnorm.py",
                "hf_reference": "transformers/models/x/modeling_x.py",
                "new_shape": {"hidden_size": 2048},
            },
            {
                "name": "a_new",
                "status": "NEW",
                "kind": "X",
                "submodule_path": "model.x",
                "new_shape": {"hidden_size": 2048},
            },
        ],
    }
    (demo / "bringup_status.json").write_text(json.dumps(status))
    monkeypatch.setattr(bl, "find_demo_dir", lambda *a, **k: demo)

    bl.autofill_stubs(model_id="org/m")

    assert (demo / "_stubs" / "reuse_norm.py").exists(), "REUSE should now get a stub"
    assert (demo / "tests" / "pcc" / "test_reuse_norm.py").exists(), "REUSE should now get a PCC test"
    # NEW still works as before
    assert (demo / "_stubs" / "a_new.py").exists()
