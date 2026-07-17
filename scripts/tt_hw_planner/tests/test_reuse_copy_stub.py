"""ADAPT/REUSE start from a native COPY of the existing tt-module.

Instead of a torch-fallback placeholder, a component that points at an existing
tt-module (tt_reuse_target) gets that module's source copied in verbatim as its
initial stub — a native starting point, editable for this model, with the
original source left untouched. NEW (or an unresolvable target) still gets the
torch-fallback template.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _bl():
    return importlib.import_module("scripts.tt_hw_planner.bringup_loop")


def _make_target(repo: Path) -> str:
    src = (
        "import ttnn\n"
        "from models.common.utils import helper\n\n\n"
        "class RMSNorm:\n"
        "    def __call__(self, x):\n"
        "        return ttnn.rms_norm(x)\n"
    )
    f = repo / "models" / "common" / "rmsnorm.py"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(src)
    return src


def test_reuse_starts_from_native_copy_original_untouched(tmp_path):
    bl = _bl()
    orig = _make_target(tmp_path)
    comp = {"name": "qwen3_r_m_s_norm", "status": "REUSE", "tt_reuse_target": "models/common/rmsnorm.py"}

    body = bl._render_component_stub(comp, model_id="org/m", repo_root=tmp_path)
    # native copy: real ttnn code, not the torch-fallback template
    assert "ttnn.rms_norm" in body and "_get_torch_submodule" not in body
    assert "class RMSNorm:" in body and "from models.common.utils import helper" in body
    assert bl._REUSE_COPY_MARKER in body
    # original source untouched
    assert (tmp_path / "models" / "common" / "rmsnorm.py").read_text() == orig


def test_adapt_also_copies(tmp_path):
    bl = _bl()
    _make_target(tmp_path)
    comp = {"name": "x", "status": "ADAPT", "tt_reuse_target": "models/common/rmsnorm.py"}
    assert "class RMSNorm:" in bl._render_component_stub(comp, model_id="org/m", repo_root=tmp_path)


def test_new_uses_torch_template(tmp_path):
    bl = _bl()
    body = bl._render_component_stub({"name": "y", "status": "NEW"}, model_id="org/m", repo_root=tmp_path)
    assert "_get_torch_submodule" in body and bl._REUSE_COPY_MARKER not in body


def test_missing_target_falls_back_to_torch(tmp_path):
    bl = _bl()
    comp = {"name": "z", "status": "REUSE", "tt_reuse_target": "models/common/does_not_exist.py"}
    body = bl._render_component_stub(comp, model_id="org/m", repo_root=tmp_path)
    assert "_get_torch_submodule" in body
