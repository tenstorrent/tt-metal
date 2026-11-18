# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-4 `single-user` packaged class (tt/package_boot_pb4.py), no device and no ttnn: the module's
source is parsed, not imported (importing PB4 imports ttnn). It must subclass update 2's Gemma4ResidentSpecPB3Packaged
(so the resident weight cache is still ensured first), key its continuation pre-warm marker to the m3_2 engine package,
run the pre-warm through contwarm.run("single-user", ...) after the PB3 prefill warmup, and the bundle must name it.
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_pb4.py"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tree():
    return ast.parse((ROOT / "tt" / "package_boot_pb4.py").read_text())


def _cls():
    return next(n for n in _tree().body if isinstance(n, ast.ClassDef))


def test_imports_and_base():
    tree = _tree()
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("models.demos.gemma4_31b_qb2.tt.package_boot_pb3", "Gemma4ResidentSpecPB3Packaged") in imports
    assert ("models.demos.gemma4_31b_qb2.tt", "contwarm") in imports
    cls = _cls()
    assert cls.name == "Gemma4ResidentSpecPB4Packaged"
    assert [ast.unparse(b) for b in cls.bases] == ["Gemma4ResidentSpecPB3Packaged"]


def test_engine_package_and_prewarm_call():
    cls = _cls()
    consts = {t.id: ast.literal_eval(s.value) for s in cls.body if isinstance(s, ast.Assign) for t in s.targets
              if isinstance(t, ast.Name)}
    assert consts.get("ENGINE_PACKAGE") == "resident_m3_2"
    warm = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "warmup_model_prefill")
    src = ast.unparse(warm)
    assert "super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)" in src
    assert "contwarm.run('single-user', self.ENGINE_PACKAGE" in src
    assert src.index("super().warmup_model_prefill") < src.index("contwarm.run(")


def test_bundle_names_the_packaged_class():
    meta = json.loads((ROOT / "vllm_models" / "resident_spec_pb4" / "vllm_metadata.json").read_text())
    assert meta == {"arch": "Gemma4QB2ResidentSpecPB4ForCausalLM",
                    "main_class": "models.demos.gemma4_31b_qb2.tt.package_boot_pb4:Gemma4ResidentSpecPB4Packaged"}
