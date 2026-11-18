# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-2 `single-user` packaged class (tt/package_boot_pb3.py), no device and no ttnn: the module's
source is parsed, not imported (importing PB3 imports ttnn). It must subclass Gemma4ResidentSpecPB3ForCausalLM, and its
initialize_vllm_model must run package_boot.ensure_resident_cache() before the PB3 construction. Update 4: PB3Packaged
is no longer a profile's class (its bundle and update 1's qb2_pb bundle are removed); it stays as the base of update 4's
`single-user` class (tt/package_boot_pb4.py, tests/test_package_boot_pb4.py), so its bundle test is gone.
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_pb3.py"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tree():
    return ast.parse((ROOT / "tt" / "package_boot_pb3.py").read_text())


def test_imports_and_base():
    tree = _tree()
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb3", "Gemma4ResidentSpecPB3ForCausalLM") in imports
    assert ("models.demos.gemma4_31b_qb2.tt.package_boot", "ensure_resident_cache") in imports
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    assert cls.name == "Gemma4ResidentSpecPB3Packaged"
    assert [ast.unparse(b) for b in cls.bases] == ["Gemma4ResidentSpecPB3ForCausalLM"]


def test_cache_before_construction():
    cls = next(n for n in _tree().body if isinstance(n, ast.ClassDef))
    init = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [ast.unparse(d) for d in init.decorator_list] == ["classmethod"]
    body = [ast.unparse(s) for s in init.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
    assert body == ["ensure_resident_cache()",
                    "return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)"]

