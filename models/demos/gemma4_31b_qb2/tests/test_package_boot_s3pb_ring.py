# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-4 `batch` packaged class (tt/package_boot_s3pb_ring.py), no device and no ttnn: the module's
source is parsed, not imported (importing the ring class imports ttnn). It must subclass the option C ring class
Gemma4ResidentMultiS32PBRingForCausalLM, its initialize_vllm_model must run package_boot.ensure_resident_cache() before
the ring construction (as update 3's package_boot_s3pb did), its continuation pre-warm marker is keyed to the m5_11
engine package and runs through contwarm.run("batch", ...) after the S3-PB prefill warmup. Update 5: no bundle names it
any more; it is the packaged base that tt/package_boot_s3pb_ring_apc.py composes (update 5's `batch` bundle).
Replaces update 3's tests/test_package_boot_s3pb.py (its class and bundle are removed in update 4).
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_s3pb_ring.py"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _tree():
    return ast.parse((ROOT / "tt" / "package_boot_s3pb_ring.py").read_text())


def _cls():
    return next(n for n in _tree().body if isinstance(n, ast.ClassDef))


def test_imports_and_base():
    tree = _tree()
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert ("models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_vllm",
            "Gemma4ResidentMultiS32PBRingForCausalLM") in imports
    assert ("models.demos.gemma4_31b_qb2.tt.package_boot", "ensure_resident_cache") in imports
    assert ("models.demos.gemma4_31b_qb2.tt", "contwarm") in imports
    cls = _cls()
    assert cls.name == "Gemma4ResidentMultiS32PBRingPackaged"
    assert [ast.unparse(b) for b in cls.bases] == ["Gemma4ResidentMultiS32PBRingForCausalLM"]


def test_cache_before_construction():
    init = next(f for f in _cls().body if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [ast.unparse(d) for d in init.decorator_list] == ["classmethod"]
    body = [ast.unparse(s) for s in init.body if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)]
    assert body == ["ensure_resident_cache()",
                    "return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)"]


def test_engine_package_and_prewarm_call():
    cls = _cls()
    consts = {t.id: ast.literal_eval(s.value) for s in cls.body if isinstance(s, ast.Assign) for t in s.targets
              if isinstance(t, ast.Name)}
    assert consts.get("ENGINE_PACKAGE") == "resident_m5_11"
    warm = next(f for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "warmup_model_prefill")
    src = ast.unparse(warm)
    assert "super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)" in src
    assert "contwarm.run('batch', self.ENGINE_PACKAGE" in src
    assert src.index("super().warmup_model_prefill") < src.index("contwarm.run(")


def test_ring_base_takes_the_wrappers_call():
    """the S3.2 base's initialize_vllm_model (the ring class inherits it) accepts (hf_config, mesh_device, max_batch_size,
    **keywords)"""
    src = (ROOT / "tt" / "generator_resident_multi_s3_2_vllm.py").read_text()
    fn = next(f for c in ast.parse(src).body if isinstance(c, ast.ClassDef) for f in c.body
              if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [a.arg for a in fn.args.args] == ["cls", "hf_config", "mesh_device", "max_batch_size"]
    assert {a.arg for a in fn.args.kwonlyargs} >= {"max_seq_len", "tt_data_parallel", "optimizations"}


def test_update5_composes_it_and_no_bundle_names_it():
    mains = {json.loads(p.read_text())["main_class"] for p in (ROOT / "vllm_models").glob("*/vllm_metadata.json")}
    assert "models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring:Gemma4ResidentMultiS32PBRingPackaged" not in mains
    apc = next(n for n in ast.parse((ROOT / "tt" / "package_boot_s3pb_ring_apc.py").read_text()).body
               if isinstance(n, ast.ClassDef))
    assert "Gemma4ResidentMultiS32PBRingPackaged" in [ast.unparse(b) for b in apc.bases]
