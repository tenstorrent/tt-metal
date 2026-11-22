# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-6 `batch` packaged class (tt/package_boot_m7.py), no device and no ttnn: the sources are
parsed, not imported (importing the class chain imports ttnn), as update 5's test_package_boot_s3pb_ring_apc.py. The
class composes the M7 class v5.6 with update 4's packaged ring class, M7 first:
- its models.* imports are exactly those two bases and package_boot, its bases are (v5.6, the packaged ring class), and
  the drafter-cache step it calls (ensure_drafter_cache, drafter_cache_key) is defined in the module itself;
- it defines ENGINE_PACKAGE (resident_m5_14) and initialize_vllm_model and nothing else: no build_drafter (v5.6's is
  the first, by M308), no serving method;
- initialize_vllm_model is a classmethod whose body is exactly the drafter-cache step, then super();
- its MRO (C3, computed on stand-in classes built from the parsed bases) is v5.6 -> v5.5 -> v5.3 -> APC v3 -> APC v2 ->
  APC v1 -> the packaged ring class -> ring, so super() reaches the packaged ring class's ensure_resident_cache() next
  and build_drafter resolves to v5.6;
- warmup_model_prefill resolves to v5.3's, which calls super() on to the packaged ring class's continuation pre-warm;
- the bundle names the class, and the package ships exactly the two profiles' bundles.
The boot-order and cache behaviour tests (fake ttnn) stay with the bench harness (bench/resident/m7/bx/tests).
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_m7.py"""

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLS, V56, V55, V53 = ("Gemma4ResidentMultiM7Packaged", "Gemma4ResidentMultiM7V56ForCausalLM",
                      "Gemma4ResidentMultiM7V55ForCausalLM", "Gemma4ResidentMultiM7V53ForCausalLM")
A3, A2, A1, PKG = ("Gemma4ResidentMultiS32PBRingAPCV3ForCausalLM", "Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM",
                   "Gemma4ResidentMultiS32PBRingAPCForCausalLM", "Gemma4ResidentMultiS32PBRingPackaged")
LEAF = "Gemma4ResidentMultiS32PBRingForCausalLM"  # the ring class: not parsed, a stand-in leaf
FILES = {
    "package_boot_m7.py": CLS,
    "generator_resident_multi_m7_v5_6_vllm.py": V56,
    "generator_resident_multi_m7_v5_5_vllm.py": V55,
    "generator_resident_multi_m7_v5_3_vllm.py": V53,
    "generator_resident_multi_s3_2_pb_ring_apc_v3_vllm.py": A3,
    "generator_resident_multi_s3_2_pb_ring_apc_v2_vllm.py": A2,
    "generator_resident_multi_s3_2_pb_ring_apc_vllm.py": A1,
    "package_boot_s3pb_ring.py": PKG,
}
ARCH = "Gemma4QB2ResidentMultiM7ForCausalLM"
MAIN = "models.demos.gemma4_31b_qb2.tt.package_boot_m7:Gemma4ResidentMultiM7Packaged"


def _tree(name):
    return ast.parse((ROOT / "tt" / name).read_text())


def _classes():
    out = {}
    for name, cls in FILES.items():
        out[cls] = next(n for n in _tree(name).body if isinstance(n, ast.ClassDef) and n.name == cls)
    return out


def _members(cdef):
    names = set()
    for s in cdef.body:
        if isinstance(s, ast.FunctionDef):
            names.add(s.name)
        elif isinstance(s, ast.Assign):
            names |= {t.id for t in s.targets if isinstance(t, ast.Name)}
    return names


def _mro():
    cdefs, made = _classes(), {}

    def make(name):
        if name not in made:
            c = cdefs.get(name)
            bases = tuple(make(ast.unparse(b)) for b in c.bases) if c else (object,)
            made[name] = type(name, bases, {})
        return made[name]

    return [k.__name__ for k in make(CLS).__mro__ if k is not object]


def _resolves_to(member, after=None):
    """the first class in the MRO (after `after`, if given) whose parsed body defines `member`"""
    cdefs, mro = _classes(), _mro()
    if after is not None:
        mro = mro[mro.index(after) + 1:]
    return next((n for n in mro if n in cdefs and member in _members(cdefs[n])), None)


def test_imports_and_bases():
    tree = _tree("package_boot_m7.py")
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) and n.module.startswith("models.")
               for a in n.names}
    assert imports == {("models.demos.gemma4_31b_qb2.tt.generator_resident_multi_m7_v5_6_vllm", V56),
                       ("models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring", PKG),
                       ("models.demos.gemma4_31b_qb2.tt", "package_boot")}
    assert [ast.unparse(b) for b in _classes()[CLS].bases] == [V56, PKG]
    defined = {f.name for f in tree.body if isinstance(f, ast.FunctionDef)}
    assert {"ensure_drafter_cache", "drafter_cache_key"} <= defined  # the D3 step is this module's own


def test_members_and_engine_package():
    cls = _classes()[CLS]
    assert _members(cls) == {"ENGINE_PACKAGE", "initialize_vllm_model"}
    consts = {t.id: ast.literal_eval(s.value) for s in cls.body if isinstance(s, ast.Assign) for t in s.targets
              if isinstance(t, ast.Name)}
    assert consts == {"ENGINE_PACKAGE": "resident_m5_14"}


def test_initialize_runs_the_drafter_cache_then_super():
    init = next(f for f in _classes()[CLS].body if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [ast.unparse(d) for d in init.decorator_list] == ["classmethod"]
    body = [ast.unparse(s) for s in init.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert body == ["ensure_drafter_cache()",
                    "return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)"]


def test_c3_mro():
    assert _mro() == [CLS, V56, V55, V53, A3, A2, A1, PKG, LEAF]


def test_resolution():
    assert _resolves_to("build_drafter") == V56  # M308: the composition never overrides it
    assert _resolves_to("initialize_vllm_model", after=CLS) == PKG  # super() reaches ensure_resident_cache() next
    assert _resolves_to("ENGINE_PACKAGE") == CLS
    assert _resolves_to("warmup_model_prefill") == V53
    warm = next(f for f in _classes()[V53].body if isinstance(f, ast.FunctionDef) and f.name == "warmup_model_prefill")
    assert "super().warmup_model_prefill(kv_cache, enable_trace, **kwargs)" in ast.unparse(warm)
    assert _resolves_to("warmup_model_prefill", after=V53) == PKG  # the packaged ring class's continuation pre-warm


def test_bundle_names_the_packaged_class():
    meta = json.loads((ROOT / "vllm_models" / "resident_multi_m7" / "vllm_metadata.json").read_text())
    assert meta == {"arch": ARCH, "main_class": MAIN}
    archs = {json.loads(p.read_text())["arch"] for p in (ROOT / "vllm_models").glob("*/vllm_metadata.json")}
    assert archs == {"Gemma4QB2ResidentSpecPB4APCForCausalLM", ARCH}
