# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-5 `batch` packaged class (tt/package_boot_s3pb_ring_apc.py), no device and no ttnn: the
sources are parsed, not imported (importing the ring class imports ttnn), as update 4's test_package_boot_s3pb_ring.py.
The class composes the S1 batch class v2 with update 4's packaged ring class by cooperative inheritance, S1 first:
- its MRO (C3, computed on stand-in classes built from the parsed bases) is APC-v2 -> APC-v1 -> RingPackaged -> ring;
- it defines nothing itself, so the package boot (initialize_vllm_model: ensure_resident_cache), ENGINE_PACKAGE and the
  continuation pre-warm (warmup_model_prefill, _cont_warm) resolve to the packaged ring class, and the capabilities,
  pools and request path to the S1 classes;
- the inherited pre-warm calls the PB generator directly, with no request ids or tail orders;
- the S1 capability is prefix_resume(tail_slots(4)): granularity 8192, min_suffix 2080, prompt_only, req_ids;
- contwarm's code hash covers every tt/ source, so adding this file changes the marker key (the first start re-warms);
- the bundle names the class.
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_s3pb_ring_apc.py"""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "package_boot_s3pb_ring_apc.py": "Gemma4ResidentMultiS32PBRingAPCPackaged",
    "generator_resident_multi_s3_2_pb_ring_apc_v2_vllm.py": "Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM",
    "generator_resident_multi_s3_2_pb_ring_apc_vllm.py": "Gemma4ResidentMultiS32PBRingAPCForCausalLM",
    "package_boot_s3pb_ring.py": "Gemma4ResidentMultiS32PBRingPackaged",
}
LEAF = "Gemma4ResidentMultiS32PBRingForCausalLM"  # the ring class: not parsed, a stand-in leaf
CLS = "Gemma4ResidentMultiS32PBRingAPCPackaged"
V2, V1, PKG = ("Gemma4ResidentMultiS32PBRingAPCV2ForCausalLM", "Gemma4ResidentMultiS32PBRingAPCForCausalLM",
               "Gemma4ResidentMultiS32PBRingPackaged")


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


def _resolves_to(member):
    cdefs = _classes()
    return next((n for n in _mro() if n in cdefs and member in _members(cdefs[n])), None)


def test_imports_and_bases():
    tree = _tree("package_boot_s3pb_ring_apc.py")
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imports == {
        ("models.demos.gemma4_31b_qb2.tt.generator_resident_multi_s3_2_pb_ring_apc_v2_vllm", V2),
        ("models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring", PKG),
    }
    cls = _classes()[CLS]
    assert [ast.unparse(b) for b in cls.bases] == [V2, PKG]


def test_c3_mro():
    assert _mro() == [CLS, V2, V1, PKG, LEAF]


def test_inherited_not_copied():
    cls = _classes()[CLS]
    body = [s for s in cls.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert body == [], "the composition must define nothing (docstring only)"
    assert _members(cls) == set()


def test_resolution():
    assert _resolves_to("initialize_vllm_model") == PKG  # ensure_resident_cache, then the ring construction
    assert _resolves_to("ENGINE_PACKAGE") == PKG
    assert _resolves_to("warmup_model_prefill") == PKG
    assert _resolves_to("_cont_warm") == PKG
    assert _resolves_to("model_capabilities") == V1
    assert _resolves_to("get_kv_cache_spec") == V1
    assert _resolves_to("allocate_pools") == V1  # the tail pool, before warmup_model (the plugin's order)
    assert _resolves_to("prefill_forward") == V2
    assert _resolves_to("release_request") == V1


def test_cooperative_init_chain():
    """every parsed __init__ on the path calls super().__init__ (so RingPackaged's cont_prewarm and S1's state are both
    set), and the package boot's classmethod still ends in super()"""
    cdefs = _classes()
    for name in (V1, PKG):
        init = next(f for f in cdefs[name].body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
        assert "super().__init__(*args, **kwargs)" in ast.unparse(init), name
    boot = next(f for f in cdefs[PKG].body if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [ast.unparse(s) for s in boot.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))] == [
        "ensure_resident_cache()",
        "return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)"]


def test_prewarm_bypasses_the_adapter():
    """the inherited _cont_warm calls the PB generator (gen.prefill_forward), never the adapter, and passes neither
    req_ids nor tail_orders: the S1 store / restore path is not on it and the tail pool stays empty"""
    warm = next(f for f in _classes()[PKG].body if isinstance(f, ast.FunctionDef) and f.name == "_cont_warm")
    calls = [c for c in ast.walk(warm) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("prefill_forward")]
    assert calls and all(ast.unparse(c.func) == "gen.prefill_forward" for c in calls)
    assert all(not {k.arg for k in c.keywords} & {"req_ids", "tail_orders"} for c in calls)


def test_capability_defaults():
    v1 = _tree("generator_resident_multi_s3_2_pb_ring_apc_vllm.py")
    consts = {t.id: ast.literal_eval(s.value) for s in v1.body if isinstance(s, ast.Assign) for t in s.targets
              if isinstance(t, ast.Name) and isinstance(s.value, ast.Constant)}
    assert consts["GRANULARITY"] == 8192 and consts["MIN_SUFFIX"] == 2080
    fn = next(f for f in v1.body if isinstance(f, ast.FunctionDef) and f.name == "prefix_resume")
    assert ast.unparse(fn.body[-1]) == ("return {'granularity': GRANULARITY, 'min_suffix': MIN_SUFFIX, 'tail_slots': slots, "
                                        "'prompt_only': True, 'req_ids': True}")
    caps = next(s for s in _classes()[V1].body if isinstance(s, ast.Assign)
                and [ast.unparse(t) for t in s.targets] == ["model_capabilities"])
    assert "'prefix_resume': prefix_resume(tail_slots(4))" in ast.unparse(caps.value)
    assert "'supports_prefix_caching': True" in ast.unparse(caps.value)


def _contwarm():
    spec = importlib.util.spec_from_file_location("contwarm_under_test", ROOT / "tt" / "contwarm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # stdlib only: no ttnn
    return mod


def test_new_tt_source_changes_the_warm_marker_key(tmp_path):
    cw = _contwarm()
    (tmp_path / "tt").mkdir()
    (tmp_path / "resident_m5_11").mkdir()
    (tmp_path / "tt" / "package_boot_s3pb_ring.py").write_text("x = 1\n")
    (tmp_path / "resident_m5_11" / "engine.py").write_text("y = 2\n")
    before = cw.code_hash("resident_m5_11", pkg_dir=tmp_path)
    (tmp_path / "tt" / "package_boot_s3pb_ring_apc.py").write_text((ROOT / "tt" / "package_boot_s3pb_ring_apc.py").read_text())
    assert cw.code_hash("resident_m5_11", pkg_dir=tmp_path) != before


def test_bundle_names_the_packaged_class():
    meta = json.loads((ROOT / "vllm_models" / "resident_multi_s3pb_ring_apc" / "vllm_metadata.json").read_text())
    assert meta == {"arch": "Gemma4QB2ResidentMultiS32PBRingAPCForCausalLM",
                    "main_class": "models.demos.gemma4_31b_qb2.tt.package_boot_s3pb_ring_apc:Gemma4ResidentMultiS32PBRingAPCPackaged"}
