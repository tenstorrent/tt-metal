# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Host test of the update-5 `single-user` packaged class (tt/package_boot_pb4_apc.py), no device and no ttnn: the
sources are parsed, not imported (importing PB3 imports ttnn), as update 4's test_package_boot_pb4.py. The class
composes the S1 single-user class (on PB3) with update 4's packaged PB4 class by cooperative inheritance, S1 first:
- its MRO (C3, computed on stand-in classes built from the parsed bases) is PB3-APC -> PB4Packaged -> PB3Packaged ->
  PB3;
- it defines nothing itself, so the package boot (initialize_vllm_model: ensure_resident_cache) resolves to
  PB3Packaged, ENGINE_PACKAGE and the continuation pre-warm (warmup_model_prefill, _cont_warm) to PB4Packaged, and the
  capabilities, pools and request path to the S1 class;
- the inherited pre-warm calls the PB generator directly, with no tail orders;
- the S1 capability is prefix_resume(tail_slots(2)) with req_ids False; the S1 class starts with `live` False;
- contwarm's code hash covers every tt/ source, so adding this file changes the marker key (the first start re-warms);
- the bundle names the class.
    pytest -q models/demos/gemma4_31b_qb2/tests/test_package_boot_pb4_apc.py"""

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "package_boot_pb4_apc.py": "Gemma4ResidentSpecPB4APCPackaged",
    "generator_resident_spec_vllm_pb3_apc.py": "Gemma4ResidentSpecPB3APCForCausalLM",
    "package_boot_pb4.py": "Gemma4ResidentSpecPB4Packaged",
    "package_boot_pb3.py": "Gemma4ResidentSpecPB3Packaged",
}
LEAF = "Gemma4ResidentSpecPB3ForCausalLM"  # PB3: not parsed, a stand-in leaf
CLS = "Gemma4ResidentSpecPB4APCPackaged"
APC, PB4, PB3P = ("Gemma4ResidentSpecPB3APCForCausalLM", "Gemma4ResidentSpecPB4Packaged",
                  "Gemma4ResidentSpecPB3Packaged")


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
    tree = _tree("package_boot_pb4_apc.py")
    imports = {(n.module, a.name) for n in tree.body if isinstance(n, ast.ImportFrom) for a in n.names}
    assert imports == {
        ("models.demos.gemma4_31b_qb2.tt.generator_resident_spec_vllm_pb3_apc", APC),
        ("models.demos.gemma4_31b_qb2.tt.package_boot_pb4", PB4),
    }
    cls = _classes()[CLS]
    assert [ast.unparse(b) for b in cls.bases] == [APC, PB4]


def test_c3_mro():
    assert _mro() == [CLS, APC, PB4, PB3P, LEAF]


def test_inherited_not_copied():
    cls = _classes()[CLS]
    body = [s for s in cls.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    assert body == [], "the composition must define nothing (docstring only)"
    assert _members(cls) == set()


def test_resolution():
    assert _resolves_to("initialize_vllm_model") == PB3P  # ensure_resident_cache, then PB3's construction
    assert _resolves_to("ENGINE_PACKAGE") == PB4
    assert _resolves_to("warmup_model_prefill") == PB4
    assert _resolves_to("_cont_warm") == PB4
    assert _resolves_to("model_capabilities") == APC
    assert _resolves_to("get_kv_cache_spec") == APC
    assert _resolves_to("allocate_pools") == APC  # the tail pool, before warmup_model (the plugin's order)
    assert _resolves_to("prefill_forward") == APC
    assert _resolves_to("release_request") == APC


def test_cooperative_init_chain_and_live():
    """the S1 class's and PB4's __init__ both call super().__init__; the S1 class sets `live` False after it"""
    cdefs = _classes()
    for name in (APC, PB4):
        init = next(f for f in cdefs[name].body if isinstance(f, ast.FunctionDef) and f.name == "__init__")
        assert "super().__init__(*args, **kwargs)" in ast.unparse(init), name
    init = ast.unparse(next(f for f in cdefs[APC].body if isinstance(f, ast.FunctionDef) and f.name == "__init__"))
    assert init.index("super().__init__(*args, **kwargs)") < init.index("self.live = False")
    boot = next(f for f in cdefs[PB3P].body if isinstance(f, ast.FunctionDef) and f.name == "initialize_vllm_model")
    assert [ast.unparse(s) for s in boot.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))] == [
        "ensure_resident_cache()",
        "return super().initialize_vllm_model(hf_config, mesh_device, max_batch_size, **kwargs)"]


def test_prewarm_bypasses_the_adapter():
    """the inherited _cont_warm calls the PB generator (gen.prefill_forward), never the adapter, with no tail orders"""
    warm = next(f for f in _classes()[PB4].body if isinstance(f, ast.FunctionDef) and f.name == "_cont_warm")
    calls = [c for c in ast.walk(warm) if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("prefill_forward")]
    assert calls and all(ast.unparse(c.func) == "gen.prefill_forward" for c in calls)
    assert all(not {k.arg for k in c.keywords} & {"req_ids", "tail_orders"} for c in calls)
    assert "self._reset_request()" in ast.unparse(warm)


def test_capability_defaults():
    caps = next(s for s in _classes()[APC].body if isinstance(s, ast.Assign)
                and [ast.unparse(t) for t in s.targets] == ["model_capabilities"])
    src = ast.unparse(caps.value)
    assert "'prefix_resume': {**prefix_resume(tail_slots(2)), 'req_ids': False}" in src
    assert "'supports_prefix_caching': True" in src


def _contwarm():
    spec = importlib.util.spec_from_file_location("contwarm_under_test", ROOT / "tt" / "contwarm.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # stdlib only: no ttnn
    return mod


def test_new_tt_source_changes_the_warm_marker_key(tmp_path):
    cw = _contwarm()
    (tmp_path / "tt").mkdir()
    (tmp_path / "resident_m3_2").mkdir()
    (tmp_path / "tt" / "package_boot_pb4.py").write_text("x = 1\n")
    (tmp_path / "resident_m3_2" / "engine.py").write_text("y = 2\n")
    before = cw.code_hash("resident_m3_2", pkg_dir=tmp_path)
    (tmp_path / "tt" / "package_boot_pb4_apc.py").write_text((ROOT / "tt" / "package_boot_pb4_apc.py").read_text())
    assert cw.code_hash("resident_m3_2", pkg_dir=tmp_path) != before


def test_bundle_names_the_packaged_class():
    meta = json.loads((ROOT / "vllm_models" / "resident_spec_pb4_apc" / "vllm_metadata.json").read_text())
    assert meta == {"arch": "Gemma4QB2ResidentSpecPB4APCForCausalLM",
                    "main_class": "models.demos.gemma4_31b_qb2.tt.package_boot_pb4_apc:Gemma4ResidentSpecPB4APCPackaged"}
