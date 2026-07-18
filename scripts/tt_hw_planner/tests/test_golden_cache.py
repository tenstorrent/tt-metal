"""Module-level golden cache: save/load roundtrip + the emitted PCC test still compiles.

Guards the tt_hw_planner optimize --module-level speedup: the per-component test,
under TT_PERF_MODULE_LEVEL, caches (submodule, inputs, golden) on the first run and
reuses it thereafter, skipping the full-model load. This checks the cache helper
roundtrips a real nn.Module + golden, and that the template still renders to valid
Python after wiring the cache in.
"""

from __future__ import annotations

import importlib
import py_compile
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _gc():
    return importlib.import_module("models.common.golden_cache")


def test_save_load_roundtrip_real_module_and_golden(tmp_path):
    import torch

    gc = _gc()
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    (demo / "tests" / "pcc").mkdir(parents=True)
    test_file = demo / "tests" / "pcc" / "test_mlp.py"
    test_file.write_text("def test_mlp():\n    pass\n")

    path = gc.golden_cache_path(str(test_file), "mlp", 0)
    assert "/_captured/mlp/golden_cache_s0.pt" in path

    mod = torch.nn.Linear(8, 8)
    kwargs = {"x": torch.randn(1, 8)}
    primary = ("x", kwargs["x"])
    golden = mod(kwargs["x"])

    assert gc.load_golden_cache(path) is None  # cold miss
    assert gc.save_golden_cache(path, mod, kwargs, primary, golden) is True
    hit = gc.load_golden_cache(path)
    assert hit is not None
    m2, k2, p2, g2 = hit
    assert isinstance(m2, torch.nn.Linear)
    assert torch.allclose(g2, golden)
    assert torch.allclose(m2(k2["x"]), golden)  # reconstructed module reproduces the golden


def test_load_miss_and_corrupt_return_none(tmp_path):
    gc = _gc()
    assert gc.load_golden_cache(str(tmp_path / "nope.pt")) is None
    bad = tmp_path / "bad.pt"
    bad.write_text("not a torch file")
    assert gc.load_golden_cache(str(bad)) is None


def test_emitted_pcc_test_still_compiles(tmp_path):
    bl = importlib.import_module("scripts.tt_hw_planner.bringup_loop")
    demo = tmp_path / "models" / "demos" / "hf_eager" / "m"
    (demo / "_stubs").mkdir(parents=True)
    (demo / "_stubs" / "mlp.py").write_text("import ttnn\n")
    test_path, generated, _ = bl._emit_pcc_template(
        demo_dir=demo,
        component_name="mlp",
        model_id="org/m",
        hf_reference="",
        new_shape={},
        repo_root=tmp_path,
        overwrite=True,
        discovered_submodule_path="model.mlp",
    )
    assert generated and Path(test_path).is_file()
    src = Path(test_path).read_text()
    assert "TT_PERF_MODULE_LEVEL" in src and "golden_cache" in src and "golden_cache_hit" in src
    py_compile.compile(str(test_path), doraise=True)  # emitted test is valid Python
