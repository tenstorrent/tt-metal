# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU guard: the in-place decode index-key export (DSV41_KEY_EXPORT_FILL=1, see test_key_export_inplace.py) must stay the default and must not be switched off by any shipped runtime spec below 524,288
tokens (B=16 / 262,144 gave wrong greedy answers with the slice_write export).

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_key_export_default.py
"""

import glob
import json
import os
import re

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_model_default_is_fill():
    src = open(os.path.join(PKG, "tt", "dsv41_model.py")).read()
    m = re.search(r'os\.environ\.get\("DSV41_KEY_EXPORT_FILL",\s*"(\d)"\)', src)
    assert m, "DSV41_KEY_EXPORT_FILL default not found in tt/dsv41_model.py"
    assert (
        m.group(1) == "1"
    ), "the decode key export must default to the in-place fill_cache path (DSV41_KEY_EXPORT_FILL=1)"


def test_shipped_specs_do_not_disable_fill():
    specs = sorted(glob.glob(os.path.join(PKG, "assets", "runtime_spec_*.json")))
    assert specs
    for p in specs:
        r = json.load(open(p))["runtime_model_spec"]
        envs = {**r.get("env_vars", {}), **r["device_model_spec"].get("env_vars", {})}
        assert envs.get("DSV41_KEY_EXPORT_FILL", "1") != "0", f"{os.path.basename(p)} sets DSV41_KEY_EXPORT_FILL=0"


def test_no_other_default_override():
    src = open(os.path.join(PKG, "tt", "generator_vllm.py")).read()
    assert not re.search(
        r'DSV41_KEY_EXPORT_FILL"\s*[,\]]\s*"0"', src
    ), "the vLLM adapter must not default DSV41_KEY_EXPORT_FILL to 0"


def _fill_max_fn():
    """``key_export_fill_max`` of tt/dsv41_model.py compiled on its own (the module needs ttnn to import)."""
    import ast

    src = open(os.path.join(PKG, "tt", "dsv41_model.py")).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "key_export_fill_max")
    ns = {"os": os}
    exec(compile(ast.Module([node], []), "dsv41_model.py", "exec"), ns)
    return ns["key_export_fill_max"]


def test_fill_max_default_covers_the_slab(monkeypatch):
    f = _fill_max_fn()
    monkeypatch.delenv("DSV41_KEY_EXPORT_FILL_MAX", raising=False)
    assert f(262176) == 524288  # 262,144-token build: unchanged
    assert f(524288) == 524288
    assert f(1048640) == 1048640  # 1M build: ratio-1 slab of 1,048,640 rows, every piece in place
    monkeypatch.setenv("DSV41_KEY_EXPORT_FILL_MAX", "2097152")
    assert f(1048640) == 2097152  # env still overrides
    monkeypatch.setenv("DSV41_KEY_EXPORT_FILL_MAX", "262144")
    assert f(1048640) == 262144
    monkeypatch.setenv("DSV41_KEY_EXPORT_FILL_MAX", "")
    assert f(1048640) == 1048640


def test_export_uses_the_helper():
    src = open(os.path.join(PKG, "tt", "dsv41_model.py")).read()
    assert "FILL_MAX = key_export_fill_max(" in src
    assert 'os.environ.get("DSV41_KEY_EXPORT_FILL_MAX", "524288")' not in src
