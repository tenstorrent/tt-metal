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
