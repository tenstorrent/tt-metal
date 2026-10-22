# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU guard of the release defaults: paged prefill (DSV41_PF_PAGED=read, fp8 pool kept), automatic prefix caching on in the shipped
262,144 / 524,288 / 1M specs, trace region of the B=4 specs. No device.

    pytest --noconftest models/demos/blackhole/deepseek_v41_flash/tests/test_release_defaults_cpu.py
"""

import glob
import json
import os
import subprocess
import sys

import pytest

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(PKG))))
ASSETS = os.path.join(PKG, "assets")
DEFAULT_SPECS = [
    "runtime_spec_b16_k5_262144_vision.json",
    "runtime_spec_b4_k5_1048576.json",
    "runtime_spec_b4_k5_1048576_vision.json",
]
APC_SPECS = sorted(glob.glob(os.path.join(ASSETS, "runtime_spec_*_apc.json")))


def _py(code, **env):
    e = {k: v for k, v in os.environ.items() if not k.startswith("DSV41_PF_PAGED")}
    e.update(env)
    e["PYTHONPATH"] = REPO + os.pathsep + e.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, cwd=REPO, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip().splitlines()[-1]


MODE = "from models.demos.blackhole.deepseek_v41_flash.tt import prefill_paged as P; print(P.MODE, P.read_on(), P.write_on())"
POOL = (
    "import ttnn; from models.demos.blackhole.deepseek_v41_flash.tt import common; "
    "print(str(common._env_setup(ttnn.bfloat8_b)).lower())"
)


def test_default_mode_is_read():
    assert _py(MODE) == "read True True"


def test_mode_zero_restores_release_path():
    assert _py(MODE, DSV41_PF_PAGED="0") == "0 False False"
    assert _py(MODE, DSV41_PF_PAGED="write") == "write False True"


def test_default_read_mode_forces_bf16_pool():
    assert "bfloat16" in _py(POOL, DSV41_POOL_DTYPE="fp8")


def test_fp8_one_keeps_fp8_pool_in_read_mode():
    assert "fp8_e4m3" in _py(POOL, DSV41_POOL_DTYPE="fp8", DSV41_PF_PAGED_FP8="1")


def test_mode_zero_keeps_fp8_pool():  # the published image: dense tables + fp8 pool
    assert "fp8_e4m3" in _py(POOL, DSV41_POOL_DTYPE="fp8", DSV41_PF_PAGED="0")


def _spec(name):
    r = json.load(open(os.path.join(ASSETS, name)))["runtime_model_spec"]
    d = r["device_model_spec"]
    return r, d, d["vllm_args"], (r["env_vars"], d["env_vars"])


@pytest.mark.parametrize(
    "name", DEFAULT_SPECS + [os.path.basename(p) for p in APC_SPECS] + ["runtime_spec_b4_k5_786432_vision.json"]
)
def test_shipped_spec_defaults(name):
    r, d, v, envs = _spec(name)
    assert v.get("enable-prefix-caching") is True and "no-enable-prefix-caching" not in v
    for e in envs:
        assert e["DSV41_APC"] == "1"
        assert e["DSV41_PF_PAGED"] == "read" and "DSV41_PF_PAGED_FP8" not in e  # bf16 pool: APC needs it
        assert e["DSV41_POOL_DTYPE"] == "fp8"
    if d["max_concurrency"] == 4:  # read mode + fp8 pool at B=4 / 1M needs >= ~766 MB of trace region (780 MB boots)
        tr = 780000000
        assert json.loads(v["additional_config"])["tt"]["trace_region_size"] == tr
        assert d["override_tt_config"]["trace_region_size"] == r["override_tt_config"]["trace_region_size"] == tr
        assert all(int(e["DSV41_TRACE_REGION"]) == tr for e in envs)


@pytest.mark.parametrize("name", DEFAULT_SPECS)
def test_default_specs_budget_is_one_key(name):
    """the token budget of the default specs is the single key vllm_args.max_num_batched_tokens (not decided here)."""
    text = open(os.path.join(ASSETS, name)).read()
    assert text.count('"max_num_batched_tokens"') == 1


def test_131072_specs_stay_on_the_old_path():
    """RELEASE_NOTES: APC must not be enabled with the 131,072-token specs; they are pinned to the dense-table prefill."""
    for name in ("runtime_spec_b16_k5_131072.json", "runtime_spec_b16_k5_131072_vision.json"):
        _, _, v, envs = _spec(name)
        assert v.get("no-enable-prefix-caching") is True
        for e in envs:
            assert e["DSV41_APC"] == "0" and e["DSV41_PF_PAGED"] == "0"


APC = (
    "import os; os.environ.setdefault('DSV41_VLLM_INTERLEAVE', '1'); "
    "import sys; sys.path.insert(0, %r); "
    "from models.demos.blackhole.deepseek_v41_flash.tt import generator_vllm as G; print(G._apc_on())"
)


def _apc(**env):
    plugin = os.environ.get("DSV41_PLUGIN_SRC", "/mnt/tt-data/ssinghal/wt/plugin_apc2/src")
    return _py(APC % plugin, **env)


def test_apc_on_with_default_bf16_pool():
    assert _apc(DSV41_APC="1", DSV41_POOL_DTYPE="fp8") == "True"


def test_apc_refused_with_read_and_fp8_pool():
    assert _apc(DSV41_APC="1", DSV41_POOL_DTYPE="fp8", DSV41_PF_PAGED_FP8="1") == "False"


def test_apc_ok_when_paged_prefill_off_even_with_fp8_flag():
    assert _apc(DSV41_APC="1", DSV41_POOL_DTYPE="fp8", DSV41_PF_PAGED="0", DSV41_PF_PAGED_FP8="1") == "True"
