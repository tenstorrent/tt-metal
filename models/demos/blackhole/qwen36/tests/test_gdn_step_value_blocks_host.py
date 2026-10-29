# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the QWEN36_GDN_STEP_VALUE_BLOCKS wiring (plain-decode gdn_decode_step value_blocks):

* the env is parsed to 1 / 2 / 4 (anything else raises);
* _forward_decode_step_op passes the op's value_blocks kwarg only when > 1 (1 = the original call, so a ttnn build
  without the attribute keeps working), next to the unchanged fast_mode / fp32_state kwargs.

The device bit-identity tests are test_gdn_decode_step.py::test_value_blocks_*.
No device, no op. Run: pytest models/demos/blackhole/qwen36/tests/test_gdn_step_value_blocks_host.py -q
"""

import types

import pytest

import ttnn
from models.demos.blackhole.qwen36.tt.gdn import tp as gdn_tp
from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, gdn_step_value_blocks


@pytest.mark.parametrize("env,want", [(None, 1), ("", 1), ("1", 1), ("2", 2), (" 4 ", 4)])
def test_value_blocks_env(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("QWEN36_GDN_STEP_VALUE_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("QWEN36_GDN_STEP_VALUE_BLOCKS", env)
    assert gdn_step_value_blocks() == want


@pytest.mark.parametrize("env", ["3", "8", "0", "-2"])
def test_value_blocks_env_rejects_other_values(monkeypatch, env):
    monkeypatch.setenv("QWEN36_GDN_STEP_VALUE_BLOCKS", env)
    with pytest.raises(ValueError, match="QWEN36_GDN_STEP_VALUE_BLOCKS"):
        gdn_step_value_blocks()


class _Shaped:
    def __init__(self, shape):
        self.shape = shape


@pytest.mark.parametrize("env,want", [(None, None), ("1", None), ("2", 2), ("4", 4)])
def test_decode_step_op_passes_value_blocks_only_when_split(monkeypatch, env, want):
    if env is None:
        monkeypatch.delenv("QWEN36_GDN_STEP_VALUE_BLOCKS", raising=False)
    else:
        monkeypatch.setenv("QWEN36_GDN_STEP_VALUE_BLOCKS", env)
    B, W = 8, 4096
    calls = []
    kda = types.SimpleNamespace(gdn_decode_step=lambda *a, **kw: calls.append((a, kw)) or _Shaped((1, B, 768)))
    monkeypatch.setattr(ttnn, "experimental", types.SimpleNamespace(kda=kda), raising=False)
    monkeypatch.setattr(ttnn, "reshape", lambda t, shape: _Shaped(shape), raising=False)
    monkeypatch.setattr(ttnn, "deallocate", lambda t: None, raising=False)
    monkeypatch.setattr(gdn_tp.tpc, "matmul_1d_decode", lambda *a, **kw: _Shaped((1, 1, B, W)))
    g = types.SimpleNamespace(
        tw={
            "qkvz": "w",
            "dt_bias_fp32": "dtb",
            "neg_exp_A_fp32": "nea",
            "norm_w_1d": "nw",
            "conv_taps_packed": "taps",
            "out": "wo",
        },
        _decode_B=B,
        B=B,
        args=types.SimpleNamespace(gdn_qkvz_decode_1d_progcfg=None),
        cfg_in=None,
        rec_state="state",
        Nv=6,
        Nk=2,
        Dk=128,
        Dv=128,
        scale=128**-0.5,
        _conv_hist_packed="hist",
        qkvz_dim_tp=2048,
        _decode_step_fp32_state=True,
        _row_proj=lambda gated, w: _Shaped((1, B, 5120)),
    )
    decode_ar = types.SimpleNamespace(all_reduce=lambda partial, keep_fp32: "reduced")
    assert TPGatedDeltaNet._forward_decode_step_op(g, _Shaped((1, 1, B, 5120)), decode_ar=decode_ar) == "reduced"
    ((args, kw),) = calls
    assert args[1:5] == ("dtb", "nea", "state", "nw")
    assert kw.get("value_blocks") == want
    assert kw["fast_mode"] is True and kw["fp32_state"] is True and kw["conv_hist"] == "hist"
