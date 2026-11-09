# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""QWEN36_MLP_SWIGLU_TO_SHARD host checks (no device): the split-SwiGLU writes the dsmc down projection's activation
shard only when _down_decode will really take the dsmc pipeline on w2_dec."""
from types import SimpleNamespace

import pytest

from models.demos.blackhole.qwen36.tt import mlp as mlp_mod
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import weight_prefetch

ACT = object()


def _self(w2_dec=True, act=ACT):
    w = SimpleNamespace(w2=object(), w2_dec=object() if w2_dec else None)
    return SimpleNamespace(weights=w, args=SimpleNamespace(mlp_w2_dsmc_act_shard_bfp4=act))


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(tpc, "verify_out_res40_enabled", lambda: True)
    monkeypatch.setattr(tpc, "verify_out_res40_down_mode", lambda: "auto")
    monkeypatch.setattr(weight_prefetch, "is_registered", lambda w: False)
    return monkeypatch


def take(s, ar=object()):
    return mlp_mod.Qwen36MLP._down_takes_dsmc_act(s, ar)


def test_off_by_default(env):
    env.delenv(mlp_mod.SWIGLU_TO_SHARD_ENV, raising=False)
    assert take(_self()) is False


def test_on_with_dsmc_down(env):
    env.setenv(mlp_mod.SWIGLU_TO_SHARD_ENV, "1")
    assert take(_self()) is True
    assert take(_self(), ar=None) is True  # no decode all-reduce: _down_decode still runs dsmc on w2_dec


def test_never_without_decode_twin_or_shard(env):
    env.setenv(mlp_mod.SWIGLU_TO_SHARD_ENV, "1")
    assert take(_self(w2_dec=False)) is False
    assert take(_self(act=None)) is False


def test_never_for_res40_1d_down(env):
    env.setenv(mlp_mod.SWIGLU_TO_SHARD_ENV, "1")
    env.setattr(tpc, "verify_out_res40_down_mode", lambda: "1d")
    assert take(_self()) is False
    # a prefetch-registered w2 bypasses the RES40 choice in _down_decode; with a decode twin it still runs dsmc
    env.setattr(weight_prefetch, "is_registered", lambda w: True)
    assert take(_self()) is True
