# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks of the weight-cache marker (ModelArgs.weight_cache_is_complete / mark_weight_cache_complete): the
settings that choose the rounded weights are recorded per build kind, so a change of them (another pinned revision,
say) makes the next build load the HF weights, while builds that read none of those weights are not affected. The
GPTQ dense weights are read by every build (prefill and decode)."""

import types

import pytest

from models.demos.gpt_oss.tt.model_config import ModelArgs

ROUND_DIR = "GPT_OSS_DECODE_SPLIT_ROUND_DIR"
DENSE = "GPT_OSS_DENSE_ROUNDED_DECODE"
KGPTQ = "GPT_OSS_PREFILL_KGPTQ"


def _args(tmp_path, routed=False):
    """A stand-in carrying what the two marker methods use; routed: the routed prefill experts are built."""
    a = types.SimpleNamespace(model_name="gpt-oss-20b", n_layers=24)
    for name in (
        "WEIGHT_CACHE_MARKER",
        "WEIGHT_CACHE_FORMAT_VERSION",
        "FUSED_DECODE_WEIGHTS_VERSION",
        "DENSE_MOE_DECODE_WEIGHTS_VERSION",
        "DENSE_SETTING_VARS",
        "FUSED_DECODE_SETTING_VARS",
        "ROUTED_PREFILL_SETTING_VARS",
    ):
        setattr(a, name, getattr(ModelArgs, name))
    a._settings = ModelArgs._settings
    a._routed_decode_bf8_layers = ModelArgs._routed_decode_bf8_layers
    a.weight_cache_path = lambda dtype: tmp_path
    a._routed_prefill_weights_version = lambda: "1" if routed else None
    a.weight_cache_is_complete = types.MethodType(ModelArgs.weight_cache_is_complete, a)
    a.mark_weight_cache_complete = types.MethodType(ModelArgs.mark_weight_cache_complete, a)
    (tmp_path / "w.tensorbin").write_bytes(b"")  # the marker also needs a tensor file next to it
    return a


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in (
        set(ModelArgs.DENSE_SETTING_VARS)
        | set(ModelArgs.FUSED_DECODE_SETTING_VARS)
        | set(ModelArgs.ROUTED_PREFILL_SETTING_VARS)
    ):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("GPT_OSS_FORCE_MODEL_LOAD", raising=False)


def test_fused_settings_change_invalidates(tmp_path, monkeypatch):
    monkeypatch.setenv(ROUND_DIR, "hf://o/r@rev1/kgptq")
    a = _args(tmp_path)
    a.mark_weight_cache_complete("bf8", fused_decode=True)
    assert a.weight_cache_is_complete("bf8", fused_decode=True)
    monkeypatch.setenv(ROUND_DIR, "hf://o/r@rev2/kgptq")
    assert not a.weight_cache_is_complete("bf8", fused_decode=True)
    # A build without the fused decode (or the routed prefill experts) reads none of these weights: still complete.
    assert a.weight_cache_is_complete("bf8", fused_decode=False)


def test_dense_settings_checked_for_every_build(tmp_path, monkeypatch):
    """Prefill and decode both read the GPTQ dense weights, so a change of them invalidates every build."""
    monkeypatch.setenv(DENSE, "hf://o/r@rev1/dense.pt")
    a = _args(tmp_path)
    a.mark_weight_cache_complete("bf8", fused_decode=False, dense_moe=True)
    assert a.weight_cache_is_complete("bf8", dense_moe=True)
    monkeypatch.setenv(DENSE, "hf://o/r@rev2/dense.pt")
    assert not a.weight_cache_is_complete("bf8", dense_moe=True)
    monkeypatch.delenv(DENSE)
    assert not a.weight_cache_is_complete("bf8", dense_moe=True)


def test_non_fused_build_keeps_fused_settings(tmp_path, monkeypatch):
    monkeypatch.setenv(ROUND_DIR, "hf://o/r@rev1/kgptq")
    a = _args(tmp_path)
    a.mark_weight_cache_complete("bf8", fused_decode=True)
    a.mark_weight_cache_complete("bf8", fused_decode=False, dense_moe=True)  # e.g. a batch-32 build afterwards
    assert a.weight_cache_is_complete("bf8", fused_decode=True)
    assert a.weight_cache_is_complete("bf8", dense_moe=True)


def test_routed_settings_checked_only_where_routed(tmp_path, monkeypatch):
    monkeypatch.setenv(KGPTQ, "1")
    monkeypatch.setenv(ROUND_DIR, "hf://o/r@rev1/kgptq")
    routed = _args(tmp_path, routed=True)
    routed.mark_weight_cache_complete("bf8")
    assert routed.weight_cache_is_complete("bf8")
    monkeypatch.setenv(ROUND_DIR, "hf://o/r@rev2/kgptq")
    assert not routed.weight_cache_is_complete("bf8")
    assert _args(tmp_path, routed=False).weight_cache_is_complete("bf8")


def test_old_marker_without_settings(tmp_path, monkeypatch):
    """A marker written before the settings were recorded: complete unless a setting is now set for that build."""
    a = _args(tmp_path)
    a.mark_weight_cache_complete("bf8", fused_decode=True)
    assert a.weight_cache_is_complete("bf8", fused_decode=True)
    monkeypatch.setenv(DENSE, "hf://o/r@rev1/dense.pt")
    assert not a.weight_cache_is_complete("bf8", fused_decode=True)
