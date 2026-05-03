# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.gemma4.tt.optimization import env_weight_dtype, precision_profile_name, profile_weight_dtype


def test_env_weight_dtype_default_has_no_cache_suffix(monkeypatch):
    monkeypatch.delenv("GEMMA4_EXPERT_WEIGHT_DTYPE", raising=False)

    dtype, suffix = env_weight_dtype("GEMMA4_EXPERT_WEIGHT_DTYPE", ttnn.bfloat16)

    assert dtype == ttnn.bfloat16
    assert suffix == ""


@pytest.mark.parametrize(
    "value, expected_dtype, expected_suffix",
    [
        ("bfp8", ttnn.bfloat8_b, "_bfp8"),
        ("bfloat8_b", ttnn.bfloat8_b, "_bfp8"),
        ("bfp4", ttnn.bfloat4_b, "_bfp4"),
        ("bf16", ttnn.bfloat16, "_bf16"),
    ],
)
def test_env_weight_dtype_overrides_are_cache_safe(monkeypatch, value, expected_dtype, expected_suffix):
    monkeypatch.setenv("GEMMA4_EXPERT_WEIGHT_DTYPE", value)

    dtype, suffix = env_weight_dtype("GEMMA4_EXPERT_WEIGHT_DTYPE", ttnn.bfloat16)

    assert dtype == expected_dtype
    assert suffix == expected_suffix


def test_env_weight_dtype_rejects_unknown_dtype(monkeypatch):
    monkeypatch.setenv("GEMMA4_EXPERT_WEIGHT_DTYPE", "int2")

    with pytest.raises(ValueError):
        env_weight_dtype("GEMMA4_EXPERT_WEIGHT_DTYPE", ttnn.bfloat16)


def test_precision_profile_default_is_mixed_bfp8(monkeypatch):
    monkeypatch.delenv("GEMMA4_PRECISION_PROFILE", raising=False)
    monkeypatch.delenv("GEMMA4_EXPERT_GATE_WEIGHT_DTYPE", raising=False)
    monkeypatch.delenv("GEMMA4_EXPERT_WEIGHT_DTYPE", raising=False)

    assert precision_profile_name() == "mixed_bfp8"

    expert_choice = profile_weight_dtype(
        "expert_gate",
        env_name="GEMMA4_EXPERT_GATE_WEIGHT_DTYPE",
        legacy_env_name="GEMMA4_EXPERT_WEIGHT_DTYPE",
    )
    lm_head_choice = profile_weight_dtype("lm_head", env_name="GEMMA4_LM_HEAD_WEIGHT_DTYPE")

    assert expert_choice.dtype == ttnn.bfloat8_b
    assert expert_choice.cache_suffix == "_bfp8"
    assert expert_choice.source == "profile:mixed_bfp8"
    assert lm_head_choice.dtype == ttnn.bfloat16
    assert lm_head_choice.cache_suffix == ""


def test_precision_profile_bf16_preserves_existing_cache_names(monkeypatch):
    monkeypatch.setenv("GEMMA4_PRECISION_PROFILE", "bf16")
    monkeypatch.delenv("GEMMA4_SHARED_MLP_DOWN_WEIGHT_DTYPE", raising=False)
    monkeypatch.delenv("GEMMA4_SHARED_MLP_WEIGHT_DTYPE", raising=False)

    choice = profile_weight_dtype(
        "shared_mlp_down",
        env_name="GEMMA4_SHARED_MLP_DOWN_WEIGHT_DTYPE",
        legacy_env_name="GEMMA4_SHARED_MLP_WEIGHT_DTYPE",
    )

    assert choice.dtype == ttnn.bfloat16
    assert choice.cache_suffix == ""
    assert choice.source == "profile:bf16"


def test_precision_specific_env_overrides_profile_and_suffixes_bf16(monkeypatch):
    monkeypatch.setenv("GEMMA4_PRECISION_PROFILE", "mixed_bfp8")
    monkeypatch.setenv("GEMMA4_ATTENTION_QKV_WEIGHT_DTYPE", "bf16")

    choice = profile_weight_dtype(
        "attention_qkv",
        env_name="GEMMA4_ATTENTION_QKV_WEIGHT_DTYPE",
        legacy_env_name="GEMMA4_ATTENTION_WEIGHT_DTYPE",
    )

    assert choice.dtype == ttnn.bfloat16
    assert choice.cache_suffix == "_bf16"
    assert choice.source == "GEMMA4_ATTENTION_QKV_WEIGHT_DTYPE"


def test_precision_legacy_env_still_controls_tensor_group(monkeypatch):
    monkeypatch.setenv("GEMMA4_PRECISION_PROFILE", "mixed_bfp8")
    monkeypatch.delenv("GEMMA4_EXPERT_DOWN_WEIGHT_DTYPE", raising=False)
    monkeypatch.setenv("GEMMA4_EXPERT_WEIGHT_DTYPE", "bfp4")

    choice = profile_weight_dtype(
        "expert_down",
        env_name="GEMMA4_EXPERT_DOWN_WEIGHT_DTYPE",
        legacy_env_name="GEMMA4_EXPERT_WEIGHT_DTYPE",
    )

    assert choice.dtype == ttnn.bfloat4_b
    assert choice.cache_suffix == "_bfp4"
    assert choice.source == "GEMMA4_EXPERT_WEIGHT_DTYPE"


def test_precision_profile_rejects_unknown_profile(monkeypatch):
    monkeypatch.setenv("GEMMA4_PRECISION_PROFILE", "int2")

    with pytest.raises(ValueError):
        profile_weight_dtype("expert_gate")
