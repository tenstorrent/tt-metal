# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.demos.gemma4.tt.optimization import env_weight_dtype


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
