# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Off-device unit tests for the shared pointwise accuracy metrics."""

import numpy as np
import pytest
from helpers.accuracy_metrics import compute_pointwise_metrics, local_ulp
from helpers.format_config import DataFormat


def test_local_ulp_bf16_is_positive_for_normal_values():
    golden = np.array([1.0, 2.0, 100.0], dtype=np.float64)
    ulp = local_ulp(golden, DataFormat.Float16_b)
    assert np.all(ulp > 0)
    # bfloat16 has 8 mantissa bits: ULP(1.0) == 2**-7.
    assert ulp[0] == pytest.approx(2.0**-7, rel=1e-6)


def test_local_ulp_block_float_is_nan():
    golden = np.array([1.0, 2.0], dtype=np.float64)
    ulp = local_ulp(golden, DataFormat.Bfp8_b)
    assert np.all(np.isnan(ulp))


def test_metrics_basic_errors():
    x = np.array([0.5, 1.0], dtype=np.float64)
    golden = np.array([2.0, 4.0], dtype=np.float64)
    hw = np.array([2.5, 3.0], dtype=np.float64)
    m = compute_pointwise_metrics(x, golden, hw, DataFormat.Float16_b)
    np.testing.assert_allclose(m["signed_error"], [0.5, -1.0])
    np.testing.assert_allclose(m["abs_error"], [0.5, 1.0])
    np.testing.assert_allclose(m["rel_error"], [0.25, 0.25])
    # signed_ulp_error == signed_error / local_ulp(golden)
    expected_ulp = np.array([0.5, -1.0]) / local_ulp(golden, DataFormat.Float16_b)
    np.testing.assert_allclose(m["signed_ulp_error"], expected_ulp)
    np.testing.assert_allclose(m["abs_ulp_error"], np.abs(expected_ulp))
    assert np.all(m["is_finite_golden"])
    assert np.all(m["is_finite_hw"])


def test_metrics_ulp_nan_when_golden_zero():
    x = np.array([0.0], dtype=np.float64)
    golden = np.array([0.0], dtype=np.float64)
    hw = np.array([0.0], dtype=np.float64)
    m = compute_pointwise_metrics(x, golden, hw, DataFormat.Float16_b)
    assert np.isnan(m["signed_ulp_error"][0])
    assert np.isnan(m["abs_ulp_error"][0])
    assert np.isnan(m["rel_error"][0])
    assert m["abs_error"][0] == 0.0  # correctness still captured


def test_metrics_ulp_nan_on_nonfinite():
    x = np.array([1.0, 2.0], dtype=np.float64)
    golden = np.array([np.inf, 4.0], dtype=np.float64)
    hw = np.array([4.0, np.nan], dtype=np.float64)
    m = compute_pointwise_metrics(x, golden, hw, DataFormat.Float16_b)
    assert np.all(np.isnan(m["signed_ulp_error"]))
    assert m["is_finite_golden"].tolist() == [False, True]
    assert m["is_finite_hw"].tolist() == [True, False]
    # abs/signed error are still computed (not gated to NaN) for non-finite inputs.
    assert m["signed_error"][0] == -np.inf  # 4.0 - inf
    assert np.isnan(m["signed_error"][1])  # nan - 4.0


def test_metrics_block_float_ulp_all_nan_but_abs_populated():
    x = np.array([1.0], dtype=np.float64)
    golden = np.array([2.0], dtype=np.float64)
    hw = np.array([2.5], dtype=np.float64)
    m = compute_pointwise_metrics(x, golden, hw, DataFormat.Bfp8_b)
    assert np.isnan(m["signed_ulp_error"][0])
    assert m["abs_error"][0] == pytest.approx(0.5)
