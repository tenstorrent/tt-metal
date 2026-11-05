# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""Nightly-only parametrizations of the sanity tests in
tests/ttnn/unit_tests/operations/fused/test_batch_norm.py (#62486)."""

import pytest

import tests.ttnn.unit_tests.operations.fused.test_batch_norm as base

pytestmark = pytest.mark.use_module_device

_BATCH_NORM_TESTS_EPS_MOMENTUM_NIGHTLY_ONLY = [
    p for p in base.BATCH_NORM_TESTS_EPS_MOMENTUM if p not in base.BATCH_NORM_TESTS_SANITY_EPS_MOMENTUM
]
_BATCH_NORM_EPS_MOMENTUM_NIGHTLY_ONLY = [
    p for p in base.BATCH_NORM_EPS_MOMENTUM if p not in base.BATCH_NORM_SANITY_EPS_MOMENTUM
]


@pytest.mark.parametrize("input_shapes", base.BATCH_NORM_TESTS_SHAPES)
@pytest.mark.parametrize("training, check_mean, check_var", base.BATCH_NORM_STAT_MODES)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps, momentum", _BATCH_NORM_TESTS_EPS_MOMENTUM_NIGHTLY_ONLY)
@pytest.mark.parametrize("testing_dtype", ["float32", "bfloat16"])
def test_batch_norm_tests(
    input_shapes, check_mean, check_var, weight, bias, eps, device, momentum, training, testing_dtype
):
    base.test_batch_norm_tests(
        input_shapes, check_mean, check_var, weight, bias, eps, device, momentum, training, testing_dtype
    )


@pytest.mark.parametrize("input_shapes", base.BATCH_NORM_SHAPES)
@pytest.mark.parametrize("training, check_mean, check_var", base.BATCH_NORM_STAT_MODES)
@pytest.mark.parametrize("weight", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("eps, momentum", _BATCH_NORM_EPS_MOMENTUM_NIGHTLY_ONLY)
def test_batch_norm(input_shapes, training, check_mean, check_var, weight, bias, eps, momentum, device):
    base.test_batch_norm(input_shapes, training, check_mean, check_var, weight, bias, eps, momentum, device)
