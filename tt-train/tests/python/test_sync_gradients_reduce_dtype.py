# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.sync_gradients(..., reduce_dtype=...)``: unset or bf16 sums as before, fp32 sums in fp32; without the
argument, the training config's ``precision.reduce_dtype`` decides.

The two devices of a ``[1, 2]`` mesh hold different bf16 gradients, as data-parallel ranks do. The C++ tests in
``tests/core/synchronize_tensor_test.cpp`` cover the fp32 routes themselves; these check the Python entry point.
"""

import numpy as np
import pytest
import ttnn
import ttml

from two_device_gradients import exact_mean, make_gradients, per_device, per_device_tensor

_TWO_DEVICES = (1, 2)
_DP_AXIS = 1


def _parameters_with_gradients(values: np.ndarray):
    """One parameter whose gradient on device i is ``values[i]``."""
    parameter = per_device_tensor(values)
    parameter.set_grad(per_device_tensor(values).get_value())
    parameters = ttml.NamedParameters()
    parameters["parameter"] = parameter
    return parameters, parameter


@pytest.fixture(scope="module")
def dp_mesh(fresh_device_mesh):
    with fresh_device_mesh(_TWO_DEVICES, ("_0", "dp"), what="sync_gradients reduce_dtype") as mesh:
        yield mesh


@pytest.mark.requires_device
@pytest.mark.parametrize("reduce_dtype", [None, ttnn.DataType.BFLOAT16])
def test_unset_and_bf16_reduce_dtype_sum_as_before(dp_mesh, reduce_dtype):
    values = make_gradients(64, 128, seed=1)
    parameters, parameter = _parameters_with_gradients(values)
    before = ttnn.multiply(ttml.core.distributed.all_reduce(parameter.get_grad(), cluster_axis=_DP_AXIS), 0.5)

    ttml.sync_gradients(parameters, axis_names=("dp",), reduce_dtype=reduce_dtype)

    np.testing.assert_array_equal(per_device(parameter.get_grad()), per_device(before))


# Phase 1 of the PR 4 measurements: 99.5% exact for the fp32 sum and 86% for the bf16 sum with two devices.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_is_closer_to_the_exact_mean(dp_mesh):
    values = make_gradients(64, 128, seed=2)
    expected = exact_mean(values)
    fp32_parameters, fp32_parameter = _parameters_with_gradients(values)
    bf16_parameters, bf16_parameter = _parameters_with_gradients(values)

    ttml.sync_gradients(fp32_parameters, axis_names=("dp",), reduce_dtype=ttnn.DataType.FLOAT32)
    ttml.sync_gradients(bf16_parameters, axis_names=("dp",))

    assert fp32_parameter.get_grad().dtype == ttnn.DataType.BFLOAT16
    fp32 = per_device(fp32_parameter.get_grad())
    np.testing.assert_array_equal(fp32[0], fp32[1])
    fp32_share = np.mean(fp32[0] == expected[0])
    bf16_share = np.mean(per_device(bf16_parameter.get_grad())[0] == expected[0])
    assert fp32_share >= 0.99
    assert fp32_share > bf16_share


@pytest.mark.requires_device
def test_fp32_reduce_dtype_on_an_axis_with_one_device_leaves_gradients_unchanged(dp_mesh):
    values = make_gradients(32, 64, seed=3)
    parameters, parameter = _parameters_with_gradients(values)
    before = per_device(parameter.get_grad())

    ttml.sync_gradients(parameters, axis_names=("_0",), reduce_dtype=ttnn.DataType.FLOAT32)

    assert parameter.get_grad().dtype == ttnn.DataType.BFLOAT16
    np.testing.assert_array_equal(per_device(parameter.get_grad()), before)


def test_rejects_reduce_dtypes_other_than_bf16_and_fp32(expect_error):
    with expect_error(ValueError, "reduce_dtype"):
        ttml.sync_gradients(ttml.NamedParameters(), reduce_dtype=ttnn.DataType.UINT32)


@pytest.fixture
def gradient_reduce_default():
    """Sets the context's gradient sync default for one test and clears it afterwards."""
    context = ttml.autograd.AutoContext.get_instance()
    yield context.set_gradient_reduce_dtype
    context.set_gradient_reduce_dtype(None)


def _synchronized(values: np.ndarray, **reduce_dtype) -> np.ndarray:
    parameters, parameter = _parameters_with_gradients(values)
    ttml.sync_gradients(parameters, axis_names=("dp",), **reduce_dtype)
    return per_device(parameter.get_grad())


@pytest.mark.requires_device
def test_configured_reduce_dtype_applies_unless_the_call_passes_one(dp_mesh, gradient_reduce_default):
    values = make_gradients(64, 128, seed=4)
    fp32 = _synchronized(values, reduce_dtype=ttnn.DataType.FLOAT32)
    bf16 = _synchronized(values, reduce_dtype=ttnn.DataType.BFLOAT16)
    assert not np.array_equal(fp32, bf16)

    # create_optimizer applies the config this way before training starts.
    ttml.core.apply_precision_config({"precision": {"reduce_dtype": "float32"}}, ttml.NamedParameters())
    np.testing.assert_array_equal(_synchronized(values), fp32)
    np.testing.assert_array_equal(_synchronized(values, reduce_dtype=ttnn.DataType.BFLOAT16), bf16)

    gradient_reduce_default(ttnn.DataType.BFLOAT16)
    np.testing.assert_array_equal(_synchronized(values), bf16)
    np.testing.assert_array_equal(_synchronized(values, reduce_dtype=ttnn.DataType.FLOAT32), fp32)


def test_reduce_dtype_from_config_reads_both_names_and_rejects_others(expect_error):
    def read(precision):
        return ttml.core.reduce_dtype_from_config({"precision": precision})

    assert read({"reduce_dtype": "float32"}) == ttnn.DataType.FLOAT32
    assert read({"reduce_dtype": "bfloat16"}) == ttnn.DataType.BFLOAT16
    assert read({"reduce_dtype": None}) is None
    assert read({"param_dtype": "float32"}) is None
    assert ttml.core.reduce_dtype_from_config({}) is None
    with expect_error(RuntimeError, "precision.reduce_dtype must be"):
        read({"reduce_dtype": "float16"})
