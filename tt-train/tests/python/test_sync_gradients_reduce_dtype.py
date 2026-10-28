# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.sync_gradients(..., reduce_dtype=...)``: unset or bf16 sums as before, fp32 sums in fp32; without the
argument, the training config's ``precision.reduce_dtype`` decides.

The two devices of a ``[1, 2]`` mesh hold different bf16 gradients, as data-parallel ranks do. The C++ tests in
``tests/core/synchronize_tensor_test.cpp`` cover the fp32 sum itself; these check the Python entry point.
"""

import numpy as np
import pytest
import ttnn
import ttml

from two_device_gradients import make_gradients, per_device, per_device_tensor

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
    gradients = per_device_tensor(values).get_value()
    fp32 = per_device(
        ttml.core.distributed.synchronize_tensor(gradients, [_DP_AXIS], reduce_dtype=ttnn.DataType.FLOAT32)
    )
    bf16 = per_device(ttml.core.distributed.synchronize_tensor(gradients, [_DP_AXIS]))
    assert not np.array_equal(fp32, bf16)
    np.testing.assert_array_equal(_synchronized(values, reduce_dtype=ttnn.DataType.FLOAT32), fp32)

    # create_optimizer applies the config this way before training starts.
    ttml.core.apply_precision_config({"precision": {"reduce_dtype": "float32"}}, ttml.NamedParameters())
    np.testing.assert_array_equal(_synchronized(values), fp32)
    np.testing.assert_array_equal(_synchronized(values, reduce_dtype=ttnn.DataType.BFLOAT16), bf16)

    gradient_reduce_default(ttnn.DataType.BFLOAT16)
    np.testing.assert_array_equal(_synchronized(values), bf16)
    np.testing.assert_array_equal(_synchronized(values, reduce_dtype=ttnn.DataType.FLOAT32), fp32)


# The C++ ParameterDtypeTest covers every name; this checks that the binding converts both kinds of result.
def test_reduce_dtype_from_config_returns_the_dtype_or_none():
    assert ttml.core.reduce_dtype_from_config({"precision": {"reduce_dtype": "float32"}}) == ttnn.DataType.FLOAT32
    assert ttml.core.reduce_dtype_from_config({}) is None
