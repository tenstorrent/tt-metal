# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.sync_gradients(..., reduce_dtype=...)``: unset or bf16 sums as before, fp32 sums in fp32.

The two devices of a ``[1, 2]`` mesh hold different bf16 gradients, as data-parallel ranks do. The C++ tests in
``tests/core/synchronize_tensor_test.cpp`` cover the fp32 routes themselves; these check the Python entry point.
"""

import numpy as np
import pytest
import ttnn
import ttml

_TWO_DEVICES = (1, 2)
_DP_AXIS = 1


def _to_bf16(values: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even), returned as float32."""
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def _make_gradients(rows: int, cols: int, seed: int) -> np.ndarray:
    """Two devices' bf16 gradients, stacked on dim 0, scattered around a per-element mean by as much as the mean."""
    rng = np.random.default_rng(seed)
    mean = rng.choice([-1.0, 1.0], size=(1, 1, rows, cols)) * 10.0 ** rng.uniform(-3, 0, size=(1, 1, rows, cols))
    return _to_bf16((mean + np.abs(mean) * rng.standard_normal((2, 1, rows, cols))).astype(np.float32))


def _exact_mean(values: np.ndarray) -> np.ndarray:
    """bf16 of the exact mean of the two devices' values: what a perfect reduction returns."""
    return _to_bf16((values.astype(np.float64).sum(axis=0) / 2.0).astype(np.float32))


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _parameters_with_gradients(values: np.ndarray):
    """One parameter whose gradient on device i is ``values[i]``."""
    mapper = ttnn.create_mesh_mapper(
        _device(), ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementShard(0)])
    )
    gradients = ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    parameter = ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    parameter.set_grad(gradients.get_value())
    parameters = ttml.NamedParameters()
    parameters["parameter"] = parameter
    return parameters, parameter


def _per_device(tensor) -> np.ndarray:
    """``tensor`` read back with device i's copy at index i."""
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig([1, 0]))
    return ttml.autograd.create_tensor(tensor).to_numpy(ttnn.DataType.FLOAT32, composer=composer)


@pytest.fixture(scope="module")
def dp_mesh(fresh_device_mesh):
    with fresh_device_mesh(_TWO_DEVICES, ("_0", "dp"), what="sync_gradients reduce_dtype") as mesh:
        yield mesh


@pytest.mark.requires_device
@pytest.mark.parametrize("reduce_dtype", [None, ttnn.DataType.BFLOAT16])
def test_unset_and_bf16_reduce_dtype_sum_as_before(dp_mesh, reduce_dtype):
    values = _make_gradients(64, 128, seed=1)
    parameters, parameter = _parameters_with_gradients(values)
    before = ttnn.multiply(ttml.core.distributed.all_reduce(parameter.get_grad(), cluster_axis=_DP_AXIS), 0.5)

    ttml.sync_gradients(parameters, axis_names=("dp",), reduce_dtype=reduce_dtype)

    np.testing.assert_array_equal(_per_device(parameter.get_grad()), _per_device(before))


# Phase 1 of the PR 4 measurements: 99.5% exact for the fp32 sum and 86% for the bf16 sum with two devices.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_is_closer_to_the_exact_mean(dp_mesh):
    values = _make_gradients(64, 128, seed=2)
    expected = _exact_mean(values)
    fp32_parameters, fp32_parameter = _parameters_with_gradients(values)
    bf16_parameters, bf16_parameter = _parameters_with_gradients(values)

    ttml.sync_gradients(fp32_parameters, axis_names=("dp",), reduce_dtype=ttnn.DataType.FLOAT32)
    ttml.sync_gradients(bf16_parameters, axis_names=("dp",))

    assert fp32_parameter.get_grad().dtype == ttnn.DataType.BFLOAT16
    fp32 = _per_device(fp32_parameter.get_grad())
    np.testing.assert_array_equal(fp32[0], fp32[1])
    fp32_share = np.mean(fp32[0] == expected[0])
    bf16_share = np.mean(_per_device(bf16_parameter.get_grad())[0] == expected[0])
    assert fp32_share >= 0.99
    assert fp32_share > bf16_share


@pytest.mark.requires_device
def test_fp32_reduce_dtype_on_an_axis_with_one_device_leaves_gradients_unchanged(dp_mesh):
    values = _make_gradients(32, 64, seed=3)
    parameters, parameter = _parameters_with_gradients(values)
    before = _per_device(parameter.get_grad())

    ttml.sync_gradients(parameters, axis_names=("_0",), reduce_dtype=ttnn.DataType.FLOAT32)

    assert parameter.get_grad().dtype == ttnn.DataType.BFLOAT16
    np.testing.assert_array_equal(_per_device(parameter.get_grad()), before)


def test_rejects_reduce_dtypes_other_than_bf16_and_fp32(expect_error):
    with expect_error(ValueError, "reduce_dtype"):
        ttml.sync_gradients(ttml.NamedParameters(), reduce_dtype=ttnn.DataType.UINT32)
