# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The fp32 gradient sync (``reduce_dtype=FLOAT32``) on a 2x2 mesh with axes ``("dp", "tp")``: a sum over both axes,
a gradient the tp axis already shards, and the topology label of the result.

Every test runs with ``ttnn.CONFIG.strict_ccl_topology`` on, so a label the CCL topology helper refuses is an error.
"""

import ml_dtypes
import numpy as np
import pytest
import ttnn
import ttml

pytestmark = [pytest.mark.requires_device, pytest.mark.usefixtures("strict_ccl_topology")]

_MESH_SHAPE = (2, 2)
_DP_AXIS = 0
_TP_AXIS = 1
_FP32 = ttnn.DataType.FLOAT32


@pytest.fixture(scope="module")
def dp_tp_mesh(fresh_device_mesh):
    with fresh_device_mesh(
        _MESH_SHAPE, ("dp", "tp"), what="fp32 gradient sync on a 2x2 mesh", require_mgd=True
    ) as mesh:
        yield mesh


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _to_bf16(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).astype(ml_dtypes.bfloat16).astype(np.float32)


def _gradients(shape: tuple, seed: int) -> np.ndarray:
    """bf16 values scattered around a per-element mean by as much as the mean, so devices' values often cancel."""
    rng = np.random.default_rng(seed)
    mean = rng.choice([-1.0, 1.0], size=shape[1:]) * 10.0 ** rng.uniform(-3, 0, size=shape[1:])
    return _to_bf16(mean + np.abs(mean) * rng.standard_normal(shape))


def _upload(values: np.ndarray, placements: list):
    mapper = ttnn.create_mesh_mapper(_device(), ttnn.MeshMapperConfig(placements))
    return ttml.autograd.Tensor.from_numpy(values.astype(np.float32), ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _per_device(tensor) -> np.ndarray:
    """``tensor`` read back as ``[device row, device column, ...]``."""
    composer = ttnn.create_mesh_composer(_device(), ttnn.MeshComposerConfig([0, 1]))
    values = ttml.autograd.create_tensor(tensor).to_numpy(ttnn.DataType.FLOAT32, composer=composer)
    return values.reshape(_MESH_SHAPE + values.shape[2:])


def _placements(tensor) -> list:
    return [
        ("Shard", p.dim) if isinstance(p, ttnn.PlacementShard) else ("Replicate",)
        for p in tensor.tensor_topology().placements()
    ]


def _fp32_all_reduce_mean(tensor, axes: list):
    """The fp32 mean written out with public ops, to check the sync against."""
    total = ttnn.typecast(tensor, _FP32)
    for axis in axes:
        total = ttml.core.distributed.all_reduce(total, cluster_axis=axis)
    return ttnn.typecast(ttnn.multiply(total, 1.0 / (2 ** len(axes))), ttnn.DataType.BFLOAT16)


def _parameters(gradient):
    parameter = ttml.autograd.create_tensor(gradient)
    parameter.set_grad(gradient)
    parameters = ttml.NamedParameters()
    parameters["parameter"] = parameter
    return parameters, parameter


# Each of the four devices holds a different gradient, as data-parallel ranks do; both axes are summed.
def test_fp32_sum_over_two_axes(dp_tp_mesh):
    values = _gradients((4, 1, 64, 128), seed=1).reshape(2, 2, 64, 128)
    gradient = _upload(values, [ttnn.PlacementShard(0), ttnn.PlacementShard(1)]).get_value()
    expected = _to_bf16(values.astype(np.float64).mean(axis=(0, 1)))

    fp32_parameters, fp32_parameter = _parameters(gradient)
    ttml.sync_gradients(fp32_parameters, axis_names=("dp", "tp"), reduce_dtype=_FP32)
    cpp_parameters, cpp_parameter = _parameters(gradient)
    ttml.core.distributed.synchronize_gradients(cpp_parameters, [_DP_AXIS, _TP_AXIS], reduce_dtype=_FP32)
    bf16 = _per_device(ttml.core.distributed.synchronize_tensor(gradient, [_DP_AXIS, _TP_AXIS]))

    fp32 = _per_device(fp32_parameter.get_grad())
    np.testing.assert_array_equal(fp32, _per_device(_fp32_all_reduce_mean(gradient, [_DP_AXIS, _TP_AXIS])))
    np.testing.assert_array_equal(_per_device(cpp_parameter.get_grad()), fp32)
    assert np.mean(fp32[0, 0] == expected) > np.mean(bf16[0, 0] == expected)
    assert _placements(fp32_parameter.get_grad()) == [("Replicate",), ("Replicate",)]
    assert _placements(cpp_parameter.get_grad()) == [("Replicate",), ("Replicate",)]


# A large gradient takes the reduce-scatter route. When the tp axis already shards the last dim, the route scatters the
# second-to-last one instead (scattering the last dim again has no topology label the strict check accepts); when only
# the last dim splits, the sum falls back to the all-reduce. Either way it equals the fp32 all-reduce and keeps the tp
# axis's shard. 1024 x 1024 bf16 values per device is above the route's 2^19-element threshold.
@pytest.mark.parametrize("rows", [1024, 32 * 33], ids=["scatters_dim_2", "falls_back_to_all_reduce"])
def test_tp_sharded_gradient_over_the_dp_axis(dp_tp_mesh, rows):
    values = _gradients((2, 1, rows, 2048), seed=2)
    gradient = _upload(values, [ttnn.PlacementShard(0), ttnn.PlacementShard(3)]).get_value()

    parameters, parameter = _parameters(gradient)
    ttml.sync_gradients(parameters, axis_names=("dp",), reduce_dtype=_FP32)

    np.testing.assert_array_equal(
        _per_device(parameter.get_grad()), _per_device(_fp32_all_reduce_mean(gradient, [_DP_AXIS]))
    )
    assert _placements(parameter.get_grad()) == [("Replicate",), ("Shard", 3)]


# The reduced axis reads Replicate after either route without synchronize_gradients relabelling it, and the other
# axis keeps its placement; the routes agree with the bf16 all-reduce's label.
@pytest.mark.parametrize("rows", [64, 1024], ids=["all_reduce_route", "reduce_scatter_route"])
def test_fp32_sum_labels_the_reduced_axis_replicate(dp_tp_mesh, rows):
    values = _gradients((2, 1, rows, 1024), seed=3)
    gradient = _upload(values, [ttnn.PlacementShard(0), ttnn.PlacementReplicate()]).get_value()

    fp32 = ttml.core.distributed.synchronize_tensor(gradient, [_DP_AXIS], reduce_dtype=_FP32)
    bf16 = ttml.core.distributed.synchronize_tensor(gradient, [_DP_AXIS])

    assert _placements(fp32) == [("Replicate",), ("Replicate",)]
    assert _placements(fp32) == _placements(bf16)
