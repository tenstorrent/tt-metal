# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.fsdp.fully_shard(..., reduce_dtype=...)``: unset or bf16 reduce-scatter as before, fp32 sums in fp32 and
keeps the mean shard in fp32.

The two devices of the ``[1, 2]`` ``fsdp_mesh`` hold different bf16 gradients, as data-parallel ranks do.
``test_sync_gradients_reduce_dtype.py`` covers the same setting on the DDP sync.
"""

import numpy as np
import pytest
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from two_device_gradients import (
    device,
    exact_mean,
    make_gradients,
    per_device,
    per_device_tensor,
    sharded_linear,
    to_bf16,
)

_FSDP_AXIS = 1


def _replicated_tensor(values: np.ndarray):
    mapper = ttml.core.distributed.replicate_to_mesh_mapper(device())
    return ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _gathered(shard) -> np.ndarray:
    """The full tensor of an FSDP shard, composed from the devices' shards by its own topology label."""
    tensor = ttml.autograd.create_tensor(shard)
    return ttml.Sharding.from_tensor(tensor).gather(tensor).astype(np.float32)


def _fp32(values: np.ndarray):
    """The devices' gradients ``values`` as one fp32 tensor, device i holding ``values[i]``."""
    return ttnn.typecast(per_device_tensor(values).get_value(), ttnn.DataType.FLOAT32)


def _reference_mean(gradients, shard_dim: int):
    """The mean shard backward_post computes, written out: the reduce-scatter over the fsdp axis, then the division."""
    return ttnn.multiply(ttml.core.distributed.reduce_scatter(gradients, shard_dim, _FSDP_AXIS), 0.5)


def _reduce_with_backward_post(values: np.ndarray, reduce_dtype, *, grad_dtype=ttnn.DataType.BFLOAT16) -> tuple:
    """Shard one weight with ``fully_shard(reduce_dtype=reduce_dtype)`` (``None`` included), give device i the gathered
    gradient ``values[i]`` in ``grad_dtype``, run backward_post.

    Returns the reduced gradient (gathered from the shards) and the shard dim. A sum in fp32, from ``reduce_dtype``, the
    stored default or an fp32 gradient, leaves an fp32 shard; any other a bf16 one.
    """
    _layer, state, tensor, shard_dim = sharded_linear(reduce_dtype, values.shape[-1], values.shape[-2])
    tensor.set_grad(_fp32(values) if grad_dtype == ttnn.DataType.FLOAT32 else per_device_tensor(values).get_value())

    state.backward_post()

    grad = tensor.get_grad()
    if reduce_dtype is None:  # resolved only for the check; fully_shard got None
        reduce_dtype = ttml.autograd.AutoContext.get_instance().get_gradient_reduce_dtype()
    fp32 = ttnn.DataType.FLOAT32 in (reduce_dtype, grad_dtype)
    assert grad.dtype == (ttnn.DataType.FLOAT32 if fp32 else ttnn.DataType.BFLOAT16)
    return _gathered(grad), shard_dim


@pytest.mark.requires_device
@pytest.mark.parametrize("reduce_dtype", [None, ttnn.DataType.BFLOAT16])
def test_unset_and_bf16_reduce_dtype_reduce_as_before(fsdp_mesh, reduce_dtype):
    values = make_gradients(64, 128, seed=1)

    reduced, shard_dim = _reduce_with_backward_post(values, reduce_dtype)

    np.testing.assert_array_equal(reduced, _gathered(_reference_mean(per_device_tensor(values).get_value(), shard_dim)))


# With two devices and this spread, the fp32 sum rounded once to bf16 gets about 99.5% of the elements exact and the
# bf16 sum about 86%. Kept in fp32, the mean is closer to the exact one still.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_is_closer_to_the_exact_mean(fsdp_mesh):
    values = make_gradients(64, 128, seed=2)
    expected = exact_mean(values)

    fp32, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)
    bf16, _ = _reduce_with_backward_post(values, None)

    fp32_share = np.mean(to_bf16(fp32) == expected)
    assert fp32_share >= 0.99
    assert fp32_share > np.mean(bf16 == expected)
    exact = values.astype(np.float64).mean(axis=0, keepdims=True)
    assert np.mean(np.abs(fp32 - exact)) < np.mean(np.abs(bf16 - exact))


# The fp32 shard is the fp32 mean itself: the reduce-scatter in fp32 and the multiply, nothing rounded to bf16.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_keeps_the_mean_shard_in_fp32(fsdp_mesh):
    values = make_gradients(64, 128, seed=4)

    reduced, shard_dim = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)

    np.testing.assert_array_equal(reduced, _gathered(_reference_mean(_fp32(values), shard_dim)))


# The next micro-batch adds to the fp32 shard: backward_pre gathers it in fp32, add_grad adds the bf16 gradient in fp32,
# and backward_post reduce-scatters the fp32 total.
@pytest.mark.requires_device
def test_fp32_shard_takes_the_next_micro_batch_in_fp32(fsdp_mesh):
    first_values, second_values = make_gradients(64, 128, seed=6), make_gradients(64, 128, seed=7)
    _layer, state, tensor, shard_dim = sharded_linear(ttnn.DataType.FLOAT32)

    # Fresh uploads each time: backward_post frees the gathered gradient it casts.
    tensor.set_grad(per_device_tensor(first_values).get_value())
    state.backward_post()
    state.backward_pre()
    assert tensor.get_grad().dtype == ttnn.DataType.FLOAT32
    tensor.add_grad(per_device_tensor(second_values).get_value())
    assert tensor.get_grad().dtype == ttnn.DataType.FLOAT32
    state.backward_post()

    gathered = ttml.core.distributed.all_gather(_reference_mean(_fp32(first_values), shard_dim), shard_dim, _FSDP_AXIS)
    expected = _reference_mean(ttnn.add(gathered, _fp32(second_values)), shard_dim)
    grad = tensor.get_grad()
    assert grad.dtype == ttnn.DataType.FLOAT32
    np.testing.assert_array_equal(_gathered(grad), _gathered(expected))


# An explicit bf16 reduce never narrows a gradient: an fp32 one (accumulated in fp32) is reduce-scattered in fp32 and its
# mean shard stays fp32, the same bits as with FLOAT32.
@pytest.mark.requires_device
def test_bf16_reduce_dtype_leaves_an_fp32_gradient_in_fp32(fsdp_mesh):
    values = make_gradients(64, 128, seed=8)

    bf16, _ = _reduce_with_backward_post(values, ttnn.DataType.BFLOAT16, grad_dtype=ttnn.DataType.FLOAT32)
    fp32, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32, grad_dtype=ttnn.DataType.FLOAT32)

    np.testing.assert_array_equal(bf16, fp32)


# Without reduce_dtype, backward_post sums in the default create_optimizer stores, as ttml.sync_gradients does; a
# reduce_dtype given to fully_shard wins over it. fully_shard gets None here, so the hook resolves it when it runs.
@pytest.mark.requires_device
def test_unset_reduce_dtype_takes_the_stored_default_unless_fully_shard_gets_one(fsdp_mesh, gradient_reduce_default):
    values = make_gradients(64, 128, seed=5)
    fp32, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)
    bf16, _ = _reduce_with_backward_post(values, ttnn.DataType.BFLOAT16)
    assert not np.array_equal(fp32, bf16)

    ttml.core.apply_precision_config({"precision": {"reduce_dtype": "float32"}}, ttml.NamedParameters())
    np.testing.assert_array_equal(_reduce_with_backward_post(values, None)[0], fp32)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, ttnn.DataType.BFLOAT16)[0], bf16)

    gradient_reduce_default(ttnn.DataType.BFLOAT16)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, None)[0], bf16)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, ttnn.DataType.FLOAT32)[0], fp32)


class _Block(AbstractModuleBase):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.fc = LinearLayer(features, features, has_bias=False)

    def forward(self, x):
        return self.fc(x)


class _Model(AbstractModuleBase):
    """Blocks wrapped one by one plus a head the root wrapper owns, as ``train.py`` wraps a transformer."""

    def __init__(self, features: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = ModuleList([_Block(features) for _ in range(num_blocks)])
        self.head = LinearLayer(features, features, has_bias=False)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def _model_gradients(reduce_dtype, *, fsdp: bool) -> dict:
    """Weight gradients after one step where each device trains on its own batch; every model starts equal."""
    features = 64
    rng = np.random.default_rng(3)
    model = _Model(features)
    for _, tensor in sorted(model.parameters().items()):
        weights = (rng.standard_normal(tuple(tensor.shape())) * 0.1).astype(np.float32)
        tensor.set_value(_replicated_tensor(weights).get_value())
    if fsdp:
        for block in model.blocks:
            ttml.fsdp.fully_shard(block, reduce_dtype=reduce_dtype)
        ttml.fsdp.fully_shard(model, reduce_dtype=reduce_dtype)

    inputs = rng.standard_normal((2, 1, 32, features)).astype(np.float32)
    targets = rng.standard_normal((2, 1, 32, features)).astype(np.float32)
    loss = ttml.ops.loss.mse_loss(model(per_device_tensor(inputs)), per_device_tensor(targets))
    loss.backward(False)

    # Per device without FSDP; with FSDP, the reduced gradient gathered from its shards.
    return {
        name: _gathered(tensor.get_grad()) if fsdp else per_device(tensor.get_grad())
        for name, tensor in model.parameters().items()
    }


@pytest.mark.requires_device
def test_blocks_and_root_both_sum_in_fp32(fsdp_mesh):
    local = _model_gradients(None, fsdp=False)
    fp32 = _model_gradients(ttnn.DataType.FLOAT32, fsdp=True)
    bf16 = _model_gradients(None, fsdp=True)

    assert len(fp32) == len(local) == 3  # two block weights and the root's head
    for name, values in local.items():
        expected = exact_mean(values)
        fp32_share = np.mean(to_bf16(fp32[name]) == expected)
        assert fp32_share >= 0.99, name
        assert fp32_share > np.mean(bf16[name] == expected), name


def test_rejects_reduce_dtypes_other_than_bf16_and_fp32(expect_error):
    with expect_error(ValueError, "reduce_dtype"):
        ttml.fsdp.fully_shard(object(), reduce_dtype=ttnn.DataType.UINT32)


def test_takes_reduce_dtype_by_keyword_only(expect_error):
    with expect_error(TypeError, "positional"):
        ttml.fsdp.fully_shard(object(), "auto", "fsdp", True, (), ttnn.DataType.FLOAT32)
