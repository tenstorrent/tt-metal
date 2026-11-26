# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.fsdp.fully_shard(..., reduce_dtype=...)``: unset or bf16 reduce-scatter as before, fp32 sums in fp32 and
keeps the mean shard in fp32.

The two devices of the ``[1, 2]`` ``fsdp_mesh`` hold different bf16 gradients, as data-parallel ranks do.
``test_sync_gradients_reduce_dtype.py`` covers the same setting on the DDP sync. Expected values come from numpy: the
fp32 sum adds on the FPU, so it is checked against the float64 sum within the FPU add's bound (``fpu_add_bound``).
"""

import numpy as np
import pytest
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from gradient_sync_helpers import (
    device,
    exact_mean,
    fpu_add_bound,
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


def _reduce_with_backward_post(
    values: np.ndarray, reduce_dtype, *, grad_dtype=ttnn.DataType.BFLOAT16, after_shard=None
) -> tuple:
    """Shard one weight with ``fully_shard(reduce_dtype=reduce_dtype)`` (``None`` included), give device i the gathered
    gradient ``values[i]`` in ``grad_dtype``, run backward_post. ``after_shard`` runs in between, where ``train.py``
    creates the optimizer.

    Returns the reduced gradient (gathered from the shards), its dtype and the shard dim.
    """
    _layer, state, tensor, shard_dim = sharded_linear(reduce_dtype, values.shape[-1], values.shape[-2])
    if after_shard is not None:
        after_shard()
    tensor.set_grad(_fp32(values) if grad_dtype == ttnn.DataType.FLOAT32 else per_device_tensor(values).get_value())

    state.backward_post()

    grad = tensor.get_grad()
    return _gathered(grad), grad.dtype, shard_dim


def _assert_fpu_mean(reduced: np.ndarray, values: np.ndarray) -> None:
    """``reduced`` is the mean of the devices' ``values`` (two of them) as one FPU add and a halving give it."""
    exact = values.astype(np.float64).sum(axis=0, keepdims=True)
    bound = fpu_add_bound(values[0], values[1])[None]
    np.testing.assert_array_less(np.abs(2.0 * reduced.astype(np.float64) - exact), bound * (1.0 + 1e-6) + 1e-30)


@pytest.mark.requires_device
@pytest.mark.parametrize("reduce_dtype", [None, ttnn.DataType.BFLOAT16])
def test_unset_and_bf16_reduce_dtype_reduce_as_before(fsdp_mesh, reduce_dtype):
    values = make_gradients(64, 128, seed=1)

    reduced, dtype, shard_dim = _reduce_with_backward_post(values, reduce_dtype)

    assert dtype == ttnn.DataType.BFLOAT16
    gradients = per_device_tensor(values).get_value()
    before = ttnn.multiply(ttml.core.distributed.reduce_scatter(gradients, shard_dim, _FSDP_AXIS), 0.5)
    np.testing.assert_array_equal(reduced, _gathered(before))


# The fp32 shard is the fp32 mean within the FPU add's bound, and kept in fp32. Rounded once to bf16, about 99.5% of
# the elements land on the exact mean, against about 86% for the bf16 sum.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_keeps_the_mean_shard_in_fp32(fsdp_mesh):
    values = make_gradients(64, 128, seed=2)

    fp32, fp32_dtype, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)
    bf16, _, _ = _reduce_with_backward_post(values, None)

    assert fp32_dtype == ttnn.DataType.FLOAT32
    _assert_fpu_mean(fp32, values)
    expected = exact_mean(values)
    fp32_share = np.mean(to_bf16(fp32) == expected)
    assert fp32_share >= 0.99
    assert fp32_share > np.mean(bf16 == expected)


# An explicit bf16 reduce never narrows a gradient: an fp32 one (accumulated in fp32) is reduce-scattered in fp32 and its
# mean shard stays fp32, the same bits as with FLOAT32.
@pytest.mark.requires_device
def test_bf16_reduce_dtype_leaves_an_fp32_gradient_in_fp32(fsdp_mesh):
    values = make_gradients(64, 128, seed=8)

    bf16, bf16_dtype, _ = _reduce_with_backward_post(values, ttnn.DataType.BFLOAT16, grad_dtype=ttnn.DataType.FLOAT32)
    fp32, _, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32, grad_dtype=ttnn.DataType.FLOAT32)

    assert bf16_dtype == ttnn.DataType.FLOAT32
    np.testing.assert_array_equal(bf16, fp32)


# Without reduce_dtype, backward_post sums in the default create_optimizer stores, as ttml.sync_gradients does; a
# reduce_dtype given to fully_shard wins over it. fully_shard gets None here, so the hook resolves it when it runs.
@pytest.mark.requires_device
def test_unset_reduce_dtype_takes_the_stored_default_unless_fully_shard_gets_one(fsdp_mesh, gradient_reduce_default):
    values = make_gradients(64, 128, seed=5)
    fp32 = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)[0]
    bf16 = _reduce_with_backward_post(values, ttnn.DataType.BFLOAT16)[0]
    assert not np.array_equal(fp32, bf16)

    ttml.core.apply_precision_config({"precision": {"reduce_dtype": "float32"}}, ttml.NamedParameters())
    np.testing.assert_array_equal(_reduce_with_backward_post(values, None)[0], fp32)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, ttnn.DataType.BFLOAT16)[0], bf16)

    gradient_reduce_default(ttnn.DataType.BFLOAT16)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, None)[0], bf16)
    np.testing.assert_array_equal(_reduce_with_backward_post(values, ttnn.DataType.FLOAT32)[0], fp32)


# train.py shards the model first and create_optimizer stores the default afterwards, so backward_post reads the default
# when it runs. A default resolved inside fully_shard would still be unset here and sum in bf16.
@pytest.mark.requires_device
def test_a_default_stored_after_fully_shard_is_used(fsdp_mesh, gradient_reduce_default):
    values = make_gradients(64, 128, seed=6)
    fp32 = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)[0]
    bf16 = _reduce_with_backward_post(values, None)[0]
    assert not np.array_equal(fp32, bf16)

    def store_fp32_default():
        ttml.core.apply_precision_config({"precision": {"reduce_dtype": "float32"}}, ttml.NamedParameters())

    reduced = _reduce_with_backward_post(values, None, after_shard=store_fp32_default)[0]
    np.testing.assert_array_equal(reduced, fp32)


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


_FEATURES = 64


def _new_model(*, fsdp: bool, reduce_dtype=None):
    """The model with the same starting weights every time, FSDP-wrapped block by block and at the root with ``fsdp``."""
    rng = np.random.default_rng(3)
    model = _Model(_FEATURES)
    for _, tensor in sorted(model.parameters().items()):
        weights = (rng.standard_normal(tuple(tensor.shape())) * 0.1).astype(np.float32)
        tensor.set_value(_replicated_tensor(weights).get_value())
    if fsdp:
        for block in model.blocks:
            ttml.fsdp.fully_shard(block, reduce_dtype=reduce_dtype)
        ttml.fsdp.fully_shard(model, reduce_dtype=reduce_dtype)
    return model


def _backward(model, micro_batch: int) -> None:
    """One micro-batch's forward and backward; each device trains on its own batch."""
    rng = np.random.default_rng(10 + micro_batch)
    inputs = rng.standard_normal((2, 1, 32, _FEATURES)).astype(np.float32)
    targets = rng.standard_normal((2, 1, 32, _FEATURES)).astype(np.float32)
    loss = ttml.ops.loss.mse_loss(model(per_device_tensor(inputs)), per_device_tensor(targets))
    loss.backward(False)
    ttml.autograd.AutoContext.get_instance().reset_graph()


def _local_gradients(micro_batch: int) -> dict:
    """Each device's own bf16 weight gradients for one micro-batch, without FSDP, stacked on dim 0."""
    model = _new_model(fsdp=False)
    _backward(model, micro_batch)
    return {name: per_device(tensor.get_grad()) for name, tensor in model.parameters().items()}


def _fsdp_gradients(reduce_dtype, micro_batches: int) -> dict:
    """The reduced weight gradients of an FSDP model after ``micro_batches`` backward passes, gathered from the shards."""
    model = _new_model(fsdp=True, reduce_dtype=reduce_dtype)
    for micro_batch in range(micro_batches):
        _backward(model, micro_batch)
    return {
        name: (_gathered(tensor.get_grad()), tensor.get_grad().dtype) for name, tensor in model.parameters().items()
    }


@pytest.mark.requires_device
def test_blocks_and_root_both_sum_in_fp32(fsdp_mesh):
    local = _local_gradients(0)
    fp32 = _fsdp_gradients(ttnn.DataType.FLOAT32, 1)
    bf16 = _fsdp_gradients(None, 1)

    assert len(fp32) == len(local) == 3  # two block weights and the root's head
    for name, values in local.items():
        reduced, dtype = fp32[name]
        assert dtype == ttnn.DataType.FLOAT32, name
        _assert_fpu_mean(reduced, values)
        expected = exact_mean(values)
        fp32_share = np.mean(to_bf16(reduced) == expected)
        assert fp32_share >= 0.99, name
        assert fp32_share > np.mean(bf16[name][0] == expected), name


# Two real micro-batches through the wrapped model: the hooks run in the order a backward gives them. The second
# backward gathers the fp32 shard back, adds its own gradient to it in fp32 and reduce-scatters the fp32 total. The
# result is the float64 mean of both micro-batches within the two FPU sums' bound.
@pytest.mark.requires_device
def test_micro_batches_accumulate_into_the_fp32_shard(fsdp_mesh):
    first, second = _local_gradients(0), _local_gradients(1)

    reduced = _fsdp_gradients(ttnn.DataType.FLOAT32, 2)

    for name, (got, dtype) in reduced.items():
        assert dtype == ttnn.DataType.FLOAT32, name
        mean_first = first[name].astype(np.float64).mean(axis=0)
        exact = mean_first + second[name].astype(np.float64).mean(axis=0)
        # Three sums, each within one FPU add's bound: the first micro-batch's reduce-scatter (halved), each device's
        # fp32 add of its second gradient to the gathered mean, and the second reduce-scatter (halved).
        totals = [mean_first + second[name][i] for i in range(2)]
        bound = 0.5 * (
            fpu_add_bound(first[name][0], first[name][1])
            + fpu_add_bound(mean_first, second[name][0])
            + fpu_add_bound(mean_first, second[name][1])
            + fpu_add_bound(*totals)
        )
        error = np.abs(got[0].astype(np.float64) - exact)
        np.testing.assert_array_less(error, bound * 1.01 + np.abs(exact) * 2.0**-22 + 1e-30, err_msg=name)


def test_rejects_reduce_dtypes_other_than_bf16_and_fp32(expect_error):
    with expect_error(ValueError, "reduce_dtype"):
        ttml.fsdp.fully_shard(object(), reduce_dtype=ttnn.DataType.UINT32)


def test_takes_reduce_dtype_by_keyword_only(expect_error):
    with expect_error(TypeError, "positional"):
        ttml.fsdp.fully_shard(object(), "auto", "fsdp", True, (), ttnn.DataType.FLOAT32)
