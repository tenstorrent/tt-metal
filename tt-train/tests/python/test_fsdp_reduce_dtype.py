# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""``ttml.fsdp.fully_shard(..., reduce_dtype=...)``: unset or bf16 reduce-scatter as before, fp32 sums in fp32.

The two devices of the ``[1, 2]`` ``fsdp_mesh`` hold different bf16 gradients, as data-parallel ranks do.
``test_sync_gradients_reduce_dtype.py`` covers the same setting on the DDP sync.
"""

import numpy as np
import pytest
import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer, ModuleList

from train_example import import_train_example
from two_device_gradients import device, exact_mean, make_gradients, per_device, per_device_tensor

_FSDP_AXIS = 1


def _replicated_tensor(values: np.ndarray):
    mapper = ttnn.create_mesh_mapper(
        device(), ttnn.MeshMapperConfig([ttnn.PlacementReplicate(), ttnn.PlacementReplicate()])
    )
    return ttml.autograd.Tensor.from_numpy(values, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _gathered(shard, shard_dim: int) -> np.ndarray:
    """The full tensor of an FSDP shard: the two devices' shards concatenated on ``shard_dim``."""
    composer = ttnn.create_mesh_composer(device(), ttnn.MeshComposerConfig([1 if shard_dim == 0 else 0, shard_dim]))
    return ttml.autograd.create_tensor(shard).to_numpy(ttnn.DataType.FLOAT32, composer=composer)


def _reduce_with_backward_post(values: np.ndarray, reduce_dtype) -> tuple:
    """Shard one weight with ``reduce_dtype``, give device i the gathered gradient ``values[i]``, run backward_post.

    Returns the reduced gradient (gathered from the shards) and the shard dim.
    """
    layer = LinearLayer(values.shape[-1], values.shape[-2], has_bias=False)
    ttml.fsdp.fully_shard(layer, reduce_dtype=reduce_dtype)
    layer.unshard()  # backward_post expects the gathered weight and restores the cached shard
    ((parameter, shard_dim),) = layer._fsdp_state.managed
    parameter.tensor.set_grad(per_device_tensor(values).get_value())

    layer._fsdp_state.backward_post()

    grad = parameter.tensor.get_grad()
    assert grad.dtype == ttnn.DataType.BFLOAT16
    return _gathered(grad, shard_dim), shard_dim


@pytest.mark.requires_device
@pytest.mark.parametrize("reduce_dtype", [None, ttnn.DataType.BFLOAT16])
def test_unset_and_bf16_reduce_dtype_reduce_as_before(fsdp_mesh, reduce_dtype):
    values = make_gradients(64, 128, seed=1)

    reduced, shard_dim = _reduce_with_backward_post(values, reduce_dtype)

    gradients = per_device_tensor(values).get_value()
    before = ttnn.multiply(ttml.core.distributed.reduce_scatter(gradients, shard_dim, _FSDP_AXIS), 0.5)
    np.testing.assert_array_equal(reduced, _gathered(before, shard_dim))


# With two devices and this spread, the fp32 sum gets about 99.5% of the elements exact and the bf16 sum about 86%.
@pytest.mark.requires_device
def test_fp32_reduce_dtype_is_closer_to_the_exact_mean(fsdp_mesh):
    values = make_gradients(64, 128, seed=2)
    expected = exact_mean(values)

    fp32, _ = _reduce_with_backward_post(values, ttnn.DataType.FLOAT32)
    bf16, _ = _reduce_with_backward_post(values, None)

    fp32_share = np.mean(fp32 == expected)
    assert fp32_share >= 0.99
    assert fp32_share > np.mean(bf16 == expected)


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
        name: _gathered(tensor.get_grad(), tensor._fsdp_shard_dim) if fsdp else per_device(tensor.get_grad())
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
        fp32_share = np.mean(fp32[name] == expected)
        assert fp32_share >= 0.99, name
        assert fp32_share > np.mean(bf16[name] == expected), name


def test_rejects_reduce_dtypes_other_than_bf16_and_fp32(expect_error):
    with expect_error(ValueError, "reduce_dtype"):
        ttml.fsdp.fully_shard(object(), reduce_dtype=ttnn.DataType.UINT32)


# train.py shards before create_optimizer applies the config, so it reads precision.reduce_dtype for FSDP itself.
@pytest.mark.requires_device
@pytest.mark.parametrize(
    "precision, expected",
    [({"reduce_dtype": "float32"}, ttnn.DataType.FLOAT32), ({}, None)],
)
def test_train_py_passes_the_configured_reduce_dtype_to_fully_shard(fsdp_mesh, precision, expected):
    train = import_train_example()
    model = _Model(64)

    train.shard_model(model, [], {"training_config": {"precision": precision}})

    assert [block._fsdp_state.reduce_dtype for block in model.blocks] == [expected, expected]
    assert model._fsdp_state.reduce_dtype == expected
