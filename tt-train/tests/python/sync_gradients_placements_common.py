# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Models and helpers shared by test_sync_gradients_placements.py and test_sync_gradients_placements_2x2.py."""

from __future__ import annotations

import numpy as np

import ttnn
import ttml
from ttml.modules import AbstractModuleBase, LinearLayer

DIM = 64  # tile-aligned per device after a 2-way split
NATIVE = ttml.autograd.PreferredPrecision.NATIVE

is_sharded_on_axis = ttml.core.distributed.is_sharded_on_axis


# The parameters of tp_block.TPBlock that are sharded on the tp axis.
TP_SHARDED_SUFFIXES = ("up/weight", "up/bias", "down/weight")


class Block(AbstractModuleBase):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = LinearLayer(DIM, DIM)
        self.fc2 = LinearLayer(DIM, DIM)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class Model(AbstractModuleBase):
    """``block`` is the FSDP target where FSDP is used; ``head`` is never FSDP-managed."""

    def __init__(self) -> None:
        super().__init__()
        self.block = Block()
        self.head = LinearLayer(DIM, DIM, has_bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _build_fsdp_model(init: str) -> Model:
    if init == "eager":
        model = Model()
        ttml.fsdp.fully_shard(model.block)
        return model
    with ttml.lazy_init():
        model = Model()
    ttml.fsdp.fully_shard(model.block)
    ttml.materialize_module(model)
    return model


def _device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _input(shape, mapper=None, *, seed: int = 0, scale: float = 0.1):
    data = np.random.default_rng(seed).standard_normal(shape, dtype=np.float32) * scale
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def _backward(model, x) -> None:
    ctx = ttml.autograd.AutoContext.get_instance()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    ctx.reset_graph()


def _placements(value) -> list:
    return list(value.tensor_topology().placements())


def _shards(value) -> list:
    """Per-device arrays in mesh row-major order (every label built here lists its coordinates in that order)."""
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def _reduce_set(value, data_axes) -> set:
    """The axes synchronize_gradients reduces a gradient along: the data axes the parameter is not sharded on."""
    return {axis for axis in data_axes if not is_sharded_on_axis(value, axis)}


def _assert_mean_over_axis(before: list, after: list, mesh_shape, axis: int, what: str) -> None:
    """``after`` holds, on every device, the mean of ``before`` over mesh axis ``axis`` (bf16 sum then exact scale)."""
    shape = tuple(mesh_shape)
    b = np.stack(before).reshape(shape + before[0].shape)
    a = np.stack(after).reshape(shape + after[0].shape)
    expected = np.broadcast_to(b.mean(axis=axis, keepdims=True), b.shape)
    np.testing.assert_allclose(
        a, expected, rtol=2e-2, atol=1e-4, err_msg=f"{what}: after vs mean of before over mesh axis {axis}"
    )


def _assert_per_axis_labels(params, n_axes: int) -> None:
    """Every parameter carries one placement per mesh axis: none of them is a collapsed 1-D label."""
    wrong = {name: _placements(t.get_value(NATIVE)) for name, t in params.items()}
    wrong = {name: p for name, p in wrong.items() if len(p) != n_axes}
    assert not wrong, f"parameters without one placement per mesh axis: {wrong}"
