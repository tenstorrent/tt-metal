# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A small tensor-parallel block, its optimizer, one training step and per-device readers, shared by the TP tests.
``dim`` is the block's width; a caller picks one that stays tile-aligned per device on its mesh."""

from __future__ import annotations

import numpy as np

import ttml
import ttnn
from ttml.modules import AbstractModuleBase, ColumnParallelLinear, LinearLayer, RowParallelLinear


class TPBlock(AbstractModuleBase):
    """Replicated linear -> column-parallel -> row-parallel: sharded weights on dims 2 and 3, a sharded bias
    (column-parallel, dim 3) and replicated params (the first linear, the row-parallel bias)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.inp = LinearLayer(dim, dim)
        self.up = ColumnParallelLinear(dim, dim, has_bias=True)
        self.down = RowParallelLinear(dim, dim, has_bias=True)

    def forward(self, x):
        return self.down(self.up(self.inp(x)))


def device_slices(value: ttnn.Tensor) -> list:
    """Each device's slice of ``value`` as a float32 numpy array, in mesh order. No composer is involved, so nothing
    here trusts the tensor's topology label."""
    return [ttnn.to_torch(shard).float().numpy() for shard in ttnn.get_device_tensors(value)]


def gather_like(value: ttnn.Tensor, param: ttml.autograd.Tensor) -> np.ndarray:
    """``value`` (``param``'s value or gradient) assembled on the host as ``param`` is laid out on the mesh."""
    sharding = ttml.Sharding.from_tensor(param)
    local_shape = tuple(param.get_value(ttml.autograd.PreferredPrecision.NATIVE).shape)
    full = np.empty(sharding.global_shape, np.float32)
    first = {}  # where a slice goes -> the first device holding it
    for coord, piece in zip(np.ndindex(*ttml.mesh().shape), device_slices(value)):
        # A piece of another shape would leave part of ``full`` unwritten.
        assert piece.shape == local_shape, f"device {coord} holds {piece.shape}, its slice of param is {local_shape}"
        where = [slice(None)] * piece.ndim
        for axis, index in enumerate(coord):
            dim = sharding.shard_dim(axis)
            if dim is not None:
                where[dim] = slice(index * piece.shape[dim], (index + 1) * piece.shape[dim])
        key = tuple((s.start, s.stop) for s in where)
        if key in first:
            np.testing.assert_array_equal(piece, full[tuple(where)], err_msg=f"device {coord} vs its copy {first[key]}")
        else:
            first[key] = coord
            full[tuple(where)] = piece
    return full


def assert_identical_across_devices(value: ttnn.Tensor, what: str) -> np.ndarray:
    """Every device holds the same bytes of ``value``; returns device 0's slice."""
    slices = device_slices(value)
    for dev, other in enumerate(slices[1:], start=1):
        assert np.array_equal(slices[0], other), f"{what}: device {dev} differs from device 0"
    return slices[0]


def make_adamw(params):
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    return ttml.optimizers.AdamW(params, cfg)


def train_step(model, opt, dim: int, input_scale: float = 1.0) -> None:
    """One forward/backward/sync/optimizer step on a ``(1, 1, dim, dim)`` input, so every moment exists and has been
    touched by the update kernel (and, for FSDP, so the gather -> compute -> reduce-scatter -> reshard hooks ran)."""
    ctx = ttml.autograd.AutoContext.get_instance()
    x = ttml.autograd.Tensor.from_numpy(
        np.random.default_rng(0).standard_normal((1, 1, dim, dim), dtype=np.float32) * input_scale,
        ttnn.Layout.TILE,
        ttnn.DataType.BFLOAT16,
    )
    opt.zero_grad()
    loss = ttml.ops.unary.mean(model(x))
    loss.backward(False)
    ttml.sync_gradients(model.parameters())
    opt.step()
    ctx.reset_graph()


def fresh_tp(dim: int, make_optimizer=make_adamw):
    """An untrained ``TPBlock(dim)`` and its optimizer: (model, params, opt)."""
    model = TPBlock(dim)
    params = model.parameters()
    return model, params, make_optimizer(params)


def trained_tp(dim: int, make_optimizer=make_adamw):
    """``fresh_tp`` after one training step: (params, opt)."""
    model, params, opt = fresh_tp(dim, make_optimizer)
    train_step(model, opt, dim)
    return params, opt
