# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A small tensor-parallel block, its optimizer and one training step, shared by the optimizer-state and checkpoint
tests. ``dim`` is the block's width; a caller picks one that stays tile-aligned per device on its mesh."""

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
        self.down = RowParallelLinear(dim, dim, has_bias=True, input_is_parallel=True)

    def forward(self, x):
        return self.down(self.up(self.inp(x)))


def make_adamw(params):
    cfg = ttml.optimizers.AdamWConfig.make(lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0)
    return ttml.optimizers.AdamW(params, cfg)


def train_step(model, opt, dim: int, input_scale: float = 1.0) -> None:
    """One forward/backward/optimizer step on a ``(1, 1, dim, dim)`` input, so every moment exists and has been
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
