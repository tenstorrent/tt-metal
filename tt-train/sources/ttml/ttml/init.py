# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter initialization functions for ttml modules.

Each function returns an initializer: a callable that takes a shape tuple
and returns a ttml.autograd.Tensor with values drawn from the specified
distribution.

Usage:
    layer = LinearLayer(
        128, 64,
        weight_init=ttml.init.uniform(-0.1, 0.1),
        bias_init=ttml.init.normal(0.0, 0.01),
    )
"""

from __future__ import annotations

import ttnn
import ttml


def _get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def uniform(low: float = 0.0, high: float = 1.0):
    """Uniform distribution over [low, high)."""

    def init_fn(shape):
        return ttml.ops.rand(shape, dtype=ttnn.DataType.BFLOAT16, low=low, high=high)

    return init_fn


def normal(mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) distribution."""

    def init_fn(shape):
        return ttml.ops.randn(shape, dtype=ttnn.DataType.BFLOAT16, mean=mean, std=std)

    return init_fn


def zeros():
    """All zeros."""

    def init_fn(shape):
        device = _get_device()
        t = ttnn.zeros(
            shape,
            device=device,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
        )
        return ttml.autograd.create_tensor(t)

    return init_fn


def ones():
    """All ones."""

    def init_fn(shape):
        device = _get_device()
        t = ttnn.ones(
            shape,
            device=device,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.Layout.TILE,
        )
        return ttml.autograd.create_tensor(t)

    return init_fn


__all__ = [
    "normal",
    "ones",
    "uniform",
    "zeros",
]
