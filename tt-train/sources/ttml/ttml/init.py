# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter initialization functions for ttml modules.

In-place variants (uniform_, normal_, constant_, zeros_, ones_, xavier_uniform_,
    xavier_normal_, kaiming_uniform_, kaiming_normal_):
    Take an existing tensor (or Parameter/Buffer wrapping one) and fill it,
    returning the same object that was passed in.

    Uniform-based (uniform_, xavier_uniform_, kaiming_uniform_) and constant-based
    (constant_, zeros_, ones_) variants are truly in-place on device via
    ttml.ops.rand_ (ttnn::uniform) and ttnn.fill - no temporary allocation.

    Normal-based (normal_, xavier_normal_, kaiming_normal_) variants still
    allocate a temporary tensor because ttnn has no in-place normal op.

    Usage:
        ttml.init.uniform_(tensor, -0.1, 0.1)
        ttml.init.xavier_uniform_(model.fc1.weight, gain=1.0)

Factory variants (uniform, normal, constant, zeros, ones, xavier_uniform,
    xavier_normal, kaiming_uniform, kaiming_normal):
    Each returns an initializer callable that takes a shape tuple and returns
    a new ttml.autograd.Tensor. Internally allocates a tensor then delegates
    to the corresponding in-place variant.

    Usage:
        layer = LinearLayer(
            128, 64,
            weight_init=ttml.init.xavier_uniform(gain=1.0),
            bias_init=ttml.init.constant(0.0),
        )
"""

from __future__ import annotations

import math
from typing import Literal

import ttnn
import ttml


_NonlinearityType = Literal[
    "linear",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose1d",
    "conv_transpose2d",
    "conv_transpose3d",
    "sigmoid",
    "tanh",
    "relu",
    "leaky_relu",
    "selu",
]

_FanMode = Literal["fan_in", "fan_out"]

_FULL_PRECISION = ttml.autograd.PreferredPrecision.FULL


def _get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


def _create_tensor(shape):
    """Allocate an autograd tensor on device. Values are uninitialized."""
    device = _get_device()
    t = ttnn.empty(
        shape,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
        device=device,
    )
    return ttml.autograd.create_tensor(t)


def _calculate_fan_in_and_fan_out(shape) -> tuple[int, int]:
    # ttml pads weight shapes to 4D with leading 1s (e.g. [1, 1, out, in]).
    # Strip them so the standard PyTorch fan calculation works correctly.
    while len(shape) > 2 and shape[0] == 1:
        shape = shape[1:]

    if len(shape) < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_output_fmaps = shape[0]
    num_input_fmaps = shape[1]
    receptive_field_size = 1
    for s in shape[2:]:
        receptive_field_size *= s

    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(shape, mode: _FanMode) -> int:
    if mode not in ("fan_in", "fan_out"):
        raise ValueError(f"Mode {mode} not supported, please use one of ('fan_in', 'fan_out')")

    fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
    return fan_in if mode == "fan_in" else fan_out


def calculate_gain(nonlinearity: _NonlinearityType, param: int | float | None = None) -> float:
    """Return the recommended gain value for the given nonlinearity function.

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    linear / conv*    1
    sigmoid           1
    tanh              5/3
    relu              sqrt(2)
    leaky_relu        sqrt(2 / (1 + negative_slope^2))
    selu              3/4
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function name.
        param: optional parameter for the non-linear function (e.g. negative
            slope for leaky_relu, default 0.01).
    """
    linear_fns = [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
    ]
    if nonlinearity in linear_fns or nonlinearity == "sigmoid":
        return 1
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, (int, float)):
            negative_slope = param
        else:
            raise ValueError(f"negative_slope {param} not a valid number")
        return math.sqrt(2.0 / (1 + negative_slope**2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity {nonlinearity}")


def _unwrap_tensor(tensor_or_param):
    """Return the underlying autograd tensor, unwrapping Parameter/Buffer if needed."""
    from ttml.modules.parameter import Buffer, Parameter

    if isinstance(tensor_or_param, (Parameter, Buffer)):
        return tensor_or_param.tensor
    return tensor_or_param


# ---------------------------------------------------------------------------
# In-place variants (core implementation, fill existing tensor, return it)
#
# Uniform-based and constant-based variants are truly in-place on device via
# ttml.ops.rand_ (wraps ttnn::uniform) and ttnn.fill.
#
# Normal-based variants still allocate a temporary tensor because ttnn has no
# in-place normal distribution op.
# ---------------------------------------------------------------------------


def uniform_(tensor, a: float = 0.0, b: float = 1.0):
    """Fill tensor in-place with values from uniform distribution [a, b)."""
    inner = _unwrap_tensor(tensor)
    ttml.ops.rand_(inner, a, b)
    return tensor


def normal_(tensor, mean: float = 0.0, std: float = 1.0):
    """Fill tensor in-place with values from normal (Gaussian) distribution.

    Note: allocates a temporary tensor (no in-place normal distribution op in ttnn).
    """
    inner = _unwrap_tensor(tensor)
    new_tensor = ttml.ops.randn(inner.shape(), mean, std)
    inner.set_value(new_tensor.get_value(_FULL_PRECISION))
    return tensor


def constant_(tensor, val: float):
    """Fill tensor in-place with a constant value."""
    inner = _unwrap_tensor(tensor)
    t = inner.get_value(_FULL_PRECISION)
    ttnn.fill(t, val, output_tensor=t)
    inner.set_value(t)
    return tensor


def zeros_(tensor):
    """Fill tensor in-place with zeros."""
    return constant_(tensor, 0.0)


def ones_(tensor):
    """Fill tensor in-place with ones."""
    return constant_(tensor, 1.0)


def xavier_uniform_(tensor, gain: float = 1.0):
    """Fill tensor in-place using Xavier uniform initialization (Glorot 2010).

    Samples from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
    """
    inner = _unwrap_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(inner.shape())
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    ttml.ops.rand_(inner, -a, a)
    return tensor


def xavier_normal_(tensor, gain: float = 1.0):
    """Fill tensor in-place using Xavier normal initialization (Glorot 2010).

    Samples from N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).

    Note: allocates a temporary tensor (no in-place normal distribution op in ttnn).
    """
    inner = _unwrap_tensor(tensor)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(inner.shape())
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return normal_(tensor, 0.0, std)


def kaiming_uniform_(
    tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Fill tensor in-place using Kaiming uniform initialization (He 2015).

    Samples from U(-bound, bound) where bound = gain * sqrt(3 / fan).

    Args:
        a: negative slope of the rectifier (only used with leaky_relu).
        mode: "fan_in" preserves forward-pass variance,
            "fan_out" preserves backward-pass variance.
        nonlinearity: nonlinearity function name, recommended "relu" or
            "leaky_relu".
    """
    inner = _unwrap_tensor(tensor)
    fan = _calculate_correct_fan(inner.shape(), mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    ttml.ops.rand_(inner, -bound, bound)
    return tensor


def kaiming_normal_(
    tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Fill tensor in-place using Kaiming normal initialization (He 2015).

    Samples from N(0, std^2) where std = gain / sqrt(fan).

    Note: allocates a temporary tensor (no in-place normal distribution op in ttnn).

    Args:
        a: negative slope of the rectifier (only used with leaky_relu).
        mode: "fan_in" preserves forward-pass variance,
            "fan_out" preserves backward-pass variance.
        nonlinearity: nonlinearity function name, recommended "relu" or
            "leaky_relu".
    """
    inner = _unwrap_tensor(tensor)
    fan = _calculate_correct_fan(inner.shape(), mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return normal_(tensor, 0.0, std)


# ---------------------------------------------------------------------------
# Factory variants (allocate a tensor, delegate to in-place variant)
# ---------------------------------------------------------------------------


def uniform(a: float = 0.0, b: float = 1.0):
    """Uniform distribution over [a, b)."""

    def _uniform_init(shape):
        return uniform_(_create_tensor(shape), a, b)

    return _uniform_init


def normal(mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) distribution."""

    def _normal_init(shape):
        return normal_(_create_tensor(shape), mean, std)

    return _normal_init


def constant(val: float):
    """All elements set to val."""

    def _constant_init(shape):
        return constant_(_create_tensor(shape), val)

    return _constant_init


def zeros():
    """All zeros."""
    return constant(0.0)


def ones():
    """All ones."""
    return constant(1.0)


def xavier_uniform(gain: float = 1.0):
    """Xavier uniform initialization (Glorot 2010).

    Samples from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
    Use ``calculate_gain`` to compute gain for a specific nonlinearity.
    """

    def _xavier_uniform_init(shape):
        return xavier_uniform_(_create_tensor(shape), gain)

    return _xavier_uniform_init


def xavier_normal(gain: float = 1.0):
    """Xavier normal initialization (Glorot 2010).

    Samples from N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).
    Use ``calculate_gain`` to compute gain for a specific nonlinearity.
    """

    def _xavier_normal_init(shape):
        return xavier_normal_(_create_tensor(shape), gain)

    return _xavier_normal_init


def kaiming_uniform(
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Kaiming uniform initialization (He 2015).

    Samples from U(-bound, bound) where bound = gain * sqrt(3 / fan).

    Args:
        a: negative slope of the rectifier (only used with leaky_relu).
        mode: "fan_in" preserves forward-pass variance,
            "fan_out" preserves backward-pass variance.
        nonlinearity: nonlinearity function name, recommended "relu" or
            "leaky_relu".
    """

    def _kaiming_uniform_init(shape):
        return kaiming_uniform_(_create_tensor(shape), a, mode, nonlinearity)

    return _kaiming_uniform_init


def kaiming_normal(
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Kaiming normal initialization (He 2015).

    Samples from N(0, std^2) where std = gain / sqrt(fan).

    Args:
        a: negative slope of the rectifier (only used with leaky_relu).
        mode: "fan_in" preserves forward-pass variance,
            "fan_out" preserves backward-pass variance.
        nonlinearity: nonlinearity function name, recommended "relu" or
            "leaky_relu".
    """

    def _kaiming_normal_init(shape):
        return kaiming_normal_(_create_tensor(shape), a, mode, nonlinearity)

    return _kaiming_normal_init


__all__ = [
    "calculate_gain",
    "constant",
    "constant_",
    "kaiming_normal",
    "kaiming_normal_",
    "kaiming_uniform",
    "kaiming_uniform_",
    "normal",
    "normal_",
    "ones",
    "ones_",
    "uniform",
    "uniform_",
    "xavier_normal",
    "xavier_normal_",
    "xavier_uniform",
    "xavier_uniform_",
    "zeros",
    "zeros_",
]
