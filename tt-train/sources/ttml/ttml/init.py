# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter initialization functions for ttml modules.

Factory variants (uniform, normal, constant, zeros, ones, xavier_uniform,
    xavier_normal, kaiming_uniform, kaiming_normal):
    Each returns an initializer callable that takes a shape tuple and returns
    a new ttml.autograd.Tensor.

    Usage:
        layer = LinearLayer(
            128, 64,
            weight_init=ttml.init.xavier_uniform(gain=1.0),
            bias_init=ttml.init.constant(0.0),
        )

In-place variants (uniform_, normal_, constant_, zeros_, ones_, xavier_uniform_,
    xavier_normal_, kaiming_uniform_, kaiming_normal_):
    Take an existing tensor (or Parameter/Buffer wrapping one) and fill it
    in-place, returning the same object that was passed in.
    Mirrors the PyTorch torch.nn.init convention.

    Note: these are not truly in-place at the device level. Each call
    allocates a temporary tensor with the new values and then copies it
    into the existing tensor via set_value. The temporary is freed
    afterwards, but callers should be aware of the transient allocation.

    Usage:
        ttml.init.uniform_(tensor, a=-0.1, b=0.1)
        ttml.init.xavier_uniform_(model.fc1.weight, gain=1.0)
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import ttnn
import ttml
from .lazy import is_lazy_init_enabled


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

# Module-private RNG so weight initialization never touches np.random's global
# state. Callers that want reproducible init must use manual_seed() below;
# seeding np.random alone has no effect, mirroring how torch.manual_seed is
# isolated from numpy.
_rng: np.random.Generator = np.random.default_rng()


def manual_seed(seed: int) -> None:
    """Seed ttml.init's RNG. Independent from np.random and torch RNGs."""
    global _rng
    _rng = np.random.default_rng(seed)


def _get_device():
    return ttml.autograd.AutoContext.get_instance().get_device()


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


# ---------------------------------------------------------------------------
# Factory variants (return initializer callables)
# ---------------------------------------------------------------------------


def _maybe_lazy(shape, init_fn, mapper=None):
    """If lazy init is active, return TensorMetadata holding ``init_fn``; else allocate."""
    shape_tuple = tuple(shape)
    if is_lazy_init_enabled():
        from ttml.modules.parameter import TensorMetadata

        return TensorMetadata(shape=shape_tuple, init_fn=init_fn, mapper=mapper)
    return init_fn(shape_tuple, mapper)


def _uniform_materialize(shape, a, b, mapper=None):
    """Host NumPy uniform then device tensor (matches eager path on main)."""
    data = _rng.uniform(low=a, high=b, size=tuple(shape)).astype(np.float32)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def uniform(a: float = 0.0, b: float = 1.0):
    """Uniform distribution over [a, b)."""

    def uniform_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _uniform_materialize(s, a, b, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return uniform_init


def _normal_materialize(shape, mean, std, mapper=None):
    """Host NumPy normal then device tensor (matches eager path on main)."""
    data = _rng.normal(loc=mean, scale=std, size=tuple(shape)).astype(np.float32)
    return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)


def normal(mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) distribution."""

    def normal_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _normal_materialize(s, mean, std, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return normal_init


def _constant_materialize(shape, val, mapper=None):
    if mapper is not None:
        data = np.full(shape, val, dtype=np.float32)
        return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    device = _get_device()
    t = ttnn.full(
        shape,
        fill_value=val,
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
    )
    return ttml.autograd.create_tensor(t)


def constant(val: float):
    """All elements set to val."""

    def constant_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _constant_materialize(s, val, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return constant_init


def _zeros_materialize(shape, mapper=None):
    if mapper is not None:
        data = np.zeros(shape, dtype=np.float32)
        return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    device = _get_device()
    t = ttnn.zeros(
        shape,
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
    )
    return ttml.autograd.create_tensor(t)


def zeros():
    """All zeros."""

    def zeros_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _zeros_materialize(s, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return zeros_init


def _ones_materialize(shape, mapper=None):
    if mapper is not None:
        data = np.ones(shape, dtype=np.float32)
        return ttml.autograd.Tensor.from_numpy(data, ttnn.Layout.TILE, ttnn.DataType.BFLOAT16, mapper)
    device = _get_device()
    t = ttnn.ones(
        shape,
        device=device,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.Layout.TILE,
    )
    return ttml.autograd.create_tensor(t)


def ones():
    """All ones."""

    def ones_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _ones_materialize(s, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return ones_init


def _xavier_uniform_materialize(shape, gain, mapper=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(list(shape))
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std
    return _uniform_materialize(shape, -a, a, mapper)


def xavier_uniform(gain: float = 1.0):
    """Xavier uniform initialization (Glorot 2010).

    Samples from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out)).
    Use ``calculate_gain`` to compute gain for a specific nonlinearity.
    """

    def xavier_uniform_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _xavier_uniform_materialize(s, gain, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return xavier_uniform_init


def _xavier_normal_materialize(shape, gain, mapper=None):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(list(shape))
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return _normal_materialize(shape, 0.0, std, mapper)


def xavier_normal(gain: float = 1.0):
    """Xavier normal initialization (Glorot 2010).

    Samples from N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out)).
    Use ``calculate_gain`` to compute gain for a specific nonlinearity.
    """

    def xavier_normal_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _xavier_normal_materialize(s, gain, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return xavier_normal_init


def _kaiming_uniform_materialize(shape, a, mode, nonlinearity, mapper=None):
    fan = _calculate_correct_fan(list(shape), mode)
    g = calculate_gain(nonlinearity, a)
    std = g / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    return _uniform_materialize(shape, -bound, bound, mapper)


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

    def kaiming_uniform_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _kaiming_uniform_materialize(s, a, mode, nonlinearity, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return kaiming_uniform_init


def _kaiming_normal_materialize(shape, a, mode, nonlinearity, mapper=None):
    fan = _calculate_correct_fan(list(shape), mode)
    g = calculate_gain(nonlinearity, a)
    std = g / math.sqrt(fan)
    return _normal_materialize(shape, 0.0, std, mapper)


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

    def kaiming_normal_init(shape, mapper=None):
        def init_fn(s, m=None):
            return _kaiming_normal_materialize(s, a, mode, nonlinearity, m)

        return _maybe_lazy(shape, init_fn, mapper)

    return kaiming_normal_init


# ---------------------------------------------------------------------------
# In-place variants (fill existing tensor, return it)
#
# NOTE: Not truly in-place - each function allocates a new temporary tensor
# via the corresponding factory variant and copies the values into the
# existing tensor with set_value(). The temporary is freed after the call.
# ---------------------------------------------------------------------------


def _unwrap_tensor(tensor_or_param):
    """Return the underlying autograd tensor, unwrapping Parameter/Buffer if needed."""
    from ttml.modules.parameter import Buffer, Parameter, TensorMetadata

    if isinstance(tensor_or_param, Parameter):
        inner = tensor_or_param.peek_tensor()
        if isinstance(inner, TensorMetadata):
            raise RuntimeError(
                "In-place initializer cannot run on a lazy Parameter. " "Call ttml.materialize_module(...) first."
            )
        return inner
    if isinstance(tensor_or_param, Buffer):
        return tensor_or_param.tensor
    return tensor_or_param


def uniform_(tensor, a: float = 0.0, b: float = 1.0):
    """Fill tensor in-place with values from uniform distribution [a, b)."""
    inner = _unwrap_tensor(tensor)
    reinit_val = uniform(a, b)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def normal_(tensor, mean: float = 0.0, std: float = 1.0):
    """Fill tensor in-place with values from normal (Gaussian) distribution."""
    inner = _unwrap_tensor(tensor)
    reinit_val = normal(mean, std)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def constant_(tensor, val: float):
    """Fill tensor in-place with a constant value."""
    inner = _unwrap_tensor(tensor)
    reinit_val = constant(val)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def zeros_(tensor):
    """Fill tensor in-place with zeros."""
    inner = _unwrap_tensor(tensor)
    reinit_val = zeros()(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def ones_(tensor):
    """Fill tensor in-place with ones."""
    inner = _unwrap_tensor(tensor)
    reinit_val = ones()(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def xavier_uniform_(tensor, gain: float = 1.0):
    """Fill tensor in-place using Xavier uniform initialization (Glorot 2010).

    See ``xavier_uniform`` for details.
    """
    inner = _unwrap_tensor(tensor)
    reinit_val = xavier_uniform(gain)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def xavier_normal_(tensor, gain: float = 1.0):
    """Fill tensor in-place using Xavier normal initialization (Glorot 2010).

    See ``xavier_normal`` for details.
    """
    inner = _unwrap_tensor(tensor)
    reinit_val = xavier_normal(gain)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def kaiming_uniform_(
    tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Fill tensor in-place using Kaiming uniform initialization (He 2015).

    See ``kaiming_uniform`` for details.
    """
    inner = _unwrap_tensor(tensor)
    reinit_val = kaiming_uniform(a, mode, nonlinearity)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def kaiming_normal_(
    tensor,
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Fill tensor in-place using Kaiming normal initialization (He 2015).

    See ``kaiming_normal`` for details.
    """
    inner = _unwrap_tensor(tensor)
    reinit_val = kaiming_normal(a, mode, nonlinearity)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


__all__ = [
    "calculate_gain",
    "constant",
    "constant_",
    "kaiming_normal",
    "kaiming_normal_",
    "kaiming_uniform",
    "kaiming_uniform_",
    "manual_seed",
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
