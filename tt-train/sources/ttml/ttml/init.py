# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parameter initialization functions for ttml modules.

Factory variants (uniform, normal, constant, zeros, ones, xavier_uniform,
    xavier_normal, kaiming_uniform, kaiming_normal):
    Each returns an initializer callable with signature::

        init_fn(shape, layout=None, mesh_device=None) -> Tensor

    When ``layout`` (a ``DistributedLayout``) and ``mesh_device`` are provided,
    the tensor is distributed across the mesh according to that layout (shard or
    replicate). This is used by the lazy-init path in ``parallelize_module``.

In-place variants (uniform_, normal_, constant_, zeros_, ones_, xavier_uniform_,
    xavier_normal_, kaiming_uniform_, kaiming_normal_):
    Take an existing tensor (or Parameter/Buffer wrapping one) and fill it
    in-place, returning the same object that was passed in.
    Mirrors the PyTorch torch.nn.init convention.

    Note: these are not truly in-place at the device level. Each call
    allocates a temporary tensor with the new values and then copies it
    into the existing tensor via set_value. The temporary is freed
    afterwards, but callers should be aware of the transient allocation.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import ml_dtypes
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


def _mesh_ndim(mesh_device):
    mesh_shape = mesh_device.shape
    return mesh_shape.dims() if hasattr(mesh_shape, "dims") else len(mesh_shape)


def _to_tensor(arr, layout=None, mesh_device=None):
    """Numpy array -> ttml Tensor on mesh.

    With ``layout`` (from TpPlan): shard/replicate per layout. With ``layout is None``
    but ``mesh_device`` set: **replicate** full tensor on the mesh so non-TP params
    (embedding, norm gamma, etc.) match distributed ops.
    """
    from ttml.distributed.layout import DistributedLayout, set_layout

    mapper = None
    stamp_layout = None
    if mesh_device is not None:
        rank = len(arr.shape)
        if layout is not None:
            mapper = layout.build_mapper(mesh_device, tensor_rank=rank)
            stamp_layout = layout
        else:
            ndim = _mesh_ndim(mesh_device)
            stamp_layout = DistributedLayout(ndim=ndim)
            mapper = stamp_layout.build_mapper(mesh_device, tensor_rank=rank)
    tensor = ttml.autograd.Tensor.from_numpy(arr, ttnn.Layout.TILE, mapper=mapper)
    if stamp_layout is not None:
        set_layout(tensor, stamp_layout)
    return tensor


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
    """Return the recommended gain value for the given nonlinearity function."""
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


def _resolve_layout(layout, mesh_device):
    if layout is not None:
        return layout
    if mesh_device is not None:
        from ttml.distributed.layout import DistributedLayout

        return DistributedLayout(ndim=_mesh_ndim(mesh_device))
    return None


def _stamp_layout(tensor, layout):
    if layout is not None:
        from ttml.distributed.layout import set_layout

        set_layout(tensor, layout)
    return tensor


def uniform(a: float = 0.0, b: float = 1.0):
    """Uniform distribution over [a, b)."""

    def uniform_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        if on_device_init and mesh_device is not None:
            mapper = _resolve_layout(layout, mesh_device).build_mapper_config(tensor_rank=len(shape))
            tensor = ttml.ops.rand(shape, a, b, mapper=mapper)
            return _stamp_layout(tensor, _resolve_layout(layout, mesh_device))
        arr = np.random.uniform(a, b, shape).astype(ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return uniform_init


def normal(mean: float = 0.0, std: float = 1.0):
    """Normal (Gaussian) distribution."""

    def normal_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        if on_device_init and mesh_device is not None:
            mapper = _resolve_layout(layout, mesh_device).build_mapper_config(tensor_rank=len(shape))
            tensor = ttml.ops.randn(shape, mean, std, mapper=mapper)
            return _stamp_layout(tensor, _resolve_layout(layout, mesh_device))
        arr = np.random.normal(mean, std, shape).astype(ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return normal_init


def constant(val: float):
    """All elements set to val."""

    def constant_init(shape, layout=None, mesh_device=None, **_kwargs):
        arr = np.full(shape, val, dtype=ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return constant_init


def zeros():
    """All zeros."""

    def zeros_init(shape, layout=None, mesh_device=None, **_kwargs):
        arr = np.zeros(shape, dtype=ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return zeros_init


def ones():
    """All ones."""

    def ones_init(shape, layout=None, mesh_device=None, **_kwargs):
        arr = np.ones(shape, dtype=ml_dtypes.bfloat16)
        return _to_tensor(arr, layout, mesh_device)

    return ones_init


def xavier_uniform(gain: float = 1.0):
    """Xavier uniform initialization (Glorot 2010)."""

    def xavier_uniform_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        a = math.sqrt(3.0) * std
        return uniform(-a, a)(shape, layout=layout, mesh_device=mesh_device, on_device_init=on_device_init)

    return xavier_uniform_init


def xavier_normal(gain: float = 1.0):
    """Xavier normal initialization (Glorot 2010)."""

    def xavier_normal_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        fan_in, fan_out = _calculate_fan_in_and_fan_out(shape)
        std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
        return normal(0.0, std)(shape, layout=layout, mesh_device=mesh_device, on_device_init=on_device_init)

    return xavier_normal_init


def kaiming_uniform(
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Kaiming uniform initialization (He 2015)."""

    def kaiming_uniform_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        fan = _calculate_correct_fan(shape, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        return uniform(-bound, bound)(shape, layout=layout, mesh_device=mesh_device, on_device_init=on_device_init)

    return kaiming_uniform_init


def kaiming_normal(
    a: float = 0,
    mode: _FanMode = "fan_in",
    nonlinearity: _NonlinearityType = "leaky_relu",
):
    """Kaiming normal initialization (He 2015)."""

    def kaiming_normal_init(shape, layout=None, mesh_device=None, *, on_device_init=False):
        fan = _calculate_correct_fan(shape, mode)
        gain = calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan)
        return normal(0.0, std)(shape, layout=layout, mesh_device=mesh_device, on_device_init=on_device_init)

    return kaiming_normal_init


# ---------------------------------------------------------------------------
# In-place variants (fill existing tensor, return it)
# ---------------------------------------------------------------------------


def _unwrap_tensor(tensor_or_param):
    """Return the underlying autograd tensor, unwrapping Parameter/Buffer if needed."""
    from ttml.modules.parameter import Buffer, Parameter

    if isinstance(tensor_or_param, (Parameter, Buffer)):
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
    """Fill tensor in-place using Xavier uniform initialization."""
    inner = _unwrap_tensor(tensor)
    reinit_val = xavier_uniform(gain)(inner.shape())
    inner.set_value(reinit_val.get_value(_FULL_PRECISION))
    return tensor


def xavier_normal_(tensor, gain: float = 1.0):
    """Fill tensor in-place using Xavier normal initialization."""
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
    """Fill tensor in-place using Kaiming uniform initialization."""
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
    """Fill tensor in-place using Kaiming normal initialization."""
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
