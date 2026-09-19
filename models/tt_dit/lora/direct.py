# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-rank payloads applied in place to a loaded parameter.

The low-rank half of an adapter goes through :class:`~..layers.lora.LoRAMixin`, which reconstructs
its delta on demand from rank-sized factors. A ``.diff`` payload has no factors: it is the delta.
It is applied the same way and for the same reason -- on device, after the base weights load, so
that one shared weight cache serves every adapter.

Unlike the low-rank path this cannot be re-derived, so undoing it requires keeping the uploaded
delta resident. That is cheap for what ``.diff`` is actually used on (norm vectors, biases, and the
few projections whose smaller dimension is already below the adapter's rank) and would not be for a
whole transformer block, which is why the two paths stay separate rather than one subsuming the
other.

A ``.set_weight`` payload is an assignment rather than an addition: it carries a whole parameter,
for a module the base checkpoint has nothing to say about. It is applied here too, because it is
full-rank and lands on a routed parameter exactly as a ``.diff`` does, but it replaces rather than
accumulates -- adding it would be right only where the base happens to be zero, and silently wrong
everywhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

import ttnn

from ..utils.tensor import from_torch

if TYPE_CHECKING:
    from ..layers.module import Parameter


@dataclass
class DirectDelta:
    """A full-rank delta held on device so it can be subtracted again."""

    path: str
    param: Parameter
    delta: ttnn.Tensor

    def undo(self) -> None:
        ttnn.subtract(self.param.data, self.delta, output_tensor=self.param.data)
        ttnn.deallocate(self.delta)


def apply_direct_delta(path: str, param: Parameter, delta: torch.Tensor, *, strength: float = 1.0) -> DirectDelta:
    """Add ``strength * delta`` into ``param`` in place, sharded the way ``param`` is.

    ``delta`` is in the parameter's own layout -- callers get that by routing it through the
    model's weight-preparation pipeline first (see :mod:`.route`), exactly as the base weight was.
    """
    if tuple(delta.shape) != tuple(param.total_shape):
        msg = f"{path}: delta shape {tuple(delta.shape)} does not match parameter {tuple(param.total_shape)}"
        raise ValueError(msg)

    # Scale in fp32: the payload ships bf16, and scaling it there would round twice.
    scaled = delta if strength == 1.0 else delta.to(torch.float32) * strength
    device_delta = from_torch(
        scaled.contiguous(),
        device=param.device,
        layout=param.layout,
        dtype=param.dtype,
        mesh_axes=param.mesh_axes,
    )
    ttnn.add(param.data, device_delta, output_tensor=param.data)
    return DirectDelta(path=path, param=param, delta=device_delta)


@dataclass
class DirectSet:
    """A whole parameter replaced on device.

    Unlike the two delta paths this keeps nothing resident and offers no ``undo``. What an
    assignment displaces is the whole parameter, and retaining a copy of it would cost as much
    device memory as the payload itself -- for MiniMax-H3's 50 gates, about a gigabyte per device.
    Reverting an assignment is a reload, not an unbind.
    """

    path: str
    param: Parameter


def apply_direct_set(path: str, param: Parameter, value: torch.Tensor, *, strength: float = 1.0) -> DirectSet:
    """Replace ``param`` with ``value`` in place, sharded the way ``param`` is.

    ``strength`` interpolates from the displaced value to ``value``, which is the only reading of a
    strength knob that an assignment admits: it is 1 at full strength and leaves the parameter
    untouched at 0, and it agrees with the additive path wherever the base is zero -- which is the
    case this exists for, a gate the base architecture zero-initialises.
    """
    if tuple(value.shape) != tuple(param.total_shape):
        msg = f"{path}: value shape {tuple(value.shape)} does not match parameter {tuple(param.total_shape)}"
        raise ValueError(msg)

    device_value = from_torch(
        value.contiguous(),
        device=param.device,
        layout=param.layout,
        dtype=param.dtype,
        mesh_axes=param.mesh_axes,
    )
    if strength != 1.0:
        # Scale against what is already there, so the result is the interpolation rather than a
        # scaled payload: `previous + strength * (value - previous)`.
        ttnn.subtract(device_value, param.data, output_tensor=device_value)
        ttnn.multiply(device_value, strength, output_tensor=device_value)
        ttnn.add(param.data, device_value, output_tensor=device_value)
    ttnn.copy(device_value, param.data)
    ttnn.deallocate(device_value)
    return DirectSet(path=path, param=param)
