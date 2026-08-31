# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-rank additive deltas applied in place to a loaded parameter.

The low-rank half of an adapter goes through :class:`~..layers.lora.LoRAMixin`, which reconstructs
its delta on demand from rank-sized factors. A ``.diff`` payload has no factors: it is the delta.
It is applied the same way and for the same reason -- on device, after the base weights load, so
that one shared weight cache serves every adapter.

Unlike the low-rank path this cannot be re-derived, so undoing it requires keeping the uploaded
delta resident. That is cheap for what ``.diff`` is actually used on (norm vectors, biases, and the
few projections whose smaller dimension is already below the adapter's rank) and would not be for a
whole transformer block, which is why the two paths stay separate rather than one subsuming the
other.
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
