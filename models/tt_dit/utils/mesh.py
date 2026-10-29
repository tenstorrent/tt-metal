# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING

import ttnn
from models.tt_dit.parallel.config import ParallelFactor

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


@contextmanager
def reshape_device(device: ttnn.MeshDevice, shape: ttnn.MeshShape | Sequence[int]) -> Iterator[None]:
    """Temporarily rearrange a mesh device into ``shape``, restoring on exit."""
    if not isinstance(shape, ttnn.MeshShape):
        shape = ttnn.MeshShape(*shape)

    # Create a new ttnn.MeshShape instance as the original will be invalidated by the reshape.
    original_shape = ttnn.MeshShape(device.shape)

    if original_shape.mesh_size() != shape.mesh_size():
        msg = f"original shape {original_shape} and target shape {shape} have different device counts"
        raise ValueError(msg)

    if original_shape == shape:
        yield
        return

    device.reshape(shape)
    try:
        yield
    finally:
        device.reshape(original_shape)


def reshape_for_factor(device: ttnn.MeshDevice, factor: ParallelFactor) -> AbstractContextManager[None]:
    """Temporarily reshapes a mesh device to fit a parallel factor.

    Args:
        device: The mesh device to reshape.
        factor: The factor whose mesh axis gets exactly ``factor.factor`` devices. The other axis
            takes the remaining devices.

    Returns:
        A context manager that applies the shape on entry and restores the original one on exit.
    """
    shape = list(device.shape)
    shape[factor.mesh_axis] = factor.factor
    shape[1 - factor.mesh_axis] = device.shape.mesh_size() // factor.factor

    return reshape_device(device, ttnn.MeshShape(*shape))
