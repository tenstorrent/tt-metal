# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

import ttnn

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
