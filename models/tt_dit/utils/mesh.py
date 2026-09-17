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

    # A reshape is a relabeling of the same physical devices, and reshaping back is not
    # guaranteed to restore the original labeling: on a 2x2 -> 1x4 -> 2x2 round trip the 1x4 is
    # laid out in ring order and the return trip transposes the 2x2 (observed on a Blackhole Galaxy
    # submesh: [0,1,4,5] -> [0,1,5,4] -> [0,4,1,5]). Tensors already resident on the devices keep
    # their physical placement, so a changed labeling silently mismatches every sharded weight.
    # Record the labeling and, if the return trip changed it, round-trip again until it matches.
    original_ids = list(device.get_device_ids())
    device.reshape(shape)
    try:
        yield
    finally:
        device.reshape(original_shape)
        for _ in range(4):
            if list(device.get_device_ids()) == original_ids:
                break
            device.reshape(shape)
            device.reshape(original_shape)
        else:
            msg = (
                f"reshape_device could not restore the original device labeling {original_ids} "
                f"(got {list(device.get_device_ids())})"
            )
            raise RuntimeError(msg)


def settle_reshape_labeling(
    device: ttnn.MeshDevice, shape: ttnn.MeshShape | Sequence[int], *, max_trips: int = 4
) -> None:
    """Drive the mesh labeling to the fixed point of a ``shape`` reshape round trip.

    Reshaping relabels the same physical devices, and the return trip does not always restore the
    original labeling: on a Blackhole Galaxy 2x2 submesh, 2x2 -> 1x4 -> 2x2 lands on a transposed
    2x2 ([0,1,4,5] -> [0,4,1,5]) and stays there on every further trip. Any tensor placed before
    such a trip then sits on the wrong device for its shard. Call this once, before loading
    weights, so every later ``reshape_device(device, shape)`` returns to the labeling the weights
    were loaded under.
    """
    if not isinstance(shape, ttnn.MeshShape):
        shape = ttnn.MeshShape(*shape)
    original_shape = ttnn.MeshShape(device.shape)
    if original_shape == shape:
        return
    after = None
    for _ in range(max_trips):
        before = list(device.get_device_ids())
        device.reshape(shape)
        device.reshape(original_shape)
        after = list(device.get_device_ids())
        if after == before:
            return
    msg = f"mesh labeling did not converge under {original_shape} <-> {shape} reshapes (last {after})"
    raise RuntimeError(msg)
