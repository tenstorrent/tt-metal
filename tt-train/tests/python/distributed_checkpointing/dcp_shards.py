# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""This process's shards of a mesh tensor as numpy arrays, by mesh coordinate.

On a mesh spread over several ranks the per-device tensors ``ttnn.get_device_tensors`` hands out still carry the
mesh distribution, so ``ttnn.to_torch`` refuses them. The host buffer of the tensor brought to the host holds one
shard per coordinate this process owns, which is what the tests and the driver compare, on one rank or many.
"""

import numpy as np
import ttnn

CARRIER = {ttnn.DataType.BFLOAT16: np.uint16, ttnn.DataType.FLOAT32: np.float32, ttnn.DataType.UINT32: np.uint32}


def local_coords(mesh_device) -> list[tuple[int, ...]]:
    """The coordinates this process holds: every coordinate on a single-host mesh."""
    return [tuple(int(x) for x in c) for c in mesh_device.get_view().get_local_mesh_coord_range()]


def shard_arrays(value, coords) -> dict[tuple[int, ...], np.ndarray]:
    """``{coord: float32 array}`` of the shards of ``value`` (a ttnn tensor) at ``coords``, in logical row-major shape."""
    host = ttnn.from_device(value) if ttnn.is_tensor_storage_on_device(value) else value
    host = host.to(ttnn.ROW_MAJOR_LAYOUT)
    buffer = host.host_buffer()
    carrier = CARRIER[value.dtype]
    sizes = tuple(int(n) for n in value.shape)
    itemsize = np.dtype(carrier).itemsize
    rows = int(np.prod(sizes[:-1]))
    out = {}
    for coord in coords:
        shard = buffer.get_shard(ttnn.MeshCoordinate(list(coord)))
        if shard is None:
            continue
        raw = np.from_dlpack(shard).reshape(-1).view(np.uint8)
        if raw.nbytes != rows * sizes[-1] * itemsize:  # the last dim may be padded to an alignment
            raw = np.ascontiguousarray(raw.reshape(rows, raw.nbytes // rows)[:, : sizes[-1] * itemsize]).reshape(-1)
        array = raw.view(carrier).reshape(sizes)
        if carrier is np.uint16:  # bf16 bits -> float32
            array = (array.astype(np.uint32) << 16).view(np.float32)
        out[tuple(coord)] = np.ascontiguousarray(array.astype(np.float32))
    return out
