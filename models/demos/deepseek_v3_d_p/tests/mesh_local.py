# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reading back the shards a process owns, for tests that also run on a mesh two hosts share."""

import ttnn


def local_device_tensors(tensor, mesh_device):
    """`{row-major mesh index: single-device tensor}` for the shards this process holds.

    On a single-host mesh that is every device. On a mesh
    that hosts share, each process holds only its own chips and the index is still the chip's position in the
    whole mesh, so a test checks its own chips against the reference for those positions.
    """
    cols = tuple(mesh_device.shape)[1]
    coords = list(tensor.device_coords())
    shards = tensor.cpu_shards(coords)
    return {int(coord[0]) * cols + int(coord[1]): shard for coord, shard in zip(coords, shards)}
