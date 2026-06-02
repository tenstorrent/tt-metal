# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inter-stage activation transport for Option B.

tt-metal does not (yet, in this build) expose a direct submesh→submesh
point-to-point copy. The portable fallback is to bounce through host DRAM:

    src_submesh → torch tensor (host) → dst_submesh

This is slower than a hypothetical D2D primitive, but it works today and the
inter-stage payloads are small:

    stage 0 → 1   image-projected hidden  [2, 256, 2048] bf8  ~1 MB
    stage 1 → 2   VLM activation          [968, 2048]    bf8  ~2 MB
    stage 2 → 3   final VLM activation    [968, 2048]    bf8  ~2 MB
    KV migration  (handled separately by kv_migration.py)

When tt-metal exposes a direct primitive (likely via `ttnn.copy` on the parent
mesh with a memcpy-style src/dst mapper, or via fabric sockets), this module
should grow a `direct_d2d_send` path.
"""

from __future__ import annotations

import ttnn


def send_activation_via_host(src_tensor: "ttnn.Tensor", dst_submesh) -> "ttnn.Tensor":
    """Move a REPLICATED activation from one submesh to another via host DRAM.

    Args:
        src_tensor: a replicated tensor allocated on the source submesh.
            (Sharded tensors should be concatenated to a single torch tensor
            via `ConcatMeshToTensor` before being shipped — that's a separate
            helper.)
        dst_submesh: the destination submesh's MeshDevice.

    Returns:
        a tensor of the same shape and dtype, placed on dst_submesh (replicated).
    """
    # For a replicated tensor every chip holds the same data, so we just grab
    # shard 0 and convert it to torch. `ttnn.to_torch(src_tensor)` without a
    # mesh_composer errors out on multi-device tensors (assert.hpp:104,
    # buffers.size() == 1).
    shards = ttnn.get_device_tensors(src_tensor)
    host_torch = ttnn.to_torch(shards[0])
    mapper = ttnn.replicate_tensor_to_mesh_mapper(dst_submesh)
    return ttnn.from_torch(
        host_torch,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        mesh_mapper=mapper,
    )
