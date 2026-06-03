# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inter-stage activation transport for Option C.

Mirrors `option_b.transport.send_activation_via_host` — tt-metal does not yet
expose a direct submesh→submesh point-to-point copy across heterogeneous
submeshes, so we bounce through host DRAM:

    src_submesh → torch tensor (host) → dst_submesh

Inter-stage payload sizes (per deployment plan §3.2):

    HOST → vision chip 0    [2, 3, 224, 224] bf16 + lang [200] u32   ~600 KB / inference
    vision chip i → i+1     [2, 256, 1152] bf8                       ~590 KB / inference
    vision-embed → prefill  [2, 256, 2048] bf8 + lang_embed          ~1.4 MB / inference
    prefill chip i → i+1    [968, 2048] bf8                          ~2 MB   / inference
    prefill chip i → denoise (KV migration, see kv_migration.py)     ~500 KB / layer (one-shot)
    denoise chip i → i+1    [50, 1024] bf8                           ~50 KB  / Euler step (10× / inf)

When tt-blaze sockets land, this should grow a `direct_d2d_send` path.
"""

from __future__ import annotations

import ttnn


def send_activation_via_host(src_tensor: "ttnn.Tensor", dst_submesh) -> "ttnn.Tensor":
    """Move a REPLICATED activation from one submesh to another via host DRAM.

    For Option C the activation is typically replicated within the source
    submesh's chip-group (or held only on the boundary chip). Sharded
    activations need a concat-via-mapper before being shipped — that's a
    separate helper we add when stage-internal sharding lands.

    Args:
        src_tensor: a replicated tensor allocated on the source submesh.
        dst_submesh: the destination submesh's MeshDevice.

    Returns:
        a tensor of the same shape and dtype, placed on dst_submesh
        (replicated, L1_MEMORY_CONFIG).
    """
    shards = ttnn.get_device_tensors(src_tensor)
    host_torch = ttnn.to_torch(shards[0])
    mapper = ttnn.replicate_tensor_to_mesh_mapper(dst_submesh)
    return ttnn.from_torch(
        host_torch,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        mesh_mapper=mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )


def send_per_chip_activation_via_host(src_tensor: "ttnn.Tensor", src_chip_idx: int, dst_submesh) -> "ttnn.Tensor":
    """Move a single chip's shard from a source submesh to a (replicated)
    placement on a destination submesh, via host DRAM.

    Useful for the prefill-pipeline → denoise hand-off, where prefill chip 17
    (last VLM layer) is the only chip holding the final activation. We pluck
    its shard, send to host, then replicate onto every denoise chip.
    """
    shards = ttnn.get_device_tensors(src_tensor)
    if src_chip_idx >= len(shards):
        raise IndexError(f"src_chip_idx={src_chip_idx} out of range (source submesh has {len(shards)} chips)")
    host_torch = ttnn.to_torch(shards[src_chip_idx])
    mapper = ttnn.replicate_tensor_to_mesh_mapper(dst_submesh)
    return ttnn.from_torch(
        host_torch,
        dtype=src_tensor.dtype,
        layout=src_tensor.layout,
        device=dst_submesh,
        mesh_mapper=mapper,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
