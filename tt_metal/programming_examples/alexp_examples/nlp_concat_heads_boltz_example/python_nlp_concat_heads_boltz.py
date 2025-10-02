#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent
# SPDX-License-Identifier: Apache-2.0

import ttnn
from ttnn.distributed import visualize_system_mesh, visualize_mesh_device, visualize_tensor


def main() -> int:
    device = ttnn.open_mesh_device(ttnn.MeshShape(4, 2))

    # Display system-wide and device mesh topology
    visualize_system_mesh()
    visualize_mesh_device(device)

    num_heads = 8
    seq = 64
    head_dim = 64

    # Create on host and explicitly distribute across the 2x4 mesh with 2D topology
    x_host = ttnn.zeros((num_heads, seq, seq, head_dim), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    mapper = ttnn.create_mesh_mapper(
        device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)],  # shard dim 0 across rows, shard dim 1 across cols
            ttnn.MeshShape(4, 2),
        ),
    )
    x = ttnn.distribute_tensor(x_host, mapper, device)

    # Show how the tensor is distributed across the mesh
    visualize_tensor(x)

    y = ttnn.experimental.nlp_concat_heads_boltz(x)

    _ = y  # prevent optimization

    ttnn.close_device(device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
