# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# MeshSocket bug 2 (uncounted alignment): when consecutive send_async ops on one socket use different page sizes and
# the receiver has not read the earlier ones yet, the pointer alignment between pages is not counted as used FIFO
# space, and a later op overwrites unread data of an earlier one. bf16 tiles: 4096 B socket pages, bfp8_b: 4352 B.
# Uses 2 neighbouring chips of the system mesh.
import time

import torch
import ttnn

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 8))  # BH LoudBox; use your system's mesh shape
tx_dev = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
rx_dev = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
end = ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0))
fifo = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 16 * 1024)
tx, rx = ttnn.create_socket_pair(tx_dev, rx_dev, ttnn.SocketConfig([ttnn.SocketConnection(end, end)], fifo))

# One socket page each: bfp8_b 4 tiles = 4352 B, bf16 2 tiles = 4096 B.
ops = [((32, 128), ttnn.bfloat8_b), ((32, 64), ttnn.bfloat16), ((32, 128), ttnn.bfloat8_b)]
srcs, dsts = [], []
for shape, dtype in ops:
    x = torch.randn(shape, dtype=torch.bfloat16)
    srcs.append(
        ttnn.from_torch(
            x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=tx_dev, mesh_mapper=ttnn.ReplicateTensorToMesh(tx_dev)
        )
    )
    dsts.append(ttnn.allocate_tensor_on_device(srcs[-1].spec, rx_dev))

for src in srcs:
    ttnn.experimental.send_async(src, tx)
time.sleep(1.0)  # the receiver starts after all three sends
for dst in dsts:
    ttnn.experimental.recv_async(dst, rx)
ttnn.synchronize_device(tx_dev)
ttnn.synchronize_device(rx_dev)

for i, (src, dst) in enumerate(zip(srcs, dsts)):
    sent = ttnn.to_torch(src, mesh_composer=ttnn.ConcatMeshToTensor(tx_dev, dim=0))
    got = ttnn.to_torch(dst, mesh_composer=ttnn.ConcatMeshToTensor(rx_dev, dim=0))
    print(f"op{i}: {ops[i][1]} {list(ops[i][0])} ->", "ok" if torch.equal(sent, got) else "CORRUPT")  # op0 CORRUPT
ttnn.close_mesh_device(mesh)
