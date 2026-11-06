# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# MeshSocket bug 1 (wrap tail + uint32 underflow): a bfp8 tensor sent right after a bf16 tensor on the same socket
# arrives corrupted, even with a host sync after every op. bf16 tiles travel in 4096 B socket pages, bfp8_b tiles in
# 4352 B; the page-size switch wraps the FIFO and the sender charges the tail before the receiver can ack it.
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


def send_recv(shape, dtype, receiver_delay_s):
    x = torch.randn(shape, dtype=torch.bfloat16)
    src = ttnn.from_torch(
        x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=tx_dev, mesh_mapper=ttnn.ReplicateTensorToMesh(tx_dev)
    )
    dst = ttnn.allocate_tensor_on_device(src.spec, rx_dev)
    ttnn.experimental.send_async(src, tx)
    time.sleep(receiver_delay_s)  # the receiver starts late
    ttnn.experimental.recv_async(dst, rx)
    ttnn.synchronize_device(tx_dev)
    ttnn.synchronize_device(rx_dev)  # host sync: the FIFO is empty again
    sent = ttnn.to_torch(src, mesh_composer=ttnn.ConcatMeshToTensor(tx_dev, dim=0))
    got = ttnn.to_torch(dst, mesh_composer=ttnn.ConcatMeshToTensor(rx_dev, dim=0))
    return "ok" if torch.equal(sent, got) else "CORRUPT"


print("op0: bf16   [32, 192] ->", send_recv((32, 192), ttnn.bfloat16, 0.0))  # ok
print("op1: bfp8_b [32, 512] ->", send_recv((32, 512), ttnn.bfloat8_b, 1.0))  # CORRUPT; ok if op1 is bf16 too
ttnn.close_mesh_device(mesh)
