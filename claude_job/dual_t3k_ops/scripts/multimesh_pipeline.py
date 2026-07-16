# SPDX-License-Identifier: Apache-2.0
"""
Multi-Mesh demo: each T3K is its OWN 2x4 MeshDevice. A tensor flows host->host over
the QSFP fabric via a MeshSocket (scale-out / pipeline parallelism).

Pipeline (row/data-parallel; no cross-shard reduction needed):
  rank 0 (mesh 0, host A): C = ADD(a, b)   [rows sharded across 8 chips] -> send_async(C)
  rank 1 (mesh 1, host B): recv_async(C) -> D = MATMUL(C, W) -> verify per local shard
Plus an intra-mesh CCL all_gather demonstrated + verified on rank 1's 2x4 mesh.

Launch (from the launcher box, repo root, venv active):

    tt-run --tcp-interface ens18 \
      --mesh-graph-descriptor tests/tt_metal/tt_fabric/custom_mesh_descriptors/dual_t3k_mesh_graph_descriptor.textproto \
      --hosts t3k-node-a,t3k-node-b \
      python3 claude_job/dual_t3k_ops/scripts/multimesh_pipeline.py

Both ranks run the SAME script and diverge on rank. Both build identical goldens with the
same seed so the socket tensor spec (shape/dtype/layout/sharding) matches on both ends.
"""

import os

import torch
import ttnn

TILE = 32
PCC_THRESHOLD = 0.99


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float32)
    b = b.flatten().to(torch.float32)
    if torch.allclose(a, b):
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0
    return float((a @ b) / denom)


def local_shards(tt_tensor, mesh_device):
    """(linear_index_in_full_mesh, local_device_torch_tensor) for THIS rank's chips only."""
    coords = list(tt_tensor.tensor_topology().mesh_coords())
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}
    view = mesh_device.get_view() if ttnn.using_distributed_env() else None
    device_tensors = ttnn.get_device_tensors(tt_tensor)
    coord_iter = coords
    if view is not None and len(device_tensors) != len(coords):
        coord_iter = [c for c in coords if view.is_local(c)]
    out = []
    for coord, dev_t in zip(coord_iter, device_tensors):
        if view is not None and not view.is_local(coord):
            continue
        out.append((coord_to_index[coord], ttnn.to_torch(dev_t)))
    return out


def main():
    torch.manual_seed(0)  # both ranks build identical tensors -> matching socket specs

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_2D)

    mesh_shape = ttnn.MeshShape(2, 4)  # each host = one 2x4 mesh (8 chips)
    device = ttnn.open_mesh_device(mesh_shape=mesh_shape)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    if not ttnn.distributed_context_is_initialized():
        raise RuntimeError("distributed context not initialized")
    rank = int(ttnn.distributed_context_get_rank())
    size = int(ttnn.distributed_context_get_size())
    if size != 2:
        raise RuntimeError(f"expected 2 processes (one per host), got {size}")
    if rank == 0:
        print(f"[rank 0] Multi-Mesh: each host owns a {tuple(device.shape)} mesh; {size} meshes total", flush=True)

    NDEV = 8  # 2x4; shard rows across all 8 chips of a mesh
    K, N = 128, 128

    # ---- socket: one connection per device position, core (0,0) both ends ----
    connections = []
    for coord in ttnn.MeshCoordinateRange(mesh_shape):
        connections.append(
            ttnn.SocketConnection(
                ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0)), ttnn.MeshCoreCoord(coord, ttnn.CoreCoord(0, 0))
            )
        )
    socket_config = ttnn.SocketConfig(
        connections,
        ttnn.SocketMemoryConfig(ttnn.BufferType.L1, 4096),
        sender_rank=0,
        receiver_rank=1,
    )

    # ---- shared inputs (identical on both ranks); rows sharded across 8 chips ----
    a = torch.randn(1, 1, TILE * NDEV, K, dtype=torch.float32)
    b = torch.randn(1, 1, TILE * NDEV, K, dtype=torch.float32)
    golden_C = a + b
    shard_rows = ttnn.ShardTensorToMesh(device, dim=2)
    ta = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=shard_rows)
    tb = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, mesh_mapper=shard_rows)

    ok = True

    if rank == 0:
        C = ttnn.add(ta, tb)  # rows-sharded add on mesh 0
        send_socket = ttnn.MeshSocket(device, socket_config)
        ttnn.experimental.send_async(C, send_socket)
        ttnn.synchronize_device(device)
        print("[rank 0] ADD done on mesh 0; C sent -> mesh 1", flush=True)
    else:
        recv_socket = ttnn.MeshSocket(device, socket_config)
        C_recv = ttnn.allocate_tensor_on_device(ta.spec, device)  # same spec as sender's C
        ttnn.experimental.recv_async(C_recv, recv_socket)

        W = torch.randn(1, 1, K, N, dtype=torch.float32)  # same seed => identical W
        golden_D = golden_C @ W
        tW = ttnn.from_torch(
            W,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            device=device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device),
        )
        D = ttnn.matmul(C_recv, tW)  # rows-sharded matmul (no cross-shard comm)
        ttnn.synchronize_device(device)

        slices = torch.chunk(golden_D, NDEV, dim=2)
        worst = 1.0
        nchk = 0
        for idx, dev_t in local_shards(D, device):
            worst = min(worst, pcc(slices[idx], dev_t))
            nchk += 1
        ok = worst >= PCC_THRESHOLD
        print(
            f"[rank 1] recv C, MATMUL(C,W) done; {nchk} local shard(s), worst PCC={worst:.5f} "
            f"-> {'PASS' if ok else 'FAIL'}",
            flush=True,
        )

    # ---- intra-mesh CCL: all_gather a width-sharded tensor on the local 2x4 mesh ----
    # (runs on both ranks/meshes; verified on rank 1)
    NCOL = 4
    x = torch.randn(1, 1, TILE, TILE * NCOL, dtype=torch.float32)
    tx = ttnn.from_torch(
        x,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        device=device,
        mesh_mapper=ttnn.ShardTensor2dMesh(device, dims=(None, 3), mesh_shape=mesh_shape),
    )
    tg = ttnn.all_gather(tx, dim=3, cluster_axis=1, topology=ttnn.Topology.Linear)
    ttnn.synchronize_device(device)
    if rank == 1:
        worst = 1.0
        for _idx, dev_t in local_shards(tg, device):
            worst = min(worst, pcc(x, dev_t))
        cclok = worst >= PCC_THRESHOLD
        ok = ok and cclok
        print(f"[rank 1] intra-mesh ALL_GATHER worst PCC={worst:.5f} -> {'PASS' if cclok else 'FAIL'}", flush=True)

    ttnn.distributed_context_barrier()

    if rank == 1:
        print("=" * 60, flush=True)
        print(f"[rank 1] Multi-Mesh pipeline OVERALL: {'PASS' if ok else 'FAIL'}", flush=True)
        print("=" * 60, flush=True)

    ttnn.close_mesh_device(device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
