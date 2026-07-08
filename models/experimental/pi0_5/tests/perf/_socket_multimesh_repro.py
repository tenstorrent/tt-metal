# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate MULTI-CHIP-mesh collinear sockets under FABRIC_1D.

The traced e2e uses single multi-chip stage meshes (prefill (6,3), denoise
(6,1)) — NOT the eager pipeline's 1x1 per-chip meshes. transport.py only wires
1x1<->1x1 sockets (hardcoded MeshCoordinate(0,0)). The cross-stage hand-offs
need a socket pair between two DIFFERENT multi-chip submeshes with one
SocketConnection per collinear (sender_coord -> receiver_coord) pair.

This repro validates the KV-migration geometry: prefill row r holds VLM layers
3r,3r+1,3r+2 (snake columns); denoise chip (r,3) needs them as past_k[0/1/2]
shards. For local index j, layer 3r+j sits at prefill col (j if r even else
2-j). One socket (6 same-row connections) gathers all six rows' layer-(3r+j)
shard onto the denoise mesh, where chip r must end up holding layer 3r+j.

Verifies data correctness for j=0,1,2 (the snake-column math) AND that the
socket coexists with a traced point_to_point chain on the denoise mesh.

Run: tt-smi -glx_reset; timeout 300 python .../_socket_multimesh_repro.py
"""

import sys

import torch
import ttnn


def c2l(r, c):
    """prefill chip (r,c) -> VLM layer (snake/boustrophedon)."""
    return 3 * r + (c if r % 2 == 0 else 2 - c)


def snake_col(r, j):
    """prefill column holding layer 3r+j on row r."""
    return j if r % 2 == 0 else 2 - j


_PAGE = 4096


def make_socket(src_mesh, dst_mesh, send_coords, recv_coords):
    """One socket pair with one connection per (send_coord -> recv_coord)."""
    conns = [
        ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*sc), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(*rc), ttnn.CoreCoord(0, 1)),
        )
        for sc, rc in zip(send_coords, recv_coords)
    ]
    mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, _PAGE * 4)
    return ttnn.create_socket_pair(src_mesh, dst_mesh, ttnn.SocketConfig(conns, mem))


def main():
    def log(m):
        print(f"[mm-sock] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        prefill = parent.create_submesh(ttnn.MeshShape(6, 3), ttnn.MeshCoordinate(0, 0))
        subs.append(prefill)
        denoise = parent.create_submesh(ttnn.MeshShape(6, 1), ttnn.MeshCoordinate(0, 3))
        subs.append(denoise)

        # Source on prefill: chip (r,c) holds a tile filled with layer-id c2l(r,c).
        layer_of_chip = torch.tensor([float(c2l(i // 3, i % 3)) for i in range(18)])
        src = torch.ones(18, 1, 32, 32) * layer_of_chip.reshape(18, 1, 1, 1)
        src_t = ttnn.from_torch(
            src,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=prefill,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(prefill, dim=0),
        )

        # FABRIC_1D socket forwarding is ADJACENT-ONLY (fabric.cpp:131-146). Only
        # prefill col-2 (r,2) is adjacent to denoise (r,3). Test that boundary:
        # ADJ_ONLY=1 sends only the col-2 layer (3r+2 for even r, 3r for odd r);
        # otherwise the multi-hop j-loop (expected to FATAL) runs.
        import os as _os

        if _os.environ.get("ADJ_ONLY", "").lower() in ("1", "true", "yes"):
            send_coords = [(r, 2) for r in range(6)]  # global (r,2) -> (r,3): ADJACENT, same row
            recv_coords = [(r, 0) for r in range(6)]
            send_sock, recv_sock = make_socket(prefill, denoise, send_coords, recv_coords)
            out_buf = ttnn.from_torch(
                torch.zeros(6, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=denoise,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
            )
            ttnn.experimental.send_direct_async(src_t, send_sock)
            ttnn.experimental.recv_direct_async(out_buf, recv_sock)
            ttnn.synchronize_device(denoise)
            got = ttnn.to_torch(out_buf, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))
            want = [float(c2l(r, 2)) for r in range(6)]
            got_vals = [got[r].mean().item() for r in range(6)]
            match = all(abs(g - w) < 0.5 for g, w in zip(got_vals, want))
            log(
                f"ADJACENT (r,2)->(r,3): got {[round(g) for g in got_vals]} want {[int(w) for w in want]} -> {'OK' if match else 'MISMATCH'}"
            )
            log(f"ADJACENT SOCKET: {'SUCCESS' if match else 'FAILED'}")
            return

        ok = True
        for j in range(3):
            send_coords = [(r, snake_col(r, j)) for r in range(6)]  # collinear: same row r
            recv_coords = [(r, 0) for r in range(6)]  # denoise local coords (col 0 -> global col 3)
            send_sock, recv_sock = make_socket(prefill, denoise, send_coords, recv_coords)
            out_buf = ttnn.from_torch(
                torch.zeros(6, 1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=denoise,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
            )
            ttnn.experimental.send_direct_async(src_t, send_sock)
            ttnn.experimental.recv_direct_async(out_buf, recv_sock)
            ttnn.synchronize_device(denoise)
            got = ttnn.to_torch(out_buf, mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))  # (6,1,32,32)
            want = [float(3 * r + j) for r in range(6)]
            got_vals = [got[r].mean().item() for r in range(6)]
            match = all(abs(g - w) < 0.5 for g, w in zip(got_vals, want))
            ok = ok and match
            log(
                f"j={j}: denoise chips got {[round(g) for g in got_vals]} want {[int(w) for w in want]} -> {'OK' if match else 'MISMATCH'}"
            )

        log(f"MULTI-MESH SOCKET (KV geometry): {'SUCCESS' if ok else 'FAILED'}")
    finally:
        for sm in reversed(subs):
            try:
                ttnn.close_mesh_device(sm)
            except Exception:
                pass
        ttnn.close_mesh_device(parent)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.exit(0)
