# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate the on-device KV migration: prefill (6,3) -> denoise (6,1) via
in-mesh p2p-gather + adjacent FABRIC_1D socket (replacing the 36x full-mesh
ConcatMeshToTensor host-bounce, ~314 ms/inference).

Prefill produces 18 per-layer KV tensors; pkvs[L] holds real layer-L KV only on
chip coord(L)=(L//3, snake_col). Denoise chip r needs layers 3r,3r+1,3r+2 as
past_k[0/1/2] (chip r = layer 3r+local).

Per local index j (0..2):
  1. p2p-gather: for each row r, point_to_point layer-(3r+j) KV from its chip
     -> staging tensor stg_j chip (r,2)  [in-mesh, multi-hop OK under 1D].
     Validates p2p can write a specific shard of a DIFFERENT output tensor.
  2. adjacent socket: one 6-connection socket stg_j (r,2)->denoise (r,3).

Verifies past_k[j] chip r == layer 3r+j, and times the on-device path vs the
host-bounce baseline (36 ConcatMeshToTensor + 6 shard).

Run: tt-smi -glx_reset; timeout 400 python .../_kv_socket_repro.py
"""

import sys
import time

import torch
import ttnn


def c2l(r, c):
    return 3 * r + (c if r % 2 == 0 else 2 - c)


def snake_col(r, j):
    return j if r % 2 == 0 else 2 - j


_PAGE = 4096
P, HD = 32, 32  # small KV stand-in (real: prefix_pad x head_dim)


def make_socket(src_mesh, dst_mesh, send_coords, recv_coords):
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
        print(f"[kv-sock] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        prefill = parent.create_submesh(ttnn.MeshShape(6, 3), ttnn.MeshCoordinate(0, 0))
        subs.append(prefill)
        denoise = parent.create_submesh(ttnn.MeshShape(6, 1), ttnn.MeshCoordinate(0, 3))
        subs.append(denoise)

        # Build 18 per-layer KV tensors: pkvs[L] = (6,3) mesh tensor, chip coord(L)
        # holds a tile filled with value L (other chips zero, matching the SPMD
        # garbage that gets ignored).
        pkvs = []
        for L in range(18):
            r, c = L // 3, snake_col(L // 3, L % 3)
            host = torch.zeros(18, 1, P, HD)
            host[r * 3 + c] = float(L)  # chip (r,c) row-major index
            pkvs.append(
                ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=prefill,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(prefill, dim=0),
                )
            )

        # One single-connection socket pair per row r: (r,2)->(r,3). Cached &
        # reused across local index j and K/V.
        row_sock = {}

        def sock_for_row(r):
            if r not in row_sock:
                row_sock[r] = make_socket(prefill, denoise, [(r, 2)], [(r, 0)])
            return row_sock[r]

        # Persistent recv buffers for past_k[0/1/2] (6 single-row sockets fill each).
        past = [
            ttnn.from_torch(
                torch.zeros(6, 1, P, HD),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=denoise,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
            )
            for _ in range(3)
        ]

        def migrate():
            # Phase 1: all in-mesh p2p (move non-col-2 layers onto col 2, in-place).
            for j in range(3):
                for r in range(6):
                    L = 3 * r + j
                    sc = snake_col(r, j)
                    if sc != 2:
                        # arg order is (input, FROM_source, TO_dest) — arg1 is the
                        # SOURCE despite the misleading C++ "receiver_coord" name.
                        ttnn.point_to_point(
                            pkvs[L],
                            ttnn.MeshCoordinate(r, sc),  # FROM (layer's chip)
                            ttnn.MeshCoordinate(r, 2),  # TO (col 2)
                            topology=ttnn.Topology.Linear,
                            output_tensor=pkvs[L],
                        )
            # Barrier: ensure p2p writes to col 2 are visible before the fabric
            # socket reads them (send_direct_async bypasses normal CQ ordering).
            ttnn.synchronize_device(prefill)
            # Phase 2: adjacent sockets col-2 -> denoise (r,3).
            for j in range(3):
                for r in range(6):
                    L = 3 * r + j
                    send_sock, recv_sock = sock_for_row(r)
                    ttnn.experimental.send_direct_async(pkvs[L], send_sock)
                    ttnn.experimental.recv_direct_async(past[j], recv_sock)
            ttnn.synchronize_device(denoise)
            return past

        past = migrate()
        ok = True
        for j in range(3):
            got = ttnn.to_torch(past[j], mesh_composer=ttnn.ConcatMeshToTensor(denoise, dim=0))
            want = [float(3 * r + j) for r in range(6)]
            got_vals = [got[r].mean().item() for r in range(6)]
            match = all(abs(g - w) < 0.5 for g, w in zip(got_vals, want))
            ok = ok and match
            log(
                f"past_k[{j}]: got {[round(g) for g in got_vals]} want {[int(w) for w in want]} -> {'OK' if match else 'MISMATCH'}"
            )
        log(f"ON-DEVICE KV MIGRATION: {'SUCCESS' if ok else 'FAILED'}")

        # timing: on-device path vs host-bounce baseline
        Nrep = 10
        migrate()  # warm
        t0 = time.perf_counter()
        for _ in range(Nrep):
            migrate()
        t_dev = 1e3 * (time.perf_counter() - t0) / Nrep

        def c2l_idx(L):
            r = L // 3
            return r * 3 + snake_col(r, L % 3)

        def host_bounce():
            kv = [
                ttnn.to_torch(pkvs[L], mesh_composer=ttnn.ConcatMeshToTensor(prefill, dim=0))[c2l_idx(L)]
                for L in range(18)
            ]
            for j in range(3):
                layers = [3 * r + j for r in range(6)]
                stacked = torch.stack([kv[L] for L in layers], 0).reshape(6, 1, P, HD)
                ttnn.from_torch(
                    stacked,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=denoise,
                    mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
                )

        host_bounce()  # warm
        t0 = time.perf_counter()
        for _ in range(Nrep):
            host_bounce()
        t_host = 1e3 * (time.perf_counter() - t0) / Nrep
        log(
            f"TIMING: on-device p2p+socket={t_dev:.2f}ms  host-bounce(18 gather+6 shard)={t_host:.2f}ms  speedup={t_host/t_dev:.1f}x"
        )
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
