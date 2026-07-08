# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal controlled test of the proposed denoise fix:

Does giving each loop iteration its OWN persistent intermediate buffers make a
MULTI-STEP per-submesh SOCKET trace replay correctly?

Structure mirrors the denoise loop minimally: a persistent state x on chip A,
threaded in-place across N steps; each step does a tiny compute on A, sockets A->B,
B computes, sockets B->A, A updates x in-place. Capture a trace on each 1x1 submesh
(A, B), replay, compare x vs eager.

  PERSIST=0 : per-step socketed intermediates are TRANSIENT (reused addresses)
  PERSIST=1 : per-step socketed intermediates are PERSISTENT, distinct per step

If PERSIST=0 degrades at N=5 (reproduces the denoise bug) and PERSIST=1 is stable
~1.0, the persistent-per-iteration approach is validated -> worth the slice rewrite.

Run:
  tt-smi -glx_reset; N=5 PERSIST=0 python .../loop_persist_minitest.py
  tt-smi -glx_reset; N=5 PERSIST=1 python .../loop_persist_minitest.py
"""

import os
import sys

import torch
import ttnn

N = int(os.environ.get("N", "5"))
PERSIST = os.environ.get("PERSIST", "0").lower() in ("1", "true", "yes")
PAGE = 4096


def main():
    def log(m):
        print(f"[loop-mini N={N} persist={int(PERSIST)}] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        A = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        B = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
        subs += [A, B]

        def mk_sock(src, dst):
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
            return ttnn.create_socket_pair(src, dst, ttnn.SocketConfig([conn], mem))

        sab = mk_sock(A, B)
        sba = mk_sock(B, A)

        def fa(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        torch.manual_seed(0)
        x0 = torch.randn(1, 32, 32)
        x = ttnn.from_torch(
            x0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=A, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ba = fa(A, torch.randn(1, 32, 32) * 0.01)
        bb = fa(B, torch.randn(1, 32, 32) * 0.01)
        recv_b = fa(B, torch.zeros(1, 32, 32))
        recv_a = fa(A, torch.zeros(1, 32, 32))
        za_bufs = [fa(A, torch.zeros(1, 32, 32)) for _ in range(N)] if PERSIST else None
        zb_bufs = [fa(B, torch.zeros(1, 32, 32)) for _ in range(N)] if PERSIST else None
        dt = 0.1

        def refresh():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(x0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), x)

        def loop():
            for i in range(N):
                xb = ttnn.typecast(x, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
                ya = ttnn.add(xb, ba)  # "block internal" intermediate on A
                if PERSIST:
                    ttnn.mul(ya, 1.01, output_tensor=za_bufs[i])
                    za = za_bufs[i]
                else:
                    za = ttnn.mul(ya, 1.01)  # transient socketed source
                za = ttnn.to_memory_config(ttnn.to_layout(za, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.send_direct_async(za, sab[0])
                ttnn.experimental.recv_direct_async(recv_b, sab[1])
                yb = ttnn.add(recv_b, bb)  # internal on B
                if PERSIST:
                    ttnn.mul(yb, 1.01, output_tensor=zb_bufs[i])
                    zb = zb_bufs[i]
                else:
                    zb = ttnn.mul(yb, 1.01)
                zb = ttnn.to_memory_config(ttnn.to_layout(zb, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                ttnn.experimental.send_direct_async(zb, sba[0])
                ttnn.experimental.recv_direct_async(recv_a, sba[1])
                va = ttnn.mul(
                    ttnn.typecast(recv_a, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG),
                    dt,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                ttnn.add(x, va, output_tensor=x)
            ttnn.synchronize_device(A)

        # warmup + eager reference
        loop()
        refresh()
        loop()
        eager = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(A, dim=0))[0].clone()
        log(f"eager done: x mean={eager.mean():.5f}")

        # capture per-submesh traces (A, B), then replay
        refresh()
        tA = ttnn.begin_trace_capture(A, cq_id=0)
        tB = ttnn.begin_trace_capture(B, cq_id=0)
        loop()
        ttnn.end_trace_capture(A, tA, cq_id=0)
        ttnn.end_trace_capture(B, tB, cq_id=0)
        log("captured per-submesh traces")
        for rnd in (1, 2, 3):
            refresh()
            ttnn.execute_trace(A, tA, cq_id=0, blocking=False)
            ttnn.execute_trace(B, tB, cq_id=0, blocking=True)
            ttnn.synchronize_device(A)
            traced = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(A, dim=0))[0]
            a, b = traced.flatten().float(), eager.flatten().float()
            pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
            log(f"replay {rnd}: PCC vs eager = {pcc:.6f}")
        log("DONE")
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
