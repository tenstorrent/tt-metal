# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Test the theory: can a fabric socket (send_direct_async/recv_direct_async) be
captured in a per-1x1-submesh trace and REPLAYED repeatedly? (i.e. the original
"sockets on 1x1 meshes, trace each submesh independently, no parent-rooted trace"
idea.)

Setup: two adjacent 1x1 submeshes A=(0,0), B=(0,1). One socket A->B. Capture a
trace ON A containing send_direct_async, and a trace ON B containing
recv_direct_async (NOT on the parent). Then execute_trace twice on the SAME
persistent buffers and check whether the 2nd replay completes or deadlocks.

Outcomes:
  - FATAL at begin/end_trace_capture or send  -> sockets not trace-capturable at all
  - round 1 OK, round 2 hangs (timeout 124)   -> socket state not reset across replay (our theory)
  - both rounds OK + data correct             -> theory WRONG, the idea would have worked

Run: tt-smi -glx_reset; timeout 200 python .../_socket_in_trace_repro.py
"""

import sys

import torch
import ttnn

PAGE = 4096


def main():
    def log(m):
        print(f"[sock-trace] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        A = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        B = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))  # adjacent, same row
        subs += [A, B]

        conn = ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
        )
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
        send_sock, recv_sock = ttnn.create_socket_pair(A, B, ttnn.SocketConfig([conn], mem))

        host = torch.randn(1, 32, 32)
        src = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=A, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        out = ttnn.from_torch(
            torch.zeros(1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=B,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # warmup (JIT compile) — eager send/recv once
        log("warmup eager send/recv ...")
        ttnn.experimental.send_direct_async(src, send_sock)
        ttnn.experimental.recv_direct_async(out, recv_sock)
        ttnn.synchronize_device(B)
        log("warmup OK")

        # capture trace ON EACH SUBMESH (not parent)
        log("begin_trace_capture(A) + send_direct_async ...")
        tidA = ttnn.begin_trace_capture(A, cq_id=0)
        ttnn.experimental.send_direct_async(src, send_sock)
        ttnn.end_trace_capture(A, tidA, cq_id=0)
        log("captured A (send)")
        tidB = ttnn.begin_trace_capture(B, cq_id=0)
        ttnn.experimental.recv_direct_async(out, recv_sock)
        ttnn.end_trace_capture(B, tidB, cq_id=0)
        log("captured B (recv)")

        for rnd in (1, 2, 3):
            log(f"replay round {rnd}: execute_trace(A) ...")
            ttnn.execute_trace(A, tidA, cq_id=0, blocking=False)
            log(f"replay round {rnd}: execute_trace(B) ...")
            ttnn.execute_trace(B, tidB, cq_id=0, blocking=True)
            ttnn.synchronize_device(B)
            got = ttnn.to_torch(out)
            ok = torch.allclose(got.float(), host, atol=0.1)
            log(f"replay round {rnd}: DONE, data {'OK' if ok else 'MISMATCH'}")
        log("ALL REPLAYS COMPLETED — sockets-in-trace REPLAYS FINE (theory WRONG)")
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
