# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal 2x2 test: does a fabric socket send/recv captured in a 1x1-submesh
trace DEADLOCK on repeated replay?

Smallest possible setup (4 devices, not 32):
  - open a 2x2 mesh ONLY to carve two 1x1 submeshes A=(0,0), B=(0,1) (adjacent).
    (sockets need both endpoints to share one fabric/parent; you can't socket
     between two independently-opened devices — but we never capture a trace on
     the 2x2 itself, only on the 1x1 submeshes.)
  - socket A->B.
  - capture a trace ON A containing send_direct_async, ON B containing
    recv_direct_async.
  - execute_trace N times on the SAME persistent buffers; check it doesn't hang
    and the data is correct each replay.

REPRO_MODE=parent (optional): capture ONE trace on the 2x2 instead -> recreates
the ORIGINAL bug (ops live on the 1x1 children, parent trace is empty, and the
full-mesh finish barrier waits on the idle chips (1,0)/(1,1) -> hang).

Run:
  tt-smi -glx_reset; timeout 150 python .../_socket_trace_2x2.py
  tt-smi -glx_reset; REPRO_MODE=parent timeout 150 python .../_socket_trace_2x2.py   # expect hang (124)
"""

import os
import sys

import torch
import ttnn

MODE = os.environ.get("REPRO_MODE", "submesh")
NREPLAY = int(os.environ.get("NREPLAY", "4"))
PAGE = 4096


def main():
    def log(m):
        print(f"[2x2 {MODE}] {m}", flush=True)

    # NOTE: a bare 2x2 (or any sub-view) CANNOT be opened with working fabric on
    # BH Galaxy — fabric-router sync times out because the sub-view's chips have
    # torus ethernet-link partners outside it that aren't present. So we must open
    # the full physical mesh (8x4) to train every torus link, then USE only 2
    # chips (A,B). Override with PARENT_R/PARENT_C if a smaller open ever works.
    pr = int(os.environ.get("PARENT_R", "8"))
    pc = int(os.environ.get("PARENT_C", "4"))
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(pr, pc), trace_region_size=134_217_728)
    log(f"opened {pr}x{pc} mesh ({parent.get_num_devices()} devices); using only chips (0,0) and (0,1)")
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
        log("socket A->B created")

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

        # warmup (JIT) — eager
        log("warmup eager send/recv ...")
        ttnn.experimental.send_direct_async(src, send_sock)
        ttnn.experimental.recv_direct_async(out, recv_sock)
        ttnn.synchronize_device(B)
        log("warmup OK")

        if MODE == "parent":
            log("begin_trace_capture(2x2 PARENT) -- ops live on 1x1 children -> expect empty trace + finish DEADLOCK")
            tid = ttnn.begin_trace_capture(parent, cq_id=0)
            ttnn.experimental.send_direct_async(src, send_sock)
            ttnn.experimental.recv_direct_async(out, recv_sock)
            log("calling end_trace_capture(PARENT) ...")
            ttnn.end_trace_capture(parent, tid, cq_id=0)
            log("end_trace_capture(PARENT) returned (did NOT hang) -- unexpected")
        else:
            log("begin_trace_capture(A) + send ...")
            tidA = ttnn.begin_trace_capture(A, cq_id=0)
            ttnn.experimental.send_direct_async(src, send_sock)
            ttnn.end_trace_capture(A, tidA, cq_id=0)
            log("begin_trace_capture(B) + recv ...")
            tidB = ttnn.begin_trace_capture(B, cq_id=0)
            ttnn.experimental.recv_direct_async(out, recv_sock)
            ttnn.end_trace_capture(B, tidB, cq_id=0)
            log("captured per-1x1-submesh traces")
            for rnd in range(1, NREPLAY + 1):
                log(f"replay {rnd}: execute_trace(A) then (B) ...")
                ttnn.execute_trace(A, tidA, cq_id=0, blocking=False)
                ttnn.execute_trace(B, tidB, cq_id=0, blocking=True)
                ttnn.synchronize_device(B)
                ok = torch.allclose(ttnn.to_torch(out).float(), host, atol=0.1)
                log(f"replay {rnd}: DONE, data {'OK' if ok else 'MISMATCH'}")
            log(f"{NREPLAY} REPLAYS COMPLETED — no deadlock; socket-in-1x1-trace replays fine")
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
