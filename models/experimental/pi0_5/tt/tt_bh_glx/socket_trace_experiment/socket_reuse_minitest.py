# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Faithful minimal repro of the denoise multi-step socket-trace defect.

The denoise snake is UNIDIRECTIONAL per hop (chip i -> chip i+1), and a given
socket pair is used ONCE PER EULER STEP -> N times inside one captured per-submesh
trace. This test reproduces exactly that: a single socket A->B, used N times in a
captured trace, each transfer carrying a DIFFERENT value (step index), and checks
whether B receives the right sequence on replay or drifts.

  - A sends src_bufs[i] (value = i+1) on step i, same socket each step.
  - B recvs into recv_b each step, copies it into out_bufs[i] (persistent).
  - After replay, out_bufs[i] should equal i+1 for all i.

If out_bufs drift / are wrong -> a single socket reused for N transfers in one
trace does not correctly advance its FIFO/credit state on replay. That is the
denoise defect, minimally. Compare against p2p (POINT2POINT=1) which should be
correct.

Run:
  tt-smi -glx_reset
  N=5 python .../socket_reuse_minitest.py            # sockets
  N=5 POINT2POINT=1 python .../socket_reuse_minitest.py   # p2p control
"""

import os
import sys

import torch
import ttnn

N = int(os.environ.get("N", "5"))
P2P = os.environ.get("POINT2POINT", "0").lower() in ("1", "true", "yes")
PAGE = 4096


def main():
    def log(m):
        print(f"[sock-reuse N={N} p2p={int(P2P)}] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        A = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        B = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
        subs += [A, B]

        def fa(dev, t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        # step i sends a tensor full of value (i+1)
        src_bufs = [fa(A, torch.full((1, 32, 32), float(i + 1))) for i in range(N)]
        recv_b = fa(B, torch.zeros(1, 32, 32))
        out_bufs = [fa(B, torch.zeros(1, 32, 32)) for _ in range(N)]  # persistent per-step

        sock = None
        if not P2P:
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
            sock = ttnn.create_socket_pair(A, B, ttnn.SocketConfig([conn], mem))

        def stepA(i):
            if P2P:
                # p2p: A is source, B is dest (whole transfer is one op on... p2p needs same mesh)
                pass
            else:
                ttnn.experimental.send_direct_async(src_bufs[i], sock[0])

        def stepB(i):
            if P2P:
                pass
            else:
                ttnn.experimental.recv_direct_async(recv_b, sock[1])
                ttnn.copy(recv_b, out_bufs[i])  # persist this step's received value

        # ---- eager reference ----
        for i in range(N):
            stepA(i)
            stepB(i)
        ttnn.synchronize_device(B)
        eager = [ttnn.to_torch(out_bufs[i]).float().mean().item() for i in range(N)]
        log(f"eager out means = {[round(v, 3) for v in eager]}  (expect {[i + 1 for i in range(N)]})")

        # ---- capture per-submesh traces ----
        tA = ttnn.begin_trace_capture(A, cq_id=0)
        for i in range(N):
            stepA(i)
        ttnn.end_trace_capture(A, tA, cq_id=0)
        tB = ttnn.begin_trace_capture(B, cq_id=0)
        for i in range(N):
            stepB(i)
        ttnn.end_trace_capture(B, tB, cq_id=0)
        log("captured per-submesh traces (A: N sends, B: N recv+copy)")

        # zero the outputs so a stale value can't masquerade as correct
        for i in range(N):
            ttnn.copy(fa(B, torch.zeros(1, 32, 32)), out_bufs[i])

        for rnd in (1, 2, 3):
            ttnn.execute_trace(A, tA, cq_id=0, blocking=False)
            ttnn.execute_trace(B, tB, cq_id=0, blocking=True)
            ttnn.synchronize_device(B)
            got = [ttnn.to_torch(out_bufs[i]).float().mean().item() for i in range(N)]
            ok = all(abs(got[i] - (i + 1)) < 0.05 for i in range(N))
            log(f"replay {rnd}: out means = {[round(v, 3) for v in got]}  {'OK' if ok else 'DRIFT/WRONG'}")
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
