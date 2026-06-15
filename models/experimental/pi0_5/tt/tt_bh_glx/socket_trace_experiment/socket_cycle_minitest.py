# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Decisive test: a CYCLIC multi-step socket loop captured SEQUENTIALLY per-submesh.

The denoise loop is a cycle (chip0->..->chip5->chip0 feedback) with x_t threaded
in-place across N Euler steps. socket_reuse_minitest proved a single socket reused
N times in a trace replays CORRECTLY when each submesh is captured SEQUENTIALLY
(begin A; record all A ops; end A; then B). The remaining unknown is the FEEDBACK
CYCLE: step i+1's input on A depends on step i's output coming back from B.

This models it minimally with 2 chips:
  per step i:  A: za = f(x);  send za A->B
               B: recv;  zb = g(recv);  send zb B->A
               A: recv;  x = x + dt*recv   (in-place, threads to step i+1)

Capture A's whole N-step trace sequentially, end A; capture B's, end B; replay both
concurrently N? no -- replay each ONCE (the trace already contains all N steps).
If the traced x matches eager x -> a cyclic in-place socket loop IS traceable with
sequential per-submesh capture, and the denoise socket path can be fixed that way.
If it hangs at capture or the PCC degrades -> the feedback cycle is the blocker.

Run:
  tt-smi -glx_reset
  N=5 python .../socket_cycle_minitest.py
"""

import os
import sys

import torch
import ttnn

N = int(os.environ.get("N", "5"))
PAGE = 4096


def main():
    def log(m):
        print(f"[sock-cycle N={N}] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        A = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        B = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
        subs += [A, B]

        def f32(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        def bf16(dev, t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        torch.manual_seed(0)
        x0 = torch.randn(1, 32, 32)
        x = f32(A, x0, ttnn.L1_MEMORY_CONFIG)
        ba = bf16(A, torch.randn(1, 32, 32) * 0.01)
        bb = bf16(B, torch.randn(1, 32, 32) * 0.01)
        recv_b = bf16(B, torch.zeros(1, 32, 32))
        recv_a = bf16(A, torch.zeros(1, 32, 32))
        # persistent per-step socketed sources (distinct addresses, trace-safe)
        za_bufs = [bf16(A, torch.zeros(1, 32, 32)) for _ in range(N)]
        zb_bufs = [bf16(B, torch.zeros(1, 32, 32)) for _ in range(N)]
        dt = 0.1

        conn = ttnn.SocketConnection(
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
            ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
        )
        mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
        sab = ttnn.create_socket_pair(A, B, ttnn.SocketConfig([conn], mem))
        sba = ttnn.create_socket_pair(B, A, ttnn.SocketConfig([conn], mem))

        def refresh():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(x0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), x)

        # A-side ops for step i (compute -> send -> later recv+update)
        def a_send(i):
            xb = ttnn.typecast(x, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            ya = ttnn.add(xb, ba)
            ttnn.mul(ya, 1.01, output_tensor=za_bufs[i])
            ttnn.experimental.send_direct_async(za_bufs[i], sab[0])

        def a_recv(i):
            ttnn.experimental.recv_direct_async(recv_a, sba[1])
            va = ttnn.mul(
                ttnn.typecast(recv_a, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG),
                dt,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.add(x, va, output_tensor=x)

        def b_step(i):
            ttnn.experimental.recv_direct_async(recv_b, sab[1])
            yb = ttnn.add(recv_b, bb)
            ttnn.mul(yb, 1.01, output_tensor=zb_bufs[i])
            ttnn.experimental.send_direct_async(zb_bufs[i], sba[0])

        # ---- eager reference (interleaved, real data) ----
        for i in range(N):
            a_send(i)
            b_step(i)
            a_recv(i)
        ttnn.synchronize_device(A)
        eager = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(A, dim=0))[0].clone()
        log(f"eager x mean = {eager.mean():.5f}")

        # ---- capture: SEQUENTIAL (per-submesh) vs INTERLEAVED (all open, like run_socket_traced) ----
        interleaved = os.environ.get("CAPTURE", "seq").lower() in ("interleaved", "inter", "il")
        refresh()
        if interleaved:
            # exactly what run_socket_traced does: open ALL captures, run interleaved body, close all
            tA = ttnn.begin_trace_capture(A, cq_id=0)
            tB = ttnn.begin_trace_capture(B, cq_id=0)
            for i in range(N):
                a_send(i)
                b_step(i)
                a_recv(i)
            ttnn.end_trace_capture(A, tA, cq_id=0)
            ttnn.end_trace_capture(B, tB, cq_id=0)
            log("INTERLEAVED capture done (both open)")
        else:
            # A's trace contains, per step: send THEN recv+update (the cycle within A).
            tA = ttnn.begin_trace_capture(A, cq_id=0)
            for i in range(N):
                a_send(i)
                a_recv(i)
            ttnn.end_trace_capture(A, tA, cq_id=0)
            log("A trace captured")
            tB = ttnn.begin_trace_capture(B, cq_id=0)
            for i in range(N):
                b_step(i)
            ttnn.end_trace_capture(B, tB, cq_id=0)
            log("B trace captured (SEQUENTIAL)")

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
