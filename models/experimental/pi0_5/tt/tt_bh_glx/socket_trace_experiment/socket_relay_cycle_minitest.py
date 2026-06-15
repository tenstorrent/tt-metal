# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""K-chip relay + feedback cycle, multi-step, captured per-submesh — the faithful
structural model of the denoise snake.

Mirrors _run_denoise_loop_device exactly in shape (not weights):
  x lives on chip0 (persistent fp32, in-place across N steps).
  per Euler step i:
    h = f0(x)                       on chip0
    h -> chip1 (socket)             ; h = f1(recv)  on chip1
    ...                             relay through K chips, compute at each
    h -> chip0 (socket, feedback)   ; x = x + dt*h  on chip0  (in-place)

Each per-chip compute reads its socket recv buffer (cached/persistent, reused every
step) and writes a persistent per-step source buffer (distinct addresses, trace-safe),
matching the transport.send(out_buf cached) + persistent intermediate pattern.

Capture each 1x1 submesh's trace sequentially; replay all concurrently; PCC vs eager.

  K=6 N=5  -> the denoise shape. If this degrades like run_socket_traced
  (0.75->0.65->0.44) the cause is the multi-hop relay cycle itself; if PCC~1.0
  the cause is the real transformer compute / FABRIC_2D, not the socket topology.

Run: tt-smi -glx_reset; K=6 N=5 python .../socket_relay_cycle_minitest.py
"""

import os
import sys

import torch
import ttnn

K = int(os.environ.get("K", "6"))
N = int(os.environ.get("N", "5"))
PAGE = 4096
# TRANSIENT=1 -> per-step socketed source is freshly allocated each step (like a real
#   transformer block's transient intermediates) instead of a persistent per-step buffer.
TRANSIENT = os.environ.get("TRANSIENT", "0").lower() in ("1", "true", "yes")
# CAPTURE=interleaved -> open ALL submesh captures, run the monolithic interleaved body,
#   close all (exactly what run_socket_traced does). seq -> capture each submesh alone.
INTERLEAVED = os.environ.get("CAPTURE", "seq").lower() in ("interleaved", "inter", "il")


def main():
    def log(m):
        print(f"[relay-cycle K={K} N={N}] {m}", flush=True)

    fabric = ttnn.FabricConfig.FABRIC_2D if os.environ.get("FABRIC", "2d") == "2d" else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        # K consecutive chips along row 0 (adjacent, collinear -> FABRIC_1D ok for K<=4;
        # for K>4 snake into row 1). Use the snake order to keep every hop adjacent.
        def snake(n):
            order = []
            for r in range(8):
                cs = range(4) if r % 2 == 0 else range(3, -1, -1)
                for c in cs:
                    order.append((r, c))
                    if len(order) == n:
                        return order
            return order

        coords = snake(K)
        chips = []
        for r, c in coords:
            sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))
            chips.append(sm)
            subs.append(sm)
        log(f"carved {K} adjacent 1x1 submeshes at {coords}")

        def f32(dev, t, mc=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        def bf16(dev, t):
            return ttnn.from_torch(
                t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        torch.manual_seed(0)
        x0 = torch.randn(1, 32, 32)
        x = f32(chips[0], x0, ttnn.L1_MEMORY_CONFIG)
        bias = [bf16(chips[k], torch.randn(1, 32, 32) * 0.01) for k in range(K)]
        dt = 0.1

        # sockets: forward hops chip[k]->chip[k+1] for k in 0..K-2, plus feedback chip[K-1]->chip[0]
        def mk(src, dst):
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
            return ttnn.create_socket_pair(src, dst, ttnn.SocketConfig([conn], mem))

        fwd = [mk(chips[k], chips[k + 1]) for k in range(K - 1)]
        fb = mk(chips[K - 1], chips[0])

        # cached recv buffers (one per hop, reused every step) + persistent per-step sources
        recv = [bf16(chips[k], torch.zeros(1, 32, 32)) for k in range(K)]  # recv[k] on chip k (recv[0]=feedback)
        src_step = [[bf16(chips[k], torch.zeros(1, 32, 32)) for _ in range(N)] for k in range(K)]

        def refresh():
            ttnn.copy_host_to_device_tensor(ttnn.from_torch(x0, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), x)

        # per-chip op emitters for step i ------------------------------------------------
        def _emit_src(k, i, h):
            if TRANSIENT:
                return ttnn.mul(h, 1.01)  # freshly allocated each step (transient)
            ttnn.mul(h, 1.01, output_tensor=src_step[k][i])  # persistent per-step buffer
            return src_step[k][i]

        def chip0_head(i):
            # f0(x) -> src; send to chip1
            xb = ttnn.typecast(x, ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            h = ttnn.add(xb, bias[0])
            za = _emit_src(0, i, h)
            ttnn.experimental.send_direct_async(za, fwd[0][0])

        def chip_mid(k, i):
            # recv from k-1; f_k; send to k+1
            ttnn.experimental.recv_direct_async(recv[k], fwd[k - 1][1])
            h = ttnn.add(recv[k], bias[k])
            za = _emit_src(k, i, h)
            ttnn.experimental.send_direct_async(za, fwd[k][0])

        def chip_last(i):
            # recv from K-2; f_{K-1}; send feedback to chip0
            k = K - 1
            ttnn.experimental.recv_direct_async(recv[k], fwd[k - 1][1])
            h = ttnn.add(recv[k], bias[k])
            za = _emit_src(k, i, h)
            ttnn.experimental.send_direct_async(za, fb[0])

        def chip0_tail(i):
            # recv feedback; x += dt*recv (in-place)
            ttnn.experimental.recv_direct_async(recv[0], fb[1])
            v = ttnn.mul(
                ttnn.typecast(recv[0], ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG),
                dt,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.add(x, v, output_tensor=x)

        # ---- eager reference (full dependency-ordered interleave) ----
        for i in range(N):
            chip0_head(i)
            for k in range(1, K - 1):
                chip_mid(k, i)
            chip_last(i)
            chip0_tail(i)
        ttnn.synchronize_device(chips[0])
        eager = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(chips[0], dim=0))[0].clone()
        log(f"eager x mean = {eager.mean():.5f}")

        # ---- capture ----
        refresh()
        tids = [None] * K
        if INTERLEAVED:
            # exactly run_socket_traced: open ALL submesh captures, run the monolithic
            # dependency-ordered body, close all.
            for k in range(K):
                tids[k] = ttnn.begin_trace_capture(chips[k], cq_id=0)
            for i in range(N):
                chip0_head(i)
                for k in range(1, K - 1):
                    chip_mid(k, i)
                chip_last(i)
                chip0_tail(i)
            for k in range(K):
                ttnn.end_trace_capture(chips[k], tids[k], cq_id=0)
            log(f"captured all traces (INTERLEAVED, transient={int(TRANSIENT)})")
        else:
            # SEQUENTIAL: each submesh captured alone (only its ops issued).
            # chip0's per-step dependency is head_i ... tail_i ... head_{i+1}.
            tids[0] = ttnn.begin_trace_capture(chips[0], cq_id=0)
            for i in range(N):
                chip0_head(i)
                chip0_tail(i)
            ttnn.end_trace_capture(chips[0], tids[0], cq_id=0)
            for k in range(1, K - 1):
                tids[k] = ttnn.begin_trace_capture(chips[k], cq_id=0)
                for i in range(N):
                    chip_mid(k, i)
                ttnn.end_trace_capture(chips[k], tids[k], cq_id=0)
            tids[K - 1] = ttnn.begin_trace_capture(chips[K - 1], cq_id=0)
            for i in range(N):
                chip_last(i)
            ttnn.end_trace_capture(chips[K - 1], tids[K - 1], cq_id=0)
            log(f"captured all traces (SEQUENTIAL, transient={int(TRANSIENT)})")

        for rnd in (1, 2, 3):
            refresh()
            for k in range(K):
                ttnn.execute_trace(chips[k], tids[k], cq_id=0, blocking=False)
            for k in range(K):
                ttnn.synchronize_device(chips[k])
            traced = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(chips[0], dim=0))[0]
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
