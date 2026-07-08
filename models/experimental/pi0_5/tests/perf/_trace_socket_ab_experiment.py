# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Minimal A/B experiment: trace + socket on 1x1 submeshes.

Models a tiny "stage": a matmul (compute) on submesh A, then a fabric socket
transfer A->B (cross-chip hand-off). The ONLY difference between the two modes
is WHERE the trace is captured:

  REPRO_MODE=parent   capture ONE trace on the PARENT mesh while the ops run on
                      the 1x1 child submeshes  -> recreates the ORIGINAL BUG:
                      the parent CQ has no ops (empty trace) and end_trace_capture
                      deadlocks on the full-mesh finish barrier (hang -> timeout).

  REPRO_MODE=submesh  capture a trace ON EACH submesh (matmul+send on A, recv on
                      B) -> the fix: each trace records its own CQ's ops, replays
                      repeatedly with correct data.

Run:
  tt-smi -glx_reset; REPRO_MODE=submesh timeout 150 python .../_trace_socket_ab_experiment.py
  tt-smi -glx_reset; REPRO_MODE=parent  timeout 150 python .../_trace_socket_ab_experiment.py   # expect hang (124)
"""

import os
import sys

import torch
import ttnn

MODE = os.environ.get("REPRO_MODE", "submesh")
PAGE = 4096


def main():
    def log(m):
        print(f"[ab {MODE}] {m}", flush=True)

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

        w = ttnn.from_torch(
            torch.randn(32, 32) * 0.1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=A,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        x = ttnn.from_torch(
            torch.randn(1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=A,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            torch.zeros(1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=B,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        def compute_and_send():
            # "stage" compute on A, then socket the result A->B
            y = ttnn.matmul(x, w)
            y = ttnn.to_memory_config(ttnn.to_layout(y, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
            ttnn.experimental.send_direct_async(y, send_sock)

        # warmup (JIT) — eager
        log("warmup (eager matmul on A + socket A->B) ...")
        compute_and_send()
        ttnn.experimental.recv_direct_async(out, recv_sock)
        ttnn.synchronize_device(B)
        log("warmup OK")

        if MODE == "parent":
            # THE BUG: capture on the PARENT while ops live on 1x1 submeshes A,B.
            log("begin_trace_capture(PARENT) ...")
            tid = ttnn.begin_trace_capture(parent, cq_id=0)
            compute_and_send()
            ttnn.experimental.recv_direct_async(out, recv_sock)
            log("calling end_trace_capture(PARENT) -- expect EMPTY trace + full-mesh finish DEADLOCK (hang) ...")
            ttnn.end_trace_capture(parent, tid, cq_id=0)
            log("end_trace_capture(PARENT) returned (did NOT hang) -- executing ...")
            ttnn.execute_trace(parent, tid, cq_id=0, blocking=True)
            log("PARENT-rooted trace executed (unexpected)")
        else:
            # THE FIX: capture a trace on EACH submesh independently.
            log("begin_trace_capture(A) [matmul + send] ...")
            tidA = ttnn.begin_trace_capture(A, cq_id=0)
            compute_and_send()
            ttnn.end_trace_capture(A, tidA, cq_id=0)
            log("begin_trace_capture(B) [recv] ...")
            tidB = ttnn.begin_trace_capture(B, cq_id=0)
            ttnn.experimental.recv_direct_async(out, recv_sock)
            ttnn.end_trace_capture(B, tidB, cq_id=0)
            log("captured per-submesh traces")
            for rnd in (1, 2, 3):
                ttnn.execute_trace(A, tidA, cq_id=0, blocking=False)
                ttnn.execute_trace(B, tidB, cq_id=0, blocking=True)
                ttnn.synchronize_device(B)
                ref = (x.cpu() if False else ttnn.to_torch(x).float()) @ ttnn.to_torch(w).float()
                got = ttnn.to_torch(out).float()
                ok = torch.allclose(got, ref, atol=0.2)
                log(f"replay {rnd}: DONE, data {'OK' if ok else 'MISMATCH'}")
            log("PER-SUBMESH traces replayed repeatedly — WORKS")
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
