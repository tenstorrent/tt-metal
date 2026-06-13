# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Validate whether fabric SOCKETS (vision/prefill transport) can coexist with a
traced p2p denoise chain under ONE fabric config.

Traced denoise needs FABRIC_1D (point_to_point hangs under 2D). The question for
the partial e2e: can the vision/prefill sockets also run under FABRIC_1D — both
same-axis hops AND cross-axis hops (v2p / KV-migration cross both row & col, which
the code comments say required FABRIC_2D)?

Tests under FABRIC_1D, each guarded so a hang shows as a timeout (124) on the
whole run and a failure shows as a caught exception:
  [A] traced p2p chain on a (6,1) mesh  (the denoise hand-off)
  [B] same-axis socket transfer (0,1)->(0,2)
  [C] cross-axis socket transfer (0,1)->(1,2)   (v2p-style)

Run: tt-smi -glx_reset; timeout 150 python ... _coexist_socket_p2p_repro.py
"""

import os
import sys

import torch
import ttnn

from models.experimental.pi0_5.tt.tt_bh_glx.transport import SocketTransport


def main():
    def log(m):
        print(f"[coexist] {m}", flush=True)

    fabric = os.environ.get("REPRO_FABRIC", "1d").lower()
    log(f"fabric={fabric}")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D if fabric == "1d" else ttnn.FabricConfig.FABRIC_2D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        denoise = parent.create_submesh(ttnn.MeshShape(6, 1), ttnn.MeshCoordinate(0, 0))
        subs.append(denoise)
        # 1x1 chips for socket tests (use rows/cols outside the denoise column 0).
        c_a = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))
        subs.append(c_a)
        c_b = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 2))
        subs.append(c_b)  # same row
        c_c = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(1, 2))
        subs.append(c_c)  # cross-axis from c_a

        # [B] same-axis socket (0,1) -> (0,2)
        transport = SocketTransport()
        t_a = ttnn.from_torch(
            torch.randn(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=c_a,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        log("[B] same-axis socket (0,1)->(0,2) ...")
        try:
            out_b = transport.send(t_a, c_b, tag="B")
            ttnn.synchronize_device(c_b)
            log("[B] same-axis socket: OK")
        except Exception as e:
            log(f"[B] same-axis socket: FAILED — {type(e).__name__}: {str(e)[:160]}")

        # [C] cross-axis socket (0,1) -> (1,2)
        log("[C] cross-axis socket (0,1)->(1,2) ...")
        try:
            out_c = transport.send(t_a, c_c, tag="C")
            ttnn.synchronize_device(c_c)
            log("[C] cross-axis socket: OK")
        except Exception as e:
            log(f"[C] cross-axis socket: FAILED — {type(e).__name__}: {str(e)[:160]}")

        # [A] traced p2p chain on the (6,1) denoise mesh
        log("[A] traced p2p chain on (6,1) ...")
        w = ttnn.from_torch(
            torch.randn(6, 32, 32) * 0.1,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=denoise,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(denoise, dim=0),
        )
        a = ttnn.from_torch(
            torch.randn(1, 1, 32, 32),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=denoise,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(denoise),
        )

        def chain(x):
            cur = x
            for c in range(6):
                cur = ttnn.matmul(cur, w, memory_config=ttnn.L1_MEMORY_CONFIG)
                if c < 5:
                    cur = ttnn.to_memory_config(ttnn.to_layout(cur, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)
                    cur = ttnn.point_to_point(
                        cur,
                        ttnn.MeshCoordinate(c, 0),
                        ttnn.MeshCoordinate(c + 1, 0),
                        topology=ttnn.Topology.Linear,
                        output_tensor=cur,
                    )
            return cur

        chain(a)  # warmup
        ttnn.synchronize_device(denoise)
        tid = ttnn.begin_trace_capture(denoise, cq_id=0)
        chain(a)
        ttnn.end_trace_capture(denoise, tid, cq_id=0)
        ttnn.execute_trace(denoise, tid, cq_id=0, blocking=True)
        log("[A] traced p2p chain: OK")
        log("SUCCESS")
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
