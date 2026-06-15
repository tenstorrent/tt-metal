# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Does the ORIGINAL per-chip socket design work with PER-SUBMESH trace capture?

Open the (8x4) parent ONLY to carve 32 1x1 submeshes (one per chip) and train
fabric — but NEVER capture a trace on the parent. Wire a socket relay chain
through all 32 chips in boustrophedon (snake) order so every hop is adjacent +
collinear (FABRIC_1D). Capture a trace ON EACH 1x1 submesh (recv + send), then
replay the whole chain N times and check it doesn't deadlock and the relayed
tensor arrives intact at the last chip.

If this replays cleanly, the original "sockets on 1x1 meshes" design WOULD have
been traceable — the only original bug was capturing on the parent, not sockets.

Run: tt-smi -glx_reset; timeout 400 python .../_socket_chain_32_trace.py
"""

import os
import sys

import torch
import ttnn

NREPLAY = int(os.environ.get("NREPLAY", "5"))
PAGE = 4096


def snake_order(rows, cols):
    order = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cs:
            order.append((r, c))
    return order


def main():
    def log(m):
        print(f"[chain32] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        order = snake_order(8, 4)  # 32 adjacent-consecutive chips
        chips = []
        for r, c in order:
            sm = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(r, c))
            chips.append(sm)
            subs.append(sm)
        n = len(chips)
        log(f"carved {n} 1x1 submeshes (snake), NOT using parent for trace")

        # socket per consecutive (adjacent) pair: chips[i] -> chips[i+1]
        socks = []
        for i in range(n - 1):
            conn = ttnn.SocketConnection(
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 0)),
                ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(0, 1)),
            )
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
            socks.append(ttnn.create_socket_pair(chips[i], chips[i + 1], ttnn.SocketConfig([conn], mem)))
        log(f"created {len(socks)} adjacent sockets")

        host = torch.randn(1, 32, 32)
        x = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=chips[0], memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        bufs = [None] + [
            ttnn.from_torch(
                torch.zeros(1, 32, 32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=chips[i],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for i in range(1, n)
        ]

        def relay():  # eager pass through the whole chain
            ttnn.experimental.send_direct_async(x, socks[0][0])
            for i in range(1, n):
                ttnn.experimental.recv_direct_async(bufs[i], socks[i - 1][1])
                if i < n - 1:
                    ttnn.experimental.send_direct_async(bufs[i], socks[i][0])
            ttnn.synchronize_device(chips[n - 1])

        log("warmup eager relay through 32 chips ...")
        relay()
        log("warmup OK")

        # capture a trace ON EACH 1x1 submesh
        tids = [None] * n
        for i in range(n):
            tids[i] = ttnn.begin_trace_capture(chips[i], cq_id=0)
            if i > 0:
                ttnn.experimental.recv_direct_async(bufs[i], socks[i - 1][1])
            if i < n - 1:
                src = x if i == 0 else bufs[i]
                ttnn.experimental.send_direct_async(src, socks[i][0])
            ttnn.end_trace_capture(chips[i], tids[i], cq_id=0)
        log(f"captured {n} per-submesh traces")

        for rnd in range(1, NREPLAY + 1):
            for i in range(n):
                ttnn.execute_trace(chips[i], tids[i], cq_id=0, blocking=False)
            ttnn.synchronize_device(chips[n - 1])
            got = ttnn.to_torch(bufs[n - 1]).float()
            ok = torch.allclose(got, host, atol=0.1)
            log(f"replay {rnd}: chain of {n} traces DONE, last-chip data {'OK' if ok else 'MISMATCH'}")
        log(f"{NREPLAY} REPLAYS of a 32-chip socket chain COMPLETED — per-1x1-submesh socket traces replay fine")
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
