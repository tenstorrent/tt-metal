# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Probe multi-core/multi-link chip-to-chip bandwidth via send_direct_async.

send_direct_async (the same op KV migration uses) derives its worker-core count
from the number of SocketConnections in the socket config and round-robins cores
across the fabric links between the two chips (send_direct_async_op_program_factory
.cpp:75,242). So a socket with N connections => N cores => up to N links.

This sweeps N = #connections for an adjacent-chip transfer of (1,1024,2048) bf16
and reports ms + GB/s. It also reveals the LINK COUNT: send_direct_async FATALs
when num_cores > num_available_links, so the largest N that succeeds (before that
error) == #forwarding links between the adjacent chips.

Run: tt-smi -glx_reset; timeout 400 python .../_p2p_multicore_probe.py
"""

import sys
import time

import torch
import ttnn

S, W = 1024, 2048
PAGE = 4096


def main():
    def log(m):
        print(f"[mc-probe] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        src = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        dst = parent.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 1))  # adjacent (same row)
        subs += [src, dst]
        mb = S * W * 2 / 1e6

        host = torch.randn(1, S, W)
        t = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=src, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        def make_socket(n):
            # n connections, each a distinct sender/receiver worker core
            conns = [
                ttnn.SocketConnection(
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 0)),
                    ttnn.MeshCoreCoord(ttnn.MeshCoordinate(0, 0), ttnn.CoreCoord(i, 1)),
                )
                for i in range(n)
            ]
            mem = ttnn.SocketMemoryConfig(ttnn.BufferType.L1, PAGE * 4)
            return ttnn.create_socket_pair(src, dst, ttnn.SocketConfig(conns, mem))

        max_links = None
        for n in (1, 2, 3, 4, 6, 8, 12, 16):
            try:
                ssock, rsock = make_socket(n)
                out = ttnn.from_torch(
                    torch.zeros(1, S, W),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=dst,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

                def _xfer():
                    ttnn.experimental.send_direct_async(t, ssock)
                    ttnn.experimental.recv_direct_async(out, rsock)
                    ttnn.synchronize_device(dst)

                for _ in range(3):
                    _xfer()
                N = 30
                t0 = time.perf_counter()
                for _ in range(N):
                    _xfer()
                ms = 1e3 * (time.perf_counter() - t0) / N
                log(f"n={n:>2} cores: {ms:.3f} ms  {mb/ms:.1f} GB/s")
                max_links = n
            except Exception as e:
                log(f"n={n:>2} cores: FAILED -> {type(e).__name__}: {str(e)[:140]}")
                break
        log(f"=> max successful connections (>= #forwarding links between adjacent chips): {max_links}")
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
