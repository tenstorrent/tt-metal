# SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Micro-benchmark a single ttnn.point_to_point hop to explain the ~4.3ms/hop
seen in the prefill snake profile. Isolated p2p (no preceding-compute
dependency), swept across payload sizes, FABRIC_1D, adjacent same-row chips.

If time is ~constant across sizes -> fixed fabric/latency/dispatch overhead.
If time scales with payload -> bandwidth-bound (reports GB/s).
Also times the to_memory_config+to_layout normalization separately, and the
p2p alone (input already DRAM-interleaved TILE) to remove conversion cost.

Run: tt-smi -glx_reset; timeout 300 python .../_p2p_microbench.py
"""

import sys
import time

import torch
import ttnn

W = 2048  # hidden width (matches VLM)


def main():
    def log(m):
        print(f"[p2p-mb] {m}", flush=True)

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    parent = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(8, 4), trace_region_size=134_217_728)
    subs = []
    try:
        mesh = parent.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))  # row 0, cols 0..3
        subs.append(mesh)
        N = 30

        def bench(fn):
            for _ in range(3):
                fn()
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(N):
                fn()
            ttnn.synchronize_device(mesh)
            return 1e3 * (time.perf_counter() - t0) / N

        for S in (256, 1024, 4096):
            mb = S * W * 2 / 1e6  # bf16 bytes per chip shard
            host = torch.randn(1, S, W)
            t = ttnn.from_torch(
                host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            # (a) p2p alone, input already DRAM-interleaved TILE (no conversion)
            def _p2p_only():
                ttnn.point_to_point(
                    t,
                    ttnn.MeshCoordinate(0, 0),
                    ttnn.MeshCoordinate(0, 1),
                    topology=ttnn.Topology.Linear,
                    output_tensor=t,
                )

            # (b) one adjacent hop, (c) two-hop (0,0)->(0,2), (d) three-hop (0,0)->(0,3)
            def _hop(dst):
                ttnn.point_to_point(
                    t,
                    ttnn.MeshCoordinate(0, 0),
                    ttnn.MeshCoordinate(0, dst),
                    topology=ttnn.Topology.Linear,
                    output_tensor=t,
                )

            t_p2p = bench(_p2p_only)
            t_2 = bench(lambda: _hop(2))
            t_3 = bench(lambda: _hop(3))
            log(
                f"S={S:>4} ({mb:.1f}MB): 1hop={t_p2p:.2f}ms ({mb/t_p2p:.1f}GB/s)  "
                f"2hop={t_2:.2f}ms  3hop={t_3:.2f}ms"
            )

        # conversion cost alone (to_memory_config+to_layout), S=1024
        host = torch.randn(1, 1024, W)
        t = ttnn.from_torch(
            host, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.L1_MEMORY_CONFIG
        )

        def _conv():
            ttnn.to_memory_config(ttnn.to_layout(t, ttnn.TILE_LAYOUT), ttnn.DRAM_MEMORY_CONFIG)

        log(f"to_memory_config+to_layout (S=1024, from L1): {bench(_conv):.2f}ms")
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
