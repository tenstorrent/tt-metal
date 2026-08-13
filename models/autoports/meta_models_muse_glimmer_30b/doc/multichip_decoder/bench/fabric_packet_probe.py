# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The three collective knobs the first sweep missed.

1. **Fabric packet size.** Every CCL dispatch in this stage logs *"Fabric packet
   size 4352 B is suboptimal for transporting 2048 B pages. Configure 8192 B
   packet size to maximize throughput"* (``ccl_common.cpp:39-71``). The runtime
   asked for a change and the first sweep never tried it, so this measures the
   shipped collectives at the default and at 8192 B, set through
   ``ttnn.FabricRouterConfig`` before the mesh is opened.
2. **``num_workers_per_link`` at the *prefill* payload.** The shipped value (1)
   was swept at the 40 KB decode payload and then applied to both modes; a 107 MB
   reduce-scatter is bandwidth-bound, not latency-bound, so it needs its own
   sweep.
3. **``l1_small_size`` above 4096.** The decode step leaves 7,296 B to give away
   and 8192 is 896 B over; this walks the values in between to record the largest
   feasible one rather than shipping the first that worked.

    python .../bench/fabric_packet_probe.py --out logs/fabric_packet_probe.log
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_MESH_SHAPE,
    FABRIC_CONFIG,
    MULTICHIP_BOUNDARY_CORES,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import width_sharded_l1

HIDDEN = 6656


def open_mesh(shape, *, packet_bytes: int | None, l1_small: int, trace: int):
    # The router config is an argument of ``set_fabric_config``, not of
    # ``open_mesh_device`` (which rejects it with a TypeError).
    if packet_bytes is None:
        ttnn.set_fabric_config(FABRIC_CONFIG)
    else:
        router = ttnn.FabricRouterConfig()
        router.max_packet_payload_size_bytes = packet_bytes
        ttnn.set_fabric_config(FABRIC_CONFIG, router_config=router)
    return ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*shape), trace_region_size=trace, l1_small_size=l1_small)


def close(mesh):
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def timed(mesh, fn, iters: int, rounds: int, *, traced: bool) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    if traced:
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        held = fn()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(rounds):
            ttnn.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(iters):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - start) / iters)
        ttnn.release_trace(mesh, trace_id)
        ttnn.deallocate(held)
        return best * 1e6
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        for _ in range(iters):
            ttnn.deallocate(fn())
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - start) / iters)
    return best * 1e6


def collectives(mesh, rows: int, workers: int, dtype=ttnn.bfloat16):
    """``(reduce_scatter, all_gather)`` closures at the shipped settings.

    ``dtype`` matters more than it looks: the packet-size warning is about the
    *page* size, and a BF16 page is 2048 B while a BFP8 one is 1088 B, so the two
    modes' payloads want opposite packet sizes.  The shipped prefill collective
    carries BFP8 and the shipped decode collective BF16, so both are measured.
    """
    grid = mesh.compute_with_storage_grid_size()
    memcfg = (
        width_sharded_l1(rows, HIDDEN, MULTICHIP_BOUNDARY_CORES, grid)
        if rows == 32 and dtype == ttnn.bfloat16
        else ttnn.DRAM_MEMORY_CONFIG
    )
    partial = ttnn.from_torch(
        torch.randn(1, 1, rows, HIDDEN, dtype=torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=memcfg,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    fractured = ttnn.from_torch(
        torch.randn(1, 1, rows, HIDDEN, dtype=torch.bfloat16),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
    )

    def rs():
        return ttnn.reduce_scatter(
            partial,
            dim=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=workers,
            use_l1_small_for_semaphores=True,
        )

    def ag():
        return ttnn.all_gather(fractured, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def ar():
        return ttnn.all_reduce(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY)

    return partial, fractured, rs, ag, ar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()
    shape = tuple(int(v) for v in args.mesh.split("x"))

    # ---- 1 + 2: packet size x payload x worker count --------------------------
    for packet in (None, 8192):
        mesh = open_mesh(shape, packet_bytes=packet, l1_small=4096, trace=90112 * 24)
        label = "default(4352)" if packet is None else str(packet)
        try:
            for rows, dtype, workers_set in (
                (32, ttnn.bfloat16, (1, 2)),
                (8192, ttnn.bfloat16, (1, 2, 4)),
                (8192, ttnn.bfloat8_b, (1, 4)),  # the shipped prefill payload
            ):
                for workers in workers_set:
                    partial, fractured, rs, ag, ar = collectives(mesh, rows, workers, dtype)
                    traced = rows == 32
                    for name, fn in (("reduce_scatter", rs), ("all_gather", ag), ("all_reduce", ar)):
                        if name != "reduce_scatter" and workers != workers_set[0]:
                            continue  # only reduce_scatter takes the worker count
                        try:
                            us = timed(mesh, fn, args.iters, args.rounds, traced=traced)
                            print(
                                f"PACKET packet={label:14s} rows={rows:5d} dtype={str(dtype).split('.')[-1]:10s} "
                                f"workers={workers} {name:15s} {us:9.2f} us",
                                flush=True,
                            )
                        except Exception as exc:  # noqa: BLE001
                            msg = " | ".join(l.strip() for l in str(exc).strip().splitlines() if l.strip())
                            print(
                                f"PACKET packet={label:14s} rows={rows:5d} dtype={str(dtype).split('.')[-1]:10s} "
                                f"workers={workers} {name:15s} FAILED {msg[:160]}",
                                flush=True,
                            )
                    ttnn.deallocate(partial)
                    ttnn.deallocate(fractured)
        finally:
            close(mesh)

    # ---- 3: the largest l1_small_size the decode step can give away -----------
    for l1_small in (4096, 5120, 6144, 7168, 8192):
        mesh = open_mesh(shape, packet_bytes=None, l1_small=l1_small, trace=90112 * 12)
        try:
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.L1)
            grid = mesh.compute_with_storage_grid_size()
            # The decode step's tightest program: the MLP matmul at in0_block_w=13
            # on the 16-core grid, with a full decode-shaped activation resident.
            act = ttnn.from_torch(
                torch.randn(1, 1, 32, HIDDEN, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=width_sharded_l1(32, HIDDEN, MULTICHIP_BOUNDARY_CORES, grid),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            weight = ttnn.from_torch(
                torch.randn(1, 1, HIDDEN, 5120, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat4_b,
                memory_config=ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(
                        ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
                        (HIDDEN, 640),
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                ),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=13, per_core_M=1, per_core_N=20, fused_activation=None
            )
            try:
                out = ttnn.linear(
                    act,
                    weight,
                    dtype=ttnn.bfloat16,
                    memory_config=width_sharded_l1(32, 5120, MULTICHIP_BOUNDARY_CORES, grid),
                    program_config=program,
                )
                ttnn.deallocate(out)
                verdict = "ok"
            except Exception as exc:  # noqa: BLE001
                verdict = "FAILS: " + " ".join(str(exc).split())[:120]
            print(
                f"L1SMALL size={l1_small:5d} pool={view.total_bytes_per_bank} "
                f"semaphores={l1_small // 256:3d} decode_mlp_matmul={verdict}",
                flush=True,
            )
            ttnn.deallocate(act)
            ttnn.deallocate(weight)
        finally:
            close(mesh)


if __name__ == "__main__":
    main()
