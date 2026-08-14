# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""``matmul_reduce_scatter_async`` on the ``wo`` -> reduce-scatter edge, standalone.

Stage 03 named this as the one untried collective lever and bounded it at 5.3%
of the decode layer (``../multichip_decoder/README.md``, limitation 2): three of
the four collective edges neighbour a norm or a residual add, but ``wo`` is a
matmul feeding a reduce-scatter directly.

Wiring it into the layer raised ``matmul_reduce_scatter_async.cpp:36
mesh_device != nullptr`` (``layer_levers2.py``), which is not a rejection, so
this probe rebuilds the edge in isolation at the shipped decode shapes -- in0
``[1,1,32,1024]`` per die, weight ``[1024,2048]`` K-sharded, scatter dim 3 --
and sweeps the spellings until one runs, then prices it against the two ops it
replaces.

    python mmrs_probe.py

Prints ``P|`` lines only.
"""
import math
import statistics
import sys
import time

import torch

import ttnn

sys.path.insert(0, "/home/raahem/tt-metal")
from models.common.modules.tt_ccl import TT_CCL

M, K_LOCAL, N, ND = 32, 1024, 2048, 4
REPS = 16

ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=60_000_000, l1_small_size=32768)


def slope(fn):
    def build(n):
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        for _ in range(n):
            fn()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        for _ in range(5):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        s = []
        for _ in range(30):
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            s.append((time.perf_counter() - t0) * 1e6)
        ttnn.release_trace(mesh, tid)
        return statistics.median(s)

    fn()
    ttnn.synchronize_device(mesh)
    return (build(REPS + 1) - build(1)) / REPS


try:
    ccl = TT_CCL(mesh)
    torch.manual_seed(0)

    x = ttnn.from_torch(
        torch.randn(1, 1, M, K_LOCAL) * 0.1,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, K_LOCAL, N) * 0.02,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )

    def rs(t):
        return ttnn.experimental.reduce_scatter_minimal_async(
            t,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )

    # --- the unfused reference: plain matmul, then reduce-scatter --------------
    mm = ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)
    print(
        f"P|unfused: matmul (2D default) {slope(lambda: ttnn.linear(x, w, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat16)):7.2f} us",
        flush=True,
    )
    print(f"P|unfused: reduce-scatter      {slope(lambda: rs(mm)):7.2f} us", flush=True)

    def mk(width, dtype=ttnn.bfloat16):
        return ttnn.from_torch(
            torch.zeros(1, 1, M, width),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    for grid, offset in (((8, 6), (0, 6)), ((8, 4), (0, 4)), ((8, 8), (0, 8))):
        per_core_n = max(1, math.ceil(N / 32 / grid[0]))
        pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=grid,
            in0_block_w=min(4, max(1, K_LOCAL // 32 // grid[0])),
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=per_core_n,
            out_block_w=max(1, per_core_n // 2),
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
            allowed_worker_cores=ttnn.CoreRangeSet(
                {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid[0] - 1, grid[1] - 1))}
            ),
        )
        interm, out_buf = mk(N), mk(N // ND)

        def leg(pc=pc, interm=interm, out_buf=out_buf, offset=offset):
            _mm, r = ttnn.experimental.matmul_reduce_scatter_async(
                x,
                w,
                persistent_intermediate_buffer=interm,
                persistent_output_buffer=out_buf,
                dim=3,
                multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(),
                reduce_scatter_core_grid_offset=offset,
                barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=2,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=None,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
            )
            return ttnn.clone(r, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        try:
            o = leg()
            ref = ttnn.to_torch(rs(mm), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            got = ttnn.to_torch(o, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).float()
            print(
                f"P|fused matmul+RS grid {grid} rs@{offset}  {slope(leg):7.2f} us   "
                f"max|diff| vs unfused {(got - ref).abs().max().item():.3e}",
                flush=True,
            )
        except Exception as exc:
            print(f"P|fused matmul+RS grid {grid} rs@{offset}  FAILED {str(exc)[:160]}", flush=True)
finally:
    ttnn.close_mesh_device(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
print("P|done")
