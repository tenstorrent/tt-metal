# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused matmul + reduce-scatter against the separate pair, at the decode shapes.

Both row-parallel decode projections end in a collective, which is the textbook
case for ``ttnn.experimental.matmul_reduce_scatter_async``.  The catch is that the
fused op takes an ordinary 2D-multicast matmul program config and reserves part
of the compute grid for the reduce-scatter workers, while this layer's decode
projections are ``MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig`` -- the
DRAM-bound decode form, where the op picks its own core set.

So the fusion cannot be judged by one API call.  Four arms per boundary, all
traced and all floor-calibrated:

* ``dram_split``  -- the shipped pair: DRAM-sharded matmul, then RS, then AG;
* ``dram_fused``  -- the fused op handed the same DRAM-sharded program config;
* ``mcast_split`` -- 2D-multicast matmul, then RS, then AG (the fused op's own
  matmul form, *unfused*, so the program-config change is separated from the
  fusion);
* ``mcast_fused`` -- the fused op as the op intends it to be called.

``mcast_split`` is the control that makes the comparison honest: if
``mcast_fused`` beats ``mcast_split`` the fusion is a real win and the question
becomes whether the 2D-multicast matmul can afford the decode shape at all.

    python .../bench/fused_ccl_probe.py
"""

from __future__ import annotations

import argparse
import math
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_L1_SMALL_SIZE,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import (
    decode_matmul_program_config,
    dram_sharded_weight_memcfg,
    width_sharded_l1,
)

TILE = 32
ROWS = 32
HIDDEN = 6656
TP = 4
BOUNDARY_CORES = 16
BF16, BFP8, BFP4 = ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b
COPIES = (1, 2, 4, 8)

#: The two row-parallel decode projections: (name, per-device K, weight dtype,
#: shipped in0_block_w).
BOUNDARIES = (("o_proj", 1024, BFP8, 2), ("mlp_down", 5120, BFP4, 10))


def slope_us(mesh, body, rounds=3):
    times = []
    for n in COPIES:
        outs = body(n)
        ttnn.synchronize_device(mesh)
        del outs
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        traced = body(n)
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        ttnn.synchronize_device(mesh)
        for _ in range(4):
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        best = float("inf")
        for _ in range(rounds):
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            best = min(best, (time.perf_counter() - t0) / 32 * 1e6)
        times.append(best)
        ttnn.release_trace(mesh, trace_id)
        del traced
    xs = torch.tensor(COPIES, dtype=torch.float64)
    ys = torch.tensor(times, dtype=torch.float64)
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    return slope, float(ys.mean() - slope * xs.mean()), times


def run(mesh, grid, name, fn):
    try:
        slope, floor, times = slope_us(mesh, fn)
        print(
            f"FUSED {name:38s} per_call={slope:8.2f} us  floor={floor:7.2f} us  "
            f"raw={'/'.join(f'{t:.1f}' for t in times)}",
            flush=True,
        )
        return slope
    except Exception as exc:  # noqa: BLE001
        msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
        print(f"FUSED-FAILED {name:38s} {msg[:400]}", flush=True)
        return None


def boundary(mesh, grid, role, k, wdtype, in0_block_w):
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, packer_l1_acc=True
    )
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)]
    barrier = ttnn.create_global_semaphore(mesh, crs, 0)
    ag_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]

    w = ttnn.from_torch(
        torch.randn(1, 1, k, HIDDEN) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=wdtype,
        memory_config=dram_sharded_weight_memcfg(k, HIDDEN, mesh),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    w_il = ttnn.from_torch(
        torch.randn(1, 1, k, HIDDEN) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=wdtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    x_sharded = ttnn.from_torch(
        torch.randn(1, 1, ROWS, k),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=width_sharded_l1(ROWS, k, BOUNDARY_CORES, grid),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    x_il = ttnn.from_torch(
        torch.randn(1, 1, ROWS, k),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    # The fused op requires caller-owned persistent buffers: an input-shaped
    # intermediate and the scattered output.
    interm = ttnn.from_torch(
        torch.zeros(1, 1, ROWS, HIDDEN),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    rs_out = ttnn.from_torch(
        torch.zeros(1, 1, ROWS, HIDDEN // TP),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    dram_pc = decode_matmul_program_config(ROWS, HIDDEN, BOUNDARY_CORES, in0_block_w)
    mcast_grid = (8, 6)
    mcast_pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=mcast_grid,
        in0_block_w=max(1, min(k // TILE // mcast_grid[0], 4)),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=max(1, math.ceil(HIDDEN / TILE / mcast_grid[0])),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )
    boundary_mem = width_sharded_l1(ROWS, HIDDEN, BOUNDARY_CORES, grid)

    def rs_ag(partial):
        scattered = ttnn.experimental.reduce_scatter_minimal_async(
            partial,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=sems,
            barrier_semaphore=barrier,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=1,
        )
        ttnn.deallocate(partial)
        out = ttnn.experimental.all_gather_async(
            scattered,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ag_sems,
            memory_config=boundary_mem,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=1,
        )
        ttnn.deallocate(scattered)
        return out

    def dram_split(n):
        outs = []
        for _ in range(n):
            outs.append(
                rs_ag(
                    ttnn.linear(
                        x_sharded,
                        w,
                        dtype=BF16,
                        memory_config=width_sharded_l1(ROWS, HIDDEN, BOUNDARY_CORES, grid),
                        program_config=dram_pc,
                        compute_kernel_config=ck,
                    )
                )
            )
        for t in outs:
            ttnn.deallocate(t)
        return []

    def mcast_split(n):
        outs = []
        for _ in range(n):
            outs.append(
                rs_ag(
                    ttnn.linear(
                        x_il,
                        w_il,
                        dtype=BF16,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=mcast_pc,
                        compute_kernel_config=ck,
                    )
                )
            )
        for t in outs:
            ttnn.deallocate(t)
        return []

    def fused(n, x, weight, pc, mm_mem):
        outs = []
        for _ in range(n):
            _mm, scattered = ttnn.experimental.matmul_reduce_scatter_async(
                x,
                weight,
                persistent_intermediate_buffer=interm,
                persistent_output_buffer=rs_out,
                dim=3,
                multi_device_global_semaphore=sems,
                reduce_scatter_core_grid_offset=ttnn.CoreCoord(0, 6),
                barrier_semaphore=barrier,
                memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                memory_config_mm=mm_mem,
                program_config=pc,
                compute_kernel_config=ck,
            )
            out = ttnn.experimental.all_gather_async(
                scattered,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=ag_sems,
                memory_config=boundary_mem,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=1,
            )
            outs.append(out)
        for t in outs:
            ttnn.deallocate(t)
        return []

    run(mesh, grid, f"{role} dram_split (shipped)", dram_split)
    run(
        mesh,
        grid,
        f"{role} dram_fused",
        lambda n: fused(n, x_sharded, w, dram_pc, width_sharded_l1(ROWS, HIDDEN, BOUNDARY_CORES, grid)),
    )
    run(mesh, grid, f"{role} mcast_split", mcast_split)
    run(mesh, grid, f"{role} mcast_fused", lambda n: fused(n, x_il, w_il, mcast_pc, ttnn.DRAM_MEMORY_CONFIG))
    for t in (w, w_il, x_sharded, x_il, interm, rs_out):
        ttnn.deallocate(t)


def gathered_input_o_proj(mesh, grid):
    """OPT-008: the *other* row-parallel decomposition, fused.

    Instead of a local-input/full-output matmul followed by a reduction, gather
    the attention output first and give each device a *column* slice of
    ``o_proj``: ``all_gather(gated) -> matmul(4096 x 1664)``, which is exactly the
    shape ``ttnn.experimental.all_gather_matmul_async`` fuses.  The multichip
    stage measured this decomposition unfused (``gather_heads``, 9.1 % slower on
    the decode boundary chain); this arm asks whether the fusion recovers that.

    The projection's output is then 1664 wide per device -- a *fractured*
    residual -- so a second all-gather restores the layer's replicated contract.
    That gather is included, because the shipped arm it is compared against also
    ends replicated.
    """
    ck = ttnn.init_device_compute_kernel_config(
        mesh.arch(), math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, packer_l1_acc=True
    )
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    ag_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    ag2_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    local_n = HIDDEN // TP
    w_col = ttnn.from_torch(
        torch.randn(1, 1, 4096, local_n) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BFP8,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    x_il = ttnn.from_torch(
        torch.randn(1, 1, ROWS, 1024),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=BF16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    mcast_grid = (8, 6)
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=mcast_grid,
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=1,
        per_core_N=max(1, math.ceil(local_n / TILE / mcast_grid[0])),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

    def tail(local):
        out = ttnn.experimental.all_gather_async(
            local,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=ag2_sems,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=CCL_TOPOLOGY,
            num_workers_per_link=1,
        )
        ttnn.deallocate(local)
        return out

    def ag_then_mm(n):
        outs = []
        for _ in range(n):
            gathered = ttnn.experimental.all_gather_async(
                x_il,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=ag_sems,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=1,
            )
            local = ttnn.linear(
                gathered,
                w_col,
                dtype=BF16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
                compute_kernel_config=ck,
            )
            ttnn.deallocate(gathered)
            outs.append(tail(local))
        for t in outs:
            ttnn.deallocate(t)
        return []

    def ag_mm_fused(n):
        outs = []
        for _ in range(n):
            _gathered, local = ttnn.experimental.all_gather_matmul_async(
                x_il,
                w_col,
                persistent_output_buffer=None,
                dim=3,
                all_gather_core_grid_offset=ttnn.CoreCoord(0, 6),
                multi_device_global_semaphore=ag_sems,
                topology=CCL_TOPOLOGY,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=pc,
                compute_kernel_config=ck,
            )
            outs.append(tail(local))
        for t in outs:
            ttnn.deallocate(t)
        return []

    run(mesh, grid, "o_proj gathered_input ag+mm", ag_then_mm)
    run(mesh, grid, "o_proj gathered_input ag_mm_fused", ag_mm_fused)
    for t in (w_col, x_il):
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roles", default="o_proj,mlp_down")
    ap.add_argument("--gathered-input", action="store_true")
    args = ap.parse_args()
    mesh = open_multichip_mesh((1, 4), trace_region_size=90112 * 12, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        wanted = args.roles.split(",")
        for role, k, wdtype, bw in BOUNDARIES:
            if role in wanted:
                boundary(mesh, grid, role, k, wdtype, bw)
        if args.gathered_input:
            gathered_input_o_proj(mesh, grid)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
