# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""The prefill reduction, at the op level, for every implementation and knob.

The 8192-row prefill collective is ~19 % of the prefill layer, and whole-layer
A/B cannot resolve it: ``logs/ab_prefill_ccl.log`` puts the *same-configuration*
prefill spread at 0.7-3.6 % inside one process, which is larger than any of the
candidates.  So the choice is made here, on the collective itself, at the shipped
payload (BFLOAT8_B), the shipped packet size and the real per-device shape.

Arms:

* ``wrapper all_reduce``        -- shipped;
* ``wrapper rs+ag``            -- the composite pair;
* ``async rs+ag ag_w=<n>``     -- ``reduce_scatter_minimal_async`` +
  ``all_gather_async``, with the all-gather worker count `ttnn.all_gather` does
  not expose, and optionally this layer's own staging buffers.

    python .../bench/prefill_ccl_probe.py --rows 8192
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_L1_SMALL_SIZE,
    close_multichip_mesh,
    open_multichip_mesh,
)

HIDDEN = 6656
TP = 4


def timed(mesh, fn, iters=8, rounds=3):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    best = float("inf")
    for _ in range(rounds):
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(iters)]
        ttnn.synchronize_device(mesh)
        best = min(best, (time.perf_counter() - t0) / iters * 1e6)
        for t in outs:
            ttnn.deallocate(t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8192)
    ap.add_argument("--dtype", default="bfloat8_b")
    args = ap.parse_args()
    dtype = {"bfloat8_b": ttnn.bfloat8_b, "bfloat16": ttnn.bfloat16}[args.dtype]

    mesh = open_multichip_mesh((1, 4), trace_region_size=0, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        rs_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)]
        ag_sems = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
        rs_barrier = ttnn.create_global_semaphore(mesh, crs, 0)
        ag_barrier = ttnn.create_global_semaphore(mesh, crs, 0)

        partial = ttnn.from_torch(
            torch.randn(1, 1, args.rows, HIDDEN),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

        def wrapper_all_reduce():
            return ttnn.all_reduce(partial, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY)

        def wrapper_rs_ag(workers):
            def fn():
                sc = ttnn.reduce_scatter(
                    partial,
                    dim=3,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=CCL_TOPOLOGY,
                    num_workers_per_link=workers,
                    use_l1_small_for_semaphores=True,
                )
                out = ttnn.all_gather(sc, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(sc)
                return out

            return fn

        def async_rs_ag(rs_workers, ag_workers, ag_barrier_on=True):
            def fn():
                sc = ttnn.experimental.reduce_scatter_minimal_async(
                    partial,
                    persistent_output_buffers=None,
                    dim=3,
                    multi_device_global_semaphore=rs_sems,
                    barrier_semaphore=rs_barrier,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=CCL_TOPOLOGY,
                    num_workers_per_link=rs_workers,
                )
                out = ttnn.experimental.all_gather_async(
                    sc,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=ag_sems,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=CCL_TOPOLOGY,
                    **({} if ag_workers is None else {"num_workers_per_link": ag_workers}),
                    # The all-gather takes a barrier semaphore too, and the layer
                    # passes one.  It is not free, so it is measured rather than
                    # assumed: ``ag_barrier_on=False`` is the arm without it.
                    **({"barrier_semaphore": ag_barrier} if ag_barrier_on else {}),
                )
                ttnn.deallocate(sc)
                return out

            return fn

        arms = [("wrapper all_reduce", wrapper_all_reduce), ("wrapper rs+ag w=4", wrapper_rs_ag(4))]
        # Every arm carries the all-gather barrier semaphore, which is what the
        # layer ships; the no-barrier arm is measured once, at the shipped worker
        # count, so its cost is on the record without being spread through the
        # table as if it were a candidate.
        for agw in (None, 1, 2, 4):
            arms.append((f"async rs_w=4 ag_w={agw}", async_rs_ag(4, agw)))
        for rsw in (1, 2):
            arms.append((f"async rs_w={rsw} ag_w=4", async_rs_ag(rsw, 4)))
        arms.append(("async rs_w=4 ag_w=None NO ag_barrier", async_rs_ag(4, None, ag_barrier_on=False)))

        for name, fn in arms:
            try:
                print(f"PREFILLCCL rows={args.rows} {args.dtype:10s} {name:34s} {timed(mesh, fn):9.1f} us", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"PREFILLCCL-FAILED {name}: {str(exc).splitlines()[0][:200]}", flush=True)
        ttnn.deallocate(partial)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
