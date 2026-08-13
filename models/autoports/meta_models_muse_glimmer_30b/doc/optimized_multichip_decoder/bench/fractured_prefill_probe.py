# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a fractured *prefill* residual is actually worth, on this stage's collectives.

The multichip stage measured the fractured residual family 1.14x faster than the
replicated one on its prefill boundary chain and left it as
"the single largest remaining prefill lever ... worth an estimated 11 % of the
prefill layer", deferred to whoever owns the layer stack.  Two things about that
number needed testing before it could be accepted or declined again:

* it was measured with the **composite wrapper** collectives.  This stage's prefill
  reduction is 15.2 % faster than the ``all_reduce`` the replicated arm used, so
  the same absolute saving is a smaller fraction of a smaller total;
* it prices the boundary chain, not the layer.  A layer-internal fractured
  residual must also pay to *get into* the fractured layout at the layer's
  replicated input and back out at its replicated output.

This probe measures both arms as complete, stackable chains at the real 8192-row
prefill shape, in the DRAM-interleaved regime prefill actually runs in:

* ``replicated`` (shipped) -- ``all_reduce`` -> residual add (6656) -> ``rms_norm`` (6656);
* ``fractured``            -- ``reduce_scatter`` -> residual add (1664) -> distributed
  ``rms_norm`` (pre_all_gather -> all_gather stats -> post_all_gather, 1664) ->
  ``all_gather`` back to 6656.

Both start from the same per-device partial and end with a full-width normalised
tensor, i.e. on the contract the next column-parallel matmul needs, so a winner is
stackable rather than a local saving that has to be undone.  ``--sublayers 2``
chains both sublayers of a layer so the entry/exit cost is amortised the way a
real layer would amortise it.

    python .../bench/fractured_prefill_probe.py --rows 8192
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
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import norm_compute_kernel_config

HIDDEN = 6656
TP = 4
LOCAL = HIDDEN // TP
EPS = 1e-6


def timed(mesh, fn, iters=4, rounds=3):
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
    ap.add_argument("--sublayers", type=int, default=2)
    args = ap.parse_args()

    mesh = open_multichip_mesh((1, 4), trace_region_size=0, l1_small_size=DEFAULT_L1_SMALL_SIZE)
    ttnn.SetDefaultDevice(mesh)
    try:
        grid = mesh.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

        def sem(n):
            return [ttnn.create_global_semaphore(mesh, crs, 0, ttnn.BufferType.L1_SMALL) for _ in range(n)]

        rs_s, ag_s, st_s = sem(3), sem(2), sem(2)
        rs_b, ag_b, st_b = sem(1)[0], sem(1)[0], sem(1)[0]
        ck = norm_compute_kernel_config(mesh.arch())

        def dev(t, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                t,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        partial = dev(torch.randn(1, 1, args.rows, HIDDEN) * 0.05, ttnn.bfloat8_b)
        residual_full = dev(torch.randn(1, 1, args.rows, HIDDEN))
        residual_local = dev(torch.randn(1, 1, args.rows, LOCAL))
        w_full = dev(torch.randn(1, 1, 1, HIDDEN) * 0.02)
        w_local = dev(torch.randn(1, 1, 1, LOCAL) * 0.02)

        def reduce_scatter(x):
            return ttnn.experimental.reduce_scatter_minimal_async(
                x,
                persistent_output_buffers=None,
                dim=3,
                multi_device_global_semaphore=rs_s,
                barrier_semaphore=rs_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                num_workers_per_link=4,
            )

        def all_gather(x, sems, barrier):
            return ttnn.experimental.all_gather_async(
                x,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=sems,
                barrier_semaphore=barrier,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
            )

        def replicated():
            """The shipped chain: reduce to full width, add, normalise at 6656."""
            out = None
            for _ in range(args.sublayers):
                sc = reduce_scatter(partial)
                reduced = all_gather(sc, ag_s, ag_b)
                ttnn.deallocate(sc)
                hidden = ttnn.add(residual_full, reduced, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(reduced)
                normed = ttnn.rms_norm(hidden, epsilon=EPS, weight=w_full, compute_kernel_config=ck)
                ttnn.deallocate(hidden)
                if out is not None:
                    ttnn.deallocate(out)
                out = normed
            return out

        def fractured():
            """Fractured: add and normalise at 1664, gather once for the next matmul."""
            out = None
            for _ in range(args.sublayers):
                sc = reduce_scatter(partial)
                hidden = ttnn.add(residual_local, sc, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(sc)
                stats = ttnn.rms_norm_pre_all_gather(hidden, compute_kernel_config=ck, dtype=ttnn.bfloat16)
                gathered = all_gather(stats, st_s, st_b)
                ttnn.deallocate(stats)
                normed_local = ttnn.rms_norm_post_all_gather(
                    hidden, gathered, epsilon=EPS, weight=w_local, compute_kernel_config=ck
                )
                ttnn.deallocate(gathered)
                ttnn.deallocate(hidden)
                normed = all_gather(normed_local, ag_s, ag_b)
                ttnn.deallocate(normed_local)
                if out is not None:
                    ttnn.deallocate(out)
                out = normed
            return out

        for name, fn in (("replicated (shipped)", replicated), ("fractured", fractured)):
            try:
                us = timed(mesh, fn)
                print(
                    f"FRACPREFILL rows={args.rows} sublayers={args.sublayers} {name:22s} {us:9.1f} us",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                print(f"FRACPREFILL-FAILED {name:22s} {msg[:420]}", flush=True)
        for t in (partial, residual_full, residual_local, w_full, w_local):
            ttnn.deallocate(t)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
