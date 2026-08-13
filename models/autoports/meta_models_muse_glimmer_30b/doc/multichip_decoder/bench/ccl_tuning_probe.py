# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tune the decode-shape collectives, which are 13 % of the decode step.

``tracy/*/decode_2048_perf_report.txt`` puts the shipped pair at ~14 us
(``ReduceScatter``) + ~13 us (``AllGather``) per sublayer for a 40 KB payload,
i.e. 2.7 GB/s -- against the 80 GB/s the same fabric reaches at the 8192-row
payload in ``logs/ccl_probe_1x4_ring.log``.  At this size the collective is pure
fixed cost, so the levers are the op variant and the sync hyperparameters, not
the payload.

Every candidate is measured **traced**, at the real decode shape
(``[1, 1, 32, 6656]`` BF16), for both memory configs the layer could hand it: DRAM
interleaved, and the width-sharded L1 boundary layout the decode step actually
uses.

    python .../bench/ccl_tuning_probe.py --out logs/ccl_tuning_probe.log
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    CCL_TOPOLOGY,
    DEFAULT_MESH_SHAPE,
    MULTICHIP_BOUNDARY_CORES,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import width_sharded_l1

HIDDEN = 6656
ROWS = 32


def traced(mesh, fn, iters: int, rounds: int) -> float:
    for _ in range(2):
        out = fn()
        _free(out)
    ttnn.synchronize_device(mesh)
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
    _free(held)
    return best * 1e6


def _free(out) -> None:
    if isinstance(out, (list, tuple)):
        for tensor in out:
            ttnn.deallocate(tensor)
    elif out is not None:
        ttnn.deallocate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()

    mesh_shape = tuple(int(v) for v in args.mesh.split("x"))
    mesh = open_multichip_mesh(mesh_shape, trace_region_size=90112 * 24)
    devices = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    try:
        torch_partial = torch.randn(1, 1, ROWS, HIDDEN, dtype=torch.bfloat16)
        boundary = width_sharded_l1(ROWS, HIDDEN, MULTICHIP_BOUNDARY_CORES, grid)
        scattered_l1 = width_sharded_l1(ROWS, HIDDEN // devices, MULTICHIP_BOUNDARY_CORES, grid)

        for layout_name, memcfg, out_memcfg in (
            ("dram", ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG),
            ("l1_sharded", boundary, boundary),
        ):
            partial = ttnn.from_torch(
                torch_partial,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            fractured = ttnn.from_torch(
                torch_partial,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if layout_name == "dram" else scattered_l1,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )

            cases = {
                "all_reduce": lambda: ttnn.all_reduce(partial, memory_config=out_memcfg, topology=CCL_TOPOLOGY),
                "reduce_scatter": lambda: ttnn.reduce_scatter(
                    partial, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY
                ),
                "all_gather": lambda: ttnn.all_gather(fractured, dim=3, memory_config=out_memcfg),
            }
            # reduce_scatter's sync hyperparameters, which the production op exposes
            # and all_gather deprecates.
            for chunks in (2, 5, 10, 20):
                for workers in (1, 2, 4):
                    cases[
                        f"reduce_scatter[c{chunks}_w{workers}]"
                    ] = lambda chunks=chunks, workers=workers: ttnn.reduce_scatter(
                        partial,
                        dim=3,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        topology=CCL_TOPOLOGY,
                        chunks_per_sync=chunks,
                        num_workers_per_link=workers,
                    )
            for buffers in (2, 4, 8):
                cases[f"reduce_scatter[b{buffers}]"] = lambda buffers=buffers: ttnn.reduce_scatter(
                    partial,
                    dim=3,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    topology=CCL_TOPOLOGY,
                    num_buffers_per_channel=buffers,
                )
            # NOTE: by the time this case runs, the sweep above has built ~18
            # distinct CCL programs and each holds a 256 B semaphore, so on a mesh
            # with the shipped 6 KB L1_SMALL region this one reports "Out of
            # Memory" -- the region is exhausted, *not* the part.  The shipped
            # decode path uses this flag successfully; see DEFAULT_L1_SMALL_SIZE.
            cases["reduce_scatter[l1_small_sem]"] = lambda: ttnn.reduce_scatter(
                partial,
                dim=3,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=CCL_TOPOLOGY,
                use_l1_small_for_semaphores=True,
            )
            cases["reduce_scatter[nl1]"] = lambda: ttnn.reduce_scatter(
                partial, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY, num_links=1
            )
            cases["reduce_scatter[nl2]"] = lambda: ttnn.reduce_scatter(
                partial, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=CCL_TOPOLOGY, num_links=2
            )
            cases["reduce_scatter[linear]"] = lambda: ttnn.reduce_scatter(
                partial, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG, topology=ttnn.Topology.Linear
            )
            cases["all_reduce[linear]"] = lambda: ttnn.all_reduce(
                partial, memory_config=out_memcfg, topology=ttnn.Topology.Linear
            )
            cases["all_reduce[nl1]"] = lambda: ttnn.all_reduce(
                partial, memory_config=out_memcfg, topology=CCL_TOPOLOGY, num_links=1
            )

            for name, fn in cases.items():
                try:
                    us = traced(mesh, fn, args.iters, args.rounds)
                    print(f"CCLTUNE {layout_name:11s} {name:34s} {us:8.2f} us", flush=True)
                except Exception as exc:  # noqa: BLE001 - a probe records failures
                    msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                    print(f"CCLTUNE {layout_name:11s} {name:34s} FAILED {msg[:220]}", flush=True)
            ttnn.deallocate(partial)
            ttnn.deallocate(fractured)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
