# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a fractured residual would actually cost and save in the **decode** regime.

``topology_probe.py`` measures the boundary contracts in the DRAM-interleaved
regime, which is the prefill regime.  The shipped decode step is different in the
one way that decides this question: its RMSNorms are **width-sharded in L1** and
cost 8.43 us each on 16 cores (``tracy/sliding/decode_2048_perf_report.csv``),
not the ~25 us an interleaved norm costs at one tile row.  So the saving a
fractured residual would buy in decode is not the saving the interleaved probe
shows, and this probe prices the three terms that actually differ, at the shipped
memory configs:

1. a sharded RMSNorm on the **full** 6656-wide residual (16 cores, what the
   replicated contract pays) against one on a **fractured** 1664-wide residual
   (4, 13 or 26 cores -- every count that divides its 52 tiles);
2. the residual add at both widths;
3. the distributed-norm stats all-gather the fractured contract has to add,
   which is the ``[rows, 32]`` payload of ``rms_norm_pre_all_gather``.

The question it answers: is ``2 x (norm_full - norm_fractured) + 2 x (add_full -
add_fractured)`` bigger or smaller than ``2 x stats_all_gather``?

    python .../bench/fractured_decode_probe.py --out logs/fractured_decode_probe.log
"""

from __future__ import annotations

import argparse
import time

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.fused_decoder import _norm_subblock_w, norm_compute_kernel_config
from models.autoports.meta_models_muse_glimmer_30b.tt.multichip_decoder import (
    DEFAULT_MESH_SHAPE,
    MULTICHIP_BOUNDARY_CORES,
    close_multichip_mesh,
    open_multichip_mesh,
)
from models.autoports.meta_models_muse_glimmer_30b.tt.optimized_decoder import width_sharded_l1

HIDDEN = 6656
TILE = 32
ROWS = 32
EPS = 1e-5


def traced(mesh, fn, iters: int, rounds: int) -> float:
    for _ in range(2):
        ttnn.deallocate(fn())
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
    ttnn.deallocate(held)
    return best * 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()

    mesh = open_multichip_mesh(tuple(int(v) for v in args.mesh.split("x")), trace_region_size=90112 * 24)
    devices = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    fractured_width = HIDDEN // devices
    kernel = norm_compute_kernel_config(mesh.arch())
    try:
        print(f"FRAC rows={ROWS} full={HIDDEN} fractured={fractured_width} devices={devices}")

        def sharded(width: int, cores: int):
            memcfg = width_sharded_l1(ROWS, width, cores, grid)
            tensor = ttnn.from_torch(
                torch.randn(1, 1, ROWS, width, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=memcfg,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            block_w = width // cores // TILE
            program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[min(cores, grid.x), -(-cores // grid.x)],
                subblock_w=_norm_subblock_w(block_w),
                block_h=ROWS // TILE,
                block_w=block_w,
                inplace=False,
            )
            weight = ttnn.from_torch(
                torch.randn(1, 1, 1, width, dtype=torch.bfloat16),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            return tensor, memcfg, program, weight

        # ---- the norms, at every legal core count for each width ---------------
        # 6656 = 208 tiles -> {4, 8, 13, 16, 26, 52}; 1664 = 52 tiles -> {4, 13, 26, 52}
        for width, counts in ((HIDDEN, (MULTICHIP_BOUNDARY_CORES,)), (fractured_width, (4, 13, 26, 52))):
            for cores in counts:
                try:
                    tensor, memcfg, program, weight = sharded(width, cores)
                    fn = lambda t=tensor, p=program, m=memcfg, w=weight: ttnn.rms_norm(
                        t, epsilon=EPS, weight=w, program_config=p, memory_config=m, compute_kernel_config=kernel
                    )
                    print(
                        f"FRAC norm      width={width:5d} cores={cores:3d} "
                        f"{traced(mesh, fn, args.iters, args.rounds):7.2f} us",
                        flush=True,
                    )
                    add = lambda t=tensor, m=memcfg: ttnn.add(t, t, memory_config=m)
                    print(
                        f"FRAC add       width={width:5d} cores={cores:3d} "
                        f"{traced(mesh, add, args.iters, args.rounds):7.2f} us",
                        flush=True,
                    )
                    ttnn.deallocate(tensor)
                    ttnn.deallocate(weight)
                except Exception as exc:  # noqa: BLE001 - a probe records failures
                    msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
                    print(f"FRAC norm      width={width:5d} cores={cores:3d} FAILED {msg[:200]}", flush=True)

        # ---- the stats all-gather the fractured contract has to add ------------
        stats_source = ttnn.from_torch(
            torch.randn(1, 1, ROWS, fractured_width, dtype=torch.bfloat16),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        stats = ttnn.rms_norm_pre_all_gather(stats_source, dtype=ttnn.bfloat16)
        stats = ttnn.reshape(stats, ttnn.Shape((1, 1, ROWS, TILE)))
        print(f"FRAC stats shape={tuple(stats.shape)}")
        gather = lambda: ttnn.all_gather(stats, dim=3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        print(f"FRAC stats_ag  width={TILE:5d} cores={'-':>3} {traced(mesh, gather, args.iters, args.rounds):7.2f} us")
        pre = lambda: ttnn.rms_norm_pre_all_gather(stats_source, dtype=ttnn.bfloat16)
        print(
            f"FRAC stats_pre width={fractured_width:5d} cores={'-':>3} "
            f"{traced(mesh, pre, args.iters, args.rounds):7.2f} us"
        )
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
