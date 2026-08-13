# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a fractured residual would actually cost and save in the **decode** regime.

``topology_probe.py`` measures the boundary contracts in the DRAM-interleaved
regime, which is the prefill regime.  The shipped decode step is different in the
one way that decides this question: its RMSNorms are **width-sharded in L1** and
cost ~8.4 us each on 16 cores (``tracy/sliding/decode_2048_perf_report.csv``),
not the ~25 us an interleaved norm costs at one tile row.  So the saving a
fractured residual would buy in decode is not the saving the interleaved probe
shows, and this probe prices the terms that actually differ, at the shipped
memory configs.

Two methodology points, both of which an earlier version of this probe got wrong
and which review round 3 caught:

1. **The replay floor is calibrated, not assumed.**  Every measurement here is a
   traced replay, and a replay costs a fixed amount before any op runs.  An
   earlier version compared floor-cancelling *differences* (two norms) against
   floor-inclusive *absolutes* (a stats gather) in the same sum, which is not a
   valid comparison.  Each op is now traced at ``--repeats`` distinct copy counts
   and the per-op cost is the **slope** of replay time against copy count, with
   the intercept reported as the floor.  Rows print both.
2. **The distributed norm is priced whole, at the sharded layout.**  A fractured
   residual does not "add a stats all-gather" to an otherwise unchanged norm: it
   replaces one full-width ``rms_norm`` with ``rms_norm_pre_all_gather`` ->
   ``all_gather`` -> ``rms_norm_post_all_gather``, all three on the fractured
   1664-wide shard.  An earlier version measured only the middle term at the
   shipped layout and the first term on DRAM-interleaved inputs.  All three now
   run on the width-sharded L1 layout the contract would actually use, and the
   comparison is path-vs-path.

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


def _free(out) -> None:
    if isinstance(out, (list, tuple)):
        for tensor in out:
            ttnn.deallocate(tensor)
    elif out is not None:
        ttnn.deallocate(out)


def replay_us(mesh, fn, copies: int, iters: int, rounds: int) -> float:
    """Min wall time per replay of a trace holding ``copies`` back-to-back calls."""
    for _ in range(2):
        _free(fn())
    ttnn.synchronize_device(mesh)
    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
    held = [fn() for _ in range(copies)]
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
    for tensor in held:
        _free(tensor)
    return best * 1e6


def cost(mesh, label: str, fn, repeats: tuple[int, ...], iters: int, rounds: int) -> float:
    """Per-op cost as the slope of replay time vs copy count; prints the floor too.

    Two points are enough for a straight line, but the extra ones are the check
    that it *is* a line -- a trace whose ops interact would not be.
    """
    try:
        times = [replay_us(mesh, fn, copies, iters, rounds) for copies in repeats]
    except Exception as exc:  # noqa: BLE001 - a probe records failures
        msg = " | ".join(line.strip() for line in str(exc).strip().splitlines() if line.strip())
        print(f"FRAC {label:44s} FAILED {msg[:200]}", flush=True)
        return float("nan")
    n = len(repeats)
    mean_x = sum(repeats) / n
    mean_y = sum(times) / n
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(repeats, times)) / sum((x - mean_x) ** 2 for x in repeats)
    floor = mean_y - slope * mean_x
    points = " ".join(f"{c}:{t:.2f}" for c, t in zip(repeats, times))
    print(f"FRAC {label:44s} per_op={slope:7.2f} us  floor={floor:6.2f} us  [{points}]", flush=True)
    return slope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", default="1,2,4,8", help="copy counts per trace, for the slope fit")
    parser.add_argument("--mesh", default="x".join(str(d) for d in DEFAULT_MESH_SHAPE))
    args = parser.parse_args()
    repeats = tuple(int(v) for v in args.repeats.split(","))

    mesh = open_multichip_mesh(tuple(int(v) for v in args.mesh.split("x")), trace_region_size=90112 * 24)
    devices = mesh.get_num_devices()
    grid = mesh.compute_with_storage_grid_size()
    fractured_width = HIDDEN // devices
    kernel = norm_compute_kernel_config(mesh.arch())
    try:
        print(
            f"FRAC rows={ROWS} full={HIDDEN} fractured={fractured_width} devices={devices} "
            f"repeats={repeats} (per_op is the slope, floor the intercept)"
        )

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

        held = []

        # ---- the two norms, at every legal core count for each width -----------
        # 6656 = 208 tiles -> {4, 8, 13, 16, 26, 52}; 1664 = 52 tiles -> {4, 13, 26, 52}
        full_norm = float("nan")
        best_fractured_norm = float("inf")
        for width, counts in ((HIDDEN, (MULTICHIP_BOUNDARY_CORES,)), (fractured_width, (4, 13, 26, 52))):
            for cores in counts:
                tensor, memcfg, program, weight = sharded(width, cores)
                held.append((tensor, weight))
                norm = cost(
                    mesh,
                    f"norm       w={width:5d} cores={cores:3d}",
                    lambda t=tensor, p=program, m=memcfg, w=weight: ttnn.rms_norm(
                        t, epsilon=EPS, weight=w, program_config=p, memory_config=m, compute_kernel_config=kernel
                    ),
                    repeats,
                    args.iters,
                    args.rounds,
                )
                cost(
                    mesh,
                    f"add        w={width:5d} cores={cores:3d}",
                    lambda t=tensor, m=memcfg: ttnn.add(t, t, memory_config=m),
                    repeats,
                    args.iters,
                    args.rounds,
                )
                if width == HIDDEN:
                    full_norm = norm
                elif norm == norm:  # not NaN
                    best_fractured_norm = min(best_fractured_norm, norm)

        # ---- the distributed norm, whole, on the shipped sharded layout --------
        # This is what the fractured contract would actually dispatch in place of
        # one full-width rms_norm: pre -> all_gather(stats) -> post, all three at
        # the fractured width, in L1.
        # Which core counts a *distributed* norm is even legal at, on this layout,
        # is itself a result: of the four that divide the fractured width's 52
        # tiles, only 4 works.  13, 26 and 52 raise
        #   "Sharded layernorm does not support a non-rectangular core grid for
        #    distributed norm" (layernorm_device_operation.cpp:197)
        # and 16 does not divide 52 at all.  A distributed norm also *requires*
        # the sharded program config -- without one it raises "std::get: wrong
        # index for variant", which is why an earlier version of this probe
        # measured the pre-op on DRAM-interleaved inputs instead.
        for cores in (13, 26, 52):
            tensor, memcfg, program, _weight = sharded(fractured_width, cores)
            try:
                out = ttnn.rms_norm_pre_all_gather(tensor, dtype=ttnn.bfloat16, program_config=program)
                print(f"FRAC dist legality cores={cores:3d} LEGAL")
                _free(out)
            except Exception as exc:  # noqa: BLE001 - the failure is the result
                first = str(exc).strip().splitlines()[0]
                print(f"FRAC dist legality cores={cores:3d} ILLEGAL {first[:150]}", flush=True)
            ttnn.deallocate(tensor)

        DIST_CORES = 4
        source, source_memcfg, source_program, gamma = sharded(fractured_width, DIST_CORES)
        held.append((source, gamma))

        def make_stats():
            return ttnn.rms_norm_pre_all_gather(source, dtype=ttnn.bfloat16, program_config=source_program)

        # ``rms_norm_post_all_gather`` requires *sharded* statistics ("Stats must be
        # sharded", layernorm_device_operation.cpp:236), so the stats all-gather is
        # measured writing back into L1, which is also what the contract would do.
        stats_memcfg = width_sharded_l1(ROWS, TILE * devices, 1, grid)
        stats = make_stats()
        gathered = ttnn.all_gather(stats, dim=3, memory_config=stats_memcfg)
        print(f"FRAC stats shape={tuple(stats.shape)} gathered={tuple(gathered.shape)}")

        pre = cost(
            mesh, f"dist pre_all_gather  (L1 sharded, {DIST_CORES}c)", make_stats, repeats, args.iters, args.rounds
        )
        gather = cost(
            mesh,
            "dist stats all_gather",
            lambda: ttnn.all_gather(stats, dim=3, memory_config=stats_memcfg),
            repeats,
            args.iters,
            args.rounds,
        )
        post = cost(
            mesh,
            f"dist post_all_gather (L1 sharded, {DIST_CORES}c)",
            lambda: ttnn.rms_norm_post_all_gather(
                source,
                gathered,
                epsilon=EPS,
                weight=gamma,
                memory_config=source_memcfg,
                program_config=source_program,
                compute_kernel_config=kernel,
            ),
            repeats,
            args.iters,
            args.rounds,
        )

        # ---- the verdict, floor-free on both sides -----------------------------
        distributed = pre + gather + post
        print(
            f"FRAC VERDICT replicated_norm={full_norm:7.2f} us  "
            f"distributed_norm={distributed:7.2f} us (pre {pre:.2f} + ag {gather:.2f} + post {post:.2f})  "
            f"delta={distributed - full_norm:+7.2f} us per distributed norm"
        )
        print(
            f"FRAC VERDICT best_fractured_plain_norm={best_fractured_norm:7.2f} us  "
            f"(what the contract would pay if the norm needed no statistics at all -- "
            f"it is not an option, a fractured residual has no full row to normalise over)"
        )
        print(f"FRAC VERDICT two distributed norms per decode step -> {2 * (distributed - full_norm):+7.2f} us")
        for tensor, weight in held:
            ttnn.deallocate(tensor)
            ttnn.deallocate(weight)
    finally:
        close_multichip_mesh(mesh)


if __name__ == "__main__":
    main()
