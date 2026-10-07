#!/usr/bin/env python3
"""Minimal repro: ttnn.all_reduce over the 8 mesh COLUMNS (cluster_axis=1, reduce_scatter + all_gather) of a [1,1,R,4096] bf16 DRAM tensor on a 4x8 Blackhole
Galaxy occasionally returns ONE stale/zero 32-column tile on ONE device (rate ~1e-3 per call chained 6..48 times in a trace, FABRIC_1D_RING).
No model weights: every device holds its own random tensor, N_CHAIN independent all_reduces are captured in one trace, the trace is replayed N times with identical
inputs, every output is compared bitwise with replay 0 (on device: ne + sum, outside the trace) and per-device counts are printed.

Usage (4x8 mesh, one process):
  python allreduce_race_repro.py                         # defaults: R=64 N_CHAIN=24 REPS=3000 MODE=rs_ag
  R=32 N_CHAIN=48 REPS=2000 python allreduce_race_repro.py
  MODE=ag_sum python allreduce_race_repro.py              # workaround: all_gather(dim 0, cluster_axis=1) + local sum  (0 events in 36000 calls in the model-level stress)
  MIX=1 python allreduce_race_repro.py                    # interleave a row-axis (cluster_axis=0) all_gather between the all_reduces (fabric traffic on both axes)
Env: R, N_CHAIN, REPS, MODE (rs_ag | ag_sum | links1), MIX (0|1), SEED.
Prints one line per glitching (replay, op index, device (row, col), number of differing elements) and a final summary
  SUMMARY host=<hostname> R=.. N_CHAIN=.. REPS=.. MODE=.. calls=<REPS*N_CHAIN> events=<n>.
"""
import os
import socket
import time

import torch

import ttnn

R = int(os.environ.get("R", "64"))
N_CHAIN = int(os.environ.get("N_CHAIN", "24"))
REPS = int(os.environ.get("REPS", "3000"))
MODE = os.environ.get("MODE", "rs_ag")
MIX = os.environ.get("MIX", "0") == "1"
D = 4096


def column_reduce(x):
    if MODE == "ag_sum":
        g = ttnn.all_gather(x, dim=0, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.sum(g, dim=0, keepdim=True)
        ttnn.deallocate(g)
        return o
    if MODE == "links1":
        return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, num_links=1)
    return ttnn.all_reduce(x, cluster_axis=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=256 * 1024 * 1024)
    try:
        rows, cols = tuple(mesh.shape)
        g = torch.Generator().manual_seed(int(os.environ.get("SEED", "0")))
        xs = []
        for i in range(N_CHAIN):  # distinct data per device and per op
            h = (torch.randn(rows, cols, R, D, generator=g) * 0.5).bfloat16()
            xs.append(
                ttnn.from_torch(
                    h,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh.shape, dims=(0, 1)),
                )
            )
        # (rows, cols, R, D) sharded on dims (0,1) -> per device [1, R, D]; make it 4D
        xs = [ttnn.reshape(x, [1, 1, R, D]) for x in xs]

        def run():
            outs = []
            for x in xs:
                o = column_reduce(x)
                if MIX:
                    gg = ttnn.all_gather(x, dim=0, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.deallocate(gg)
                outs.append(o)
            return outs

        run()  # compile
        ttnn.synchronize_device(mesh)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = run()
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        comp = ttnn.ConcatMesh2dToTensor(mesh, dims=(0, 1), mesh_shape=tuple(mesh.shape))

        def f32(t):
            return ttnn.typecast(t, ttnn.float32)

        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
        ref = [ttnn.clone(f32(o)) for o in outs]  # replay 0 = reference
        # the reference itself must be right: compare with the host sum of the 8 column inputs for the first op
        o0 = ttnn.to_torch(ttnn.get_device_tensors(outs[0])[0]).float().reshape(R, D)
        print("ref sanity: max |dev0 out - fp32 sum of row-0 inputs| =", end=" ")
        h0 = ttnn.to_torch(xs[0], mesh_composer=comp).float().reshape(rows, cols, R, D)
        print(float((o0 - h0[0].sum(0)).abs().max()), flush=True)

        def flags():
            fl = []
            for o, r in zip(outs, ref):
                fl.append(
                    ttnn.to_layout(ttnn.reshape(ttnn.sum(ttnn.ne(f32(o), r)), [1, 1, 1, 1]), ttnn.ROW_MAJOR_LAYOUT)
                )
            return ttnn.concat(fl, dim=3)

        flags()
        events = 0
        t0 = time.time()
        for rep in range(REPS):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            f = ttnn.to_torch(flags(), mesh_composer=comp).reshape(rows, cols, N_CHAIN)
            for r_, c_, j in f.nonzero().tolist():
                events += 1
                print(
                    f"GLITCH replay={rep} op={j} device=({r_},{c_}) differing_elements={int(f[r_, c_, j])}", flush=True
                )
        print(
            f"SUMMARY host={socket.gethostname()} R={R} N_CHAIN={N_CHAIN} REPS={REPS} MODE={MODE} MIX={int(MIX)} calls={REPS * N_CHAIN} "
            f"events={events} secs={time.time() - t0:.0f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
