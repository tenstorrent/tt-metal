#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Fixed, deterministic multi-chip workload for the Phase 2.2 A/B measurement
# (see PLAN_ETH_AGGREGATOR.md §5).
#
# Runs a matmul loop on every chip of the mesh and reports wall-clock for the
# timed region as machine-readable JSON on stdout.
#
# Deliberately compute-only: on a T3K, dispatch to the remote chips (4-7) is
# itself tunneled over ethernet, so any slowdown when the collector is running
# is dispatch-path interference — which is exactly the effect under test. Adding
# CCL traffic would confound "collector steals ERISC cycles" with "collector
# competes with the workload's own ethernet traffic".
#
# Usage:
#   python3 ab_workload.py --iters 200 --warmup 20 --size 2048

import argparse
import json
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=2, help="mesh rows (T3K: 2)")
    ap.add_argument("--cols", type=int, default=4, help="mesh cols (T3K: 4)")
    ap.add_argument("--size", type=int, default=2048, help="square matmul dim")
    ap.add_argument("--iters", type=int, default=50000, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=500, help="discarded iterations")
    ap.add_argument("--label", type=str, default="", help="free-form tag echoed in output")
    # Run for a bounded WALL TIME and exit through the normal `finally`, closing the mesh
    # device cleanly.
    #
    # This exists because the alternative -- oversize --iters and SIGKILL the process when
    # the measurement window closes -- leaves the fabric ERISC firmware stopped mid-loop
    # on an active ethernet core. Its heartbeat word then holds FABRIC_HEARTBEAT_SIGNATURE
    # (0xAABB) with a frozen counter, and UMD's TopologyDiscovery::eth_heartbeat_running
    # throws on a valid-but-frozen signature, so the NEXT tt-metal device open fails
    # outright and the board needs a reset. Measured 2026-08-30: killed the workload,
    # next run died with "Stuck at 0xaabb2d45", tt-smi -r all to recover.
    ap.add_argument(
        "--seconds", type=float, default=0.0, help="run the timed loop for this many wall seconds instead of --iters"
    )
    args = ap.parse_args()

    import torch
    import ttnn

    result = {"label": args.label, "ok": False}

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(args.rows, args.cols))
    try:
        num_devices = mesh.get_num_devices()
        result["num_devices"] = num_devices

        n = args.size
        torch.manual_seed(0)
        a_t = torch.randn(1, 1, n, n, dtype=torch.bfloat16)
        b_t = torch.randn(1, 1, n, n, dtype=torch.bfloat16)

        mapper = ttnn.ReplicateTensorToMesh(mesh)
        a = ttnn.from_torch(a_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)
        b = ttnn.from_torch(b_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=mapper)

        for _ in range(args.warmup):
            c = ttnn.matmul(a, b)
            ttnn.deallocate(c)
        ttnn.synchronize_device(mesh)

        t0 = time.perf_counter()
        done = 0
        if args.seconds > 0:
            # Check the clock every 50 iterations rather than every one: perf_counter()
            # per matmul would show up in a 0.18 ms/iter loop.
            while time.perf_counter() - t0 < args.seconds:
                for _ in range(50):
                    c = ttnn.matmul(a, b)
                    ttnn.deallocate(c)
                done += 50
        else:
            for _ in range(args.iters):
                c = ttnn.matmul(a, b)
                ttnn.deallocate(c)
            done = args.iters
        ttnn.synchronize_device(mesh)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        result.update(
            ok=True,
            elapsed_s=elapsed,
            iters=done,
            size=n,
            per_iter_ms=1000.0 * elapsed / done,
            # 2*n^3 FLOPs per matmul, per device
            tflops=(2.0 * n * n * n * done * num_devices) / elapsed / 1e12,
        )
    finally:
        try:
            ttnn.close_mesh_device(mesh)
        except Exception:
            pass

    print("TTNVTOP_AB_RESULT " + json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
