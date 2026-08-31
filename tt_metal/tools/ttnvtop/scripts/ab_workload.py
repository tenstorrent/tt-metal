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
    # HEAVY CCL: all-gathers per iteration, so the workload and the monitor contend for
    # TT-FABRIC and not merely for compute.
    #
    # The matmul-only loop is deliberately fabric-free (see the header) because it was
    # built to isolate the collector's COST. That is the wrong shape for a COEXISTENCE
    # test: fabric is exactly where the interesting failures live, since tt-metal places
    # EDMs on link-free ethernet cores and that is what decides whether the aggregator can
    # find a core at all.
    # NOTE cluster_axis=0. On a T3K reshaped to 2x4 only ONE mesh axis carries fabric
    # links; gathering along axis 1 dies with "Requested link index 0 is out of bounds.
    # 0 ethernet channels available to forward" -- on a CLEAN board with no monitor
    # running, so it is a topology mistake, not contention. The working reference is
    # tests/ttnn/unit_tests/base_functionality/test_multi_device.py's
    # test_line_all_gather_after_reshape: all_gather(dim=2, cluster_axis=0).
    ap.add_argument("--ccl", type=int, default=0, help="all_gathers per iteration over tt-fabric (0 = compute only)")
    ap.add_argument("--ccl-dim", type=int, default=2048, help="width of the CCL tensor")
    # Matmuls per CCL group. The loop was hard-wired to 1 matmul per `--ccl` all-gathers,
    # which is fabric-dominated: at --ccl 4 the all-gathers take ~97% of wall time and the
    # monitor correctly reported ~0.5% FPU against the 22% that calibration says is 100%
    # compute duty. Saturating compute AND fabric together needs this ratio to be a knob,
    # not a constant, so the loss measurement can be swept across the whole spectrum
    # instead of asserted at one arbitrary point.
    ap.add_argument("--mm", type=int, default=1, help="matmuls per CCL group (raise to shift toward compute)")
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

    result = {"label": args.label, "ok": False, "ccl_per_iter": args.ccl, "mm_per_iter": args.mm}

    # FABRIC MUST BE CONFIGURED BEFORE THE MESH DEVICE IS CREATED.
    #
    # Without this, all_gather dies with "Requested link index 0 is out of bounds. 0
    # ethernet channels available to forward" -- on a CLEAN board with no monitor running,
    # and on every mesh shape tried (2x4 direct, 1x8 line, both cluster axes). The tree's
    # own reference test passes on the same machine, and the only difference is that its
    # pytest fixture calls set_fabric_config first (conftest.py set_fabric: "Must be called
    # before creating the mesh device"). Fabric appears in the log as initialised either
    # way, which is what made this look like a topology problem rather than a missing call.
    if args.ccl > 0:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)

    # Open as the 8-chip LINE and reshape, exactly as the reference test does; opening a
    # 2x4 mesh directly does not give the same fabric node mapping.
    if args.ccl > 0:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.rows * args.cols))
        mesh.reshape(ttnn.MeshShape(args.rows, args.cols))
    else:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(args.rows, args.cols))
    try:
        num_devices = mesh.get_num_devices()
        result["num_devices"] = num_devices

        n = args.size
        torch.manual_seed(0)
        a_t = torch.randn(1, 1, n, n, dtype=torch.bfloat16)
        b_t = torch.randn(1, 1, n, n, dtype=torch.bfloat16)

        # CCL tensor: sharded across BOTH mesh axes, gathered along the 4-device axis each
        # iteration. Follows the ShardTensor2dMesh + all_gather(cluster_axis=) pattern from
        # tests/ttnn/unit_tests/base_functionality/test_multi_device.py.
        ccl_t = None
        if args.ccl > 0:
            # Mirrors test_line_all_gather_after_reshape: shard both dims over the 2x4
            # mesh, gather dim 2 back along cluster_axis 0.
            ccl_axis, ccl_gather_dim = 0, 2
            ccl_torch = torch.rand((1, 1, 64 * args.rows, args.ccl_dim), dtype=torch.bfloat16)
            ccl_t = ttnn.from_torch(
                ccl_torch,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=[args.rows, args.cols], dims=(2, 3)),
            )

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
                    for _ in range(args.mm):
                        c = ttnn.matmul(a, b)
                        ttnn.deallocate(c)
                    for _ in range(args.ccl):
                        g = ttnn.all_gather(ccl_t, dim=ccl_gather_dim, cluster_axis=ccl_axis)
                        ttnn.deallocate(g)
                done += 50
        else:
            for _ in range(args.iters):
                for _ in range(args.mm):
                    c = ttnn.matmul(a, b)
                    ttnn.deallocate(c)
                for _ in range(args.ccl):
                    g = ttnn.all_gather(ccl_t, dim=ccl_gather_dim, cluster_axis=ccl_axis)
                    ttnn.deallocate(g)
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
        # conftest's reset_fabric: set DISABLED after the device is closed.
        if args.ccl > 0:
            try:
                ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
            except Exception:
                pass

    print("TTNVTOP_AB_RESULT " + json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
