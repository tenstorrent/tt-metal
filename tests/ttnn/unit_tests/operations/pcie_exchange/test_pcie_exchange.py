# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.experimental.pcie_exchange on two PCIe-only Blackhole chips (desktop-0 chips 4 and 5).

Run standalone (no pytest machinery, so nothing else opens the devices):
    source env_vars_setup.sh
    python tests/ttnn/unit_tests/operations/pcie_exchange/test_pcie_exchange.py [--chips 4,5] [--iters 200]
"""
import argparse
import os
import time

import torch
import ttnn

TILE = 32
ROWS = 32


def open_isolated_mesh(chips, trace_region_size=32 * 1024 * 1024):
    p2p = ttnn._ttnn.pcie_p2p
    p2p.use_isolated_mesh(list(chips))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, len(chips)), trace_region_size=trace_region_size)
    grid = mesh.compute_with_storage_grid_size()
    # Row-major first 64 cores of the compute grid: the residual/matmul-output layout of the model and the
    # core walk pcie_exchange's all-gather output uses.
    cores = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(64)]
    p2p.setup(mesh, cores)
    return mesh, cores


def width_sharded(cores, shard_w):
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [ROWS, shard_w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def make_exchange(mesh, cores, ntiles):
    """Persistent landing buffer: [landing0][landing1][control] per core, zeroed on both devices."""
    n = len(cores)
    shard_w = (2 * ntiles + 1) * TILE
    z = torch.zeros(1, 1, ROWS, n * shard_w, dtype=torch.bfloat16)
    return ttnn.from_torch(
        z,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=width_sharded(cores, shard_w),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )


def to_device(mesh, cores, x, shard_w):
    """x: [num_devices, 1, ROWS, W]; device d gets x[d] width-sharded over `cores`."""
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=width_sharded(cores, shard_w),
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )


def from_device(mesh, t):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def check_all_reduce(mesh, cores, n_cores, shard_w, iters):
    cores_n = cores[:n_cores]
    W = n_cores * shard_w
    ntiles = shard_w // TILE
    xb = make_exchange(mesh, cores_n, ntiles)
    worst = 0.0
    for it in range(iters):
        x = torch.randn(2, 1, ROWS, W, dtype=torch.bfloat16)
        t = to_device(mesh, cores_n, x, shard_w)
        out = ttnn.experimental.pcie_exchange(t, xb, 0)
        got = from_device(mesh, out)
        exp = (x[0].float() + x[1].float()).bfloat16().float()
        for d in range(2):
            err = (got[d].float() - exp).abs().max().item()
            worst = max(worst, err)
            assert err <= 0.0625 * exp.abs().max().item() + 1e-3, f"all_reduce iter {it} device {d}: max err {err}"
        ttnn.deallocate(out)
        ttnn.deallocate(t)
    print(f"all_reduce {n_cores} cores x [{ROWS},{shard_w}] ok over {iters} iters, worst abs err {worst:.4f}")
    return xb


def check_all_gather(mesh, cores, n_cores, shard_w, iters):
    cores_n = cores[:n_cores]
    W = n_cores * shard_w
    ntiles = shard_w // TILE
    xb = make_exchange(mesh, cores_n, ntiles)
    for it in range(iters):
        x = torch.randn(2, 1, ROWS, W, dtype=torch.bfloat16)
        t = to_device(mesh, cores_n, x, shard_w)
        out = ttnn.experimental.pcie_exchange(t, xb, 1)
        assert out.shape[-1] == 2 * W, out.shape
        got = from_device(mesh, out)
        exp = torch.cat([x[0], x[1]], dim=-1)
        for d in range(2):
            assert torch.equal(got[d], exp), f"all_gather iter {it} device {d} mismatch"
        ttnn.deallocate(out)
        ttnn.deallocate(t)
    print(f"all_gather {n_cores} cores x [{ROWS},{shard_w}] -> [{ROWS},{2 * W}] ok over {iters} iters")


def time_traced(mesh, cores, n_cores, shard_w, mode, reps=50, loops=20):
    """Capture `loops` back-to-back exchanges in a trace, replay `reps` times, report per-op time."""
    cores_n = cores[:n_cores]
    W = n_cores * shard_w
    ntiles = shard_w // TILE
    xb = make_exchange(mesh, cores_n, ntiles)
    x = torch.randn(2, 1, ROWS, W, dtype=torch.bfloat16)
    t = to_device(mesh, cores_n, x, shard_w)
    out = ttnn.experimental.pcie_exchange(t, xb, mode)  # compile
    ttnn.deallocate(out)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    for _ in range(loops):
        outs.append(ttnn.experimental.pcie_exchange(t, xb, mode))
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t0) / (reps * loops)
    # Correctness after a long replay (sequence numbers advanced reps*loops times).
    got = from_device(mesh, outs[-1])
    if mode == 0:
        exp = (x[0].float() + x[1].float()).bfloat16()
    else:
        exp = torch.cat([x[0], x[1]], dim=-1)
    for d in range(2):
        assert torch.allclose(got[d].float(), exp.float(), atol=0.0625, rtol=0.0625), f"traced mode {mode} device {d}"
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    ttnn.deallocate(t)
    name = "all_reduce" if mode == 0 else "all_gather"
    print(
        f"traced {name} {n_cores} cores x [{ROWS},{shard_w}] ({n_cores * shard_w * ROWS * 2 / 1024:.0f} KiB/device): {dt * 1e6:.2f} us per op"
    )


def l1_report(mesh, tag):
    probe = ttnn.from_torch(
        torch.zeros(1, 1, 32, 32),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    print(f"[{tag}] next L1 alloc at {probe.buffer_address()} (L1 grows top-down; lower = fuller)", flush=True)
    ttnn.deallocate(probe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default=os.environ.get("TT_PCIE_P2P_CHIPS", "4,5"))
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    chips = [int(c) for c in args.chips.split(",")]
    mesh, cores = open_isolated_mesh(chips)
    try:
        # Llama-3.1-8B decode shapes: residual [32, 4096] on 64 cores (2 tiles/core); gather of [32, 2048] on 32 cores.
        l1_report(mesh, "start")
        check_all_reduce(mesh, cores, 64, 64, args.iters)
        l1_report(mesh, "after all_reduce 64")
        check_all_reduce(mesh, cores, 32, 64, args.iters)
        check_all_gather(mesh, cores, 32, 64, args.iters)
        l1_report(mesh, "after checks")
        time_traced(mesh, cores, 64, 64, 0)
        l1_report(mesh, "after traced all_reduce")
        time_traced(mesh, cores, 32, 64, 1)
        l1_report(mesh, "after traced all_gather")
        # A bigger shard (lm_head-ish, 32 tiles per core = 64 KiB) to see the bandwidth regime.
        check_all_reduce(mesh, cores, 64, 1024, max(2, args.iters // 10))
        l1_report(mesh, "after all_reduce 64x1024")
        time_traced(mesh, cores, 64, 1024, 0, loops=4)
    finally:
        ttnn._ttnn.pcie_p2p.teardown()
        ttnn.close_mesh_device(mesh)
    print("ALL OK")


if __name__ == "__main__":
    main()
