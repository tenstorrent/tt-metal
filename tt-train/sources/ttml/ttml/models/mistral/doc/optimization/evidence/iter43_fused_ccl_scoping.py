# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 43 scoping on the 1x4 mesh: ttnn's experimental minimal matmul and its fused collectives against the
step's routes at the TP shapes. (a) minimal_matmul vs the mcast matmul (w2 forward [1024, 8256] x [8256, 5120],
out-proj [1024, 1024] x [1024, 5120]); (b) minimal_matmul + strided reduce-scatter over the token rows (dim 2) vs
matmul then reduce_scatter; (c) strided all-gather (dim 2, the sequence-parallel shard [256, K]) + minimal matmul vs
all_gather then matmul (qkv [1024, 5120] x [5120, 1536]). Wall per call over the mesh, outputs vs the separate route.

    TT_MESH_GRAPH_DESC_PATH=... python iter43_fused_ccl_scoping.py
"""

import time
import traceback

import numpy as np
import torch
import ttnn
import ttml

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
device = ttml.autograd.AutoContext.get_instance().get_device()
grid = device.compute_with_storage_grid_size()
all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
tp_axis = 1
rng = np.random.default_rng(0)
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)


def dev(a, shard_dim=None):
    t = torch.from_numpy(a.astype(np.float32)).bfloat16()
    mapper = (
        ttnn.ShardTensorToMesh(device, dim=shard_dim) if shard_dim is not None else ttnn.ReplicateTensorToMesh(device)
    )
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)


def host(t, concat_dim=None):
    """chip 0's copy of a replicated tensor, or the chips' shards concatenated along concat_dim."""
    if concat_dim is None:
        return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0)).float()[:1]
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=concat_dim)).float()


def timed(fn, reps=20):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


def sems():
    return [ttnn.create_global_semaphore(device, all_cores, 0) for _ in range(3)]


def try_run(label, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"{label}: FAILED {str(e).splitlines()[0][:160]}")
        return None


# ---------------------------------------------------------------- (a) minimal_matmul vs the mcast matmul
for name, M, K, N, blocks in (
    ("w2 fwd", 1024, 8256, 5120, [(4, 6, 5, 4, 1, (11, 8)), (4, 3, 5, 4, 1, (11, 8)), (2, 6, 5, 2, 1, (11, 10))]),
    ("out fwd", 1024, 1024, 5120, [(4, 4, 5, 4, 1, (11, 8)), (2, 8, 5, 2, 1, (11, 10))]),
    ("gate_up fwd", 1024, 5184, 16384, [(4, 9, 5, 4, 1, (11, 8)), (2, 9, 8, 2, 4, (11, 10))]),
):
    a = dev(rng.standard_normal((1, 1, M, K)))
    w = dev(rng.standard_normal((1, 1, K, N)) * 0.02)
    t_ref, ref = timed(
        lambda: ttnn.matmul(a, w, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))
    )
    ref_h = host(ref)
    print(f"(a) {name} [{M}, {K}] x [{K}, {N}]: ttnn mcast matmul (its choice) {t_ref:.0f} us")
    for mb, kb, nb, sh, sw, (gx, gy) in blocks:
        mc = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
        )
        res = try_run(
            f"  minimal_matmul M{mb} K{kb} N{nb} sub {sh}x{sw} grid {gx}x{gy}",
            lambda: timed(lambda: ttnn.experimental.minimal_matmul(a, w, compute_kernel_config=cfg, config=mc)),
        )
        if res:
            t, out = res
            print(
                f"  minimal_matmul M{mb} K{kb} N{nb} sub {sh}x{sw} grid {gx}x{gy}: {t:.0f} us  max|diff| {float((host(out) - ref_h).abs().max()):.2e}"
            )
    ttnn.deallocate(a)
    ttnn.deallocate(w)

# ---------------------------------------------------------------- (b) matmul + reduce-scatter over the token rows
for name, M, K, N, mc_args in (
    ("w2 fwd", 1024, 8256, 5120, (4, 6, 5, 4, 1, (11, 8))),
    ("out fwd", 1024, 1024, 5120, (4, 4, 5, 4, 1, (11, 8))),
):
    a = dev(rng.standard_normal((1, 1, M, K)))
    w = dev(rng.standard_normal((1, 1, K, N)) * 0.02)

    def separate():
        mm = ttnn.matmul(a, w, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))
        return ttml.core.distributed.reduce_scatter(mm, 2, tp_axis)

    t_sep, rs_ref = timed(separate)
    print(f"(b) {name}: matmul then reduce_scatter (dim 2) {t_sep:.0f} us per call")
    mb, kb, nb, sh, sw, (gx, gy) = mc_args
    for mm_grid_y in (gy, gy - 2):
        mc = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, mm_grid_y),
        )
        s = sems()

        def fused():
            return ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                a,
                w,
                2,
                s,
                ttnn.CoreCoord(0, mm_grid_y),
                compute_kernel_config=cfg,
                num_links=2,
                topology=ttnn.Topology.Ring,
                cluster_axis=tp_axis,
                config=mc,
            )

        res = try_run(f"  fused minimal_matmul + strided RS (matmul grid {gx}x{mm_grid_y})", lambda: timed(fused))
        if res:
            t, outs = res
            rs = outs[-1]
            print(
                f"  fused minimal_matmul + strided RS (matmul grid {gx}x{mm_grid_y}): {t:.0f} us  max|diff| vs separate {float((host(rs, 2) - host(rs_ref, 2)).abs().max()):.2e}"
            )
    ttnn.deallocate(a)
    ttnn.deallocate(w)

# ---------------------------------------------------------------- (c) all-gather over the token rows + matmul
for name, M, K, N, mc_args in (
    ("qkv fwd", 1024, 5120, 1536, (4, 4, 4, 4, 1, (11, 8))),
    ("gate_up fwd", 1024, 5184, 16384, (4, 9, 5, 4, 1, (11, 8))),
):
    shard = dev(rng.standard_normal((1, 1, M, K)), shard_dim=2)  # [256, K] per chip
    w = dev(rng.standard_normal((1, 1, K, N)) * 0.02)

    def separate():
        g = ttml.core.distributed.all_gather(shard, 2, tp_axis)
        return ttnn.matmul(g, w, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))

    t_sep, ref = timed(separate)
    print(f"(c) {name}: all_gather (dim 2) then matmul {t_sep:.0f} us per call")
    mb, kb, nb, sh, sw, (gx, gy) = mc_args
    for mm_grid_y in (gy, gy - 2):
        mc = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=sh,
            subblock_w=sw,
            compute_with_storage_grid_size=ttnn.CoreCoord(gx, mm_grid_y),
        )
        s = sems()

        def fused():
            return ttnn.experimental.strided_all_gather_minimal_matmul_async(
                shard,
                w,
                dim=2,
                multi_device_global_semaphore=s,
                strided_all_gather_core_grid_offset=ttnn.CoreCoord(0, mm_grid_y),
                num_links=2,
                topology=ttnn.Topology.Ring,
                cluster_axis=tp_axis,
                config=mc,
                compute_kernel_config=cfg,
            )

        res = try_run(f"  fused strided AG + minimal_matmul (matmul grid {gx}x{mm_grid_y})", lambda: timed(fused))
        if res:
            t, outs = res
            mm = outs[-1] if isinstance(outs, (list, tuple)) else outs
            print(
                f"  fused strided AG + minimal_matmul (matmul grid {gx}x{mm_grid_y}): {t:.0f} us  max|diff| vs separate {float((host(mm) - host(ref)).abs().max()):.2e}"
            )
    ttnn.deallocate(shard)
    ttnn.deallocate(w)
ttml.autograd.AutoContext.get_instance().close_device()
