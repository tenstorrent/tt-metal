# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Iteration 43 scoping on the 1x4 mesh: ttnn's mcast-matmul-based fused collectives (the FUSE_OP hooks of the
mcast kernels) against the step's separate routes. (a) matmul_reduce_scatter_async for the sequence-parallel w2^T
forward ([5120, 8256] x [8256, 1024], the output scattered over the tokens, dim 3) and the gate_up dX^T
([5184, 16384] x [16384, 1024]) with the reduce-scatter workers on the last one or two core rows; (b)
all_gather_matmul_async with transpose_a for the sequence-parallel qkv / gate_up forward (x^T shards [K, 256]
gathered over dim 3, then x @ W). Wall per call over the mesh, outputs vs the separate route.
"""

import time

import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import _compute_config, dx_plain_matmul_kwargs, forward_transposed_matmul_kwargs

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
ttml.open_device_mesh(mesh)
device = ttml.autograd.AutoContext.get_instance().get_device()
grid = device.compute_with_storage_grid_size()
all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
tp_axis = 1
rng = np.random.default_rng(0)
cfg = _compute_config()


def dev(a, shard_dim=None):
    t = torch.from_numpy(a.astype(np.float32)).bfloat16()
    mapper = (
        ttnn.ShardTensorToMesh(device, dim=shard_dim) if shard_dim is not None else ttnn.ReplicateTensorToMesh(device)
    )
    return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mapper)


def zeros(shape):
    return ttnn.from_torch(
        torch.zeros(shape).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def host(t, concat_dim):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=concat_dim)).float()


def timed(fn, reps=10):
    fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps, out


def sems():
    return [ttnn.create_global_semaphore(device, all_cores, 0) for _ in range(3)]


def pc(gx, gy, k, pcm, pcn, sh, sw, obh=None):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=k,
        out_subblock_h=sh,
        out_subblock_w=sw,
        out_block_h=obh or pcm,
        out_block_w=pcn,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=True,
    )


def run(label, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        msg = [l for l in str(e).splitlines() if "FATAL" in l or "Error" in l or "rror" in l]
        print(f"{label}: FAILED {(msg[0] if msg else str(e))[:200]}", flush=True)
        return None


print("(a) matmul + reduce-scatter over the tokens (dim 3)", flush=True)
cases = [
    # name, M, K, N, committed kwargs, fused configs [(gx, gy, k, pcm, pcn, sh, sw, obh)]
    (
        "w2^T fwd [5120, 8256] x [8256, 1024]",
        5120,
        8256,
        1024,
        forward_transposed_matmul_kwargs(8256, 5120, 1024),
        [(11, 8, 6, 20, 3, 1, 3, 10), (11, 9, 6, 18, 3, 1, 3, 9)],
    ),
    (
        "gate_up dX^T [5184, 16384] x [16384, 1024]",
        5184,
        16384,
        1024,
        dx_plain_matmul_kwargs(5184, 16384, 1024),
        [(11, 8, 8, 21, 3, 1, 3, 21), (11, 9, 8, 18, 3, 1, 3, 18)],
    ),
]
for name, M, K, N, kw, fused_cfgs in cases:
    a = dev(rng.standard_normal((1, 1, M, K)) * 0.02)
    b = dev(rng.standard_normal((1, 1, K, N)))

    def separate():
        mm = ttnn.matmul(a, b, compute_kernel_config=cfg, **kw)
        return ttml.core.distributed.reduce_scatter(mm, 3, tp_axis)

    t_mm, _ = timed(lambda: ttnn.matmul(a, b, compute_kernel_config=cfg, **kw))
    t_sep, rs_ref = timed(separate)
    print(f"  {name}: committed matmul {t_mm:.0f} us; matmul + reduce_scatter {t_sep:.0f} us", flush=True)
    for gx, gy, k, pcm, pcn, sh, sw, obh in fused_cfgs:
        t_cfg, _ = run(
            f"  matmul alone on {gx}x{gy}",
            lambda: timed(
                lambda: ttnn.matmul(
                    a, b, compute_kernel_config=cfg, program_config=pc(gx, gy, k, pcm, pcn, sh, sw, obh)
                )
            ),
        ) or (None, None)
        inter = zeros((1, 1, M, N))
        outp = zeros((1, 1, M, N // 4))
        s = sems()

        def fused():
            return ttnn.experimental.matmul_reduce_scatter_async(
                a,
                b,
                inter,
                outp,
                3,
                s,
                ttnn.CoreCoord(0, gy),
                num_links=2,
                topology=ttnn.Topology.Ring,
                program_config=pc(gx, gy, k, pcm, pcn, sh, sw, obh),
                compute_kernel_config=cfg,
            )

        res = run(f"  fused (matmul {gx}x{gy}, RS on rows {gy}..)", lambda: timed(fused))
        if res:
            t, outs = res
            rs = outs[1] if isinstance(outs, (list, tuple)) else outs
            d = float((host(rs, 3) - host(rs_ref, 3)).abs().max())
            print(
                f"  fused (matmul {gx}x{gy}, RS on rows {gy}..): {t:.0f} us (the matmul alone on that grid {t_cfg:.0f}); max|diff| vs separate {d:.2e}",
                flush=True,
            )
        ttnn.deallocate(inter)
        ttnn.deallocate(outp)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

print("(b) all-gather over the tokens (dim 3 of x^T) + matmul with transpose_a", flush=True)
for name, K, N, fused_cfgs in (
    ("qkv fwd x [1024, 5120] @ W [5120, 1536]", 5120, 1536, [(11, 8, 4, 4, 6, 4, 1, 4)]),
    ("gate_up fwd x_cat [1024, 5184] @ W [5184, 16384]", 5184, 16384, [(11, 8, 9, 4, 47, 4, 1, 4)]),
):
    x_t_shard = dev(rng.standard_normal((1, 1, K, 1024)), shard_dim=3)  # x^T [K, 256] per chip
    w = dev(rng.standard_normal((1, 1, K, N)) * 0.02)

    def separate():
        g = ttml.core.distributed.all_gather(x_t_shard, 3, tp_axis)  # x^T [K, 1024]
        x = ttnn.transpose(g, -2, -1)
        return ttnn.matmul(x, w, compute_kernel_config=cfg, core_grid=ttnn.CoreGrid(y=grid.y, x=grid.x))

    t_sep, ref = timed(separate)
    print(f"  {name}: all_gather + transpose + matmul (ttnn's choice) {t_sep:.0f} us", flush=True)
    for gx, gy, k, pcm, pcn, sh, sw, obh in fused_cfgs:
        s = sems()

        def fused():
            return ttnn.experimental.all_gather_matmul_async(
                x_t_shard,
                w,
                None,
                3,
                s,
                ttnn.CoreCoord(0, gy),
                num_links=2,
                topology=ttnn.Topology.Ring,
                transpose_a=True,
                program_config=pc(gx, gy, k, pcm, pcn, sh, sw, obh),
                compute_kernel_config=cfg,
            )

        res = run(f"  fused AG + matmul(transpose_a) on {gx}x{gy}", lambda: timed(fused))
        if res:
            t, outs = res
            mm = outs[-1] if isinstance(outs, (list, tuple)) else outs
            d = float((host(mm, 0)[:1] - host(ref, 0)[:1]).abs().max())
            print(
                f"  fused AG + matmul(transpose_a) on {gx}x{gy}: {t:.0f} us; max|diff| vs separate {d:.2e}", flush=True
            )
    ttnn.deallocate(x_t_shard)
    ttnn.deallocate(w)
ttml.autograd.AutoContext.get_instance().close_device()
