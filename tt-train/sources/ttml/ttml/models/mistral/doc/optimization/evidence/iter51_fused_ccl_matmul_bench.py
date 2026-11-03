"""Iteration 51 scoping: the collective fused into its matmul (ttnn's all_gather_minimal_matmul_async and
minimal_matmul_strided_reduce_scatter_async, the kernels the k2 QB2 demo runs) against today's pair (ttml's
all-gather or reduce-scatter on the sequence dim + the swept 2D-mcast matmul). The fused kernels gather / scatter
the hidden dim (dim 3), so the comparison assumes a hidden-sharded residual stream: the per-chip activation is
[S, K/4] before an all-gather and the partial product [S, N] is scattered to [S, N/4]. Same bytes on the links as
the sequence-sharded layout. HiFi4, fp32 accumulate. Per pair prints: today's pieces (collective, matmul, sum),
minimal_matmul alone, and the fused op, in us per call over the 1x4 mesh (host-timed, 20 reps, synchronised)."""
import sys
import time
import numpy as np
import torch
import ttnn
import ttml
from ttml.modules.frozen_linear import matmul_kwargs_2d

import os

mesh = ttml.Mesh((1, 4), ("dp", "tp"))
if os.environ.get("FABRIC", "2d") == "1d":
    # The fused all-gather kernels hard-code the 1D fabric packet header; ttml's open_device_mesh installs the 2D
    # torus config inferred from the MGD. Same MGD, FABRIC_1D_RING instead (what the k2 QB2 demo runs with).
    ttml._mesh._validate_mgd(mesh)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    ttml.autograd.AutoContext.get_instance().open_device(list(mesh.shape), [])
    ttml._mesh._mesh = mesh
    print("fabric: FABRIC_1D_RING")
else:
    ttml.open_device_mesh(mesh)
    print("fabric: ttml default (2D torus from the MGD)")
device = ttml.autograd.AutoContext.get_instance().get_device()
cfg = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False, packer_l1_acc=True
)
grid = device.compute_with_storage_grid_size()
all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
TP, S = 4, 1024
LINKS = 2
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def dev(a, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        torch.from_numpy(a),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device),
    )


def timed(fn, reps=REPS):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / reps


def sems(n):
    return [ttnn.create_global_semaphore(device, all_cores, 0) for _ in range(n)]


rng = np.random.default_rng(0)


def mm_config(m_blk, k_blk, n_blk, sub_h, sub_w, gx, gy):
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_blk,
        K_block_size=k_blk,
        N_block_size=n_blk,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
    )


def bench_ag(name, K, N, swiglu=False):
    """x_hs [S, K/4] gathered on dim 3 then x [S, K] @ W [K, N]; today: x_sp [S/4, K] gathered on dim 2."""
    x_hs = dev(rng.normal(0, 1, (1, 1, S, K // TP)).astype(np.float32))
    x_sp = dev(rng.normal(0, 1, (1, 1, S // TP, K)).astype(np.float32))
    x_full = dev(rng.normal(0, 1, (1, 1, S, K)).astype(np.float32))
    w = dev(rng.normal(0, 0.02, (1, 1, K, N)).astype(np.float32))
    kw = matmul_kwargs_2d(S, K, N, False) or {}
    t_ag = timed(lambda: ttml.core.distributed.all_gather(x_sp, 2, 1))
    t_mm = timed(lambda: ttnn.matmul(x_full, w, compute_kernel_config=cfg, **kw))
    out = {}
    for label, conf in [
        ("mm11x9 k8 n8 2x2", mm_config(8, 8, 8, 2, 2, grid.x, grid.y - 1)),  # row 9 holds the all-gather muxes
        ("mm11x9 k8 n4 2x2", mm_config(8, 8, 4, 2, 2, grid.x, grid.y - 1)),
        ("mm11x9 k10 n8 2x2", mm_config(8, 10, 8, 2, 2, grid.x, grid.y - 1)),
        ("mm11x8 k8 n8 2x2", mm_config(8, 8, 8, 2, 2, grid.x, grid.y - 2)),
    ]:
        try:
            t_min = timed(lambda: ttnn.experimental.minimal_matmul(x_full, w, compute_kernel_config=cfg, config=conf))
        except Exception as e:  # noqa: BLE001
            t_min = float("nan")
            print(f"  minimal_matmul {label}: {type(e).__name__} {str(e).splitlines()[0][:120]}")
        sem_sets = [sems(2) for _ in range(4)]
        bar = sems(4)
        i = [0]

        def fused():
            j = i[0] % 4
            i[0] += 1
            return ttnn.experimental.all_gather_minimal_matmul_async(
                x_hs,
                w,
                config=conf,
                multi_device_global_semaphore=sem_sets[j],
                topology=ttnn.Topology.Ring,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=cfg,
                num_links=LINKS,
                cluster_axis=1,
                barrier_semaphore=bar[j],
                force_transpose=True,
                num_workers_per_link=6,
                num_buffers_per_channel=24,
                fuse_swiglu=swiglu,
            )

        try:
            t_f = timed(fused)
        except Exception as e:  # noqa: BLE001
            t_f = float("nan")
            print(f"  fused {label}: {type(e).__name__} {repr(e)[:300]}")
        out[label] = (t_min, t_f)
    print(f"AG {name:14s} K={K} N={N}: today ag {t_ag:7.0f} + mm2d {t_mm:7.0f} = {t_ag + t_mm:7.0f} us")
    for label, (t_min, t_f) in out.items():
        print(
            f"   {label:22s} minimal_mm {t_min:7.0f}  fused ag+mm {t_f:7.0f} us  ({t_f / (t_ag + t_mm) * 100:5.1f} % of today)"
        )
    sys.stdout.flush()


def bench_rs(name, K, N):
    """x [S, K] @ W [K, N] then the partial scattered on dim 3 to [S, N/4]; today: scattered on dim 2."""
    x = dev(rng.normal(0, 1, (1, 1, S, K)).astype(np.float32))
    w = dev(rng.normal(0, 0.02, (1, 1, K, N)).astype(np.float32))
    y = dev(rng.normal(0, 1, (1, 1, S, N)).astype(np.float32))
    kw = matmul_kwargs_2d(S, K, N, False) or {}
    t_rs = timed(lambda: ttml.core.distributed.reduce_scatter(y, 2, 1))
    t_mm = timed(lambda: ttnn.matmul(x, w, compute_kernel_config=cfg, **kw))
    out = {}
    for label, conf, workers in [
        ("mm11x8 k8 n8 2x2 w4", mm_config(8, 8, 8, 2, 2, grid.x, grid.y - 2), 4),
        ("mm11x8 k8 n4 2x2 w4", mm_config(8, 8, 4, 2, 2, grid.x, grid.y - 2), 4),
        ("mm11x9 k8 n8 2x2 w2", mm_config(8, 8, 8, 2, 2, grid.x, grid.y - 1), 2),
    ]:
        try:
            t_min = timed(lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=cfg, config=conf))
        except Exception as e:  # noqa: BLE001
            t_min = float("nan")
            print(f"  minimal_matmul {label}: {type(e).__name__} {str(e).splitlines()[0][:120]}")
        sem_sets = [sems(3) for _ in range(4)]
        bar = sems(4)
        i = [0]
        offset = ttnn.CoreCoord(0, conf.compute_with_storage_grid_size.y)

        def fused():
            j = i[0] % 4
            i[0] += 1
            return ttnn.experimental.minimal_matmul_strided_reduce_scatter_async(
                x,
                w,
                3,
                sem_sets[j],
                offset,
                compute_kernel_config=cfg,
                num_links=LINKS,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                rs_output_mem_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                cluster_axis=1,
                config=conf,
                barrier_semaphore=bar[j],
                num_workers_per_link=workers,
                num_buffers_per_channel=2,
                chunk_width_in_mm_blocks=1,
            )

        try:
            t_f = timed(fused)
        except Exception as e:  # noqa: BLE001
            t_f = float("nan")
            import traceback

            print(f"  fused {label}: {type(e).__name__} {repr(e)[:300]}")
            traceback.print_exc(limit=3)
        out[label] = (t_min, t_f)
    print(f"RS {name:14s} K={K} N={N}: today mm2d {t_mm:7.0f} + rs {t_rs:7.0f} = {t_rs + t_mm:7.0f} us")
    for label, (t_min, t_f) in out.items():
        print(
            f"   {label:22s} minimal_mm {t_min:7.0f}  fused mm+rs {t_f:7.0f} us  ({t_f / (t_rs + t_mm) * 100:5.1f} % of today)"
        )
    sys.stdout.flush()


which = sys.argv[2] if len(sys.argv) > 2 else "all"
if which in ("all", "ag"):
    bench_ag("qkv fw", 5120, 1536)
    bench_ag("wo bw dX", 5120, 1024)
    bench_ag("w2 bw dX", 5120, 8192)
    bench_ag("gate_up fw", 5120, 16384)
if which in ("all", "rs"):
    bench_rs("wo fw", 1024, 5120)
    bench_rs("qkv bw dX", 1536, 5120)
    bench_rs("w2 fw", 8192, 5120)
    bench_rs("gate_up bw dX", 16384, 5120)
ttml.autograd.AutoContext.get_instance().close_device()
