# DP4 x TP8 with a HIDDEN-sharded residual ([M, 8192/8] per chip, all rows) instead of sequence-parallel rows: every big
# collective then sits inside an existing fused op that overlaps it with compute:
#   QKV and FF1|FF3(+SwiGLU): all_gather_minimal_matmul_async (gathers K = hidden over cluster_axis 0, BH-tuned configs)
#   WO and FF2: matmul_reduce_scatter_prefill(scatter_dim=-1) (reduce-scatter on N = hidden)
#   RMSNorm: local sum of squares on 1024 cols, all-gather of the [M, 32] stats, apply
# Times each fused piece next to the same matmul without communication (the overlap ceiling) and the plain all-gather.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_hshard_bench.py -s     env: HS_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.matmul import get_agmm_config
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

H, TP, FF, QKV = 8192, 8, 3584, 1280
MS = [int(m) for m in (os.environ.get("HS_M") or "4096,16384").split(",")]
ITERS = int(os.environ.get("HS_ITERS") or "10")
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_hshard_bench(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(4)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(4)]
    k = [0]
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def nxt():
        k[0] = (k[0] + 1) % 4
        return sems[k[0]], bars[k[0]]

    def ck(f, fp32=False):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=fp32, packer_l1_acc=True
        )

    lofi, hifi2 = ck(ttnn.MathFidelity.LoFi), ck(ttnn.MathFidelity.HiFi2)

    def t(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=DRAM,
            mesh_mapper=rep,
        )

    wqkv = t((1, 1, H, QKV), ttnn.bfloat8_b, 0.02)
    w13 = ttnn.from_torch(
        prepare_for_fused_swiglu(torch.randn(H, 2 * FF) * 0.02, ndev=1, gate_is_first=True).reshape(1, 1, H, 2 * FF),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    wo = t((1024, H), ttnn.bfloat8_b, 0.02)
    w2 = t((FF, H), ttnn.bfloat8_b, 0.02)

    def timed(name, fn, M):
        try:
            out = fn()
            ttnn.synchronize_device(mesh)
            for o in out if isinstance(out, (list, tuple)) else [out]:
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                out = fn()
                for o in out if isinstance(out, (list, tuple)) else [out]:
                    ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / ITERS * 1e3
            logger.info(f"HSHARD M={M} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"HSHARD M={M} {name} ERROR {str(e)[:300]}")
            return float("nan")

    def agmm(x, w, ckc, N, swiglu, dtype, M):
        _, cfg, wpl = get_agmm_config(
            M, H, N, full_grid=grid, cluster_size=TP, num_links=2, force_transpose=True, fuse_swiglu=swiglu
        )
        s, b = nxt()
        out = ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=x,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=ckc,
            multi_device_global_semaphore=s,
            barrier_semaphore=b,
            num_links=2,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            memory_config=DRAM,
            dtype=dtype,
            force_transpose=True,
            num_workers_per_link=wpl,
            num_buffers_per_channel=24,
            fuse_swiglu=swiglu,
        )
        for o in out[:-1]:
            ttnn.deallocate(o)
        return out[-1]

    def mm(x, w, ckc, swiglu, dtype):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=ckc,
            memory_config=DRAM,
            dtype=dtype,
            fuse_swiglu=swiglu,
        )

    def mmrs(a, w, fid, dim):
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            a,
            w,
            cluster_axis=0,
            scatter_dim=dim,
            topology=ttnn.Topology.Ring,
            num_links=2,
            compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fid),
            memory_config=DRAM,
        )

    def ag(x, dim):
        s, b = nxt()
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=s,
            num_links=2,
            barrier_semaphore=b,
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
        )

    for M in MS:
        xh = t((1, 1, M, H // TP), ttnn.bfloat8_b)  # hidden-sharded activation
        xh16 = t((1, 1, M, H // TP), ttnn.bfloat16)
        xfull = t((1, 1, M, H), ttnn.bfloat8_b)  # what the matmul sees after the gather
        a_wo = t((1, 1, M, 1024), ttnn.bfloat8_b)
        a_ff2 = t((1, 1, M, FF), ttnn.bfloat8_b)
        r = {}
        r["ag_K"] = timed("all_gather hidden (dim 3) alone", lambda: ag(xh, 3), M)
        r["qkv_agmm"] = timed("QKV agmm fused", lambda: agmm(xh, wqkv, hifi2, QKV, False, ttnn.bfloat16, M), M)
        r["qkv_mm"] = timed("QKV mm only", lambda: mm(xfull, wqkv, hifi2, False, ttnn.bfloat16), M)
        r["qkv_agmm_lofi"] = timed("QKV agmm fused LoFi", lambda: agmm(xh, wqkv, lofi, QKV, False, ttnn.bfloat16, M), M)
        r["ff13_agmm"] = timed(
            "FF13+swiglu agmm fused", lambda: agmm(xh, w13, lofi, 2 * FF, True, ttnn.bfloat8_b, M), M
        )
        r["ff13_mm"] = timed("FF13+swiglu mm only", lambda: mm(xfull, w13, lofi, True, ttnn.bfloat8_b), M)
        for fid, fn in ((ttnn.MathFidelity.HiFi2, "HiFi2"), (ttnn.MathFidelity.LoFi, "LoFi")):
            r[f"wo_rsN_{fn}"] = timed(f"WO mmrs scatter N {fn}", lambda: mmrs(a_wo, wo, fid, -1), M)
            r[f"wo_rsM_{fn}"] = timed(f"WO mmrs scatter M {fn}", lambda: mmrs(a_wo, wo, fid, -2), M)
            r[f"ff2_rsN_{fn}"] = timed(f"FF2 mmrs scatter N {fn}", lambda: mmrs(a_ff2, w2, fid, -1), M)
            r[f"ff2_rsM_{fn}"] = timed(f"FF2 mmrs scatter M {fn}", lambda: mmrs(a_ff2, w2, fid, -2), M)

        def dnorm():
            st = ttnn.rms_norm_pre_all_gather(xh16, dtype=ttnn.bfloat16)
            g = ag(st, 3)
            ttnn.deallocate(st)
            o = ttnn.rms_norm_post_all_gather(xh16, g, epsilon=1e-5, memory_config=DRAM, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(g)
            return o

        r["dnorm"] = timed("distributed rms norm (1024 cols, stats all-gather)", dnorm, M)
        logger.info(f"HSHARD M={M} SUMMARY " + " ".join(f"{a}={b:.3f}" for a, b in r.items()))
        for z in (xh, xh16, xfull, a_wo, a_ff2):
            ttnn.deallocate(z)
