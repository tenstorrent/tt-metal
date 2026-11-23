# DP4 x TP8 prefill pieces on the column axis (cluster_axis 0, 8 chips), every column its own data:
#   1. all_gather_async along rows (dim 2), bf8 [M/8, 8192] -> [M, 8192]: links x buffers-per-channel sweep
#      (ttnn.all_gather ignores num_links / topology / workers; the experimental op honours them)
#   2. matmul_reduce_scatter_prefill (fused, fp32 adds per hop, scatter rows) vs minimal_matmul + reduce_scatter for
#      WO [M, 1024] @ [1024, 8192] and FF2 [M, 3584] @ [3584, 8192]: links x fidelity
#   3. RMSNorm over a full 8192-wide row on M/8 rows (plain rms_norm overflows L1): variants
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_tp8_ccl_norm_sweep.py -s
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, TP = 8192, 8
MS = [int(m) for m in (os.environ.get("TP8_M") or "4096").split(",")]
ITERS = int(os.environ.get("TP8_ITERS") or "10")
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _time(mesh, fn):
    out = fn()
    ttnn.synchronize_device(mesh)
    ttnn.deallocate(out)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(mesh)
    return (time.perf_counter() - t0) / ITERS * 1e3


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0, "l1_small_size": 32768}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_tp8_ccl_norm_sweep(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    idx = [0]
    rep = ttnn.ReplicateTensorToMesh(mesh)
    hifi2 = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def t(shape, dt, scale=1.0):
        return ttnn.from_torch(
            torch.randn(shape) * scale,
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dt,
            memory_config=DRAM,
            mesh_mapper=rep,
        )

    def ag_async(x, links, bufs=None, **kw):
        i = idx[0]
        idx[0] ^= 1
        extra = {} if bufs is None else {"num_buffers_per_channel": bufs}
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sems[i],
            num_links=links,
            barrier_semaphore=bars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            **extra,
            **kw,
        )

    for M in MS:
        tag = f"SWEEP M={M}"
        x8 = t((1, 1, M // TP, H), ttnn.bfloat8_b)
        nbytes = 7 / 8 * M * H * 1088 / 1024
        # golden check once: rows of chip r land at rows r*M/8
        for links in (1, 2):
            for bufs in (None, 8, 16, 24, 32):
                for kw in ({}, {"chunks_per_sync": 16}, {"num_workers_per_link": 4}):
                    try:
                        ms = _time(mesh, lambda: ag_async(x8, links, bufs, **kw))
                        logger.info(
                            f"{tag} AG_ASYNC links={links} bufs={bufs} {kw}: {ms:.3f} ms = {nbytes / ms / 1e6:.0f} GB/s"
                        )
                    except Exception as e:
                        logger.warning(f"{tag} AG_ASYNC links={links} bufs={bufs} {kw} ERROR {str(e)[:160]}")
        ttnn.deallocate(x8)

        for name, K in (("wo", 1024), ("ff2", 3584)):
            a = t((1, 1, M, K), ttnn.bfloat8_b)
            w2d = t((K, H), ttnn.bfloat8_b, 0.02)
            w4d = t((1, 1, K, H), ttnn.bfloat8_b, 0.02)
            for fid in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi):
                for links in (1, 2):
                    try:
                        ms = _time(
                            mesh,
                            lambda: ttnn.experimental.matmul_reduce_scatter_prefill(
                                a,
                                w2d,
                                cluster_axis=0,
                                scatter_dim=-2,
                                topology=ttnn.Topology.Ring,
                                num_links=links,
                                compute_kernel_config=ttnn.ComputeConfigDescriptor(math_fidelity=fid),
                                memory_config=DRAM,
                            ),
                        )
                        logger.info(f"{tag} MMRS {name} {fid} links={links}: {ms:.3f} ms")
                    except Exception as e:
                        logger.warning(f"{tag} MMRS {name} {fid} links={links} ERROR {str(e)[:200]}")
            cfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=grid,
            )
            mm_ms = _time(
                mesh,
                lambda: ttnn.experimental.minimal_matmul(
                    input_tensor=a,
                    weight_tensor=w4d,
                    config=cfg,
                    compute_kernel_config=hifi2,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat8_b,
                ),
            )
            p = ttnn.experimental.minimal_matmul(
                input_tensor=a,
                weight_tensor=w4d,
                config=cfg,
                compute_kernel_config=hifi2,
                memory_config=DRAM,
                dtype=ttnn.bfloat8_b,
            )
            rs_ms = _time(mesh, lambda: ttnn.reduce_scatter(p, 2, cluster_axis=0, memory_config=DRAM))
            logger.info(f"{tag} SEPARATE {name}: mm {mm_ms:.3f} + rs {rs_ms:.3f} = {mm_ms + rs_ms:.3f} ms")
            for z in (a, w2d, w4d, p):
                ttnn.deallocate(z)

        x = t((1, 1, M // TP, H), ttnn.bfloat16)
        nw = ttnn.from_torch(
            torch.ones(1, 1, H // 32, 32),
            device=mesh,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=rep,
        )
        ref = None
        variants = {
            "rms_norm": lambda: ttnn.rms_norm(x, epsilon=1e-5, weight=nw, memory_config=DRAM),
            "rms_norm_nolegacy": lambda: ttnn.rms_norm(
                x,
                epsilon=1e-5,
                weight=nw,
                memory_config=DRAM,
                program_config=ttnn.LayerNormDefaultProgramConfig(legacy_reduction=False, legacy_rsqrt=False),
            ),
            "pre_post": lambda: ttnn.rms_norm_post_all_gather(
                x, ttnn.rms_norm_pre_all_gather(x, dtype=ttnn.bfloat16), epsilon=1e-5, weight=nw, memory_config=DRAM
            ),
            "split4_pre_post": None,
        }

        def split4():
            parts = ttnn.split(x, H // 4, dim=-1)
            st = [ttnn.rms_norm_pre_all_gather(p, dtype=ttnn.bfloat16) for p in parts]
            s = ttnn.concat(st, dim=-1)
            outs = [ttnn.rms_norm_post_all_gather(p, s, epsilon=1e-5, memory_config=DRAM) for p in parts]
            return ttnn.concat(outs, dim=-1)

        variants["split4_pre_post"] = split4
        tx = ttnn.to_torch(ttnn.get_device_tensors(x)[0]).float()
        gold = tx * torch.rsqrt(tx.pow(2).mean(-1, keepdim=True) + 1e-5)
        for name, fn in variants.items():
            try:
                ms = _time(mesh, fn)
                o = fn()
                got = ttnn.to_torch(ttnn.get_device_tensors(o)[0]).float()
                ttnn.deallocate(o)
                err = (got - gold).abs().max().item()
                logger.info(f"{tag} NORM {name}: {ms:.3f} ms max_abs_err {err:.4f}")
            except Exception as e:
                logger.warning(f"{tag} NORM {name} ERROR {str(e)[:200]}")
        ttnn.deallocate(x)
