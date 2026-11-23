# Block-config sweep of the fused, overlapping all-gather + matmul (ttnn.experimental.all_gather_minimal_matmul_async,
# K-gather over cluster_axis 0 = one 8-chip mesh column) at the DP4 x TP8 hidden-sharded shapes:
#   QKV:        [M, 1024 per chip] gathered to K=8192 @ [8192, 1280]   bf8 weight
#   FF13+SwiGLU:[M, 1024 per chip] gathered to K=8192 @ [8192, 7168]   bf4 weight, fused SwiGLU (out N/2)
# Separate all_gather (dim 3) + minimal_matmul is the reference; max(matmul, gather) is the overlap ceiling.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_agmm_sweep.py -s
# env: SW_M (4096,16384), SW_SHAPES (qkv,ff13), SW_ITERS (5)
import itertools
import math
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

H, TP = 8192, 8
MS = [int(m) for m in (os.environ.get("SW_M") or "4096,16384").split(",")]
SHAPES = (os.environ.get("SW_SHAPES") or "qkv,ff13").split(",")
ITERS = int(os.environ.get("SW_ITERS") or "5")
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_agmm_sweep(mesh_device, device_params):
    mesh = mesh_device
    full = mesh.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full.x - 1, full.y - 1))])
    sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(4)]
    bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(4)]
    k = [0]
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def nxt():
        k[0] = (k[0] + 1) % 4
        return sems[k[0]], bars[k[0]]

    def ck(f):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )

    weights = {
        "qkv": (
            ttnn.from_torch(
                torch.randn(1, 1, H, 1280) * 0.02,
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            ),
            1280,
            False,
            ck(ttnn.MathFidelity.LoFi),
        ),
        "ff13": (
            ttnn.from_torch(
                prepare_for_fused_swiglu(torch.randn(H, 7168) * 0.02, ndev=1, gate_is_first=True).reshape(
                    1, 1, H, 7168
                ),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat4_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            ),
            7168,
            True,
            ck(ttnn.MathFidelity.LoFi),
        ),
    }

    def timed(fn):
        o = fn()
        ttnn.synchronize_device(mesh)
        for z in o if isinstance(o, (list, tuple)) else [o]:
            ttnn.deallocate(z)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            o = fn()
            for z in o if isinstance(o, (list, tuple)) else [o]:
                ttnn.deallocate(z)
        ttnn.synchronize_device(mesh)
        return (time.perf_counter() - t0) / ITERS * 1e3

    for M in MS:
        xh = ttnn.from_torch(
            torch.randn(1, 1, M, H // TP),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=DRAM,
            mesh_mapper=rep,
        )
        for name in SHAPES:
            w, N, swiglu, ckc = weights[name]

            # reference: separate gather + matmul on the full grid
            def ag():
                s, b = nxt()
                return ttnn.experimental.all_gather_async(
                    input_tensor=xh,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=s,
                    num_links=2,
                    barrier_semaphore=b,
                    memory_config=DRAM,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=0,
                )

            g = ag()
            mmcfg = ttnn.MinimalMatmulConfig(
                M_block_size=8,
                K_block_size=8,
                N_block_size=8,
                subblock_h=1,
                subblock_w=8,
                compute_with_storage_grid_size=full,
            )
            t_mm = timed(
                lambda: ttnn.experimental.minimal_matmul(
                    input_tensor=g,
                    weight_tensor=w,
                    config=mmcfg,
                    compute_kernel_config=ckc,
                    memory_config=DRAM,
                    dtype=ttnn.bfloat8_b,
                    fuse_swiglu=swiglu,
                )
            )
            t_ag = timed(ag)
            ttnn.deallocate(g)
            logger.info(
                f"SWEEP M={M} {name}: separate gather {t_ag:.3f} + matmul {t_mm:.3f} = {t_ag + t_mm:.3f} ms; "
                f"overlap ceiling {max(t_ag, t_mm):.3f}"
            )
            best = None
            n_ok = 0
            for transpose in (True, False):
                grid = ttnn.CoreCoord(full.x, full.y - 1) if transpose else ttnn.CoreCoord(full.x - 1, full.y)
                in0_axis = grid.x if transpose else grid.y
                workers = math.ceil(in0_axis / 2)
                for mb, kb, nb, (sh, sw), bufs in itertools.product(
                    (4, 8, 16), (4, 8, 16), (4, 8, 16), ((2, 4), (1, 8), (1, 4)), (8, 24)
                ):
                    if swiglu and nb % 2:
                        continue
                    if mb % sh or nb % sw:
                        continue
                    cfg = ttnn.MinimalMatmulConfig(
                        M_block_size=mb,
                        K_block_size=kb,
                        N_block_size=nb,
                        subblock_h=sh,
                        subblock_w=sw,
                        compute_with_storage_grid_size=grid,
                    )

                    def fused():
                        s, b = nxt()
                        out = ttnn.experimental.all_gather_minimal_matmul_async(
                            input_tensor=xh,
                            weight_tensor=w,
                            config=cfg,
                            compute_kernel_config=ckc,
                            multi_device_global_semaphore=s,
                            barrier_semaphore=b,
                            num_links=2,
                            topology=ttnn.Topology.Ring,
                            cluster_axis=0,
                            memory_config=DRAM,
                            dtype=ttnn.bfloat8_b,
                            force_transpose=transpose,
                            num_workers_per_link=workers,
                            num_buffers_per_channel=bufs,
                            fuse_swiglu=swiglu,
                        )
                        return list(out)

                    try:
                        t = timed(fused)
                    except Exception as e:
                        continue
                    n_ok += 1
                    tag = f"T{int(transpose)} mb{mb} kb{kb} nb{nb} sb{sh}x{sw} buf{bufs}"
                    if best is None or t < best[0]:
                        best = (t, tag)
                        logger.info(f"SWEEP M={M} {name} new best {t:.3f} ms {tag}")
            logger.info(
                f"SWEEP M={M} {name} RESULT best fused {best[0] if best else float('nan'):.3f} ms "
                f"({best[1] if best else '-'}) over {n_ok} legal configs vs separate {t_ag + t_mm:.3f}, "
                f"ceiling {max(t_ag, t_mm):.3f}"
            )
        ttnn.deallocate(xh)
