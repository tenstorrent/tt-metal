# Fused row all-gather + matmul (ttnn.experimental.strided_all_gather_minimal_matmul_async, dim=2: each band of the
# matmul's M split starts as soon as its source device's rows arrive) vs today's all_gather_async + minimal_matmul, at
# the DP4 x TP8 shapes on every 8-chip mesh column (cluster_axis 0):
#   QKV:         rows [M/8, 8192] bf8 -> [M, 8192] @ [8192, 1280] bf8, fp32 acc, out bf16
#   FF13+SwiGLU: rows [M/8, 8192] bf8 -> [M, 8192] @ [8192, 7168] bf4 (fused SwiGLU, out 3584), fp32 acc, out bf8
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_row_gather_mm.py -s
# env: RG_M (4096,16384), RG_SHAPES (qkv,ff13), RG_ITERS (10), RG_GRID_Y (8), RG_OFF_Y (8), RG_WPL (4)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

H, TP = 8192, 8
MS = [int(m) for m in (os.environ.get("RG_M") or "4096,16384").split(",")]
SHAPES = (os.environ.get("RG_SHAPES") or "qkv,ff13").split(",")
ITERS = int(os.environ.get("RG_ITERS") or "10")
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    n = min(a.numel(), 4_000_000)
    return torch.corrcoef(torch.stack([a[:n], b[:n]]))[0, 1].item()


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_row_gather_mm(mesh_device, device_params):
    mesh = mesh_device
    full = mesh.compute_with_storage_grid_size()
    gy = int(os.environ.get("RG_GRID_Y") or "8")
    gx = int(os.environ.get("RG_GRID_X") or str(full.x))
    # transport cores: below the matmul grid (rows), or right of it when the grid is narrower (columns)
    off_y = int(os.environ.get("RG_OFF_Y") or (str(gy) if gx == full.x else "0"))
    off_x = int(os.environ.get("RG_OFF_X") or ("0" if gx == full.x else str(gx)))
    wpl = int(os.environ.get("RG_WPL") or "4")
    logger.info(f"RG grid {full.x}x{full.y}, fused matmul grid {gx}x{gy}, transport at ({off_x},{off_y}), wpl {wpl}")
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(full.x - 1, full.y - 1))])
    ag_sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)]
    ag_bars = [ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)]
    rg_sems = [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(3)] for _ in range(2)]
    k = [0, 0]
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(8, 4))
    rep = ttnn.ReplicateTensorToMesh(mesh)

    def ck(f):
        return ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    torch.manual_seed(0)
    w_qkv = torch.randn(H, 1280) * 0.02
    w_13 = torch.randn(H, 7168) * 0.02
    weights = {
        "qkv": (
            ttnn.from_torch(
                w_qkv.reshape(1, 1, H, 1280),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            ),
            w_qkv,
            False,
            ttnn.bfloat16,
        ),
        "ff13": (
            ttnn.from_torch(
                prepare_for_fused_swiglu(w_13, ndev=1, gate_is_first=True).reshape(1, 1, H, 7168),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat4_b,
                memory_config=DRAM,
                mesh_mapper=rep,
            ),
            w_13,
            True,
            ttnn.bfloat8_b,
        ),
    }
    lofi = ck(ttnn.MathFidelity.LoFi)

    def timed(name, fn, M):
        try:
            o = fn()
            ttnn.synchronize_device(mesh)
            for z in o:
                ttnn.deallocate(z)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                for z in fn():
                    ttnn.deallocate(z)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / ITERS * 1e3
            logger.info(f"RG M={M} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"RG M={M} {name} ERROR {str(e)[:600]}")
            return float("nan")

    for M in MS:
        x = torch.randn(1, 1, M, H)
        tx = ttnn.from_torch(
            x, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=DRAM, mesh_mapper=rows
        )
        for name in SHAPES:
            tw, w, swiglu, odt = weights[name]

            def ag():
                i = k[0]
                k[0] ^= 1
                return ttnn.experimental.all_gather_async(
                    input_tensor=tx,
                    persistent_output_buffer=None,
                    dim=2,
                    multi_device_global_semaphore=ag_sems[i],
                    num_links=2,
                    barrier_semaphore=ag_bars[i],
                    memory_config=DRAM,
                    topology=ttnn.Topology.Ring,
                    cluster_axis=0,
                )

            def mm(a, grid):
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=8,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=grid,
                )
                return ttnn.experimental.minimal_matmul(
                    input_tensor=a,
                    weight_tensor=tw,
                    config=cfg,
                    compute_kernel_config=lofi,
                    memory_config=DRAM,
                    dtype=odt,
                    fuse_swiglu=swiglu,
                )

            def separate():
                g = ag()
                return [g, mm(g, full)]

            def fused(mb=8):
                i = k[1]
                k[1] ^= 1
                cfg = ttnn.MinimalMatmulConfig(
                    M_block_size=mb,
                    K_block_size=8,
                    N_block_size=8,
                    subblock_h=1,
                    subblock_w=4,
                    compute_with_storage_grid_size=ttnn.CoreCoord(gx, gy),
                )
                return list(
                    ttnn.experimental.strided_all_gather_minimal_matmul_async(
                        tx,
                        tw,
                        None,
                        2,
                        rg_sems[i],
                        ttnn.CoreCoord(off_x, off_y),
                        num_links=2,
                        memory_config_ag=DRAM,
                        topology=ttnn.Topology.Ring,
                        cluster_axis=0,
                        config=cfg,
                        memory_config_mm=DRAM,
                        compute_kernel_config=lofi,
                        num_workers_per_link=wpl,
                        fuse_swiglu=swiglu,
                    )
                )

            g = ag()
            t_sep = timed("separate all_gather_async + minimal_matmul (full grid)", separate, M)
            t_ag = timed("  all_gather_async alone", lambda: [ag()], M)
            t_mm = timed("  minimal_matmul alone (full grid)", lambda: [mm(g, full)], M)
            t_mm8 = timed(f"  minimal_matmul alone ({gx}x{gy} grid)", lambda: [mm(g, ttnn.CoreCoord(gx, gy))], M)
            # each core row takes a 1/8 sub-band of every chip's slice: M blocks must tile it
            sub = M // 32 // (TP * TP)
            mbs = [m for m in (8, 4, 2) if sub % m == 0]
            # correctness first (a hang here is the transport/semaphore protocol)
            try:
                outs = fused(mbs[0])
                ttnn.synchronize_device(mesh)
                ref = mm(g, full)
                d_out, d_ref, d_g, d_gr = (ttnn.get_device_tensors(t) for t in (outs[1], ref, outs[0], g))
                msg = []
                for dev in (0, 13, 31):
                    msg.append(
                        f"dev{dev}: gathered==ag {torch.equal(ttnn.to_torch(d_g[dev]), ttnn.to_torch(d_gr[dev]))} "
                        f"pcc vs separate {_pcc(ttnn.to_torch(d_out[dev]), ttnn.to_torch(d_ref[dev])):.5f}"
                    )
                if not swiglu:
                    gold = x[0, 0] @ w
                    msg.append(f"dev5 pcc vs torch {_pcc(ttnn.to_torch(d_out[5]).reshape(M, -1), gold):.5f}")
                logger.info(f"RG M={M} {name} fused out {tuple(outs[1].shape)} CHECK " + " | ".join(msg))
                for z in outs + [ref]:
                    ttnn.deallocate(z)
            except Exception as e:
                logger.warning(f"RG M={M} {name} fused ERROR {str(e)[:800]}")
                ttnn.deallocate(g)
                continue

            # serial latency (sync after every call) and program-cache growth: host-side cost shows up here
            def serial(fn, n=5):
                t0 = time.perf_counter()
                for _ in range(n):
                    for z in fn():
                        ttnn.deallocate(z)
                    ttnn.synchronize_device(mesh)
                return (time.perf_counter() - t0) / n * 1e3

            n0 = mesh.num_program_cache_entries()
            s_f = serial(lambda: fused(mbs[0]))
            n1 = mesh.num_program_cache_entries()
            s_s = serial(separate)
            t_h0 = time.perf_counter()
            outs_h = fused(mbs[0])
            t_host = (time.perf_counter() - t_h0) * 1e3
            ttnn.synchronize_device(mesh)
            for z in outs_h:
                ttnn.deallocate(z)
            logger.info(
                f"RG M={M} {name} SERIAL fused {s_f:.3f} ms vs separate {s_s:.3f} ms; cache entries "
                f"{n0} -> {n1} over 5 fused calls; one fused call returns to host in {t_host:.3f} ms"
            )
            res = {}
            for mb in mbs:
                res[mb] = timed(f"FUSED row-gather matmul mb{mb}", lambda: fused(mb), M)
            ttnn.deallocate(g)
            best = min(res.values())
            logger.info(
                f"RG M={M} {name} SUMMARY separate {t_sep:.3f} (ag {t_ag:.3f} + mm {t_mm:.3f}; mm {gx}x{gy} "
                f"{t_mm8:.3f}) -> fused best {best:.3f} ms ({' '.join(f'mb{m}={v:.3f}' for m, v in res.items())})"
            )
        ttnn.deallocate(tx)
