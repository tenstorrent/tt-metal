# DP4 x TP8 attention block with all-to-all instead of all-gather + reduce-scatter (column axis, 8 chips):
#   today:  AG rows of the norm output [R, 8192] -> QKV [M, 8192] @ [8192, 1280] -> ... -> WO [M, 1024] @ [1024, 8192]
#           + reduce-scatter rows
#   a2a:    QKV on own rows with the FULL weight [R, 8192] @ [8192, 10240] -> all-to-all (split heads, concat rows)
#           -> ... -> all-to-all (split rows, concat heads) -> WO on own rows with the FULL weight [R, 8192] @ [8192, 8192]
# Same matmul FLOPs; traffic per chip ~18 MB at 16K instead of ~124 MB + the WO reduce-scatter. Times each piece.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_a2a_bench.py -s        env: A2A_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, TP, QKV, HD = 8192, 8, 1280, 1024
MS = [int(m) for m in (os.environ.get("A2A_M") or "4096,16384").split(",")]
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_a2a_bench(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    rows = ttnn.ShardTensor2dMesh(mesh, dims=(2, None), mesh_shape=(8, 4))
    rep = ttnn.ReplicateTensorToMesh(mesh)
    ckc = {
        f: ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=f, math_approx_mode=False, fp32_dest_acc_en=False, packer_l1_acc=True
        )
        for f in (ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi)
    }

    def mm(a, w, ck, dtype):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a, weight_tensor=w, config=cfg, compute_kernel_config=ck, memory_config=DRAM, dtype=dtype
        )

    def a2a(x, ind, outd):
        return ttnn.experimental.all_to_all_async_generic(
            x, ind, outd, num_links=2, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
        )

    def timed(name, fn, M):
        try:
            o = fn()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(o)
            t0 = time.perf_counter()
            for _ in range(10):
                ttnn.deallocate(fn())
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / 10 * 1e3
            logger.info(f"A2A M={M} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"A2A M={M} {name} ERROR {str(e)[:300]}")
            return float("nan")

    wqkv_full = ttnn.from_torch(
        torch.randn(1, 1, H, TP * QKV) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        mesh_mapper=rep,
    )
    wo_full = ttnn.from_torch(
        torch.randn(1, 1, H, H) * 0.02, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
    )
    for M in MS:
        R = M // TP
        torch.manual_seed(0)
        q_rows = torch.randn(1, 1, M, TP * QKV)  # full QKV output, rows sharded: chip r holds rows r*R..
        att = torch.randn(1, 1, M, H)  # attention output of all heads, all rows
        tq = ttnn.from_torch(q_rows, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rows)
        heads = ttnn.ShardTensor2dMesh(mesh, dims=(3, None), mesh_shape=(8, 4))
        ta = ttnn.from_torch(att, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=heads)
        h = ttnn.from_torch(
            torch.randn(1, 1, R, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
        )
        # correctness: a2a#1 chip j must hold columns j*1280.. of every row; a2a#2 chip r must hold rows r*R.. of all heads
        o1 = a2a(tq, 3, 2)
        o2 = a2a(ta, 2, 3)
        d1, d2 = ttnn.get_device_tensors(o1), ttnn.get_device_tensors(o2)
        ok = []
        for j in (0, 5):
            g1 = q_rows[0, 0, :, j * QKV : (j + 1) * QKV]
            t1 = ttnn.to_torch(d1[j * 4]).float().reshape(M, QKV)
            g2 = att[0, 0, j * R : (j + 1) * R, :]
            t2 = ttnn.to_torch(d2[j * 4]).float().reshape(R, H)
            ok.append(
                f"chip{j}: a2a1 shape {tuple(o1.shape)} pcc {torch.corrcoef(torch.stack([g1.flatten(), t1.flatten()]))[0, 1]:.4f}, "
                f"a2a2 shape {tuple(o2.shape)} pcc {torch.corrcoef(torch.stack([g2.flatten(), t2.flatten()]))[0, 1]:.4f}"
            )
        logger.info(f"A2A M={M} check " + " | ".join(ok))
        ttnn.deallocate(o1), ttnn.deallocate(o2)
        r = {}
        r["a2a_qkv"] = timed("all-to-all after QKV (split heads, concat rows)", lambda: a2a(tq, 3, 2), M)
        r["a2a_attn"] = timed("all-to-all after attention (split rows, concat heads)", lambda: a2a(ta, 2, 3), M)
        for fid, fn in ((ttnn.MathFidelity.HiFi2, "HiFi2"), (ttnn.MathFidelity.LoFi, "LoFi")):
            r[f"qkv_full_{fn}"] = timed(
                f"QKV full weight on own rows {fn}", lambda: mm(h, wqkv_full, ckc[fid], ttnn.bfloat16), M
            )
            r[f"wo_full_{fn}"] = timed(
                f"WO full weight on own rows {fn}", lambda: mm(h, wo_full, ckc[fid], ttnn.bfloat16), M
            )
        logger.info(f"A2A M={M} SUMMARY " + " ".join(f"{k}={v:.3f}" for k, v in r.items()))
        for z in (tq, ta, h):
            ttnn.deallocate(z)
