# WO for the DP4 x TP8 prefill without the reduce-scatter: today chip r computes the partial a_r @ WO_r for ALL M rows
# ([M, 8192] per chip, 8x the attention output) and matmul_reduce_scatter_prefill sums them down the column. Instead an
# all-to-all hands chip r every head of ITS rows ([M/8, 8192], ~15 MB moved at 16K instead of ~124 MB) and chip r runs
# the full WO [8192, 8192] on them. Same FLOPs, the output lands in the residual's sequence-parallel layout.
# Checks both all_to_all_async_generic argument orders against torch, then times MMRS vs a2a + full matmul.
#   pytest models/demos/llama3_70b_galaxy/tests/test_wo_a2a.py -s        env: WA_M (4096,16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, TP, HD = 8192, 8, 1024
DRAM = ttnn.DRAM_MEMORY_CONFIG


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.flatten().float(), b.flatten().float()]))[0, 1].item()


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_wo_a2a(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    lofi = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    mmrs_ck = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi)
    heads = ttnn.ShardTensor2dMesh(mesh, dims=(3, None), mesh_shape=(8, 4))  # chip r <- columns r*1024..
    wo_rows = ttnn.ShardTensor2dMesh(mesh, dims=(0, None), mesh_shape=(8, 4))  # chip r <- WO rows r*1024.. (2D)
    rep = ttnn.ReplicateTensorToMesh(mesh)
    torch.manual_seed(0)
    wo = torch.randn(H, H) * 0.02  # [in = heads*128, out]
    t_wo_sh = ttnn.from_torch(wo, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=wo_rows)
    t_wo_full = ttnn.from_torch(
        wo.reshape(1, 1, H, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
    )

    def mm(a, w, mb=8, kb=16, nb=8):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=mb,
            K_block_size=kb,
            N_block_size=nb,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=grid,
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=ttnn.bfloat8_b,
        )

    def a2a(x, d0, d1):
        return ttnn.experimental.all_to_all_async_generic(
            x, d0, d1, num_links=2, memory_config=DRAM, topology=ttnn.Topology.Ring, cluster_axis=0
        )

    def mmrs(a):
        return ttnn.experimental.matmul_reduce_scatter_prefill(
            a,
            t_wo_sh,
            cluster_axis=0,
            scatter_dim=-2,
            topology=ttnn.Topology.Ring,
            num_links=2,
            compute_kernel_config=mmrs_ck,
            memory_config=DRAM,
        )

    def timed(name, fn, M, n=20):
        try:
            o = fn()
            ttnn.synchronize_device(mesh)
            for t in o if isinstance(o, (list, tuple)) else [o]:
                ttnn.deallocate(t)
            t0 = time.perf_counter()
            for _ in range(n):
                o = fn()
                for t in o if isinstance(o, (list, tuple)) else [o]:
                    ttnn.deallocate(t)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / n * 1e3
            logger.info(f"WOA2A M={M} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:  # noqa: BLE001
            logger.warning(f"WOA2A M={M} {name} ERROR {str(e)[:300]}")
            return float("nan")

    for M in [int(m) for m in (os.environ.get("WA_M") or "4096,16384").split(",")]:
        R = M // TP
        att = torch.randn(1, 1, M, H)  # attention output, all heads (concat), all rows of one prompt
        ta = ttnn.from_torch(att, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=heads)
        gold = att[0, 0].float() @ wo.float()  # [M, H]
        for d0, d1 in ((2, 3), (3, 2)):
            try:
                o = a2a(ta, d0, d1)
                dts = ttnn.get_device_tensors(o)
                s = []
                for r in (0, 5):
                    t = ttnn.to_torch(dts[r * 4]).float()
                    if tuple(t.shape[-2:]) == (R, H):
                        s.append(f"chip{r} rows pcc {_pcc(t.reshape(R, H), att[0, 0, r * R:(r + 1) * R]):.4f}")
                    else:
                        s.append(f"chip{r} shape {tuple(t.shape)}")
                logger.info(f"WOA2A M={M} a2a({d0},{d1}) out {tuple(o.shape)}: " + ", ".join(s))
                ttnn.deallocate(o)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"WOA2A M={M} a2a({d0},{d1}) ERROR {str(e)[:200]}")
        # end to end: a2a then full WO; compare with gold rows
        for d0, d1 in ((2, 3), (3, 2)):
            try:
                o = a2a(ta, d0, d1)
                if tuple(o.shape)[-2:] != (R, H):
                    ttnn.deallocate(o)
                    continue
                y = mm(o, t_wo_full)
                dts = ttnn.get_device_tensors(y)
                pc = [
                    _pcc(ttnn.to_torch(dts[r * 4]).float().reshape(R, H), gold[r * R : (r + 1) * R]) for r in (0, 3, 7)
                ]
                logger.info(f"WOA2A M={M} a2a({d0},{d1})+WO_full vs torch: pcc " + " ".join(f"{p:.4f}" for p in pc))
                ym = mmrs(ta)
                dts = ttnn.get_device_tensors(ym)
                pm = [
                    _pcc(ttnn.to_torch(dts[r * 4]).float().reshape(R, H), gold[r * R : (r + 1) * R]) for r in (0, 3, 7)
                ]
                logger.info(f"WOA2A M={M} MMRS vs torch: pcc " + " ".join(f"{p:.4f}" for p in pm))
                ttnn.deallocate(o), ttnn.deallocate(y), ttnn.deallocate(ym)
                t_mmrs = timed("wo_mmrs", lambda: mmrs(ta), M)
                t_a2a = timed("a2a", lambda: a2a(ta, d0, d1), M)
                t_a2a_mm = timed("a2a + wo_full", lambda: mm(a2a(ta, d0, d1), t_wo_full), M)
                for mb, kb, nb in ((8, 16, 8), (8, 32, 8), (4, 32, 8), (8, 16, 16), (4, 16, 16)):
                    o2 = a2a(ta, d0, d1)
                    timed(f"wo_full only M{mb}K{kb}N{nb}", lambda: mm(o2, t_wo_full, mb, kb, nb), M)
                    ttnn.deallocate(o2)
                logger.info(f"WOA2A M={M} SUMMARY mmrs {t_mmrs:.3f} a2a {t_a2a:.3f} a2a+wo {t_a2a_mm:.3f} ms")
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(f"WOA2A M={M} e2e a2a({d0},{d1}) ERROR {str(e)[:300]}")
        ttnn.deallocate(ta)
