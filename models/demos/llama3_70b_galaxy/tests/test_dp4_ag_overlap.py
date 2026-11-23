# Can a row all-gather on the column axis run beside a matmul on Blackhole? The DP4 x TP8 prefill layer gathers the norm
# output (bf8 rows, cluster_axis 0) before QKV and FF1|FF3. Split the rows into two halves A, B: while FF1|FF3 runs on
# A's gathered rows (compute sub-device: all but the last CCL_ROWS worker rows), gather B (CCL sub-device). Measures:
#   serial:  AG(A) mm(A) AG(B) mm(B) on the full grid
#   overlap: AG(A) full grid; [mm(A) on compute rows || AG(B) on CCL rows]; mm(B) full grid
#   pieces alone (mm on compute rows / full grid, AG on CCL rows / full grid)
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_dp4_ag_overlap.py -s
# env: OVL_M (16384), OVL_CCL_ROWS (1), OVL_ITERS (10)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, N = 8192, 7168  # FF1|FF3 packed per chip (TP8)
M = int(os.environ.get("OVL_M") or "16384")
CCL_ROWS = int(os.environ.get("OVL_CCL_ROWS") or "1")
CCL_COL = os.environ.get("OVL_CCL_COL", "0") == "1"  # give the CCL the last core column instead of the last row(s)
ITERS = int(os.environ.get("OVL_ITERS") or "10")
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_dp4_ag_overlap(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    mm_rows = grid.y - CCL_ROWS
    mm_cols = grid.x
    if CCL_COL:
        mm_rows, mm_cols = grid.y, grid.x - 1
    rep = ttnn.ReplicateTensorToMesh(mesh)
    lofi = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    R2 = M // 8 // 2
    xa = ttnn.from_torch(
        torch.randn(1, 1, R2, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
    )
    xb = ttnn.from_torch(
        torch.randn(1, 1, R2, H), device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=rep
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, H, N) * 0.02, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat4_b, mesh_mapper=rep
    )

    full_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    mm_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(mm_cols - 1, mm_rows - 1))])
    if CCL_COL:
        ccl_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(mm_cols, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
    else:
        ccl_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, mm_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )

    def sems(crs):
        return [[ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)] for _ in range(2)], [
            ttnn.create_global_semaphore(mesh, crs, 0) for _ in range(2)
        ]

    full_sems, full_bars = sems(full_crs)
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([mm_crs]), ttnn.SubDevice([ccl_crs])], 0)
    idx = [0]

    def ag(x, sem_set, bar_set, sub=None):
        i = idx[0]
        idx[0] ^= 1
        kw = {} if sub is None else {"subdevice_id": sub}
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sem_set[i],
            num_links=2,
            barrier_semaphore=bar_set[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            **kw,
        )

    def mm(a, rows, cols=None):
        cfg = ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=ttnn.CoreCoord(cols or grid.x, rows),
        )
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=ttnn.bfloat8_b,
        )

    def timed(name, fn):
        try:
            outs = fn()
            ttnn.synchronize_device(mesh)
            for o in outs:
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                for o in fn():
                    ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / ITERS * 1e3
            logger.info(f"AGOVL M={M} ccl={'col' if CCL_COL else CCL_ROWS} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"AGOVL M={M} {name} ERROR {str(e)[:400]}")
            try:
                mesh.clear_loaded_sub_device_manager()
            except Exception:
                pass
            return float("nan")

    def serial():
        ga = ag(xa, full_sems, full_bars)
        oa = mm(ga, grid.y)
        gb = ag(xb, full_sems, full_bars)
        ob = mm(gb, grid.y)
        ttnn.deallocate(ga), ttnn.deallocate(gb)
        return [oa, ob]

    # semaphores for the sub-device run: AG lives on the CCL sub-device cores
    sub_sems = {}

    def overlap():
        ga = ag(xa, full_sems, full_bars)
        mesh.load_sub_device_manager(manager)
        if "s" not in sub_sems:
            sub_sems["s"] = sems(ccl_crs)
        ss, sb = sub_sems["s"]
        oa = mm(ga, mm_rows, mm_cols)  # sub-device 0 (compute cores)
        gb = ag(xb, ss, sb, ttnn.SubDeviceId(1))  # sub-device 1 (CCL rows), beside the matmul
        mesh.clear_loaded_sub_device_manager()  # join
        ob = mm(gb, grid.y)
        ttnn.deallocate(ga), ttnn.deallocate(gb)
        return [oa, ob]

    t_serial = timed("serial", serial)
    t_ovl = timed("overlap", overlap)
    ga = ag(xa, full_sems, full_bars)
    t_mm_full = timed("mm alone full grid", lambda: [mm(ga, grid.y)])
    t_mm_sub = timed("mm alone compute cores", lambda: [mm(ga, mm_rows, mm_cols)])
    t_ag = timed("ag alone full grid", lambda: [ag(xb, full_sems, full_bars)])
    logger.info(
        f"AGOVL M={M} SUMMARY serial {t_serial:.3f} overlap {t_ovl:.3f} | ideal overlap ~ ag + max(mm_sub, ag) + mm "
        f"= {t_ag + max(t_mm_sub, t_ag) + t_mm_full:.3f} vs serial pieces {2 * t_ag + 2 * t_mm_full:.3f}"
    )
    mesh.remove_sub_device_manager(manager)
