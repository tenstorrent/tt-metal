# Go / no-go for a two-program row-gather matmul: do a column all-gather (CCL sub-device: one core column) and a matmul
# (compute sub-device: the other columns) really run at the same time on Blackhole? Times each alone (manager loaded)
# and both issued back to back (no join between them); concurrent if both ~= max(each), serial if ~= sum.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_subdev_concurrency.py -s     env: SDC_M (16384)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

H, N, TP = 8192, 7168, 8
DRAM = ttnn.DRAM_MEMORY_CONFIG


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_subdev_concurrency(mesh_device, device_params):
    mesh = mesh_device
    M = int(os.environ.get("SDC_M") or "16384")
    grid = mesh.compute_with_storage_grid_size()
    rep = ttnn.ReplicateTensorToMesh(mesh)
    # SDC_SPLIT=col: CCL = last core column, matmul = the rest. =rows: matmul = rows 0..7 (8 rows: one per source device
    # along M), CCL = the remaining rows.
    split = os.environ.get("SDC_SPLIT") or "col"
    if split == "rows":
        mm_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, 7))])
        ccl_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 8), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        mm_grid = ttnn.CoreCoord(grid.x, 8)
    else:
        mm_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 2, grid.y - 1))])
        ccl_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(grid.x - 1, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        mm_grid = ttnn.CoreCoord(grid.x - 1, grid.y)
    iters = int(os.environ.get("SDC_ITERS") or "10")
    # Serial baseline on the full grid (no sub-device manager): all_gather_async + full-grid matmul, the layer as it
    # runs today. With SDC_ITERS large (seconds of work) the chips reach their power limit, so this compares overlap
    # against serial under sustained (throttled) load.
    full_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    fsems = [[ttnn.create_global_semaphore(mesh, full_crs, 0) for _ in range(2)] for _ in range(2)]
    fbars = [ttnn.create_global_semaphore(mesh, full_crs, 0) for _ in range(2)]
    fk = [0]
    lofi0 = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    xs = ttnn.from_torch(
        torch.randn(1, 1, M // TP, H),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    a0 = ttnn.from_torch(
        torch.randn(1, 1, M, H),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    w0 = ttnn.from_torch(
        torch.randn(1, 1, H, int(os.environ.get("SDC_N") or str(N))) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    cfg0 = ttnn.MinimalMatmulConfig(
        M_block_size=8, K_block_size=8, N_block_size=8, subblock_h=1, subblock_w=8, compute_with_storage_grid_size=grid
    )

    def ag_full():
        i = fk[0]
        fk[0] ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=xs,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=fsems[i],
            num_links=2,
            barrier_semaphore=fbars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            num_workers_per_link=8,
        )

    def mm_full():
        return ttnn.experimental.minimal_matmul(
            input_tensor=a0,
            weight_tensor=w0,
            config=cfg0,
            compute_kernel_config=lofi0,
            memory_config=DRAM,
            dtype=ttnn.bfloat8_b,
        )

    def timed0(name, fn):
        out = fn()
        ttnn.synchronize_device(mesh)
        for o in out:
            ttnn.deallocate(o)
        t0 = time.perf_counter()
        for _ in range(iters):
            for o in fn():
                ttnn.deallocate(o)
        ttnn.synchronize_device(mesh)
        ms = (time.perf_counter() - t0) / iters * 1e3
        logger.info(f"SDC M={M} iters={iters} {name}: {ms:.3f} ms")
        return ms

    t_ag0 = timed0("FULL GRID all-gather alone", lambda: [ag_full()])
    t_mm0 = timed0("FULL GRID matmul alone", lambda: [mm_full()])
    t_ser0 = timed0("FULL GRID serial all-gather + matmul (today)", lambda: [ag_full(), mm_full()])
    for t in (xs, a0, w0):
        ttnn.deallocate(t)
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([mm_crs]), ttnn.SubDevice([ccl_crs])], 0)
    mesh.load_sub_device_manager(manager)
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    sems = [[ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)] for _ in range(2)]
    bars = [ttnn.create_global_semaphore(mesh, ccl_crs, 0) for _ in range(2)]
    k = [0]
    lofi = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )
    x = ttnn.from_torch(
        torch.randn(1, 1, M // TP, H),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    a = ttnn.from_torch(
        torch.randn(1, 1, M, H),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    w = ttnn.from_torch(
        torch.randn(1, 1, H, N) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    NW = int(os.environ.get("SDC_N") or str(N))
    w = ttnn.from_torch(
        torch.randn(1, 1, H, NW) * 0.02,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=DRAM,
        mesh_mapper=rep,
    )
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=8,
        K_block_size=8,
        N_block_size=8,
        subblock_h=1,
        subblock_w=8,
        compute_with_storage_grid_size=mm_grid,
    )

    def ag():
        i = k[0]
        k[0] ^= 1
        return ttnn.experimental.all_gather_async(
            input_tensor=x,
            persistent_output_buffer=None,
            dim=2,
            multi_device_global_semaphore=sems[i],
            num_links=2,
            barrier_semaphore=bars[i],
            memory_config=DRAM,
            topology=ttnn.Topology.Ring,
            cluster_axis=0,
            subdevice_id=ttnn.SubDeviceId(1),
            **({"num_workers_per_link": int(os.environ["SDC_WPL"])} if os.environ.get("SDC_WPL") else {}),
        )

    def mm():
        return ttnn.experimental.minimal_matmul(
            input_tensor=a,
            weight_tensor=w,
            config=cfg,
            compute_kernel_config=lofi,
            memory_config=DRAM,
            dtype=ttnn.bfloat8_b,
        )

    def timed(name, fn, n=iters):
        try:
            out = fn()
            ttnn.synchronize_device(mesh)
            for o in out:
                ttnn.deallocate(o)
            t0 = time.perf_counter()
            for _ in range(n):
                for o in fn():
                    ttnn.deallocate(o)
            ttnn.synchronize_device(mesh)
            ms = (time.perf_counter() - t0) / n * 1e3
            logger.info(f"SDC M={M} split={split} N={NW} {name}: {ms:.3f} ms")
            return ms
        except Exception as e:
            logger.warning(f"SDC M={M} {name} ERROR {str(e)[:400]}")
            return float("nan")

    t_ag = timed("all-gather alone (CCL column)", lambda: [ag()])
    t_mm = timed("matmul alone (11 columns)", lambda: [mm()])
    t_both = timed("both, back to back", lambda: [ag(), mm()])
    t_both2 = timed("both, matmul first", lambda: [mm(), ag()])
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0)])
    t_both3 = timed("both, stall group = compute only", lambda: [ag(), mm()])
    mesh.set_sub_device_stall_group([ttnn.SubDeviceId(0), ttnn.SubDeviceId(1)])
    logger.info(
        f"SDC M={M} SUMMARY ag {t_ag:.3f} mm {t_mm:.3f} -> both {t_both:.3f} / {t_both2:.3f} / {t_both3:.3f} "
        f"(concurrent ~ {max(t_ag, t_mm):.3f}, serial ~ {t_ag + t_mm:.3f}) | FULL GRID ag {t_ag0:.3f} mm {t_mm0:.3f} "
        f"serial {t_ser0:.3f} | iters {iters}"
    )
    mesh.clear_loaded_sub_device_manager()
    mesh.remove_sub_device_manager(manager)
