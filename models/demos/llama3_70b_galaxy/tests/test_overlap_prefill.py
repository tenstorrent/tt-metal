# Can a prefill reduce-scatter run beside a matmul on Blackhole? Microbench of the CCL/compute overlap pattern the
# deepseek_v3_d_p prefill uses (two sub-devices, load manager, issue both programs, clear = join), at the 4K FF1 shape
# of Llama-3.3-70B TP32 on the 8x4 galaxy: x [4096, 2048] @ w1 [2048, 3584] (bf4, LoFi), partials summed over
# cluster_axis 1 (4 chips), scattered on N.
#   full:     MM(4096 rows) -> RS                              (what the model runs today)
#   halves:   MM(h0), MM(h1), RS(o0), RS(o1)                    (same work in halves, no overlap)
#   overlap:  MM(h0); [MM(h1) on the matmul rows || RS(o0) on the CCL rows]; RS(o1)
# Status: the overlap variant fails: ttnn.reduce_scatter on a sub-device that does not start at (0, 0) applies the
# offset twice and finds 0 cores. Measured on BH (FF1 shape): full 408 us, halves 508-536 us, mm 180 us, RS 225 us.
# Timed without trace (a sub-device manager load cannot be captured by a plain trace); every variant the same way.
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_overlap_prefill.py -s
# env: OVL_CCL_ROWS (1 | 2: worker rows given to the CCL sub-device), OVL_LINKS (2), OVL_WORKERS (4), OVL_ITERS (20)
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

M, K, N = 4096, 2048, 3584
CCL_ROWS = int(os.environ.get("OVL_CCL_ROWS", "1"))
LINKS = int(os.environ.get("OVL_LINKS", "2"))
WORKERS = int(os.environ.get("OVL_WORKERS", "4"))
ITERS = int(os.environ.get("OVL_ITERS", "20"))


@pytest.mark.parametrize(
    "device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING, "trace_region_size": 0}], indirect=True
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_overlap_prefill(mesh_device, device_params):
    mesh = mesh_device
    grid = mesh.compute_with_storage_grid_size()
    mm_rows = grid.y - CCL_ROWS
    tag = f"OVL grid={grid.x}x{grid.y} mm_rows={mm_rows} ccl_rows={CCL_ROWS} links={LINKS} workers={WORKERS}"
    logger.info(tag)
    ckc = ttnn.init_device_compute_kernel_config(
        mesh.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    def mm_cfg(rows):
        return ttnn.MinimalMatmulConfig(
            M_block_size=8,
            K_block_size=8,
            N_block_size=8,
            subblock_h=1,
            subblock_w=8,
            compute_with_storage_grid_size=ttnn.CoreCoord(grid.x, rows),
        )

    torch.manual_seed(0)
    ta = torch.randn(1, 1, M, K)
    tw = torch.randn(K, N) / K**0.5
    rep = ttnn.ReplicateTensorToMesh(mesh)
    dram = ttnn.DRAM_MEMORY_CONFIG
    a = ttnn.from_torch(
        ta, device=mesh, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=dram, mesh_mapper=rep
    )
    halves = [
        ttnn.from_torch(
            ta[:, :, i * M // 2 : (i + 1) * M // 2],
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=dram,
            mesh_mapper=rep,
        )
        for i in range(2)
    ]
    w = ttnn.from_torch(
        tw.reshape(1, 1, K, N),
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat4_b,
        memory_config=dram,
        mesh_mapper=rep,
    )
    G = tuple(mesh.shape)[1]
    golden = torch.chunk(G * (ta[0, 0] @ tw), G, dim=-1)[0]  # device (0, 0) holds N chunk 0

    def mm(x, rows):
        return ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=w,
            config=mm_cfg(rows),
            compute_kernel_config=ckc,
            memory_config=dram,
            dtype=ttnn.bfloat8_b,
        )

    def rs(x, sub=None):
        kw = {} if sub is None else {"subdevice_id": sub}
        return ttnn.reduce_scatter(
            x,
            3,
            cluster_axis=1,
            memory_config=dram,
            topology=ttnn.Topology.Ring,
            num_links=LINKS,
            num_workers_per_link=WORKERS,
            **kw,
        )

    mm_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, mm_rows - 1))])
    ccl_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, mm_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
    manager = mesh.create_sub_device_manager([ttnn.SubDevice([mm_crs]), ttnn.SubDevice([ccl_crs])], 0)
    ccl_sub = ttnn.SubDeviceId(1)

    def full():
        o = mm(a, mm_rows)
        r = rs(o)
        ttnn.deallocate(o)
        return [r]

    def halves_serial():
        o0, o1 = mm(halves[0], mm_rows), mm(halves[1], mm_rows)
        r0, r1 = rs(o0), rs(o1)
        ttnn.deallocate(o0), ttnn.deallocate(o1)
        return [r0, r1]

    def overlap():
        o0 = mm(halves[0], grid.y)  # full grid
        mesh.load_sub_device_manager(manager)
        o1 = mm(halves[1], mm_rows)  # sub-device 0
        r0 = rs(o0, ccl_sub)  # sub-device 1, beside the matmul
        mesh.clear_loaded_sub_device_manager()  # join
        r1 = rs(o1)
        ttnn.deallocate(o0), ttnn.deallocate(o1)
        return [r0, r1]

    def bench(name, fn):
        try:
            outs = fn()
            ttnn.synchronize_device(mesh)
            got = torch.cat([ttnn.to_torch(ttnn.get_device_tensors(r)[0]).float().reshape(-1, N // G) for r in outs])
            _, pcc = comp_pcc(golden, got, 0.99)
            for r in outs:
                ttnn.deallocate(r)
            t0 = time.perf_counter()
            for _ in range(ITERS):
                for r in fn():
                    ttnn.deallocate(r)
            ttnn.synchronize_device(mesh)
            us = (time.perf_counter() - t0) / ITERS * 1e6
            logger.info(f"{tag} {name}: {us:.1f} us pcc={pcc}")
        except Exception as e:
            logger.warning(f"{tag} {name} ERROR {str(e)[:500]}")
            try:
                mesh.clear_loaded_sub_device_manager()
            except Exception:
                pass

    for name, fn in (("full", full), ("halves", halves_serial), ("overlap", overlap), ("full", full)):
        bench(name, fn)
    # each piece alone (full grid and the matmul rows; the RS alone): the overlap ceiling
    for rows in (grid.y, mm_rows):
        o = mm(a, rows)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.deallocate(mm(a, rows))
        ttnn.synchronize_device(mesh)
        logger.info(f"{tag} mm_only rows={rows}: {(time.perf_counter() - t0) / ITERS * 1e6:.1f} us")
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.deallocate(rs(o))
        ttnn.synchronize_device(mesh)
        logger.info(f"{tag} rs_only (M={M}): {(time.perf_counter() - t0) / ITERS * 1e6:.1f} us")
        ttnn.deallocate(o)
    mesh.remove_sub_device_manager(manager)
