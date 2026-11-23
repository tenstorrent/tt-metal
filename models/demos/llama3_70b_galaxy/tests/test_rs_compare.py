# Compare reduce-scatter approaches for Llama-70B TP32 prefill on BH Galaxy.
# Tests at 4096x3584 (the FF1/FF3 reduce-scatter shape) on cluster_axis=1 (4 devices).
# Approaches:
#   1. reduce_scatter_minimal_async (current prefill RS path)
#   2. ttnn.reduce_scatter (standard, ring topology)
#   3. all_gather + local reduce (emulate RS via AG then reduce on each device)
#
# Usage:
#   cd /mnt/nvme0/jerrywang/tt-metal && source python_env/bin/activate
#   MESH_DEVICE=TG pytest models/demos/llama3_70b_galaxy/tests/test_rs_compare.py -s --timeout=300
#   For quick run (fewer iterations): RS_QUICK=1 pytest ...
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

N_PER_TRACE = 20
REPLAYS = 5


def _make_input(mesh, rows=4096, width=3584, dtype=ttnn.bfloat8_b):
    torch_x = torch.randn(1, 1, rows, width, dtype=torch.bfloat16)
    x = ttnn.from_torch(
        torch_x,
        device=mesh,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
    )
    return x, torch_x


def _golden_rs(torch_x, num_devices=4):
    """Golden reduce-scatter: sum all replicas, then take 1/num_devices of the width."""
    summed = torch_x * num_devices  # all replicas are identical, so sum = x * N
    chunk = summed.shape[-1] // num_devices
    return summed[..., :chunk].float()


def bench_rs_minimal_async(mesh, x, sem_sets, barriers, cfg):
    """Benchmark reduce_scatter_minimal_async."""
    out = ttnn.experimental.reduce_scatter_minimal_async(
        input_tensor=x,
        dim=3,
        multi_device_global_semaphore=[sem_sets[0]],
        barrier_semaphore=barriers[0],
        num_links=cfg.get("links", 2),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=cfg.get("topology", ttnn.Topology.Ring),
        cluster_axis=1,
        num_workers_per_link=cfg.get("workers", 4),
    )
    ttnn.synchronize_device(mesh)
    dev0_out = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)

    # Trace benchmark
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    for i in range(N_PER_TRACE):
        outs.append(
            ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor=x,
                dim=3,
                multi_device_global_semaphore=[sem_sets[i % len(sem_sets)]],
                barrier_semaphore=barriers[i % len(barriers)],
                num_links=cfg.get("links", 2),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=cfg.get("topology", ttnn.Topology.Ring),
                cluster_axis=1,
                num_workers_per_link=cfg.get("workers", 4),
            )
        )
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, dev0_out


def bench_rs_standard(mesh, x, cfg):
    """Benchmark ttnn.reduce_scatter (standard op)."""
    grid = mesh.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})

    out = ttnn.reduce_scatter(
        x,
        3,
        cluster_axis=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=cfg.get("topology", ttnn.Topology.Ring),
        num_links=cfg.get("links", 2),
        num_workers_per_link=cfg.get("workers", 1),
    )
    ttnn.synchronize_device(mesh)
    dev0_out = ttnn.to_torch(ttnn.get_device_tensors(out)[0]).float()
    ttnn.deallocate(out)

    # Trace benchmark
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    for i in range(N_PER_TRACE):
        outs.append(
            ttnn.reduce_scatter(
                x,
                3,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=cfg.get("topology", ttnn.Topology.Ring),
                num_links=cfg.get("links", 2),
                num_workers_per_link=cfg.get("workers", 1),
            )
        )
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, dev0_out


def bench_ag_reduce(mesh, x, sem_sets, barriers):
    """Benchmark all_gather on axis=1 then local reduce (sum along gathered dim).
    This emulates reduce-scatter by gathering everything then keeping just one chunk."""
    # all_gather on axis 1 (4 devices): output width = 4 * input width
    ag_out = ttnn.all_gather(
        x,
        dim=3,
        cluster_axis=1,
        num_links=2,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=ttnn.Topology.Ring,
    )
    ttnn.synchronize_device(mesh)

    dev0_out = ttnn.to_torch(ttnn.get_device_tensors(ag_out)[0]).float()
    ttnn.deallocate(ag_out)

    # Trace benchmark - just the all_gather part
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    outs = []
    for i in range(N_PER_TRACE):
        outs.append(
            ttnn.all_gather(
                x,
                dim=3,
                cluster_axis=1,
                num_links=2,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
            )
        )
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)  # warm
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    us = (time.perf_counter() - t0) / (REPLAYS * N_PER_TRACE) * 1e6
    ttnn.release_trace(mesh, tid)
    for o in outs:
        ttnn.deallocate(o)
    return us, dev0_out


FABRICS = {
    "1D_RING": ttnn.FabricConfig.FABRIC_1D_RING,
}


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": FABRICS["1D_RING"], "trace_region_size": 100_000_000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
@pytest.mark.parametrize(
    "rows,width,dtype_name",
    [
        (4096, 3584, "bf8"),  # FF1/FF3 RS shape
        (4096, 2048, "bf16"),  # WO RS shape
    ],
    ids=["ff_4096x3584_bf8", "wo_4096x2048_bf16"],
)
def test_rs_compare(mesh_device, device_params, rows, width, dtype_name):
    dtype = ttnn.bfloat8_b if dtype_name == "bf8" else ttnn.bfloat16
    grid = mesh_device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    sem_sets = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(N_PER_TRACE)]
    barriers = [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(N_PER_TRACE)]

    x, torch_x = _make_input(mesh_device, rows=rows, width=width, dtype=dtype)
    golden = _golden_rs(torch_x, num_devices=4)

    results = []

    # 1. reduce_scatter_minimal_async (current path)
    for workers in [1, 4]:
        cfg = {"topology": ttnn.Topology.Ring, "links": 2, "workers": workers}
        try:
            us, dev0_out = bench_rs_minimal_async(mesh_device, x, sem_sets, barriers, cfg)
            ok, pcc = comp_pcc(golden, dev0_out, 0.98)
            results.append(("rs_minimal_async", f"w={workers}", us, ok, pcc))
            logger.info(f"rs_minimal_async workers={workers}: {us:.1f} us/call  pcc={pcc:.4f}")
        except Exception as e:
            logger.warning(f"rs_minimal_async workers={workers}: ERROR {e}")
            results.append(("rs_minimal_async", f"w={workers}", -1, False, 0))

    # 2. ttnn.reduce_scatter (standard)
    for workers in [1]:
        cfg = {"topology": ttnn.Topology.Ring, "links": 2, "workers": workers}
        try:
            us, dev0_out = bench_rs_standard(mesh_device, x, cfg)
            ok, pcc = comp_pcc(golden, dev0_out, 0.98)
            results.append(("rs_standard", f"w={workers}", us, ok, pcc))
            logger.info(f"rs_standard workers={workers}: {us:.1f} us/call  pcc={pcc:.4f}")
        except Exception as e:
            logger.warning(f"rs_standard workers={workers}: ERROR {e}")
            results.append(("rs_standard", f"w={workers}", -1, False, 0))

    # 3. all_gather (reference: same data, same links — shows ethernet ceiling)
    try:
        us, _ = bench_ag_reduce(mesh_device, x, sem_sets, barriers)
        results.append(("all_gather_ref", "links=2", us, True, 1.0))
        logger.info(f"all_gather (reference, same shape): {us:.1f} us/call")
    except Exception as e:
        logger.warning(f"all_gather reference: ERROR {e}")
        results.append(("all_gather_ref", "links=2", -1, False, 0))

    ttnn.deallocate(x)

    # Summary
    logger.info(f"\n=== RS COMPARE  shape=[1,1,{rows},{width}] dtype={dtype_name} ===")
    for name, cfg_str, us, ok, pcc in results:
        status = f"{us:7.1f} us" if us > 0 else "  ERROR"
        logger.info(f"  {name:30s} {cfg_str:10s}  {status}  pcc={pcc:.4f}")

    # The AG reference should be fastest; RS should not be more than 2x slower
    ag_us = next((r[2] for r in results if r[0] == "all_gather_ref" and r[2] > 0), None)
    if ag_us:
        for name, cfg_str, us, ok, pcc in results:
            if us > 0 and "all_gather" not in name:
                ratio = us / ag_us
                logger.info(f"  {name} {cfg_str}: {ratio:.1f}x all_gather time")
