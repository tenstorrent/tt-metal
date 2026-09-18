# SPDX-License-Identifier: Apache-2.0
"""Times the collectives and matmuls a Megatron-SP Llama block issues (from issue #52944), plus bfp8 payload variants.
Run from the repo root with `-p conftest -s`; TT_MESH_GRAPH_DESC_PATH selects the 1x4 RING MGD for -k torus_ring and
the LINE MGD for -k mesh_linear."""
import os
import time

import pytest
import torch
from loguru import logger

import ttnn

HIDDEN, INTERMEDIATE, SEQ, TP = 4096, 14336, 2048, 4
QKV = (32 + 2 * 8) * 128
REPS, ITERS, TRACE = 10, 20, 90000000


def _sems(mesh_device, cores, n):
    return [ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(n)]


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY, "trace_region_size": TRACE},
        {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "trace_region_size": TRACE},
    ],
    ids=["torus_ring", "mesh_linear"],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(1, 4)], indirect=True)
def test_verify(mesh_device, device_params):
    fabric = device_params["fabric_config"]
    torus = "TORUS" in str(fabric)
    topology = ttnn.Topology.Ring if torus else ttnn.Topology.Linear
    name = "torus_ring" if torus else "mesh_linear"
    logger.info(f"=== {name}: fabric={fabric}, MGD={os.getenv('TT_MESH_GRAPH_DESC_PATH')} ===")
    grid = mesh_device.compute_with_storage_grid_size()
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    ag_sem, rs_sem = _sems(mesh_device, crs, 2), _sems(mesh_device, crs, 3)
    ar_sem = (_sems(mesh_device, crs, 2), _sems(mesh_device, crs, 3), _sems(mesh_device, crs, 2))
    barrier = ttnn.create_global_semaphore(mesh_device, crs, 0)

    def mk(shape, shard_dim=None, dtype=ttnn.bfloat16):
        t = torch.randn(shape, dtype=torch.bfloat16)
        mapper = (
            ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(1, 4), dims=(None, shard_dim))
            if shard_dim is not None
            else ttnn.ReplicateTensorToMesh(mesh_device)
        )
        return ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype, mesh_mapper=mapper)

    def timed(fn):
        for _ in range(2):
            fn()
        ttnn.synchronize_device(mesh_device)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        for _ in range(REPS):
            fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        dt = (time.perf_counter() - t0) / (ITERS * REPS) * 1e6
        ttnn.release_trace(mesh_device, tid)
        return dt

    results = {}
    ag_in2, ag_in3, rs_in = mk([1, 1, SEQ, HIDDEN], 2), mk([1, 1, SEQ, HIDDEN], 3), mk([1, 1, SEQ, HIDDEN])
    for links in (1, 2):
        results[f"all_gather(dim=2) links={links}"] = timed(
            lambda l=links: ttnn.experimental.all_gather_async(
                ag_in2,
                dim=2,
                multi_device_global_semaphore=ag_sem,
                num_links=l,
                topology=topology,
                cluster_axis=1,
                barrier_semaphore=barrier,
            )
        )
        results[f"reduce_scatter(dim=2) links={links}"] = timed(
            lambda l=links: ttnn.experimental.reduce_scatter_minimal_async(
                rs_in,
                dim=2,
                multi_device_global_semaphore=rs_sem,
                barrier_semaphore=barrier,
                num_links=l,
                topology=topology,
                cluster_axis=1,
            )
        )
    for links in (1, 2):
        results[f"all_gather(dim=3) links={links}"] = timed(
            lambda l=links: ttnn.experimental.all_gather_async(
                ag_in3,
                dim=3,
                multi_device_global_semaphore=ag_sem,
                num_links=l,
                topology=topology,
                cluster_axis=1,
                barrier_semaphore=barrier,
            )
        )
        results[f"reduce_scatter(dim=3) links={links}"] = timed(
            lambda l=links: ttnn.experimental.reduce_scatter_minimal_async(
                rs_in,
                dim=3,
                multi_device_global_semaphore=rs_sem,
                barrier_semaphore=barrier,
                num_links=l,
                topology=topology,
                cluster_axis=1,
            )
        )
        results[f"all_reduce links={links} (classic TP)"] = timed(
            lambda l=links: ttnn.experimental.all_reduce_async(
                rs_in,
                1,
                mesh_device,
                ar_sem[0],
                ar_sem[1],
                ar_sem[2],
                ttnn.ReduceType.Sum,
                topology=topology,
                num_links=l,
            )
        )
    ag_in2_b8, rs_in_b8 = mk([1, 1, SEQ, HIDDEN], 2, ttnn.bfloat8_b), mk([1, 1, SEQ, HIDDEN], dtype=ttnn.bfloat8_b)
    for links in (2,):
        try:
            results[f"all_gather(dim=2) bfp8 links={links}"] = timed(
                lambda l=links: ttnn.experimental.all_gather_async(
                    ag_in2_b8,
                    dim=2,
                    multi_device_global_semaphore=ag_sem,
                    num_links=l,
                    topology=topology,
                    cluster_axis=1,
                    barrier_semaphore=barrier,
                )
            )
            results[f"reduce_scatter(dim=2) bfp8 links={links}"] = timed(
                lambda l=links: ttnn.experimental.reduce_scatter_minimal_async(
                    rs_in_b8,
                    dim=2,
                    multi_device_global_semaphore=rs_sem,
                    barrier_semaphore=barrier,
                    num_links=l,
                    topology=topology,
                    cluster_axis=1,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"bfp8 variant failed: {e}")

    def bench_mm(label, m, k, n):
        a, w = mk([1, 1, m, k]), mk([1, 1, n, k])
        results[label] = timed(lambda: ttnn.linear(a, w, transpose_b=True))

    bench_mm(f"mm qkv      [{SEQ},{HIDDEN}]@[{HIDDEN},{QKV // TP}]", SEQ, HIDDEN, QKV // TP)
    bench_mm(f"mm out_proj [{SEQ},{HIDDEN // TP}]@[{HIDDEN // TP},{HIDDEN}]", SEQ, HIDDEN // TP, HIDDEN)
    bench_mm(f"mm gate_up  [{SEQ},{HIDDEN}]@[{HIDDEN},{2 * INTERMEDIATE // TP}]", SEQ, HIDDEN, 2 * INTERMEDIATE // TP)
    bench_mm(
        f"mm w2       [{SEQ},{INTERMEDIATE // TP}]@[{INTERMEDIATE // TP},{HIDDEN}]", SEQ, INTERMEDIATE // TP, HIDDEN
    )
    bench_mm("mm tiny     [32,32]@[32,32] (dispatch floor)", 32, 32, 32)
    logger.info(f"##### {name} RESULTS (us/op, traced {REPS}x{ITERS}) #####")
    for k, v in results.items():
        logger.info(f"[{name}] {k:52s} {v:9.1f}")
