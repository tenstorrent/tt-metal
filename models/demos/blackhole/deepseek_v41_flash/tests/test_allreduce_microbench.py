# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Traced micro-benchmark of the decode attention-output all-reduce on the 8-device mesh axis (cluster_axis=1) of the 4x8 Galaxy:
the current reduce_scatter + all_gather pair (gpt_oss MeshConfig.allreduce) against ttnn.experimental.all_reduce_async with L1 width-sharded
input / intermediate / output, [32, 5120] bf16 per device. Prints AR lines: name = us per call (20 calls in one trace) and the PCC / max abs
difference of every variant against the pair."""

import time

import pytest
import torch

import ttnn
from models.demos.gpt_oss.config import mesh_4x8
from models.demos.gpt_oss.tt.ccl import CCLManager

M, N = 32, 5120
REP = 20
AXIS = 1
RING = 8


def traced_us(mesh, fn, n=30):
    def body():
        for _ in range(REP):
            fn()

    body()
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    for _ in range(3):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    dt = (time.perf_counter() - t) / n / REP
    ttnn.release_trace(mesh, tid)
    return dt * 1e6


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 100_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_allreduce_microbench(mesh_device):
    md = mesh_device
    mesh_config = mesh_4x8()
    res = {}
    # a different partial on every device: [4, 8, M, N] sharded over both mesh axes
    host = torch.randn(4, 8, M, N).to(torch.bfloat16)
    mk = lambda mc: ttnn.from_torch(
        host,
        device=md,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=mc,
        mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(4, 8)),
    )
    part = mk(ttnn.DRAM_MEMORY_CONFIG)
    gold = host.float().sum(dim=1)  # [4, M, N] per mesh row

    def dev_out(t):
        outs = ttnn.get_device_tensors(t)  # row-major over (4, 8)
        return torch.stack([ttnn.to_torch(outs[r * 8]).reshape(M, N) for r in range(4)])

    for links in (1, 2, 3, 4):
        ccl = CCLManager(md, num_links=links, topology=ttnn.Topology.Ring)
        try:
            res[f"pair_links{links}"] = traced_us(md, lambda: mesh_config.allreduce(ttnn.clone(part), ccl, axis=AXIS))
            clone_us = traced_us(md, lambda: ttnn.clone(part))
            res[f"pair_links{links}_minus_clone"] = res[f"pair_links{links}"] - clone_us
            res[f"pair_links{links}_pcc_vs_torch"] = pcc(
                dev_out(mesh_config.allreduce(ttnn.clone(part), ccl, axis=AXIS)), gold
            )
        except Exception as e:  # noqa: BLE001
            res[f"pair_links{links}"] = f"ERR {str(e)[:200]}"

    all_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    for ncores in (20,):
        npc = N // ncores
        if N % ncores or npc % 32:
            continue
        crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(8, 8), row_wise=True)
        in_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [M, npc], ttnn.ShardOrientation.ROW_MAJOR),
        )
        mid_mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, [M, npc * RING], ttnn.ShardOrientation.ROW_MAJOR),
        )
        for links in (1, 2, 3, 4):
            tag = f"ar_async_c{ncores}_l{links}"
            try:
                xin = mk(in_mc)
                mid = ttnn.from_torch(
                    torch.zeros(4, 8, M, N * RING),
                    device=md,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=mid_mc,
                    mesh_mapper=ttnn.ShardTensor2dMesh(md, dims=(0, 1), mesh_shape=(4, 8)),
                )
                sems = [ttnn.create_global_semaphore(md, all_cores, 0) for _ in range(2)]
                idx = [0]

                def run():
                    i = idx[0]
                    idx[0] ^= 1
                    return ttnn.experimental.all_reduce_async(
                        xin,
                        mid,
                        cluster_axis=AXIS,
                        mesh_device=md,
                        multi_device_global_semaphore=sems[i],
                        memory_config=in_mc,
                        dtype=ttnn.bfloat16,
                        topology=ttnn.Topology.Ring,
                        num_links=links,
                    )

                out = run()
                ttnn.synchronize_device(md)
                res[f"{tag}_pcc_vs_torch"] = pcc(dev_out(out), gold)
                res[tag] = traced_us(md, run)
                # end to end incl. DRAM -> sharded L1 and back (what a drop-in replacement of the pair costs today)
                res[tag + "_with_reshards"] = traced_us(
                    md,
                    lambda: ttnn.to_memory_config(run_in(run, xin, part, in_mc), ttnn.DRAM_MEMORY_CONFIG),
                )
            except Exception as e:  # noqa: BLE001
                res[tag] = f"ERR {str(e)[:300]}"
    for k, v in res.items():
        print(f"AR {k} = {v if isinstance(v, str) else f'{v:.4g}'}", flush=True)


def run_in(run, xin, part, in_mc):
    # reshard the DRAM partial into the (persistent) L1 input then reduce
    ttnn.copy(ttnn.to_memory_config(part, in_mc), xin)
    return run()
