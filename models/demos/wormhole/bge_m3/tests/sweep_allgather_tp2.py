# SPDX-License-Identifier: Apache-2.0
"""Standalone sweep of the seq-parallel K/V all_gather on 1x2 N300.

Shape mirrors attention: K/V = [B=12, H=16, Sq=4096, d=64] bf8, sharded on the
sequence dim across mesh axis 0 (mesh (2,1)), gathered to Sk=8192.

Sweeps sub_core_grids (more worker cores), num_workers_per_link,
num_buffers_per_channel, chunks_per_sync. Prints a ranked table so the in-model
config can be chosen without one-knob-at-a-time e2e runs.

Run:
  TT_VISIBLE_DEVICES=0 pytest models/demos/wormhole/bge_m3/tests/sweep_allgather_tp2.py -s -q
"""
import time

import pytest
import torch

import ttnn


def _corerangeset(mesh_device, gx, gy):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


@pytest.mark.parametrize("mesh_device", [(2, 1)], indirect=True, ids=["tp2_n300"])
@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 50_000_000, "num_command_queues": 1, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_sweep_allgather(mesh_device):
    B, H, Sq, d = 12, 16, 4096, 64
    axis = 0
    grid = mesh_device.compute_with_storage_grid_size()
    print(f"\ndevice grid = {grid.x}x{grid.y}")

    # Full replicated tensor [B,H,Sq,d] per shard; shard the seq dim over axis 0.
    torch_kv = torch.randn(B, H, Sq * 2, d, dtype=torch.bfloat16)
    kv = ttnn.from_torch(
        torch_kv,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(2, 1), dims=(2, None)),
    )

    def bench(label, **kw):
        try:
            # warmup
            for _ in range(2):
                o = ttnn.all_gather(kv, dim=2, cluster_axis=axis, topology=ttnn.Topology.Linear, **kw)
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            N = 20
            t0 = time.perf_counter()
            for _ in range(N):
                o = ttnn.all_gather(kv, dim=2, cluster_axis=axis, topology=ttnn.Topology.Linear, **kw)
                ttnn.deallocate(o)
            ttnn.synchronize_device(mesh_device)
            us = (time.perf_counter() - t0) / N * 1e6
            print(f"  {label:52s} {us:8.1f} us")
            return us
        except Exception as e:
            print(f"  {label:52s} CRASH: {str(e)[:60]}")
            return None

    results = {}
    # baseline
    results["baseline (default)"] = bench("baseline (default)", num_links=1)
    # sub_core_grids: more cores
    for gx, gy in [(8, 8), (8, 4), (8, 2), (4, 8), (8, 7), (8, 6)]:
        crs = _corerangeset(mesh_device, gx, gy)
        results[f"sub_core_grids {gx}x{gy}"] = bench(f"sub_core_grids {gx}x{gy}", num_links=1, sub_core_grids=crs)
    # worker / buffer hyperparams
    for w in [2, 4, 8]:
        results[f"num_workers_per_link={w}"] = bench(f"num_workers_per_link={w}", num_links=1, num_workers_per_link=w)
    for bpc in [2, 4, 8]:
        results[f"num_buffers_per_channel={bpc}"] = bench(
            f"num_buffers_per_channel={bpc}", num_links=1, num_buffers_per_channel=bpc
        )
    for cps in [2, 4, 8]:
        results[f"chunks_per_sync={cps}"] = bench(f"chunks_per_sync={cps}", num_links=1, chunks_per_sync=cps)
    # best sub_core_grid + workers combo
    crs88 = _corerangeset(mesh_device, 8, 8)
    for w in [4, 8]:
        results[f"8x8 + workers={w}"] = bench(
            f"8x8 + workers={w}", num_links=1, sub_core_grids=crs88, num_workers_per_link=w
        )

    print("\n=== RANKED (fastest first) ===")
    ok = {k: v for k, v in results.items() if v is not None}
    base = results.get("baseline (default)")
    for k, v in sorted(ok.items(), key=lambda x: x[1]):
        delta = f"({(v/base-1)*100:+.1f}% vs base)" if base else ""
        print(f"  {v:8.1f} us  {k:40s} {delta}")
