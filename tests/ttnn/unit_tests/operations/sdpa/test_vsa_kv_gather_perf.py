# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark for the fine-stage K/V all-gather (the largest VSA-only cost in the block):
[1, H, S_local, d] bf16 gathered along the SP axis (8 devices) on the 4x8 galaxy. Sweeps the
all_gather_async configuration; reports device-side time per gather from a traced replay and the
achieved per-device receive bandwidth. Run: ./run_h3_safe.sh kv_gather <this file> -q -s"""

import os
import time

import pytest
import torch
import ttnn

from models.common.utility_functions import skip_for_wormhole_b0
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.test import ring_params_8k_req_exact_devices

# fabric (ring) must be configured on the mesh for the async CCLs, as in the block tests
_DEVICE_PARAMS = {**ring_params_8k_req_exact_devices, "trace_region_size": 200_000_000, "l1_small_size": 65536}

SP_AXIS = 1  # 4x8 mesh, SP over the 8-wide axis (matches the block tests' sp1tp0)
H, D = 14, 128
S_LOCAL = {"5s": 4768, "10s": 9216, "15s": 14464}


def _bench(mesh_device, fn, out_shape_bytes, n_iters=10):
    """Trace n_iters gathers back to back, replay 3x, return ms per gather and GB/s received/device."""
    out = fn()  # compile
    ttnn.synchronize_device(mesh_device)
    tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
    for _ in range(n_iters):
        out = fn()
    ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
    ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=True)  # warm
    ttnn.synchronize_device(mesh_device)
    t0 = time.perf_counter()
    reps = 3
    for _ in range(reps):
        ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh_device)
    ms = (time.perf_counter() - t0) * 1e3 / (reps * n_iters)
    ttnn.release_trace(mesh_device, tid)
    recv_bytes = out_shape_bytes * 7 / 8  # a device receives the other 7 shards
    return ms, recv_bytes / (ms * 1e-3) / 1e9, out


@skip_for_wormhole_b0("galaxy only")
@pytest.mark.parametrize(
    "mesh_device, device_params",
    [((4, 8), _DEVICE_PARAMS)],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize("dur", ["15s", "10s"])
def test_kv_gather_sweep(mesh_device, dur):
    s_local = S_LOCAL[dur]
    num_links = int(os.environ.get("KV_GATHER_LINKS", "2"))
    torch_k = torch.randn(1, H, s_local * 8, D)  # the gathered result; each shard gets its slice
    k = ttnn.from_torch(
        torch_k,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 2), mesh_shape=tuple(mesh_device.shape)),
    )
    out_bytes = torch_k.numel() * 2
    results = []

    for topology in (ttnn.Topology.Ring, ttnn.Topology.Linear):
        ccl = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)

        # 1. exactly the model path: persistent-buffer all-gather, default hyperparams
        ms, gbs, out = _bench(
            mesh_device, lambda: ccl.all_gather_persistent_buffer(k, dim=2, mesh_axis=SP_AXIS), out_bytes
        )
        results.append((f"{topology.name} persistent default", ms, gbs))
        # 2. persistent + tuned hyperparams
        ms, gbs, out = _bench(
            mesh_device,
            lambda: ccl.all_gather_persistent_buffer(k, dim=2, mesh_axis=SP_AXIS, use_hyperparams=True),
            out_bytes,
        )
        results.append((f"{topology.name} persistent hyper", ms, gbs))
        # 3. non-persistent (barrier semaphore) variants with explicit knobs
        for cps, wpl, nbuf in ((None, None, None), (16, 3, 2), (8, 4, 4), (32, 2, 8), (4, 8, 2)):

            def fn():
                return ttnn.experimental.all_gather_async(
                    k,
                    persistent_output_buffer=None,
                    dim=2,
                    multi_device_global_semaphore=ccl.get_ag_ping_pong_semaphore(SP_AXIS),
                    barrier_semaphore=ccl.get_barrier_semaphore(SP_AXIS),
                    num_links=num_links,
                    topology=topology,
                    cluster_axis=SP_AXIS,
                    **(
                        {}
                        if cps is None
                        else dict(chunks_per_sync=cps, num_workers_per_link=wpl, num_buffers_per_channel=nbuf)
                    ),
                )

            try:
                ms, gbs, out = _bench(mesh_device, fn, out_bytes)
                results.append((f"{topology.name} barrier cps={cps} wpl={wpl} nbuf={nbuf}", ms, gbs))
                ttnn.deallocate(out)
            except Exception as e:  # noqa: BLE001 - report and keep sweeping
                results.append((f"{topology.name} barrier cps={cps} wpl={wpl} nbuf={nbuf}", float("nan"), str(e)[:80]))

    # 4. production generic all_gather
    try:
        ms, gbs, out = _bench(
            mesh_device, lambda: ttnn.all_gather(k, dim=2, cluster_axis=SP_AXIS, num_links=num_links), out_bytes
        )
        results.append(("ttnn.all_gather generic", ms, gbs))
    except Exception as e:  # noqa: BLE001
        results.append(("ttnn.all_gather generic", float("nan"), str(e)[:80]))

    print(f"\nKV_GATHER dur={dur} S_local={s_local} per-device shard {s_local*H*D*2/1e6:.1f} MB, links={num_links}")
    for name, ms, gbs in results:
        print(f"KV_GATHER {name:42s} {ms:8.3f} ms  {gbs if isinstance(gbs, str) else f'{gbs:6.1f} GB/s recv/device'}")
