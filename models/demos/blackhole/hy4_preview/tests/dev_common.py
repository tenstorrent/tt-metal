# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers of the standalone device scripts (python -m ...tests.dev_*): mesh open/close, PCC, real-activation loader."""


import os

import ttnn

DEFAULT_WORKER_L1_BH = (
    1460864  # Blackhole default worker L1 above the 69 KB kernel-config ring (probe_l1.py: l1_bank_size)
)
ACTS = "/mnt/tt-data/ssinghal/hy4-logs/program/layer-moe/acts/"


def p(*a):
    print("LM", *a, flush=True)


def open_mesh(trace=200_000_000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    kw = {}
    # B10 (default OFF): HY4_RING_KB=N grows the per-core kernel-config ring buffer (69 KB default) by N KB at the cost of N KB of worker L1 (CBs / L1 tensors start N KB higher).
    # The trace replay stalls the dispatcher whenever the next program's config does not fit in the ring (go signal sent ~5-10 us after the previous program ended).
    ring_kb = int(os.environ.get("HY4_RING_KB", "0"))
    if ring_kb:
        kw["worker_l1_size"] = DEFAULT_WORKER_L1_BH - ring_kb * 1024
    return ttnn.open_mesh_device(ttnn.MeshShape(4, 8), l1_small_size=16384, trace_region_size=trace, **kw)


def close_mesh(md):
    ttnn.close_mesh_device(md)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))


def rep(md, t, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16):
    return ttnn.from_torch(
        t,
        device=md,
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(md),
    )


def dev0(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0])


def traced_us(md, fn, n=15, rep_=10):
    """median us per call: fn() x rep_ inside one trace, n replays timed one by one."""
    import statistics
    import time

    def body():
        for _ in range(rep_):
            o = fn()
            for t in o if isinstance(o, (tuple, list)) else [o]:
                if isinstance(t, ttnn.Tensor):
                    ttnn.deallocate(t)

    body()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    body()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(md)
        ts.append((time.perf_counter() - t0) / rep_ * 1e6)
    ttnn.release_trace(md, tid)
    return statistics.median(ts)
