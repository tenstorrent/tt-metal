# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Real per-op cost inside a long trace: N chained copies of one op in ONE trace; (t(N)-t(1))/(N-1).
The single-op traces of test_mhc_microbench.py include the trace launch latency, so they overstate small ops."""

import time

import pytest
import torch

import ttnn

D, T = 5120, 4


def replay_ms(md, fn, n=30):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    ttnn.synchronize_device(md)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.perf_counter() - t) / n * 1e3
    ttnn.release_trace(md, tid)
    return dt


@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
@pytest.mark.parametrize(
    "device_params",
    [
        pytest.param(
            {
                "l1_small_size": 16384,
                "fabric_config": ttnn.FabricConfig.FABRIC_1D_RING,
                "trace_region_size": 200_000_000,
            },
            id="ring",
        )
    ],
    indirect=True,
)
@torch.no_grad()
def test_trace_op_cost(mesh_device):
    md = mesh_device
    rep = ttnn.ReplicateTensorToMesh(md)
    up = lambda t, dt=ttnn.float32: ttnn.from_torch(
        t, device=md, dtype=dt, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep
    )
    a, b = up(torch.randn(T, 1, 4, D)), up(torch.randn(T, 1, 4, D))
    small = up(torch.randn(1, 1, 32, 32))
    bf = up(torch.randn(1, 1, T, D), ttnn.bfloat16)
    w = up(torch.randn(1, 1, D, 1280) * 0.02, ttnn.bfloat8_b)
    ops = {
        "add fp32 [T,4,D]": lambda: ttnn.add(a, b),
        "add tiny [32,32]": lambda: ttnn.add(small, small),
        "matmul bf16 [T,D]x[D,1280] bfp8": lambda: ttnn.matmul(bf, w),
        "typecast bf16->fp32": lambda: ttnn.typecast(bf, ttnn.float32),
        "sum last dim fp32": lambda: ttnn.sum(a, dim=-1, keepdim=True),
    }
    for name, op in ops.items():

        def many(n):
            def f():
                for _ in range(n):
                    op()

            return f

        t1, t41 = replay_ms(md, many(1)), replay_ms(md, many(41))
        print(
            f"OPCOST {name:36s} single-op trace {t1:.3f} ms | per op in long trace {(t41 - t1) / 40 * 1e3:6.1f} us",
            flush=True,
        )
