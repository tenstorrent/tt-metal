# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Router cost study on ONE card (the router is replicated per device, so per-device time is the same on 4x8): real gate of layer MIMO_LAYER (5) and real dump
ffn_in (first 64 tokens). Stage-cumulative traced times (host-timed, 30 replays of 10 calls) of the exact router, then variants (MiMoGate.route_v) timed the same way.
Agreement of every variant vs fp32 CPU routing is measured in test_router_variants_agreement."""
import os
import time

import pytest
import torch

import ttnn
from models.demos.blackhole.mimo_v26.tests.test_moe_decode_4x8 import load_gate
from models.demos.blackhole.mimo_v26.tt.router import MiMoGate

DUMP = os.environ.get("MIMO_DUMP", "/mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16")
L = int(os.environ.get("MIMO_LAYER", "5"))
T = 64


def timed(md, fn, n=10, reps=30):
    fn()
    ttnn.synchronize_device(md)
    tid = ttnn.begin_trace_capture(md, cq_id=0)
    for _ in range(n):
        fn()
    ttnn.end_trace_capture(md, tid, cq_id=0)
    for _ in range(3):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    t0 = time.time()
    for _ in range(reps):
        ttnn.execute_trace(md, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(md)
    dt = (time.time() - t0) / reps / n * 1e3
    ttnn.release_trace(md, tid)
    return dt


@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 1), (1, 1))], indirect=["mesh_device"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384, "trace_region_size": 100000000}], indirect=True)
def test_router_cost(mesh_device, mesh_shape):
    md = mesh_device
    a = torch.load(f"{DUMP}/layer_{L}.pt", mmap=True)["prefill"]
    x = a["ffn_in"].reshape(-1, 4096)[:T]
    gw, gb = load_gate(L)
    gate = MiMoGate(md, gw.to(torch.bfloat16), gb, bias_shift=0.0, batch_per_device=None)
    xd = ttnn.from_torch(x.reshape(1, 1, T, 4096), device=md, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    prev = 0.0
    for st in range(1, 8):

        def fn(st=st):
            outs = gate.route_stage(xd, st)
            for o in outs if isinstance(outs, tuple) else (outs,):
                ttnn.deallocate(o)

        v = timed(md, fn)
        print(
            f"RESULT router stage {st} ({gate.STAGES[st]}): cumulative {v:.3f} ms (delta {v - prev:+.3f})", flush=True
        )
        prev = v
    for var in gate.VARIANTS:

        def fn(var=var):
            w, i = gate.route_v(xd, var)
            ttnn.deallocate(w)
            ttnn.deallocate(i)

        print(f"RESULT router variant {var}: {timed(md, fn):.3f} ms", flush=True)
