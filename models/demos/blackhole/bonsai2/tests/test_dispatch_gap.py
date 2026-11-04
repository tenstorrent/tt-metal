# Traced wall time per call of single ops, to compare with their device kernel time: the difference is the per-op
# launch cost inside a trace (traced decode = 42.1 ms while its kernels sum to ~32.3 ms).
import time

import numpy as np
import pytest
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.ternary.decode_linear3 import TernaryDecodeLinear3
from models.demos.blackhole.bonsai2.tt.ternary.rotate import DecodeRotation

N_CALLS = 200


def _timed_trace(device, fn):
    fn()  # compile
    ttnn.synchronize_device(device)
    tid = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(N_CALLS):
        fn()
    ttnn.end_trace_capture(device, tid, cq_id=0)
    ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
    t = time.perf_counter()
    for _ in range(5):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    us = (time.perf_counter() - t) / (5 * N_CALLS) * 1e6
    ttnn.release_trace(device, tid)
    return us


OPS = [
    "ternary_6144x5120",
    "ternary_1024x4160",
    "ternary_6144x5120_c40",
    "ternary_6144x5120_eager",
    "ternary_17408x5120",
]


@pytest.mark.parametrize("device_params", [{"trace_region_size": 200_000_000}], indirect=True)
@pytest.mark.parametrize("op", OPS)
def test_dispatch_gap(device, op):
    rng = np.random.default_rng(0)
    if op.startswith("ternary"):
        K, N = (int(v) for v in op.split("_")[1].split("x"))
        cores = 40 if op.endswith("_c40") else None
        t = rng.integers(-1, 2, size=(K, N)).astype(np.int8)
        s = (rng.random((N, K // 128)) * 0.02 + 0.005).astype(np.float16)
        lin = TernaryDecodeLinear3.from_host(device, t, s, inline_scales=True, cores=cores)
        act = ttnn.from_torch(
            torch.randn(8, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        fn = lambda: lin(act)
    elif op == "rot5120":
        rot = DecodeRotation(device, np.sign(rng.standard_normal(5120)).astype(np.float32))
        x = ttnn.from_torch(
            torch.randn(1, 1, 5120).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        fn = lambda: rot(x)
    else:
        a = ttnn.from_torch(
            torch.randn(1, 1, 5120).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        fn = lambda: ttnn.add(a, a)
    if op.endswith("_eager"):
        fn()
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        for _ in range(N_CALLS):
            fn()
        ttnn.synchronize_device(device)
        us = (time.perf_counter() - t0) / N_CALLS * 1e6
    else:
        us = _timed_trace(device, fn)
    print(f"GAP op={op} traced_us_per_call={us:.2f}")
