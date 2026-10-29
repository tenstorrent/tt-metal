"""One program (ttnn.var over H and W) vs the composite (mean, subtract + square, mean) on many-plane shapes.

MODE=kernel: run under `python -m tracy -r -p -v`; sum_signposts.py totals DEVICE KERNEL DURATION per tag.
MODE=trace: prints `TRACE <tag> <us per iteration>` from replaying a captured trace REPLAYS times.
"""
import os
import time

import torch
import ttnn

MODE = os.environ.get("MODE", "kernel")
REPEATS, REPLAYS = 3, 20
SQUARE = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
PLANES = [1, 2, 4, 6, 8, 10, 11, 12, 16, 24, 32, 64, 65, 128]
PLANE_SHAPES = [(256, 256), (256, 512), (512, 512), (32, 2048), (1024, 1024)]


def one_program(x):
    return ttnn.var(x, dim=[-2, -1], keepdim=True, correction=False)


def composite(x):
    m = ttnn.mean(x, dim=[-2, -1], keepdim=True)
    return ttnn.mean(ttnn.subtract(x, m, activations=SQUARE), dim=[-2, -1], keepdim=True)


PATHS = {"one": one_program, "comp": composite}


def cases():
    for dtype in ("bf16", "fp32"):
        for h, w in PLANE_SHAPES:
            for p in PLANES:
                if h * w * p > 32 * 1024 * 1024 or (dtype == "fp32" and (h, w) not in ((256, 256), (512, 512))):
                    continue
                yield dtype, (p, 1, h, w)


device = ttnn.open_device(device_id=0, trace_region_size=64 * 1024 * 1024)
if MODE == "kernel":
    from tracy import signpost
for dtype, shape in cases():
    tdtype = ttnn.bfloat16 if dtype == "bf16" else ttnn.float32
    x = ttnn.from_torch(torch.randn(shape) * 3 + 1, dtype=tdtype, layout=ttnn.TILE_LAYOUT, device=device)
    for name, op in PATHS.items():
        tag = f"{name}|{dtype}|{'x'.join(map(str, shape))}"
        try:
            op(x).deallocate()
            ttnn.synchronize_device(device)
        except Exception as e:
            print(f"SKIP {tag}: {str(e).splitlines()[0][:150]}", flush=True)
            continue
        if MODE == "kernel":
            for _ in range(REPEATS):
                signpost(f"BEGIN {tag}")
                op(x).deallocate()
                ttnn.synchronize_device(device)
                signpost(f"END {tag}")
        else:
            tid = ttnn.begin_trace_capture(device, cq_id=0)
            out = op(x)
            ttnn.end_trace_capture(device, tid, cq_id=0)
            ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            start = time.perf_counter()
            for _ in range(REPLAYS):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            per_iter = (time.perf_counter() - start) / REPLAYS * 1e6
            ttnn.release_trace(device, tid)
            out.deallocate()
            print(f"TRACE {tag} {per_iter:.1f}", flush=True)
    x.deallocate()
    if MODE == "kernel":
        ttnn.ReadDeviceProfiler(device)
ttnn.close_device(device)
