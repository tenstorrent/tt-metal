# Op-level device time of reduces over W at production shapes, then logical output bits of each reduce and of its
# consumer, with the reduce's output buffer reusing freed NaN-filled buffers. One side per process.
# usage: python -m tracy -r -p --no-web-server -o OUT prof_w.py <side> <run>
import hashlib
import sys

import torch
import ttnn
from tracy import signpost

REPS = 6
# name, input shape, op, dims, keepdim, consumer
CASES = [
    ("q36_router_topk_sum_s4096", (1, 1, 4096, 8), "sum", -1, True, "div"),
    ("q36_router_topk_sum_s1024", (1, 1, 1024, 8), "sum", -1, True, "div"),
    ("q36_router_topk_sum_b32", (1, 1, 32, 8), "sum", -1, True, "div"),
    ("q36_greedy_maxval_1952x32", (1, 1, 1952, 32), "max", -1, False, "maxval"),
    ("sampling_maxv_32x256", (1, 1, 32, 256), "max", 3, True, "eq"),
    ("logprobs_max_32x16032", (1, 1, 32, 16032), "max", -1, True, "subexp"),
    ("logprobs_sum_32x16032", (1, 1, 32, 16032), "sum", -1, True, "div"),
    ("rms_mean_4096x1152", (1, 1, 4096, 1152), "mean", -1, True, "rms"),
    ("rms_mean_1024x5120", (1, 1, 1024, 5120), "mean", -1, True, "rms"),
    ("rms_mean_4x1024x1024", (4, 1, 1024, 1024), "mean", -1, True, "rms"),
    ("hw_sum_1024x1024", (1, 1, 1024, 1024), "sum", [-2, -1], True, "sub"),
]
FN = {"max": ttnn.max, "sum": ttnn.sum, "mean": ttnn.mean}


def consume(kind, x, y):
    if kind == "div":
        return ttnn.div(x, y)
    if kind == "maxval":
        return ttnn.max(ttnn.reshape(y, (1, 1, 1, y.shape[-1])), dim=-1, keepdim=True)
    if kind == "eq":
        return ttnn.add(ttnn.eq(x, y), ttnn.lt(x, y))
    if kind == "subexp":
        return ttnn.sum(ttnn.exp(ttnn.subtract(x, y)), dim=-1, keepdim=True)
    if kind == "rms":
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(y, 1e-6)))
    if kind == "sub":
        return ttnn.subtract(x, y)


def dump(side, run, name, t):
    o = ttnn.to_torch(t).contiguous().float()
    torch.save(o, f"/tmp/hr_out/{side}_{run}_{name}.pt")
    return hashlib.sha256(o.numpy().tobytes()).hexdigest()[:16], bool(torch.isnan(o).any())


def main():
    side, run = sys.argv[1], sys.argv[2]
    device = ttnn.open_device(device_id=0)
    for name, shape, op, dim, keep, kind in CASES:
        torch.manual_seed(58709)
        x = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        y = None
        for k in range(REPS):
            ttnn.synchronize_device(device)
            signpost(header=f"{name}#{k}")
            y = FN[op](x, dim=dim, keepdim=keep)
        ttnn.synchronize_device(device)
        ttnn.ReadDeviceProfiler(device)
        y.deallocate()
        # Poison: free NaN buffers of the reduce output's padded size so its allocation reuses one.
        oshape = list(shape)
        oshape[-1] = 32
        poison = [ttnn.full(oshape, float("nan"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(4)]
        if dim == [-2, -1]:
            # the W step's fp32 intermediate, then the final 32x32 output
            poison += [ttnn.full(oshape, float("nan"), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(2)]
            poison += [ttnn.full([1, 1, 32, 32], float("nan"), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device) for _ in range(4)]
        addrs = {p.buffer_address() for p in poison}
        for p in poison:
            p.deallocate()
        y = FN[op](x, dim=dim, keepdim=keep)
        reused = y.buffer_address() in addrs
        c = consume(kind, x, y)
        hy, ny = dump(side, run, name, y)
        hc, nc = dump(side, run, name + "_consumer", c)
        print(f"CASE {name} ok BITS {side} r{run} out {hy} nan {ny} consumer {hc} nan {nc} poisoned_buffer {reused}", flush=True)
    ttnn.close_device(device)


if __name__ == "__main__":
    main()
