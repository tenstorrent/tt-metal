# Device time of the rand_tile users under `python -m tracy -r -p --no-web-server -o <dir> prof_rand.py <cases>`, every
# case in one process; a signpost names each case before its R3_NMEAS measured calls (2 warm-up calls first).
import gc, os, sys
import torch
import ttnn

try:
    from tracy import signpost
except ImportError:
    signpost = None

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))
BIG = (1, 1, 4096, 2048)


def _sampling_setup(d):
    shape = (1, 1, 32, 32 * 8)
    g = torch.Generator().manual_seed(7)
    v = ttnn.from_torch(torch.randn(shape, generator=g), device=d, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    i = ttnn.from_torch(torch.arange(0, shape[-1], dtype=torch.int32).expand(shape), device=d, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    k = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=d, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    p = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    t = ttnn.from_torch(torch.ones(32), device=d, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    return (v, i, k, p, t)


def _tt(x, d, dt):
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d)


CASES = {
    "rand_bf16_4096x2048": (lambda d: (), lambda d: ttnn.rand(BIG, dtype=ttnn.bfloat16, device=d, seed=1234)),
    "rand_fp32_4096x2048": (lambda d: (), lambda d: ttnn.rand(BIG, dtype=ttnn.float32, device=d, seed=1234)),
    "rand_bf16_1024x1024": (lambda d: (), lambda d: ttnn.rand((1, 1, 1024, 1024), dtype=ttnn.bfloat16, device=d, seed=1234)),
    "rand_bf16_32x32": (lambda d: (), lambda d: ttnn.rand((1, 1, 32, 32), dtype=ttnn.bfloat16, device=d, seed=1234)),
    "rand_bf16_8192x8192": (lambda d: (), lambda d: ttnn.rand((1, 1, 8192, 8192), dtype=ttnn.bfloat16, device=d, seed=1234)),
    "uniform_fp32_4096x2048": (lambda d: (_tt(torch.zeros(BIG), d, ttnn.float32),), lambda d, x: ttnn.uniform(x, 0.0, 1.0, 1234)),
    "uniform_bf16_4096x2048": (lambda d: (_tt(torch.zeros(BIG), d, ttnn.bfloat16),), lambda d, x: ttnn.uniform(x, 0.0, 1.0, 1234)),
    "bernoulli_fp32_4096x2048": (
        lambda d: (_tt(torch.rand(BIG, generator=torch.Generator().manual_seed(0)), d, ttnn.float32),),
        lambda d, p: ttnn.bernoulli(p, 1234, dtype=ttnn.float32)),
    "sampling_32x256": (_sampling_setup, lambda d, v, i, k, p, t: ttnn.sampling(v, i, k=k, p=p, temp=t, seed=2024)),
}


def bits(dev, names, out):
    import hashlib, json
    res = {}
    for name in names:
        setup, run = CASES[name]
        args = setup(dev)
        y = run(dev, *args)
        y = y[0] if isinstance(y, (list, tuple)) else y
        t = ttnn.to_torch(y)
        res[name] = hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]
    json.dump(res, open(out, "w"), indent=0)
    print("saved", len(res))


def main():
    names = sys.argv[1:] or list(CASES)
    if os.environ.get("R_BITS"):
        dev = ttnn.open_device(device_id=0)
        try:
            bits(dev, names, os.environ["R_BITS"])
        finally:
            ttnn.close_device(dev)
        return
    dev = ttnn.open_device(device_id=0)
    try:
        for name in names:
            setup, run = CASES[name]
            gc.collect()
            try:
                args = setup(dev)
                for _ in range(N_WARM):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                signpost(header=name)
                for _ in range(N_MEAS):
                    run(dev, *args)
                ttnn.synchronize_device(dev)
                print(f"CASE {name} ok", flush=True)
                del args
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
        ttnn.ReadDeviceProfiler(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
