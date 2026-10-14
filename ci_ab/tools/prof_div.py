# Device time and output words of ttnn.divide on Int32 tensors (div_int32_tile), every case in one process under
# `python -m tracy -r -p --no-web-server -o <dir> prof_div.py <cases>`; R_BITS=<json>: one call per case, words hashed.
import gc, hashlib, json, os, sys
import torch
import ttnn

try:
    from tracy import signpost
except ImportError:
    signpost = None

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "10"))
IMIN, IMAX = -(2**31), 2**31 - 1


def ints(shape, seed, lo=IMIN, hi=IMAX, edges=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)
    if edges:
        e = torch.tensor([IMIN, IMIN + 1, -(2**24) - 1, -(2**24), -7, -3, -2, -1, 0, 1, 2, 3, 7, 2**24, 2**24 + 1, IMAX - 1, IMAX])
        n = x.numel()
        x.view(-1)[: 2 * len(e) ** 2] = torch.cat([e.repeat_interleave(len(e)), e.repeat(len(e))])[: 2 * len(e) ** 2] if n >= 2 * len(e) ** 2 else x.view(-1)[:0]
    return x.to(torch.int32)


def dram(x, d):
    return ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=d)


def hs(x, d, tiles_h=8):
    grid = d.compute_with_storage_grid_size(); n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
                            ttnn.ShardSpec(crs, (32 * tiles_h, x.shape[-1]), ttnn.ShardOrientation.ROW_MAJOR))
    return ttnn.from_torch(x, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem), mem


def hs_shape(d, tiles_h=8, w=256):
    g = d.compute_with_storage_grid_size()
    return (1, 1, 32 * tiles_h * g.x * g.y, w)


CASES = {
    "div_dram_full": (lambda d: (dram(ints((1, 1, 1024, 1024), 1), d), dram(ints((1, 1, 1024, 1024), 2), d)),
                      lambda d, a, b: ttnn.divide(a, b)),
    "div_dram_small": (lambda d: (dram(ints((1, 1, 1024, 1024), 3, -1000, 1000, False), d), dram(ints((1, 1, 1024, 1024), 4, -50, 50, False), d)),
                       lambda d, a, b: ttnn.divide(a, b)),
    "div_hs": (lambda d: (*hs(ints(hs_shape(d), 5), d), hs(ints(hs_shape(d), 6), d)[0]),
               lambda d, a, m, b: ttnn.divide(a, b, memory_config=m)),
}
for k in range(6):
    CASES[f"bits_{k}"] = (lambda d, k=k: (dram(ints((1, 1, 1024, 1024), 100 + k, *( (IMIN, IMAX) if k % 2 == 0 else (-(2**k * 1000), 2**k * 1000))), d),
                                          dram(ints((1, 1, 1024, 1024), 200 + k, *((IMIN, IMAX) if k < 3 else (-(10**k), 10**k))), d)),
                          lambda d, a, b: ttnn.divide(a, b))


def main():
    names = sys.argv[1:] or list(CASES)
    dev = ttnn.open_device(device_id=0)
    try:
        if os.environ.get("R_BITS"):
            res = {}
            for name in CASES:
                setup, run = CASES[name]
                y = ttnn.to_torch(run(dev, *setup(dev)))
                res[name] = hashlib.sha256(y.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:20]
            json.dump(res, open(os.environ["R_BITS"], "w"), indent=0)
            print("saved", len(res))
            return
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
