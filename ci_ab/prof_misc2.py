# Issue resolution round (#58774, #58772): device time of softshrink and softsign (their Blackhole row loops) and of every
# ttnn op that seeds the PRNG (init_prng_seed), run under
# `python -m tracy -r -p --no-web-server -o <dir> prof_misc2.py <cases>`; a signpost precedes the R3_NMEAS measured calls
# of every case, after 2 warm-up calls.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

if not os.environ.get("HWLOCK_HELD") and not os.path.isfile(os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH", "")):
    sys.exit("not under hwlock")  # card rule of 2026-10-07 12:35 UTC; the mock cluster opens no card

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def dram(t, dev, dt=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dt, layout=layout, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def hs_mem(dev, tiles_h=8, tiles_w=8):
    grid = dev.compute_with_storage_grid_size()
    ncores = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(ncores, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, 32 * tiles_w), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec), ncores


def case_unary(fn, sharded, shape=(1, 1, 1024, 1024)):
    def setup(dev):
        torch.manual_seed(0)
        if sharded:
            mem, ncores = hs_mem(dev)
            shp = (1, 1, 32 * 8 * ncores, 32 * 8)
            x = ttnn.from_torch(torch.randn(shp, dtype=torch.bfloat16) * 2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mem)
            return (x, mem)
        return (dram(torch.randn(shape, dtype=torch.bfloat16) * 2, dev), ttnn.DRAM_MEMORY_CONFIG)

    def run(dev, x, mem):
        if fn == "softshrink":
            return ttnn.softshrink(x, lambd=0.5, memory_config=mem)
        if fn == "logical_not":
            return ttnn.logical_not(x, memory_config=mem)
        return ttnn.softsign(x, memory_config=mem)

    return setup, run


def case_rand(shape):
    def setup(dev):
        return ()

    def run(dev):
        return ttnn.rand(shape, dtype=ttnn.bfloat16, device=dev, seed=1234)

    return setup, run


def case_uniform(shape):
    def setup(dev):
        return (dram(torch.zeros(shape, dtype=torch.float32), dev, ttnn.float32),)

    def run(dev, x):
        return ttnn.uniform(x, 0.0, 1.0, 1234)

    return setup, run


def case_bernoulli(shape):
    def setup(dev):
        torch.manual_seed(0)
        return (dram(torch.rand(shape, dtype=torch.float32), dev, ttnn.float32),)

    def run(dev, p):
        return ttnn.bernoulli(p, 1234, dtype=ttnn.float32)

    return setup, run


def case_dropout(shape):
    def setup(dev):
        torch.manual_seed(0)
        return (dram(torch.randn(shape, dtype=torch.bfloat16), dev),)

    def run(dev, x):
        return ttnn.experimental.dropout(x, probability=0.5, scale=2.0, seed=1234)

    return setup, run


def case_sampling(users=32, width=256):
    def setup(dev):
        torch.manual_seed(0)
        shape = [1, 1, users, width]
        vals = ttnn.from_torch(torch.randn(shape), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        idx = ttnn.from_torch(torch.arange(0, width, dtype=torch.int32).expand(shape), device=dev, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        k = ttnn.from_torch(torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=dev, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.from_torch(torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp = ttnn.from_torch(torch.ones(32), device=dev, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
        return (vals, idx, k, p, temp)

    def run(dev, vals, idx, k, p, temp):
        return ttnn.sampling(vals, idx, k=k, p=p, temp=temp, seed=2024)

    return setup, run


def case_manual_seed():
    def setup(dev):
        return ()

    def run(dev):
        return ttnn.manual_seed(seeds=42, device=dev)

    return setup, run


CASES = {
    "softshrink_dram_1024x1024": case_unary("softshrink", False),
    "softshrink_hs_8x8": case_unary("softshrink", True),
    "softsign_dram_1024x1024": case_unary("softsign", False),
    "softsign_hs_8x8": case_unary("softsign", True),
    "logical_not_dram_1024x1024": case_unary("logical_not", False),
    "rand_32x32": case_rand((1, 1, 32, 32)),
    "rand_1024x1024": case_rand((1, 1, 1024, 1024)),
    "uniform_32x32": case_uniform((1, 1, 32, 32)),
    "uniform_1024x1024": case_uniform((1, 1, 1024, 1024)),
    "bernoulli_32x32": case_bernoulli((1, 1, 32, 32)),
    "bernoulli_1024x1024": case_bernoulli((1, 1, 1024, 1024)),
    "dropout_32x32": case_dropout((1, 1, 32, 32)),
    "dropout_1024x1024": case_dropout((1, 1, 1024, 1024)),
    "sampling_32x256": case_sampling(),
    "manual_seed": case_manual_seed(),
}


def main():
    names = sys.argv[1:] or list(CASES)
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
                ttnn.ReadDeviceProfiler(dev)
                print(f"CASE {name} ok", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"CASE {name} FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                ttnn.synchronize_device(dev)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
