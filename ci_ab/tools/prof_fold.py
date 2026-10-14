# Device time of the ttnn ops whose kernels run copy_tile before an SFPU op (scratch, not for merge), under
# `python -m tracy -r -p --no-web-server -o <dir> prof_fold.py <case>`: 2 warm-up calls, a signpost, R3_NMEAS measured calls.
# hs = height-sharded in L1 on every core (8 tiles x 8 per core, 4 x 8 for 32-bit), dram = DRAM interleaved.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "12"))
DT = {"bf16": ttnn.bfloat16, "bf8": ttnn.bfloat8_b, "fp32": ttnn.float32, "i32": ttnn.int32, "u32": ttnn.uint32, "u16": ttnn.uint16}
DRAM = ttnn.DRAM_MEMORY_CONFIG


def hs(d, tiles_h, width):
    grid = d.compute_with_storage_grid_size()
    n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec), (1, 1, 32 * tiles_h * n, width)


def tens(d, shape, mem, dt="bf16", seed=70, scale=3.0, positive=False):
    g = torch.Generator().manual_seed(seed)
    if dt in ("i32", "u32", "u16"):
        hi = {"i32": 2**31 - 1, "u32": 2**31 - 1, "u16": 65535}[dt]
        x = torch.randint(0, hi, shape, generator=g, dtype=torch.int64).to(torch.int32)
    else:
        x = torch.randn(shape, generator=g) * scale
        if positive:
            x = x.abs() + 0.5
    return ttnn.from_torch(x, dtype=DT[dt], layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)


def typecast_case(src, dst, sharded=True, shape=None):
    def setup(d):
        if sharded:
            th = 4 if src in ("fp32", "i32", "u32") or dst in ("fp32", "i32", "u32") else 8
            mem, shp = hs(d, th, 256)
        else:
            mem, shp = DRAM, shape
        return (tens(d, shp, mem, src, scale=1000.0), mem)

    return setup, lambda d, x, mem: ttnn.typecast(x, DT[dst], memory_config=mem)


UNARY = {
    "exp": lambda t, m: ttnn.exp(t, memory_config=m),
    "recip": lambda t, m: ttnn.reciprocal(t, memory_config=m),
    "gelu": lambda t, m: ttnn.gelu(t, memory_config=m),
}


def unary_case(op, shape=None, sharded=False):
    def setup(d):
        mem, shp = hs(d, 8, 256) if sharded else (DRAM, shape)
        return (tens(d, shp, mem, positive=(op == "recip")), mem)

    return setup, lambda d, x, mem: UNARY[op](x, mem)


def softmax_case(shape, dt):
    def setup(d):
        return (tens(d, shape, DRAM, dt),)

    return setup, lambda d, x: ttnn.softmax(x, dim=-1, numeric_stable=False)


def quant_case(shape, dt):
    def setup(d):
        return (tens(d, shape, DRAM, dt),)

    return setup, lambda d, x: ttnn.quantize(x, 0.05, 3)


CASES = {
    "tc_bf16_u16_hs": typecast_case("bf16", "u16"),
    "tc_bf16_i32_hs": typecast_case("bf16", "i32"),
    "tc_bf16_fp32_hs": typecast_case("bf16", "fp32"),
    "tc_bf16_u32_hs": typecast_case("bf16", "u32"),
    "tc_i32_bf16_hs": typecast_case("i32", "bf16"),
    "tc_bf16_fp32_dram_32x128256": typecast_case("bf16", "fp32", sharded=False, shape=(1, 1, 32, 128256)),
    "exp_dram_1024x1024": unary_case("exp", (1, 1, 1024, 1024)),
    "exp_hs": unary_case("exp", sharded=True),
    "recip_dram_1024x1024": unary_case("recip", (1, 1, 1024, 1024)),
    "recip_hs": unary_case("recip", sharded=True),
    "gelu_dram_1024x4096": unary_case("gelu", (1, 1, 1024, 4096)),
    "gelu_hs": unary_case("gelu", sharded=True),
    "softmax_bf8_1x1x16384x256": softmax_case((1, 1, 16384, 256), "bf8"),
    "softmax_bf8_1x8x256x256": softmax_case((1, 8, 256, 256), "bf8"),
    "quant_bf16_1024x1024": quant_case((1, 1, 1024, 1024), "bf16"),
    "quant_fp32_1024x1024": quant_case((1, 1, 1024, 1024), "fp32"),
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
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
