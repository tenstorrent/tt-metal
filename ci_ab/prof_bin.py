# Regression audit headroom round: device time of the ttnn ops that reach the generic binary loop and the bf16 multiply
# (binary_ng SFPU), run under `python -m tracy -r -p --no-web-server -o <dir> prof_bin.py <cases>`; a signpost precedes
# the R3_NMEAS measured calls of every case, after 2 warm-up calls.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def tens(d, shape, dt, seed, mem):
    g = torch.Generator().manual_seed(seed)
    if dt in (ttnn.int32, ttnn.uint32):
        x = torch.randint(-(2**30), 2**30, shape, generator=g, dtype=torch.int32)
        if dt == ttnn.uint32:
            x = x.abs()
    else:
        x = torch.randn(shape, generator=g) * 4
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)


def mem_shape(d, sharded, tiles_h):
    if not sharded:
        return ttnn.DRAM_MEMORY_CONFIG, (1, 1, 1024, 1024)
    grid = d.compute_with_storage_grid_size()
    n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, 32 * 8), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec), (1, 1, 32 * tiles_h * n, 32 * 8)


OPS = {"mul": ttnn.multiply, "add": ttnn.add, "sub": ttnn.subtract, "rsub": ttnn.rsub, "xlogy": ttnn.xlogy}


def case(op, dt, sharded, shape=None):
    th = 8 if dt == ttnn.bfloat16 else 4

    def setup(d):
        mem, shp = mem_shape(d, sharded, th)
        shape = shp if shape_arg is None else shape_arg
        return (tens(d, shape, dt, 70, mem), tens(d, shape, dt, 71, mem), mem)

    shape_arg = shape

    def run(d, a, b, mem):
        # accurate mode: the SFPU kernels (the default for bf16 add, sub and mul is the FPU)
        return OPS[op](a, b, memory_config=mem, fast_and_approximate_mode=False)

    return setup, run


CASES = {}
for op, dt, dn in (("add", ttnn.bfloat16, "bf16"), ("sub", ttnn.bfloat16, "bf16"), ("rsub", ttnn.bfloat16, "bf16"), ("add", ttnn.int32, "i32"), ("sub", ttnn.int32, "i32"), ("rsub", ttnn.int32, "i32"), ("mul", ttnn.bfloat16, "bf16"), ("add", ttnn.float32, "f32"), ("sub", ttnn.float32, "f32"), ("rsub", ttnn.float32, "f32"), ("xlogy", ttnn.bfloat16, "bf16"), ("mul", ttnn.float32, "f32")):
    CASES[f"{op}_{dn}_hs"] = case(op, dt, True)
    CASES[f"{op}_{dn}_dram_1024x1024"] = case(op, dt, False)
EUP = ttnn._ttnn.activation.EltwiseUnaryWithParam


def ucmp_case(op):
    def setup(d):
        mem, shp = mem_shape(d, True, 8)
        return (tens(d, shp, ttnn.int32, 72, mem), mem)

    def run(d, a, mem):
        return ttnn.unary_chain(a, [EUP(getattr(ttnn.UnaryOpType, op), 5)], memory_config=mem)

    return setup, run


# the residual adds of models/demos/llama_3p1_8b_d_p/tt/decoder.py (accurate mode), decode and prefill rows
CASES["add_bf16_llama_res_32x4096"] = case("add", ttnn.bfloat16, False, (1, 1, 32, 4096))
CASES["add_bf16_llama_res_2048x4096"] = case("add", ttnn.bfloat16, False, (1, 1, 2048, 4096))
CASES["ucmp_lt_hs"] = ucmp_case("UNARY_LT")
CASES["ucmp_gt_hs"] = ucmp_case("UNARY_GT")
# the sampling penalties' int32 count update (models/common/sampling/tt_penalties.py), batch 32 by the Llama-3 vocabulary
CASES["add_i32_penalties_32x128256"] = case("add", ttnn.int32, False, (1, 1, 32, 128256))


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
