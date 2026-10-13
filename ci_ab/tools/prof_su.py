# Device time of the ttnn ops reached by the stacked activations change (gelu, silu, sigmoid, and silu fused into a
# binary_ng multiply and gelu fused into a matmul), run under `python -m tracy -r -p --no-web-server -o <dir> prof_su.py
# <cases>`; a signpost precedes the R3_NMEAS measured calls of every case, after 2 warm-up calls.
import gc
import os
import sys

import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))


def tens(d, shape, seed, mem, dt=ttnn.bfloat16):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(shape, generator=g) * 3
    return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)


def hs(d, tiles_h, width):
    grid = d.compute_with_storage_grid_size()
    n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec), (1, 1, 32 * tiles_h * n, width)


DRAM = ttnn.DRAM_MEMORY_CONFIG
UNARY = {"gelu": lambda t, m: ttnn.gelu(t, memory_config=m), "silu": lambda t, m: ttnn.silu(t, memory_config=m),
         "sigmoid": lambda t, m: ttnn.sigmoid(t, memory_config=m)}


def unary_case(op, shape=None, sharded=False):
    def setup(d):
        mem, shp = hs(d, 8, 256) if sharded else (DRAM, shape)
        return (tens(d, shp, 70, mem), mem)

    def run(d, a, mem):
        return UNARY[op](a, mem)

    return setup, run


def silu_mul_case(shape):
    def setup(d):
        return (tens(d, shape, 70, DRAM), tens(d, shape, 71, DRAM))

    def run(d, a, b):
        return ttnn.multiply(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM)

    return setup, run


def gelu_linear_case(m, k, n):
    def setup(d):
        return (tens(d, (1, 1, m, k), 70, DRAM), tens(d, (1, 1, k, n), 71, DRAM))

    def run(d, a, w):
        return ttnn.linear(a, w, activation="gelu", memory_config=DRAM)

    return setup, run


CASES = {
    "gelu_dram_1024x4096": unary_case("gelu", (1, 1, 1024, 4096)),
    "gelu_hs": unary_case("gelu", sharded=True),
    "silu_dram_32x14336": unary_case("silu", (1, 1, 32, 14336)),
    "silu_hs": unary_case("silu", sharded=True),
    "sigmoid_dram_1024x1024": unary_case("sigmoid", (1, 1, 1024, 1024)),
    "sigmoid_dram_1024x4096": unary_case("sigmoid", (1, 1, 1024, 4096)),
    "sigmoid_dram_256x1024": unary_case("sigmoid", (1, 1, 256, 1024)),
    "sigmoid_hs": unary_case("sigmoid", sharded=True),
    "silu_dram_1024x1024": unary_case("silu", (1, 1, 1024, 1024)),
    "gelu_dram_1024x1024": unary_case("gelu", (1, 1, 1024, 1024)),
    "exp_dram_1024x1024": (lambda d: (tens(d, (1, 1, 1024, 1024), 70, DRAM), DRAM), lambda d, a, m: ttnn.exp(a, memory_config=m)),
    "silu_mul_32x14336": silu_mul_case((1, 1, 32, 14336)),
    "silu_mul_128x14336": silu_mul_case((1, 1, 128, 14336)),
    "gelu_linear_1024x1024x4096": gelu_linear_case(1024, 1024, 4096),
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
                print(f"CASE {name} FAILED {type(e).__name__}: {e}", flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
