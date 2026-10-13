# Device time of the unary ops the stacked activations change, at DRAM interleaved and height-sharded shapes and the model
# shapes (Llama MLP silu, SDXL silu, MoE gate sigmoid), every case after a signpost, R3_NMEAS measured calls after 2
# warm-up calls; run under `python -m tracy -r -p --no-web-server -o <dir> prof_all.py [cases]`.
import gc, os, sys
import torch
import ttnn
from tracy import signpost

N_WARM, N_MEAS = 2, int(os.environ.get("R3_NMEAS", "6"))
DRAM = ttnn.DRAM_MEMORY_CONFIG


def tens(d, shape, seed, mem):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(shape, generator=g) * 3).to(torch.bfloat16)
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d, memory_config=mem)


def hs(d, tiles_h, width):
    grid = d.compute_with_storage_grid_size()
    n = grid.x * grid.y
    crs = ttnn.num_cores_to_corerangeset(n, ttnn.CoreCoord(grid.x, grid.y), row_wise=True)
    spec = ttnn.ShardSpec(crs, (32 * tiles_h, width), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec), (1, 1, 32 * tiles_h * n, width)


OPS = {
    "sigmoid": lambda t, m: ttnn.sigmoid(t, memory_config=m),
    "silu": lambda t, m: ttnn.silu(t, memory_config=m),
    "gelu": lambda t, m: ttnn.gelu(t, memory_config=m),
    "exp": lambda t, m: ttnn.exp(t, memory_config=m),
    "expm1": lambda t, m: ttnn.expm1(t, memory_config=m),
    "elu": lambda t, m: ttnn.elu(t, alpha=1.0, memory_config=m),
    "celu": lambda t, m: ttnn.celu(t, alpha=1.5, memory_config=m),
    "selu": lambda t, m: ttnn.selu(t, memory_config=m),
    "mish": lambda t, m: ttnn.mish(t, memory_config=m),
    "softplus": lambda t, m: ttnn.softplus(t, beta=1.0, threshold=20.0, memory_config=m),
    "tanhshrink": lambda t, m: ttnn.tanhshrink(t, memory_config=m),
    "logsigmoid": lambda t, m: ttnn.log_sigmoid(t, memory_config=m),
    "relu": lambda t, m: ttnn.relu(t, memory_config=m),
}


def unary(op, shape=None, sharded=False):
    def setup(d):
        mem, shp = hs(d, 8, 256) if sharded else (DRAM, shape)
        return (tens(d, shp, 70, mem), mem)

    return setup, lambda d, a, m: OPS[op](a, m)


def silu_mul(shape):
    return (lambda d: (tens(d, shape, 70, DRAM), tens(d, shape, 71, DRAM)),
            lambda d, a, b: ttnn.multiply(a, b, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], memory_config=DRAM))


CASES = {}
for op in OPS:
    CASES[f"{op}_dram_1024x1024"] = unary(op, (1, 1, 1024, 1024))
    CASES[f"{op}_hs"] = unary(op, sharded=True)
for name, op, shape in (("silu_dram_32x14336", "silu", (1, 1, 32, 14336)), ("silu_dram_128x14336", "silu", (1, 1, 128, 14336)),
                        ("silu_dram_4096x320", "silu", (1, 1, 4096, 320)), ("silu_dram_1024x1280", "silu", (1, 1, 1024, 1280)),
                        ("sigmoid_dram_32x256", "sigmoid", (1, 1, 32, 256)), ("sigmoid_dram_1024x256", "sigmoid", (1, 1, 1024, 256)),
                        ("sigmoid_dram_256x1024", "sigmoid", (1, 1, 256, 1024)), ("sigmoid_dram_1024x4096", "sigmoid", (1, 1, 1024, 4096)),
                        ("sigmoid_dram_2048x2048", "sigmoid", (1, 1, 2048, 2048)), ("silu_dram_2048x2048", "silu", (1, 1, 2048, 2048))):
    CASES[name] = unary(op, shape)
CASES["silu_mul_32x14336"] = silu_mul((1, 1, 32, 14336))
CASES["silu_mul_128x14336"] = silu_mul((1, 1, 128, 14336))


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
