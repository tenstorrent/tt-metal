import importlib.util
from pathlib import Path
import torch
import ttnn

MOD_PATH = Path("ttnn/ttnn/operations/moe_fused_swiglu/perf_experiments/reduce_accum_mechanism/bench.py").resolve()
spec = importlib.util.spec_from_file_location("bench_probe", MOD_PATH)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

device = ttnn.open_device(device_id=0)
try:
    TILE = 32
    fan_in = 4
    block_tiles = 6  # small for quick eyeballing
    cfg = bench.create_sharded_memory_config(block_tiles)

    seed = torch.full((TILE, block_tiles * TILE), 100.0)
    children = [torch.full((TILE, block_tiles * TILE), float(10 * (i + 1))) for i in range(fan_in)]
    # expected = 100 + 10 + 20 + 30 + 40 = 200

    tt_seed = ttnn.from_torch(seed, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
    tt_children = [
        ttnn.from_torch(c, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
        for c in children
    ]

    out = bench.run_reduce_accum(
        tt_seed,
        tt_children,
        variant=bench.VARIANT_PACK_L1_ACC,
        fan_in=fan_in,
        block_tiles=block_tiles,
        dtype=ttnn.bfloat16,
    )
    torch_out = ttnn.to_torch(out).to(torch.float32)
    print("EXPECTED = 200.0")
    print("ACTUAL[0,0:8] =", torch_out[0, 0:8])
    print("ACTUAL unique values:", torch.unique(torch_out))
finally:
    ttnn.close_device(device)
