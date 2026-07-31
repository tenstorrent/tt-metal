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
    block_tiles = 48
    cfg = bench.create_sharded_memory_config(block_tiles)

    torch.manual_seed(0)
    seed = torch.randn(TILE, block_tiles * TILE) * 0.1
    children = [torch.randn(TILE, block_tiles * TILE) * 0.1 for _ in range(fan_in)]
    expected = seed.clone()
    for c in children:
        expected = expected + c

    for dtype, name in ((ttnn.bfloat16, "bf16"), (ttnn.bfloat8_b, "bfp8_b")):
        tt_seed = ttnn.from_torch(seed, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg)
        tt_children = [
            ttnn.from_torch(c, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=cfg) for c in children
        ]
        out = bench.run_reduce_accum(
            tt_seed, tt_children, variant=bench.VARIANT_PACK_L1_ACC, fan_in=fan_in, block_tiles=block_tiles, dtype=dtype
        )
        torch_out = ttnn.to_torch(out).to(torch.float32)
        diff = (expected - torch_out).abs().max().item()
        print(f"{name}: max abs diff = {diff}")
        print(f"{name}: expected[0,:4]={expected[0,:4].tolist()}  actual[0,:4]={torch_out[0,:4].tolist()}")
finally:
    ttnn.close_device(device)
