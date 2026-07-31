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
    print("expected[0,0:8] =", expected[0, 0:8])
    print("actual[0,0:8]   =", torch_out[0, 0:8])
    print("expected[0,32:40] =", expected[0, 32:40])
    print("actual[0,32:40]   =", torch_out[0, 32:40])
    print("max abs diff (tile0):", (expected[0, :32] - torch_out[0, :32]).abs().max().item())
    print("max abs diff (whole):", (expected - torch_out).abs().max().item())
    # check per-tile-column max diff, TILE=32 wide tiles across block_tiles=48 columns of 32
    for t in range(block_tiles):
        d = (expected[:, t * 32 : (t + 1) * 32] - torch_out[:, t * 32 : (t + 1) * 32]).abs().max().item()
        if d > 0.01:
            print(f"tile {t}: max diff {d}")
finally:
    ttnn.close_device(device)
