import sys

sys.stdout.reconfigure(line_buffering=True)
import torch
import ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_tree_shape.program_descriptor import (
    TILE,
    make_sharded_config,
    run_tree_variant,
    run_twophase,
)

device = ttnn.open_device(device_id=0)
try:
    k, n_tiles = 4, 12
    config = make_sharded_config(device, k, n_tiles)
    torch_local = torch.empty((k * TILE, n_tiles * TILE), dtype=torch.float32)
    col_pattern = (torch.arange(n_tiles * TILE, dtype=torch.float32) % 13).reshape(1, -1) / 32.0
    for row in range(k):
        torch_local[row * TILE : (row + 1) * TILE] = (row + 1) * 0.25 + col_pattern

    local_tensor = ttnn.from_torch(
        torch_local, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    quantized = ttnn.to_torch(local_tensor).to(torch.float32)
    ref = sum(quantized[row * TILE : (row + 1) * TILE] for row in range(k))

    for name, fn in [
        ("hillis_steele", lambda rt: run_tree_variant(device, local_tensor, rt, "hillis_steele", k, n_tiles)),
        ("fanin2", lambda rt: run_tree_variant(device, local_tensor, rt, "fanin2", k, n_tiles)),
        ("twophase", lambda rt: run_twophase(device, local_tensor, rt, k, n_tiles)),
    ]:
        zero = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
        result_tensor = ttnn.from_torch(
            zero, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
        )
        print(f"about to run {name} k={k} n={n_tiles}", flush=True)
        fn(result_tensor)
        actual = ttnn.to_torch(result_tensor).to(torch.float32)[0:TILE]
        diff = (actual - ref).abs().max()
        print(f"{name} max abs diff", diff.item(), flush=True)
    print("ALL THREE VARIANTS RAN WITHOUT HANGING", flush=True)
finally:
    ttnn.close_device(device)
