import sys

sys.stdout.reconfigure(line_buffering=True)
import torch
import ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_tree_shape.program_descriptor import (
    TILE,
    make_sharded_config,
    run_tree_variant,
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

    zero = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
    result_tensor = ttnn.from_torch(
        zero, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    print("about to run hillis_steele k=4 n=12", flush=True)
    run_tree_variant(device, local_tensor, result_tensor, "hillis_steele", k, n_tiles)
    actual = ttnn.to_torch(result_tensor).to(torch.float32)[0:TILE]
    diff = (actual - ref).abs().max()
    print("HILLIS_STEELE max abs diff", diff.item(), flush=True)

    zero2 = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
    result_tensor2 = ttnn.from_torch(
        zero2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    print("about to run fanin2 k=4 n=12", flush=True)
    run_tree_variant(device, local_tensor, result_tensor2, "fanin2", k, n_tiles)
    actual2 = ttnn.to_torch(result_tensor2).to(torch.float32)[0:TILE]
    diff2 = (actual2 - ref).abs().max()
    print("FANIN2 max abs diff", diff2.item(), flush=True)

    zero3 = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
    result_tensor3 = ttnn.from_torch(
        zero3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    print("about to run twophase k=4 n=12", flush=True)
    from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_tree_shape.program_descriptor import run_twophase

    run_twophase(device, local_tensor, result_tensor3, k, n_tiles)
    actual3 = ttnn.to_torch(result_tensor3).to(torch.float32)[0:TILE]
    diff3 = (actual3 - ref).abs().max()
    print("TWOPHASE max abs diff", diff3.item(), flush=True)
    print("ALL THREE VARIANTS RAN WITHOUT HANGING", flush=True)
finally:
    ttnn.close_device(device)
