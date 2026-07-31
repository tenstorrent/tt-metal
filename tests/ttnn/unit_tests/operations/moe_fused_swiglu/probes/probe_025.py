import sys

sys.stdout.reconfigure(line_buffering=True)
import torch
import ttnn
from ttnn.operations.moe_fused_swiglu.perf_experiments.reduce_tree_shape.program_descriptor import (
    TILE,
    make_sharded_config,
    run_tree_variant,
)

print("opening device", flush=True)
device = ttnn.open_device(device_id=0)
try:
    k, n_tiles = 4, 12
    print("making sharded config", flush=True)
    config = make_sharded_config(device, k, n_tiles)
    torch_local = torch.empty((k * TILE, n_tiles * TILE), dtype=torch.float32)
    col_pattern = (torch.arange(n_tiles * TILE, dtype=torch.float32) % 13).reshape(1, -1) / 32.0
    for row in range(k):
        torch_local[row * TILE : (row + 1) * TILE] = (row + 1) * 0.25 + col_pattern

    print("creating local_tensor", flush=True)
    local_tensor = ttnn.from_torch(
        torch_local, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    print("reading back local_tensor for reference", flush=True)
    quantized = ttnn.to_torch(local_tensor).to(torch.float32)
    ref = sum(quantized[row * TILE : (row + 1) * TILE] for row in range(k))
    print("prologue readback ok", flush=True)

    zero = torch.zeros((k * TILE, n_tiles * TILE), dtype=torch.float32)
    result_tensor = ttnn.from_torch(
        zero, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=config
    )
    print("about to run kernel", flush=True)

    run_tree_variant(device, local_tensor, result_tensor, "hillis_steele", k, n_tiles)
    print("kernel launched+synced (generic_op returned)", flush=True)
    actual = ttnn.to_torch(result_tensor).to(torch.float32)[0:TILE]
    print("ACTUAL[0,:8]", actual[0, :8], flush=True)
    print("REF[0,:8]   ", ref[0, :8], flush=True)
    diff = (actual - ref).abs().max()
    print("max abs diff", diff.item(), flush=True)
finally:
    ttnn.close_device(device)
