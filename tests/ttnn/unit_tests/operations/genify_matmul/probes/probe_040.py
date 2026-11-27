import torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    a = ttnn.from_torch(torch.randn(1, 2048, 5120), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    b = ttnn.from_torch(torch.randn(5120, 8704), dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 2048, 4352]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    geom, grid, blk, g = pd.plan(a, b, out, True, dev, epilogue="swiglu")
    print("GEOM", geom)
    print("GRID", grid)
    print("BLK", blk, g)
    print(
        "MODE",
        pd.operand_mode(geom, grid, blk),
        "clock",
        dev.get_clock_rate_mhz() if hasattr(dev, "get_clock_rate_mhz") else None,
    )
finally:
    ttnn.close_device(dev)
