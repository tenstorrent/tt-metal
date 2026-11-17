import torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
try:
    for dt in ("bfloat16", "bfloat8_b"):
        for bmem in (ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG):
            a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 32, 512, 128]), getattr(ttnn, dt), ttnn.TILE_LAYOUT, dev)
            b = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 32, 128, 1024]), getattr(ttnn, dt), ttnn.TILE_LAYOUT, dev, bmem
            )
            out = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 32, 512, 1024]), getattr(ttnn, dt), ttnn.TILE_LAYOUT, dev
            )
            geom, grid, blk, g = pd.plan(a, b, out, False, dev, math_fidelity=ttnn.MathFidelity.HiFi2)
            print("CASE", dt, bmem.buffer_type)
            print("GRID", grid)
            print("BLK", blk)
            print("MODE", pd.operand_mode(geom, grid, blk))
            for t in (a, b, out):
                t.deallocate()
finally:
    ttnn.close_device(dev)
