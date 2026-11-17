import torch, ttnn
from ttnn.operations.genify_matmul.genify_matmul_program_descriptor import plan

dev = ttnn.open_device(device_id=0)
try:
    for m, k, n, wd, ep in [
        (2048, 5120, 8704, ttnn.bfloat4_b, "swiglu"),
        (2048, 5120, 8704, ttnn.bfloat4_b, "none"),
        (2048, 6144, 3072, ttnn.bfloat16, "addcmul"),
    ]:
        a = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, m, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        b = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), wd, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG)
        no = n // 2 if ep == "swiglu" else n
        o = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, m, no]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
        )
        row = (
            ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
            )
            if ep == "addcmul"
            else None
        )
        res = o if ep == "addcmul" else None
        g, grid, blk, _ = plan(a, b, o, True, dev, epilogue=ep, row=row, residual=res)
        print("PLAN", m, k, n, ep, grid, blk)
finally:
    ttnn.close_device(dev)
