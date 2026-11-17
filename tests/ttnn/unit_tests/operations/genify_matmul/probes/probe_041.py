import torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

CASES = [
    ("proj_out", 2048, 3072, 128, "bfloat16", "bfloat16", "none"),
    ("te_qkv", 512, 5120, 3072, "bfloat16", "bfloat8_b", "none"),
    ("te_o", 512, 4096, 2560, "bfloat16", "bfloat8_b", "none"),
    ("te_down", 512, 16384, 5120, "bfloat16", "bfloat8_b", "none"),
    ("s_qkv", 2304, 6144, 9216, "bfloat16", "bfloat16", "none"),
    ("s_out", 2304, 12288, 6144, "bfloat16", "bfloat16", "none"),
    ("s_ff1", 2304, 6144, 18432, "bfloat16", "bfloat16", "swiglu"),
    ("q_qkvg4096", 4096, 5120, 3584, "bfloat16", "bfloat4_b", "none"),
    ("q_in1024", 1024, 5120, 4160, "bfloat16", "bfloat4_b", "none"),
    ("q512_in", 512, 5120, 4160, "bfloat16", "bfloat4_b", "none"),
    ("q512_out", 512, 1536, 5120, "bfloat16", "bfloat4_b", "none"),
    ("q512_gu", 512, 5120, 8704, "bfloat16", "bfloat4_b", "swiglu"),
    ("q512_down", 512, 4352, 5120, "bfloat16", "bfloat4_b", "none"),
    ("q512_qkvg", 512, 5120, 3584, "bfloat16", "bfloat4_b", "none"),
]
dev = ttnn.open_device(device_id=0)
try:
    print("clock", dev.get_clock_rate_mhz() if hasattr(dev, "get_clock_rate_mhz") else None)
    for name, m, k, n, ad, bd, ep in CASES:
        a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, k]), getattr(ttnn, ad), ttnn.TILE_LAYOUT, dev)
        b = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), getattr(ttnn, bd), ttnn.TILE_LAYOUT, dev)
        on = n // 2 if ep == "swiglu" else n
        out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, on]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
        geom, grid, blk, g = pd.plan(a, b, out, True, dev, epilogue=ep)
        print("CASE", name)
        print("GRID", grid)
        print("BLK", blk)
        print(
            "MODE",
            pd.operand_mode(geom, grid, blk),
            "TRAF MB",
            pd.read_traffic(geom, grid, blk) / 1e6,
            "W",
            pd.grid_work(grid),
            "ramp",
            pd.k_head_ramp(blk, pd.operand_mode(geom, grid, blk), ep),
        )
        for t in (a, b, out):
            t.deallocate()
finally:
    ttnn.close_device(dev)
