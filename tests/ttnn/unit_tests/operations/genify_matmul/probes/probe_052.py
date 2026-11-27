import ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
L, H2, H4 = ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4
cases = [
    ("te_qkv", 512, 5120, 3072, ttnn.bfloat8_b, H2),
    ("te_o", 512, 4096, 2560, ttnn.bfloat8_b, H2),
    ("te_down", 512, 16384, 5120, ttnn.bfloat8_b, H2),
    ("q512_in", 512, 5120, 4160, ttnn.bfloat4_b, L),
    ("q512_out", 512, 1536, 5120, ttnn.bfloat4_b, L),
    ("q1024_down", 1024, 4352, 5120, ttnn.bfloat4_b, L),
    ("q4096_qkvg", 4096, 5120, 3584, ttnn.bfloat4_b, L),
    ("proj_out_h4", 2048, 3072, 128, ttnn.bfloat16, H4),
    ("proj_out_h2", 2048, 3072, 128, ttnn.bfloat16, H2),
    ("x_emb", 2048, 128, 3072, ttnn.bfloat16, H2),
    ("g512_h4", 512, 5120, 3072, ttnn.bfloat8_b, H4),
    ("384x1536x1024", 384, 1536, 1024, ttnn.bfloat16, H2),
]
pd_split = pd.R1A_SPLIT_LINES
for name, m, k, n, wd, fid in cases:
    a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, k]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    b = ttnn.allocate_tensor_on_device(ttnn.Shape([k, n]), wd, ttnn.TILE_LAYOUT, dev)
    o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, m, n]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
    geom, grid, blk, _ = pd.plan(a, b, o, True, dev, math_fidelity=fid)
    walk, dram, core, wr = pd.split_cost_terms(geom, grid, blk)
    print(
        f"RATIO {name:14s} mode={pd.operand_mode(geom,grid,blk)} split={pd.r1a_split(geom,grid,blk)} core/walk={core/walk:.2f} grid={grid.grid_m}x{grid.grid_n}"
    )
    for t in (a, b, o):
        ttnn.deallocate(t)
ttnn.close_device(dev)
