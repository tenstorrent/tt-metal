import os, ttnn, torch, math
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd
from eval.golden_tests.genify_matmul.axes import observed as genify_matmul

dev = ttnn.open_device(device_id=0)
a = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 64, 16384]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
b = ttnn.allocate_tensor_on_device(ttnn.Shape([16384, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
o = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 64, 256]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
for bl, inj in ((True, 3), (False, 3), (False, 1)):
    pd.ACT_STREAM_BANK_LOCAL = bl
    pd.ACT_STREAM_INJECTORS = inj
    geom, grid, blk, _ = pd.plan(a, b, o, False, dev, math_fidelity=ttnn.MathFidelity.HiFi4)
    print(
        "OUT",
        bl,
        inj,
        pd.operand_mode(geom, grid, blk),
        grid,
        blk.num_k_blocks,
        blk.block_k_tiles,
        [round(t / 1350, 1) for t in pd.split_cost_terms(geom, grid, blk)],
    )
ttnn.close_device(dev)
