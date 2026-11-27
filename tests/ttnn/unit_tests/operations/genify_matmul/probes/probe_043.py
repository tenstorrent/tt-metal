import torch, ttnn
from ttnn.operations.genify_matmul import genify_matmul_program_descriptor as pd

dev = ttnn.open_device(device_id=0)
shapes = [
    ((1, 32, 512, 128), (1, 32, 128, 1024)),
    ((1, 16, 256, 128), (1, 16, 128, 256)),
    ((1, 64, 64, 64), (1, 64, 64, 64)),
    ((1, 8, 512, 512), (1, 8, 512, 512)),
    ((1, 128, 64, 128), (1, 128, 128, 64)),
    ((1, 4, 128, 128), (1, 4, 128, 256)),
    ((1, 32, 256, 512), (1, 32, 512, 256)),
    ((1, 64, 128, 256), (1, 64, 256, 128)),
]
try:
    for mc in (True, False):
        pd.OPERAND_MCAST = mc
        for ep in ("none", "bias", "addcmul"):
            for sa, sb in shapes:
                a = ttnn.allocate_tensor_on_device(ttnn.Shape(list(sa)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
                b = ttnn.allocate_tensor_on_device(ttnn.Shape(list(sb)), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
                out = ttnn.allocate_tensor_on_device(
                    ttnn.Shape(list(sa[:-1]) + [sb[-1]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev
                )
                row = (
                    ttnn.allocate_tensor_on_device(ttnn.Shape([1, sb[-1]]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev)
                    if ep != "none"
                    else None
                )
                res = out if ep == "addcmul" else None
                geom, grid, blk, g = pd.plan(
                    a, b, out, False, dev, epilogue=ep, row=row, residual=res, math_fidelity=ttnn.MathFidelity.HiFi2
                )
                print(
                    "S",
                    mc,
                    ep,
                    sa,
                    sb,
                    "bb",
                    blk.block_batches,
                    "nk",
                    blk.num_k_blocks,
                    "mode",
                    pd.operand_mode(geom, grid, blk),
                    "cb",
                    grid.core_batches,
                    "m",
                    blk.block_m_tiles,
                    "n",
                    blk.block_n_tiles,
                    "wb",
                    blk.write_batch,
                )
                for t in (a, b, out, row):
                    if t is not None:
                        t.deallocate()
finally:
    ttnn.close_device(dev)
