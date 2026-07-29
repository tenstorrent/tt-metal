import torch, ttnn
from ttnn.operations.tilize.tilize_program_descriptor import build_plan
import ttnn as t


def plan_for(device, shape, dtype=ttnn.bfloat16, multicore=True):
    x = torch.zeros(shape, dtype=torch.bfloat16)
    ti = ttnn.from_torch(x, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    to = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.TILE_LAYOUT, device, ti.memory_config())
    p = build_plan(ti, to, device, use_multicore=multicore)
    return p


def rpt(device, name, shape):
    p = plan_for(device, shape)
    blocks = [u["row_count"] * u["chunk_count"] for u in p["work"]]
    tot = sum(blocks)
    mx = max(blocks)
    mn = min(blocks)
    nc = len(blocks)
    ideal = tot / nc
    print(
        f"{name:28s} {str(shape):22s} nt_h={p['nt_h']:5d} Wt={p['wt']:4d} chunk={p['chunk_wt']:3d} "
        f"cores={p['ncores']:3d} blk max/min/mean={mx}/{mn}/{ideal:.2f} imbalance={mx/ideal:.3f} total_blk={tot}"
    )


def main(device):
    for name, shape in [
        ("even_square_64row", (1, 1, 2048, 2048)),
        ("imbal_square_65row", (1, 1, 2080, 2048)),
        ("imbal_square_66row", (1, 1, 2112, 2048)),
        ("imbal_square_96row", (1, 1, 3072, 2048)),
        ("imbal_square_127row", (1, 1, 4064, 2048)),
        ("even_tall_narrow", (1, 1, 2048, 32)),
        ("imbal_tall_narrow_65", (1, 1, 2080, 32)),
        ("imbal_tall_narrow_33", (1, 1, 1056, 32)),
        ("wide_short", (1, 1, 32, 16384)),
        ("wide_short_awk_Wt5", (1, 1, 32, 160)),
        ("awk_Wt7", (1, 1, 64, 224)),
        ("awk_Wt63", (1, 1, 2048, 2016)),
    ]:
        rpt(device, name, shape)


device = ttnn.open_device(device_id=0)
try:
    main(device)
finally:
    ttnn.close_device(device)
