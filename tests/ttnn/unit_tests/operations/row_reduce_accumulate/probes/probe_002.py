import torch, ttnn
from ttnn.operations.examples.row_reduce_accumulate import run_op, create_sharded_memory_config

TILE = 32


def make_input(device, wt, seed=13):
    torch.manual_seed(seed)
    w = wt * TILE
    row_base = torch.linspace(0.25, 4.0, TILE).unsqueeze(1)
    x = row_base + (torch.rand(TILE, w) - 0.5) * 0.5
    xb = x.to(torch.bfloat16)
    golden = xb.to(torch.float32).mean(dim=1)
    xd = ttnn.from_torch(
        xb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=create_sharded_memory_config(wt)
    )
    return xd, golden


device = ttnn.open_device(device_id=0)
try:
    xd, golden = make_input(device, 1)
    print("golden        :", [round(v, 2) for v in golden.tolist()])
    for v in ("dest_accum.fp32", "l1_accum.bf16"):
        out = run_op(xd, variant=v, width_tiles=1, kernel_iters=1)
        col0 = ttnn.to_torch(out).to(torch.float32)[:, 0]
        print(f"{v:16s}:", [round(x, 2) for x in col0.tolist()])
    # full output tile (not just col0) for dest_accum.fp32 to see if reduce wrote elsewhere
    out = run_op(xd, variant="dest_accum.fp32", width_tiles=1, kernel_iters=1)
    full = ttnn.to_torch(out).to(torch.float32)
    print("dest_accum.fp32 tile col-sums:", [round(x, 2) for x in full.sum(0).tolist()[:6]], "...")
    print("dest_accum.fp32 row0 first8 :", [round(x, 2) for x in full[0, :8].tolist()])
    xd2, golden2 = make_input(device, 2)
    out = run_op(xd2, variant="l1_accum.fp32", width_tiles=2, kernel_iters=1)
    col0 = ttnn.to_torch(out).to(torch.float32)[:, 0]
    print("l1_accum.fp32 W=2:", [round(x, 2) for x in col0.tolist()])
    print("golden W=2       :", [round(v, 2) for v in golden2.tolist()])
finally:
    ttnn.close_device(device)
