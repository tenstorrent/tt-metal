import torch, ttnn
from ttnn.operations.examples.row_reduce_accumulate import run_op, create_sharded_memory_config, VARIANTS

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


def pcc(a, g):
    return torch.corrcoef(torch.stack([a.flatten().double(), g.flatten().double()]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    for wt in (1, 2, 8, 32):
        for ki in (1, 5):
            xd, golden = make_input(device, wt)
            print(f"\n== W={wt}t kernel_iters={ki}")
            for v in VARIANTS:
                out = run_op(xd, variant=v, width_tiles=wt, kernel_iters=ki)
                col0 = ttnn.to_torch(out).to(torch.float32)[:, 0]
                mx = (col0 - golden).abs().max().item()
                flag = "" if pcc(col0, golden) > 0.99 else "  <-- BAD"
                print(f"   {v:26s} pcc={pcc(col0,golden):+.5f}  max_abs_err={mx:.4f}{flag}")
finally:
    ttnn.close_device(device)
