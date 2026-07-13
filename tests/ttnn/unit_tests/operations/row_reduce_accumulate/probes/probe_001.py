import torch, ttnn
from ttnn.operations.examples.row_reduce_accumulate import run_op, create_sharded_memory_config

TILE = 32


def make_input(device, wt, seed=13):
    torch.manual_seed(seed)
    w = wt * TILE
    row_base = torch.linspace(0.25, 4.0, TILE).unsqueeze(1)
    noise = (torch.rand(TILE, w) - 0.5) * 0.5
    x = row_base + noise
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
    for wt in (1, 2):
        xd, golden = make_input(device, wt)
        print(f"\n===== W={wt}t  golden[:5]={golden[:5].tolist()}")
        for v in (
            "reduce_fold.fp32",
            "l1_accum.fp32",
            "l1_accum.bf16",
            "dest_accum.fp32",
            "dest_accum.bf16",
            "dest_accum_pairs.fp32",
        ):
            out = run_op(xd, variant=v, width_tiles=wt, kernel_iters=1)
            col0 = ttnn.to_torch(out).to(torch.float32)[:, 0]
            print(f"  {v:24s} pcc={pcc(col0,golden):+.4f}  col0[:5]={[round(x,3) for x in col0[:5].tolist()]}")
finally:
    ttnn.close_device(device)
