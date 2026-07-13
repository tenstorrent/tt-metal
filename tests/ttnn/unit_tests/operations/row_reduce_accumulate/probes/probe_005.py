import torch, ttnn
from ttnn.operations.examples.row_reduce_accumulate import run_op, create_sharded_memory_config

TILE = 32


def make_input(device, wt, input_dtype, dist="positive", seed=13):
    torch.manual_seed(seed)
    w = wt * TILE
    data = (
        torch.rand(TILE, w)
        if dist == "positive"
        else (torch.linspace(0.25, 4.0, TILE).unsqueeze(1) + (torch.rand(TILE, w) - 0.5) * 0.5)
    )
    golden = data.to(torch.float64).mean(dim=1)
    td = {"fp32": (torch.float32, ttnn.float32), "bf16": (torch.bfloat16, ttnn.bfloat16)}[input_dtype]
    xd = ttnn.from_torch(
        data.to(td[0]),
        dtype=td[1],
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config(wt),
    )
    return xd, golden


def pcc(a, g):
    return torch.corrcoef(torch.stack([a.flatten().double(), g.flatten().double()]))[0, 1].item()


device = ttnn.open_device(device_id=0)
try:
    for prec in ("fp32-fp32", "bf16-fp32", "bf16-bf16"):
        for wt in (1, 8):
            idt = prec.split("-")[0]
            xd, golden = make_input(device, wt, idt, "signal")
            for m in ("dest_accum", "dest_accum_sfpu", "dest_accum_pairs_sfpu"):
                out = run_op(xd, method=m, precision=prec, width_tiles=wt, kernel_iters=1)
                t = ttnn.to_torch(out).to(torch.float64)
                col0, row0 = t[:, 0], t[0, :]
                print(
                    f"{m:22s} {prec:10s} W={wt:2d}t  col0 pcc={pcc(col0,golden):+.4f} maxerr={(col0-golden).abs().max():.4f}  | row0[:4]={[round(x,3) for x in row0[:4].tolist()]}"
                )
finally:
    ttnn.close_device(device)
