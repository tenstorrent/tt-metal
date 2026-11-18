import torch, ttnn
import ttnn.operations.wq_b_linear.wq_b_linear_program_descriptor as pd
from ttnn.operations.wq_b_linear import wq_b_linear

device = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * 0.02).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
ref = torch.matmul(x.double(), ttnn.to_torch(tw).double())
for split in (False, True):
    pd.K_REMAINDER_SPLIT = split
    y = ttnn.to_torch(wq_b_linear(tx, tw)).double()
    pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
    rms = ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
    lo = (y - ref)[..., 110 * 32 :]
    print(
        "PREC split",
        split,
        "pcc",
        round(pcc, 7),
        "rel_rms",
        round(rms, 5),
        "leftover-col max abs err",
        round(lo.abs().max().item(), 4),
        "main max",
        round((y - ref)[..., : 110 * 32].abs().max().item(), 4),
    )
ttnn.close_device(device)
