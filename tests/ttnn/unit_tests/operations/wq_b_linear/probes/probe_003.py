import torch, ttnn
from ttnn.operations.wq_b_linear import wq_b_linear
from eval.golden_tests.wq_b_linear.helpers import WEIGHT_SCALE

d = ttnn.open_device(device_id=0)
torch.manual_seed(0)
x = torch.randn(1, 1, 160, 1536).bfloat16()
w = (torch.randn(1536, 4096) * WEIGHT_SCALE).bfloat16()
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=d)
tw = ttnn.from_torch(w, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=d)
ref = torch.matmul(x.double(), ttnn.to_torch(tw).double())
y = ttnn.to_torch(wq_b_linear(tx, tw)).double()
pcc = torch.corrcoef(torch.stack([y.flatten(), ref.flatten()]))[0, 1].item()
rr = ((y - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item()
print(f"ACC pcc={pcc:.6f} rel_rms={rr:.4f}")
ttnn.close_device(d)
