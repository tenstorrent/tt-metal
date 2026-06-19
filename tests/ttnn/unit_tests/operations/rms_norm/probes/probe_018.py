import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from ttnn.operations.rms_norm import rms_norm_program_descriptor as pd

dev = device
# Wide-W, few rows -> should route through Regime B RM (mcast all-gather)
shape = (1, 1, 32, 8192)
torch.manual_seed(0)
x = torch.randn(shape)
ti = ttnn.from_torch(x.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)

# Prove it routes through Regime B RM: build the descriptor and check semaphores present.
out_t = ttnn.allocate_tensor_on_device(
    ttnn.Shape(list(shape)), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, dev, ttnn.DRAM_MEMORY_CONFIG
)
prog, io = pd.create_program_descriptor(ti, out_t, None, 1e-6, None)
print("num semaphores:", len(prog.semaphores), "(Regime B => 2)")

out = ttnn.to_torch(rms_norm(ti)).float()
var = x.pow(2).mean(dim=-1, keepdim=True)
exp = (x * torch.rsqrt(var + 1e-6)).float()
pcc = torch.corrcoef(torch.stack([out.flatten(), exp.flatten()]))[0, 1].item()
relrms = ((out - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
print(f"PCC={pcc:.6f} relRMS={relrms:.5f} mean(out)={out.mean().item():.4f}")
