import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)


def cfg(acc):
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi4
    c.fp32_dest_acc_en = acc
    c.math_approx_mode = False
    return c


shape = (1, 1, 32, 7168)
W = shape[-1]
torch.manual_seed(0)
x = torch.randn(shape, dtype=torch.bfloat16)
xf = x.float()
print(f"MARK true_sum row0..3 = {[round(v,2) for v in (xf**2).sum(-1).flatten()[:4].tolist()]}")
print(
    f"MARK true 1/rms row0..3 = {[round(v,6) for v in (1.0/torch.sqrt((xf**2).mean(-1)+1e-6)).flatten()[:4].tolist()]}"
)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
for acc in (True, False):
    print(f"MARK === acc={acc} ===")
    ttnn.to_torch(rms_norm(tx, compute_kernel_config=cfg(acc)))
ttnn.close_device(device)
