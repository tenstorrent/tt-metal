import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

device = ttnn.open_device(device_id=0)
shape = (1, 1, 6400, 64)
x = (
    (torch.arange(6400).float().reshape(6400, 1) / 100 + 1 + torch.arange(64).float().reshape(1, 64) / 64)
    .reshape(shape)
    .to(torch.bfloat16)
)
tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
y = ttnn.to_torch(rms_norm(tx)).float().reshape(6400, 64)
xf = x.float().reshape(6400, 64)
rms = torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
ref = xf / rms
for r in [0, 31, 32, 33, 63, 64, 95, 96]:
    ratio = xf[r] / y[r]
    print(
        f"D row {r}: y[:4]={y[r,:4].tolist()} ref[:4]={ref[r,:4].tolist()} x/y={ratio[:3].tolist()} rms={rms[r].item():.4f}"
    )
ttnn.close_device(device)
