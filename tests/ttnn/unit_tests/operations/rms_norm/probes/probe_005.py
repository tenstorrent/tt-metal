import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)
torch.manual_seed(0)


def tf32(t):
    return (t.view(torch.int32) & ~((1 << 13) - 1)).view(torch.float32)


for W in (64, 512, 4096):
    x = torch.randn(1, 1, 64, W)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.to_torch(rms_norm(xt)).float()
    true_rinv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
    ratio = out / (x * true_rinv)
    eff_rinv = (out / tf32(x)).median(-1).values.unsqueeze(-1)
    print(
        "RES W",
        W,
        "ratio median",
        ratio.median().item(),
        "eff_rinv/true median",
        (eff_rinv / true_rinv).median().item(),
        "tf32-only predicted ratio median",
        (tf32(x) * tf32(true_rinv) / (x * true_rinv)).median().item(),
    )
    # emulate sumsq with tf32 inputs
    rinv_tf = torch.rsqrt((tf32(x) ** 2).mean(-1, keepdim=True) + 1e-6)
    print("RES   rinv(tf32 x)/true", (rinv_tf / true_rinv).median().item())
ttnn.close_device(dev)
