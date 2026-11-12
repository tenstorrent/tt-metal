import torch, ttnn
from ttnn.operations.rms_norm import rms_norm

dev = ttnn.open_device(device_id=0)


def run(x, gamma=None):
    tx = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    tg = (
        ttnn.from_torch(gamma.reshape(1, 1, 1, -1), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        if gamma is not None
        else None
    )
    got = ttnn.to_torch(rms_norm(tx, gamma=tg)).double()
    xd = x.double()
    ref = xd / torch.sqrt((xd * xd).mean(-1, keepdim=True) + 1e-6)
    if gamma is not None:
        ref = ref * gamma.double()
    m = ref.abs() > 1e-3
    r = got[m] / ref[m]
    return r.median().item(), r.std().item()


try:
    torch.manual_seed(0)
    x = torch.randn(1, 1, 256, 1024)
    g = torch.randn(1024)
    print("RATIO fp32 random, no gamma      ", run(x))
    print("RATIO fp32 random, gamma         ", run(x, g))
    xb = x.to(torch.bfloat16).float()
    gb = g.to(torch.bfloat16).float()
    print("RATIO fp32 bf16-exact x, no gamma", run(xb))
    print("RATIO fp32 bf16-exact x, gamma   ", run(xb, gb))
    # per-row constant input: x = 1.5 everywhere -> rstd = 1/1.5 exactly-ish, out should be 1.0
    print("RATIO constant 1.5 rows          ", run(torch.full((1, 1, 64, 1024), 1.5)))
finally:
    ttnn.close_device(dev)
