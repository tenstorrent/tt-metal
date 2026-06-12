import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


device = ttnn.open_device(device_id=0)
try:
    B, H, Sq, Skv, D = 1, 1, 128, 1024, 1024
    torch.manual_seed(0)
    q = torch.randn(B, H, Sq, D)
    k = torch.randn(B, H, Skv, D)
    v = torch.randn(B, H, Skv, D)
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    qt = ttnn.from_torch(q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(k, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(v, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    ot = scaled_dot_product_attention(qt, kt, vt)
    out = ttnn.to_torch(ot)
    print(
        "RESULT shape",
        out.shape,
        "PCC",
        float(pcc(out, ref)),
        "max_abs",
        float((out.float() - ref.float()).abs().max()),
    )
finally:
    ttnn.close_device(device)
