import torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    if a.std() == 0 or b.std() == 0:
        return float(torch.allclose(a, b))
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def run(B, H, S, D, dtype, mask=False, scale=None):
    torch.manual_seed(0)
    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)
    m = torch.randn(B, 1, S, S) * 0.1 if mask else None
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale)
    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tm = ttnn.from_torch(m, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device) if mask else None
    try:
        o = sdpa(tq, tk, tv, attn_mask=tm, scale=scale)
        r = ttnn.to_torch(o).float()
        print(f"OK   B{B}H{H}S{S}D{D} {str(dtype)[9:]} mask={mask} scale={scale}: PCC={pcc(r,ref):.6f}")
    except Exception as e:
        print(f"FAIL B{B}H{H}S{S}D{D} {str(dtype)[9:]}: {str(e).splitlines()[0][:120]}")


for D in [256, 512, 1024]:
    run(1, 1, 128, D, ttnn.bfloat16)
    run(1, 1, 128, D, ttnn.float32)
    run(1, 1, 128, D, ttnn.bfloat16, mask=True)
    run(1, 1, 128, D, ttnn.bfloat16, scale=0.125)
ttnn.close_device(device)
