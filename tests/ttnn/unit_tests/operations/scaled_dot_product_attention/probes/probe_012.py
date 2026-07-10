import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa


def run(B, H, S, D, dtype):
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    k = torch.randn(B, H, S, D, dtype=torch.float32)
    v = torch.randn(B, H, S, D, dtype=torch.float32)
    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        o = sdpa(tq, tk, tv)
        print(f"OK   B{B}H{H}S{S}D{D} {dtype}")
    except Exception as e:
        msg = str(e).split("\n")[0]
        print(f"FAIL B{B}H{H}S{S}D{D} {dtype}: {msg[:200]}")


for D in [256, 512, 1024]:
    run(1, 1, 128, D, ttnn.bfloat16)
