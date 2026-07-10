import torch, ttnn

device = ttnn.open_device(device_id=0)
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention as sdpa


def run(B, H, S, D, dtype):
    q = torch.randn(B, H, S, D, dtype=torch.float32)
    tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    try:
        o = sdpa(tq, tk, tv)
        ttnn.to_torch(o)
        print(f"OK   B{B}H{H}S{S}D{D} {dtype}")
    except Exception as e:
        msg = str(e).replace("\n", " ")
        print(f"FAIL B{B}H{H}S{S}D{D} {dtype}: {msg[:260]}")


for D in [256, 512, 1024]:
    run(1, 1, 128, D, ttnn.bfloat16)
ttnn.close_device(device)
