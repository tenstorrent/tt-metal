import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)


def pcc(B, H, S, D):
    torch.manual_seed(0)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())
    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(scaled_dot_product_attention(q, k, v)).float()
    a = o.flatten()
    b = ref.float().flatten()
    print(f"PCC {(B,H,S,D)} = {torch.corrcoef(torch.stack([a,b]))[0,1].item():.6f}")


try:
    for s in [(1, 1, 32, 32), (1, 4, 128, 64), (2, 8, 256, 64), (1, 8, 512, 128)]:
        pcc(*s)
finally:
    ttnn.close_device(dev)
