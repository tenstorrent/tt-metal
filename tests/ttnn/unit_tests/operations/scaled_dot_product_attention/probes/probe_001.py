import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
try:
    for B, H, S, D in [(1, 12, 512, 64), (1, 8, 1024, 128)]:
        torch.manual_seed(0)
        Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
        ref = torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float()).to(torch.bfloat16)
        q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        o = scaled_dot_product_attention(q, k, v)
        ot = ttnn.to_torch(o).float()
        a = ot.flatten()
        b = ref.float().flatten()
        pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        print(f"shape {(B,H,S,D)} units={B*H*(S//32)} PCC={pcc:.5f}")
finally:
    ttnn.close_device(dev)
