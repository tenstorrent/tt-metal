import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from eval.golden_tests.scaled_dot_product_attention.helpers import pytorch_scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)


def run(B, H, S, D, maskB, maskH):
    torch.manual_seed(1)
    Q = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H, S, D, dtype=torch.bfloat16)
    m = torch.bernoulli(torch.full((maskB, maskH, S, S), 0.25)) * -1e9
    ref = pytorch_scaled_dot_product_attention(Q, K, V, attn_mask=m.to(torch.bfloat16))
    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tm = ttnn.from_torch(m.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(scaled_dot_product_attention(q, k, v, attn_mask=tm)).float()
    a = o.flatten()
    b = ref.float().flatten()
    pcc = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    print(f"case B={B} H={H} maskB={maskB} maskH={maskH} PCC={pcc:.5f}")


try:
    run(2, 4, 128, 32, 1, 1)  # batch+head bcast
    run(2, 4, 128, 32, 1, 4)  # batch bcast, per-head
    run(2, 4, 128, 32, 2, 1)  # per-batch, head bcast
    run(2, 4, 128, 32, 2, 4)  # full
finally:
    ttnn.close_device(dev)
