import torch, ttnn
import ttnn.operations.scaled_dot_product_attention as sdpa_mod
from eval.golden_tests.scaled_dot_product_attention.helpers import pytorch_scaled_dot_product_attention

# Temporarily widen SUPPORTED to test the kernel's GQA/MQA path directly
sdpa_mod.SUPPORTED["kv_heads_mode"] = ["mha", "gqa", "mqa"]
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)


def run(qs, ks, label):
    torch.manual_seed(2)
    Q = torch.randn(qs, dtype=torch.bfloat16)
    K = torch.randn(ks, dtype=torch.bfloat16)
    V = torch.randn(ks, dtype=torch.bfloat16)
    ref = pytorch_scaled_dot_product_attention(Q, K, V)
    q = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    o = ttnn.to_torch(scaled_dot_product_attention(q, k, v)).float()
    a = o.flatten()
    b = ref.float().flatten()
    print(f"case {label} PCC={torch.corrcoef(torch.stack([a,b]))[0,1].item():.5f}")


try:
    run((1, 8, 128, 64), (1, 2, 128, 64), "GQA 4:1")
    run((1, 8, 128, 64), (1, 1, 128, 64), "MQA")
finally:
    ttnn.close_device(dev)
