import math
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    shape = (1, 1, 64, 64)
    Q = (torch.randn(shape) * 0.3).to(torch.float32)
    K = (torch.randn(shape) * 0.3).to(torch.float32)
    V = (torch.randn(shape) * 0.3).to(torch.float32)
    scale = 1.0 / math.sqrt(shape[-1])
    expected = torch.matmul(torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * scale, dim=-1), V)

    q = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)
    abs_err = (actual - expected).abs()
    print("fp32 path: shape match=", actual.shape == expected.shape)
    print("fp32 max abs err=", abs_err.max().item())
    print("fp32 mean abs err=", abs_err.mean().item())
    print("fp32 PCC≈", torch.corrcoef(torch.stack([actual.flatten(), expected.flatten()]))[0, 1].item())
finally:
    ttnn.close_device(device)
