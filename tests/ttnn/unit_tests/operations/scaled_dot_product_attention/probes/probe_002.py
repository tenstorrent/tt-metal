import math
import torch
import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

device = ttnn.open_device(device_id=0)
try:
    torch.manual_seed(42)
    shape = (1, 1, 64, 64)
    Q = (torch.randn(shape) * 0.3).to(torch.bfloat16)
    K = (torch.randn(shape) * 0.3).to(torch.bfloat16)
    V = (torch.randn(shape) * 0.3).to(torch.bfloat16)
    scale = 1.0 / math.sqrt(shape[-1])
    Q32, K32, V32 = Q.float(), K.float(), V.float()
    expected = torch.matmul(torch.softmax(torch.matmul(Q32, K32.transpose(-2, -1)) * scale, dim=-1), V32).to(
        torch.bfloat16
    )

    q = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    k = ttnn.from_torch(K, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    v = ttnn.from_torch(V, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(q, k, v)
    actual = ttnn.to_torch(out)
    abs_err = (actual.float() - expected.float()).abs()
    print("bf8b path: actual shape=", actual.shape, " dtype=", actual.dtype)
    print("bf8b max abs err=", abs_err.max().item())
    print("bf8b mean abs err=", abs_err.mean().item())
    print("bf8b PCC≈", torch.corrcoef(torch.stack([actual.flatten().float(), expected.flatten().float()]))[0, 1].item())
finally:
    ttnn.close_device(device)
