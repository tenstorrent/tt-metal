import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from models.common.utility_functions import comp_pcc


def ref(q, k, v, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    w = torch.softmax(s, dim=-1)
    return torch.matmul(w, v)


device = ttnn.open_device(device_id=0)
try:
    for dtype in [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b]:
        for shape in [(1, 1, 32, 32), (1, 8, 128, 64), (1, 4, 1024, 64)]:
            torch.manual_seed(0)
            q = torch.randn(shape)
            k = torch.randn(shape)
            v = torch.randn(shape)
            out = ref(q, k, v)
            tq = ttnn.from_torch(q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            tk = ttnn.from_torch(k, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            tv = ttnn.from_torch(v, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
            to = scaled_dot_product_attention(tq, tk, tv)
            r = ttnn.to_torch(to).to(torch.float32)
            _, pcc = comp_pcc(out, r, 0.99)
            print(f"{dtype} {shape}: {pcc}")
finally:
    ttnn.close_device(device)
