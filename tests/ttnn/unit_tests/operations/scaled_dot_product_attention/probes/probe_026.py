import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)


def pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    a, b = a - a.mean(), b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-12)).item()


try:
    torch.manual_seed(0)
    shapes = [(1, 1, 1, 64, 64), (1, 8, 1, 256, 64), (1, 1, 1, 4096, 64), (1, 8, 2, 4096, 128), (1, 32, 1, 128, 128)]
    for B, H, Hkv, S, D in shapes:
        q = torch.randn(B, H, S, D)
        k = torch.randn(B, Hkv, S, D)
        v = torch.randn(B, Hkv, S, D)
        kb = k.repeat_interleave(H // Hkv, dim=1)
        vb = v.repeat_interleave(H // Hkv, dim=1)
        ref = torch.nn.functional.scaled_dot_product_attention(q.float(), kb.float(), vb.float(), is_causal=True)
        for dt in [ttnn.bfloat8_b]:
            tq = ttnn.from_torch(q, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tk = ttnn.from_torch(k, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            tv = ttnn.from_torch(v, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)
            o = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, is_causal=True))
            print(f"shape {B}x{H}x{S}x{D} bf8b: causal pcc={pcc(o, ref):.4f}")
finally:
    ttnn.close_device(dev)
