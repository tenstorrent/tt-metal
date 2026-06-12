import math, torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def ref(Q, K, V, scale):
    Qf, Kf, Vf = (t.float() for t in (Q, K, V))
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    sc = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    return torch.matmul(torch.softmax(sc, -1), Vf)


device = ttnn.open_device(device_id=0)
try:
    for shape in [(1, 1, 128, 64), (1, 1, 256, 64), (1, 2, 128, 64)]:
        torch.manual_seed(0)
        Q = torch.randn(shape, dtype=torch.float32)
        K = torch.randn(shape, dtype=torch.float32)
        V = torch.randn(shape, dtype=torch.float32)
        r = ref(Q, K, V, None)
        tQ = ttnn.from_torch(Q, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tK = ttnn.from_torch(K, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        tV = ttnn.from_torch(V, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        o = ttnn.to_torch(scaled_dot_product_attention(tQ, tK, tV)).float()
        pcc = torch.corrcoef(torch.stack([o.flatten(), r.flatten()]))[0, 1].item()
        rms = (torch.sqrt(torch.mean((o - r) ** 2)) / r.std()).item()
        print(f"RESULT shape={shape} PCC={pcc:.6f} rms={rms:.6f} nan={torch.isnan(o).any().item()}")
finally:
    ttnn.close_device(device)
