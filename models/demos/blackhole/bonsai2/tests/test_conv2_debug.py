# conv2 debug: all four taps at once, zero state, against torch.
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill.forward import ConvPrefill


def test_conv2_taps(device):
    torch.manual_seed(0)
    T, D, W = 2048, 10240, 16480
    x = (torch.randn(1, T, W) * 0.5).bfloat16().float()
    B16 = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    taps = (torch.randn(4, D) * 0.3).bfloat16().float()
    st = torch.zeros(1, 32, D)
    out = ttnn.to_torch(ConvPrefill(device, taps)(B16(x), B16(st))).float()[0]
    acc = torch.zeros(T, D)
    for k in range(4):
        s = 3 - k
        xs = torch.cat([torch.zeros(s, D), x[0, : T - s, :D]], 0) if s else x[0, :, :D]
        acc += xs * taps[k]
    ref = torch.nn.functional.silu(acc)
    r = ((out - ref).norm() / ref.norm()).item()
    e = (out - ref).abs().reshape(T // 32, 32, D // 32, 32).amax(dim=(0, 2))
    print(
        f"ALL rel {r:.4f} bad rows {(e.amax(1) > 0.01).nonzero().flatten().tolist()[:16]} bad cols {(e.amax(0) > 0.01).nonzero().flatten().tolist()[:16]}"
    )
