# ConvPrefill (one-op causal conv + SiLU, state in place) against the shared _causal_conv1d_fir.
import torch

import ttnn
from models.demos.blackhole.bonsai2.tt.gdn_prefill.forward import ConvPrefill
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir


def test_conv_prefill(device):
    torch.manual_seed(0)
    T, D, W = 2048, 10240, 16480
    x = torch.randn(1, T, W) * 0.5
    taps = torch.randn(4, D) * 0.3
    st = torch.zeros(1, 32, D)
    st[0, :3] = torch.randn(3, D) * 0.5
    B16 = lambda t: ttnn.from_torch(t.bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    taps_dev = [B16(taps[k].reshape(1, 1, D)) for k in range(4)]
    ref, ref_state = _causal_conv1d_fir(
        B16(x[:, :, :D]), None, None, 4, device, conv_state=B16(st[:, :3]), weight_taps=taps_dev
    )
    ref, ref_state = ttnn.to_torch(ref).float(), ttnn.to_torch(ref_state).float()[0, :3]
    state = B16(st)
    out = ttnn.to_torch(ConvPrefill(device, taps)(B16(x), state)).float()
    new_state = ttnn.to_torch(state).float()[0, :3]
    r = ((out - ref).norm() / ref.norm()).item()
    rs = ((new_state - ref_state).norm() / ref_state.norm()).item()
    print(f"CONV out rel {r:.5f} state rel {rs:.6f}")
    assert r < 0.01 and rs == 0.0
