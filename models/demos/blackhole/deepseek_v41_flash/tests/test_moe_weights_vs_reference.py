# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU test: the host weight layout fed to TTMoEGate/TTMoEDecode reproduces the checkpoint's own MoE.

Golden = exactly the math TTMoEDecode implements (silu(x@w0) * (x@w1) @ w2, no clamp, weights from the
unbiased sqrtsoftplus scores renormalised and x1.5, plus the shared expert). Reference = the checkpoint's
`MoE` module run by model.py. The PCC between them is also the cost of dropping the swiglu clamp.
"""

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.reference import ref_layer as R
from models.demos.blackhole.deepseek_v41_flash.tt.moe_weights import load_moe_layer


def golden_moe(x, w, k=6, scale=1.5):
    s = torch.nn.functional.softplus(x.float() @ w["gate_weight"]).sqrt()  # [n, 384]
    idx = (s + w["gate_bias"]).topk(k, dim=-1).indices
    wt = s.gather(1, idx)
    wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * scale
    out = torch.zeros(x.size(0), x.size(1))
    for t in range(x.size(0)):
        xt = x[t : t + 1].float()
        for j in range(k):
            e = int(idx[t, j])
            h = torch.nn.functional.silu(xt @ w["w0"][0, e].float()) * (xt @ w["w1"][0, e].float())
            out[t] += wt[t, j] * (h @ w["w2"][0, e].float())[0]
        sid = 384
        h = torch.nn.functional.silu(xt @ w["shared_w0"][sid][0, 0].float()) * (xt @ w["shared_w1"][sid][0, 0].float())
        out[t] += (h @ w["shared_w2"][sid][0, 0].float())[0]
    return out, idx


@pytest.mark.parametrize("layer_id", [0, 2, 20])
def test_moe_weight_layout_matches_reference(layer_id):
    torch.manual_seed(0)
    blk = R.build_layer(layer_id, max_batch_size=2, max_seq_len=256)
    tok = torch.randint(1000, 100000, (2, 8))
    h, pm = R.embed_tokens(tok)
    captured = {}
    blk.ffn.register_forward_hook(lambda m, i, o: captured.update(x=i[0].detach(), y=o.detach()))
    with torch.inference_mode():
        blk(h, 0, pm, None)
    x = captured["x"].reshape(-1, 5120)
    ref = captured["y"].reshape(-1, 5120).float()

    w = load_moe_layer(layer_id)
    got, idx = golden_moe(x, w)
    ref_idx = blk.ffn.gate(x)[1]
    same_routing = (idx.sort(-1).values == ref_idx.sort(-1).values).all(-1).float().mean().item()
    p = R.pcc(got, ref)
    print(f"layer {layer_id}: routing identical for {same_routing*100:.0f}% of tokens; MoE PCC vs reference {p:.5f}")
    assert same_routing > 0.9
    assert p > 0.99
