# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone mHC reference must track the model's own DeepseekV4HyperConnection.

mhc_reference is a lightweight fp32 extract the op / composite PCC tests validate against; this
pins it to the canonical model so the two cannot silently drift. Pure CPU (torch vs torch).
"""

import pytest
import torch
import torch.nn.functional as F

from models.demos.deepseek_v3_d_p.reference.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperConnection
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, MHCWrap, parametrize


def _pcc(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.parametrize("H, D, scale_val", [(4, 64, 1.0), (4, 128, 0.01)], ids=["H4D64s1.0", "H4D128s0.01"])
def test_reference_matches_canonical(H, D, scale_val):
    cfg = MHCConfig(dim=D, n=H)
    dcfg = DeepseekV4Config(
        hidden_size=D, hc_mult=H, hc_sinkhorn_iters=cfg.sinkhorn_iters, hc_eps=cfg.eps, rms_norm_eps=cfg.norm_eps
    )
    hc = DeepseekV4HyperConnection(dcfg).eval()

    g = torch.Generator().manual_seed(1)
    fn = torch.randn(hc.fn.shape, generator=g) * 0.02
    base = torch.randn(hc.base.shape, generator=g)
    scale = torch.full((3,), float(scale_val))
    with torch.no_grad():
        hc.fn.copy_(fn)
        hc.base.copy_(base)
        hc.scale.copy_(scale)

    x = torch.randn(1, 8, H, D, generator=g)
    with torch.no_grad():
        post_c, comb_c, collapsed_c = hc(x)

    # End-to-end: our RMS is linear-then-scalar vs the model's norm-then-linear -- equal up to
    # fp32 rounding because the RMS factor is a per-token scalar and commutes through the matmul.
    wrap = MHCWrap(cfg, constraint="sinkhorn")
    with torch.no_grad():
        wrap.fn.copy_(fn)
        wrap.base.copy_(base)
        wrap.scale.copy_(scale)
        y_o, post_o, comb_o = wrap.hc_pre(x)
    for name, o, c in [("post", post_o, post_c), ("comb", comb_o, comb_c), ("collapsed", y_o, collapsed_c)]:
        pcc, md = _pcc(o, c), (o.float() - c.float()).abs().max().item()
        assert pcc > 0.99999 and md < 1e-4, f"{name}: pcc={pcc:.8f} maxabs={md:.2e}"

    # Parametrization is bit-identical when both consume the model's own projected mixes.
    with torch.no_grad():
        mixes = F.linear(hc.input_norm(x.flatten(2).float()), hc.fn.float())
        pre_a, post_a, comb_a = parametrize(mixes, scale, base, cfg, constraint="sinkhorn")
        pre_c = torch.sigmoid(mixes[..., :H] * scale[0] + base[:H]) + hc.hc_eps
    for name, a, c in [("pre", pre_a, pre_c), ("post", post_a, post_c), ("comb", comb_a, comb_c)]:
        md = (a.float() - c.float()).abs().max().item()
        assert md < 1e-5, f"parametrize {name}: maxabs={md:.2e}"
