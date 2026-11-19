# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone iHC reference must track HF transformers' own ``hy_v4`` modules.

reference/ihc.py is the lightweight extract the device PCC tests validate against; this pins it
to HYV4HyperConnection / HYV4HyperHead and to the residual line of HYV4DecoderLayer.forward so the
two cannot silently drift. Pure CPU (torch vs torch), no device. Skipped when the installed
transformers predates ``hy_v4``.
"""

from types import SimpleNamespace

import pytest
import torch

from models.experimental.hy4_preview.reference.ihc import IHCConfig, IHCHead, IHCWrap, ihc_expand

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")

CASES = [(4, 64, 1.0, 2.0), (4, 128, 0.01, 2.0), (3, 96, 0.5, 1.5)]
IDS = ["H4D64s1.0", "H4D128s0.01", "H3D96s0.5m1.5"]


def _hf_config(cfg: IHCConfig):
    """The five fields the HF hyper-connection modules read from the model config."""
    return SimpleNamespace(
        hidden_size=cfg.dim, hc_mult=cfg.n, hc_eps=cfg.eps, rms_norm_eps=cfg.norm_eps, hc_magnitude=cfg.magnitude
    )


def _max_diff(a, b):
    return (a.double() - b.double()).abs().max().item()


@pytest.mark.parametrize("H, D, scale_val, magnitude", CASES, ids=IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_wrap_matches_hf(H, D, scale_val, magnitude, dtype):
    cfg = IHCConfig(dim=D, n=H, magnitude=magnitude)
    hc = hy_v4.HYV4HyperConnection(_hf_config(cfg)).eval()

    g = torch.Generator().manual_seed(1)
    fn = torch.randn(hc.fn.shape, generator=g) * 0.02
    base = torch.randn(hc.base.shape, generator=g)
    scale = torch.tensor([scale_val, 0.5 * scale_val])  # distinct, so a pre/post swap shows
    ref = IHCWrap(cfg).eval()
    with torch.no_grad():
        for m in (hc, ref):
            m.fn.copy_(fn)
            m.base.copy_(base)
            m.scale.copy_(scale)

    x = torch.randn(2, 5, H, D, generator=g).to(dtype)
    f_out = torch.randn(2, 5, D, generator=g).to(dtype)
    with torch.no_grad():
        hf_post, hf_y = hc(x)
        # HYV4DecoderLayer.forward, the line after each sublayer
        hf_new = (hf_post.float().unsqueeze(-1) * f_out.float().unsqueeze(-2) + x.float()).to(dtype)
        y, post = ref.hc_pre(x)
        new = ref.hc_post(f_out, x, post)

    assert y.dtype == hf_y.dtype and new.dtype == hf_new.dtype
    assert _max_diff(post, hf_post) == 0.0
    assert _max_diff(y, hf_y) == 0.0
    assert _max_diff(new, hf_new) == 0.0


@pytest.mark.parametrize("H, D, scale_val, magnitude", CASES, ids=IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
def test_head_matches_hf(H, D, scale_val, magnitude, dtype):
    cfg = IHCConfig(dim=D, n=H, magnitude=magnitude)
    hf_head = hy_v4.HYV4HyperHead(_hf_config(cfg)).eval()

    g = torch.Generator().manual_seed(2)
    fn = torch.randn(hf_head.hc_fn.shape, generator=g) * 0.02
    base = torch.randn(hf_head.hc_base.shape, generator=g)
    scale = torch.full((1,), float(scale_val))
    ref = IHCHead(cfg).eval()
    with torch.no_grad():
        hf_head.hc_fn.copy_(fn)
        hf_head.hc_base.copy_(base)
        hf_head.hc_scale.copy_(scale)
        ref.fn.copy_(fn)
        ref.base.copy_(base)
        ref.scale.copy_(scale)

    x = torch.randn(2, 5, H, D, generator=g).to(dtype)
    with torch.no_grad():
        hf_y, y = hf_head(x), ref(x)
    assert y.dtype == hf_y.dtype
    assert _max_diff(y, hf_y) == 0.0


def test_expand_matches_hf():
    """HYV4Model.forward: inputs_embeds.unsqueeze(2).expand(-1, -1, hc_mult, -1).contiguous()."""
    h = torch.randn(2, 5, 64)
    out = ihc_expand(h, 4)
    assert out.shape == (2, 5, 4, 64)
    assert all(torch.equal(out[:, :, i], h) for i in range(4))
