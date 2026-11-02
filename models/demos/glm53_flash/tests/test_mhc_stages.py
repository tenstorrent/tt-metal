# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage error table of ``TtMHCFast`` against the composite ``TtMHCWrap`` and a float64 reference on real
layer-4 streams, at the prefill shape T=128 and the decode shape T=4 (the accuracy-gate shapes): mixes (the
Sinkhorn input), pre / post / comb (the Sinkhorn outputs of each implementation's own mixes), h (the sublayer
input before the norm), the normalised bf16 sublayer input, the post-block on identical coefficients, and the
whole wrap. Numbers are rel rms in float64; the table goes to the log (and ``_logs/optmhc_stages.md`` when
``GLM53_STAGES_OUT`` names a file).

    GLM53_MHC_POST_KERNEL=2 TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/test_mhc_stages.py -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.test_mhc import _build, _down, _sinkhorn, _streams, _up
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import mhc_kernel_config

LAYER = 4
SHAPES = [4, 128]


def reference64(x: torch.Tensor, fn, base, scale, kcfg):
    """The HF hyper-connection math in float64: mixes [T,24], pre [T,n], post [T,n], comb [T,n*n], h [T,C]."""
    n, C = kcfg.n, kcfg.dim
    xd = x.double()
    rs = torch.rsqrt(xd.square().mean(-1, keepdim=True) + kcfg.norm_eps)
    mixes = (xd @ fn.double().t()) * rs
    b = base.double()
    pre = torch.sigmoid(mixes[:, :n] * float(scale[0]) + b[:n]) + kcfg.eps
    post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * float(scale[1]) + b[n : 2 * n])
    comb = _sinkhorn(
        (mixes[:, 2 * n :] * float(scale[2]) + b[2 * n :]).reshape(-1, n, n), kcfg.sinkhorn_iters, kcfg.eps
    )
    h = (pre[:, :, None] * xd.reshape(-1, n, C)).sum(1)
    return mixes, pre, post, comb.reshape(-1, n * n), h


def _rel(ref: torch.Tensor, got: torch.Tensor) -> float:
    ref, got = ref.double().reshape(-1), got.double().reshape(-1)
    return ((ref - got).norm() / ref.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("which", ["attn", "ffn"])
def test_mhc_stage_table(mesh_device, which):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    wrap, fast, (fn, base, scale) = _build(mesh_device, cfg, raw, which)
    n, C = kcfg.n, kcfg.dim
    g = torch.Generator().manual_seed(0)
    lines = []
    for T in SHAPES:
        x = _streams(T, offset=100)
        r_mixes, r_pre, r_post, r_comb, r_h = reference64(x, fn, base, scale, kcfg)
        xd = _up(mesh_device, x.reshape(1, 1, T, n * C))
        rows = []  # (stage, fast vs wrap, wrap vs f64, fast vs f64)

        m_w = _down(wrap.project(xd))[0, 0][:, :24]
        m_f = _down(fast.project(xd))[0, 0][:, :24]
        rows.append(("mixes (Sinkhorn input)", _rel(m_w, m_f), _rel(r_mixes, m_w), _rel(r_mixes, m_f)))

        # the Sinkhorn kernel on each implementation's own mixes -> pre / post / comb
        def sink(mx):
            md = _up(mesh_device, mx.reshape(T, 24))
            pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
                md, wrap.consts, n, wrap.iters, wrap.eps
            )
            return _down(pre)[:, :n], _down(post)[:, :n], _down(comb)[:, : n * n]

        pre_w, post_w, comb_w = sink(m_w)
        pre_f, post_f, comb_f = sink(m_f)
        rows.append(("pre", _rel(pre_w, pre_f), _rel(r_pre, pre_w), _rel(r_pre, pre_f)))
        rows.append(("post", _rel(post_w, post_f), _rel(r_post, post_w), _rel(r_post, post_f)))
        rows.append(("comb", _rel(comb_w, comb_f), _rel(r_comb, comb_w), _rel(r_comb, comb_f)))

        h_w, post_wd, comb_wd = wrap.hc_pre(xd)
        h_f, post_fd, comb_fd = fast.hc_pre(xd)
        h_w_t, h_f_t = _down(h_w)[0, 0], _down(h_f)[0, 0]
        rows.append(("h = sum_i pre_i x_i", _rel(h_w_t, h_f_t), _rel(r_h, h_w_t), _rel(r_h, h_f_t)))
        # h on identical coefficients (isolates the mix arithmetic): torch64 with the wrap's pre
        r_h_w = (pre_w.double()[:, :, None] * x.double().reshape(T, n, C)).sum(1)
        rows.append(
            (
                "h vs f64 on own pre",
                float("nan"),
                _rel(r_h_w, h_w_t),
                _rel((pre_f.double()[:, :, None] * x.double().reshape(T, n, C)).sum(1), h_f_t),
            )
        )

        # the post-block on identical (wrap) coefficients and a bf16 sublayer-like output
        y = (h_w_t * (0.5 + torch.rand(1, C, generator=g))).to(torch.bfloat16).float()
        yd = _up(mesh_device, y.reshape(1, 1, T, C))
        o_w = _down(wrap.hc_post(yd, xd, post_wd, comb_wd))[0, 0]
        o_f = _down(fast.hc_post(yd, xd, post_wd, comb_wd))[0, 0]
        post_wt, comb_wt = _down(post_wd)[0, 0].double(), _down(comb_wd)[0, 0].double().reshape(T, n, n)
        r_o = (
            post_wt[:, :, None] * y.double()[:, None, :]
            + torch.einsum("tij,tic->tjc", comb_wt, x.double().reshape(T, n, C))
        ).reshape(T, n * C)
        rows.append(("post-block (same coefficients)", _rel(o_w, o_f), _rel(r_o, o_w), _rel(r_o, o_f)))

        # the whole wrap with the fused / closure RMSNorm through a bf16 sublayer (per-column gain)
        gain_bf = ttnn.typecast(_up(mesh_device, 0.5 + torch.rand(1, 1, 1, C, generator=g)), ttnn.bfloat16)
        w_norm = ttnn.from_torch(
            0.8 + 0.4 * torch.rand(1, 1, 1, C, generator=g),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        eps = float(cfg.rms_norm_eps)
        hs = {}

        def normed_sublayer(t):
            hn = ttnn.rms_norm(t, epsilon=eps, weight=w_norm, compute_kernel_config=wrap.ckc)
            hs["wrap_n"] = _down(hn)[0, 0]
            return ttnn.multiply(hn, gain_bf)

        def fused_sublayer(t):
            hs["fast_n"] = _down(t)[0, 0]
            return ttnn.multiply(t, gain_bf)

        f_w = _down(wrap.forward(xd, normed_sublayer, sublayer_dtype=ttnn.bfloat16))[0, 0]
        f_f = _down(fast.forward(xd, fused_sublayer, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps)))[0, 0]
        r_hn = r_h * torch.rsqrt(r_h.pow(2).mean(-1, keepdim=True) + eps) * _down(w_norm).reshape(1, C).double()
        rows.append(
            (
                "normed hidden (bf16 sublayer input)",
                _rel(hs["wrap_n"], hs["fast_n"]),
                _rel(r_hn, hs["wrap_n"]),
                _rel(r_hn, hs["fast_n"]),
            )
        )
        yb = (hs["wrap_n"].double() * _down(gain_bf).reshape(1, C).double()).to(torch.bfloat16).double()
        r_fw = (
            post_wt[:, :, None] * yb[:, None, :] + torch.einsum("tij,tic->tjc", comb_wt, x.double().reshape(T, n, C))
        ).reshape(T, n * C)
        rows.append(("wrap output (streams), same sublayer", _rel(f_w, f_f), _rel(r_fw, f_w), float("nan")))

        hdr = f"\n### hc_{which}, T={T}: rel rms (float64)\n| stage | fast vs wrap | wrap vs f64 | fast vs f64 |\n|---|---|---|---|"
        body = "\n".join(f"| {s} | {a:.2e} | {b:.2e} | {c:.2e} |" for s, a, b, c in rows)
        logger.info(hdr + "\n" + body)
        lines.append(hdr + "\n" + body)
        ttnn.deallocate(xd)
    out = os.environ.get("GLM53_STAGES_OUT")
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
