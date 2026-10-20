# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``TtMHCFast`` (tt/mhc.py) vs the DeepSeek composite ``TtMHCWrap`` on real residual streams.

Streams: rows of ``bringup/references/glm53_flash/R4097/layer4_prefill.pt`` (``x_in``, the HF input
of layer 4 at 4097 real tokens), parameters: the layer-4 attention and FFN hyper-connections of the
checkpoint. Every stage is compared at T = 1, 2, 32, 256, 1024 (decode B=1/2/32 and prefill shapes):
``project``, ``hc_pre`` (h, post, comb), ``hc_post``, the whole ``forward`` through a bf16
sublayer and ``forward`` with the fused input RMSNorm (vs the composite + typecast + rms_norm); bar
PCC >= 0.9999 plus the rel-rms delta, and both against a float64 reference of the HF math (PCC >=
0.999: the TT fp32 matmul rounds its operands to bf16). The second test checks the pad-row contract
(T < 32): finite garbage in the input's pad rows must not change the real rows.

    TT_VISIBLE_DEVICES=1 pytest models/demos/glm53_flash/tests/test_mhc.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.glm53_flash.reference.checkpoint import default_snapshot_dir, load_layer_raw, shard_headers
from models.demos.glm53_flash.tests.utils import pcc
from models.demos.glm53_flash.tt.config import GLM53FlashConfig
from models.demos.glm53_flash.tt.mhc import TtMHCFast, TtMHCWrap, build_mhc, mhc_kernel_config

DUMP = Path("bringup/references/glm53_flash/R4097/layer4_prefill.pt")
LAYER = 4
PCC_BAR = 0.9999  # fast vs composite (the block-diagonal matmuls round their operands to bf16, see TtMHCFast)
PCC_REF = 0.999
SHAPES = [1, 2, 32, 256, 1024]


def _sinkhorn(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def torch_reference(x: torch.Tensor, fn, base, scale, kcfg):
    """HF ``Glm5NextTextHyperConnection`` math in fp32 on packed ``[T, n*C]`` streams.
    Returns mixes [T,24], h [T,C], post [T,n], comb [T,n,n]."""
    n, C = kcfg.n, kcfg.dim
    xf = x.float()
    rs = torch.rsqrt(xf.square().mean(-1, keepdim=True) + kcfg.norm_eps)
    mixes = (xf @ fn.float().t()) * rs
    b = base.float()
    pre = torch.sigmoid(mixes[:, :n] * float(scale[0]) + b[:n]) + kcfg.eps
    post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * float(scale[1]) + b[n : 2 * n])
    comb = _sinkhorn(
        (mixes[:, 2 * n :] * float(scale[2]) + b[2 * n :]).reshape(-1, n, n), kcfg.sinkhorn_iters, kcfg.eps
    )
    h = (pre[:, :, None] * xf.reshape(-1, n, C)).sum(1)
    return mixes, h, post, comb


def torch_post(y: torch.Tensor, res: torch.Tensor, post: torch.Tensor, comb: torch.Tensor, n: int) -> torch.Tensor:
    """new_j = post_j * y + sum_i comb[i,j] res_i -> packed [T, n*C]."""
    T = y.shape[0]
    r = res.float().reshape(T, n, -1)
    out = post[:, :, None] * y.float()[:, None, :] + torch.einsum("tij,tic->tjc", comb, r)
    return out.reshape(T, -1)


def _up(device, t: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _down(t: ttnn.Tensor) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def _cmp(name: str, ref: torch.Tensor, got: torch.Tensor, bar: float) -> float:
    ref, got = ref.double().reshape(-1), got.double().reshape(-1)
    p = pcc(ref, got)
    d = ref - got
    rel_rms = (d.norm() / ref.norm().clamp_min(1e-30)).item()
    rel_max = (d.abs().max() / ref.abs().max().clamp_min(1e-30)).item()
    logger.info(f"    {name}: pcc {p:.7f} rel rms {rel_rms:.2e} max|d|/max|ref| {rel_max:.2e}")
    assert p >= bar, f"{name}: pcc {p} < {bar}"
    return p


def _streams(T: int, offset: int = 0) -> torch.Tensor:
    x_in = torch.load(DUMP)["x_in"]  # [1, 4097, 4, 4096] bf16
    assert offset + T <= x_in.shape[1]
    return x_in[0, offset : offset + T].reshape(T, -1).float()  # packed [T, n*C]


def _build(device, cfg, raw, which: str):
    fn, base, scale = raw[f"hc_{which}_fn"].float(), raw[f"hc_{which}_base"].float(), raw[f"hc_{which}_scale"].float()
    return (
        build_mhc(device, cfg, fn, base, scale, impl="wrap"),
        build_mhc(device, cfg, fn, base, scale, impl="fast"),
        (fn, base, scale),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("which", ["attn", "ffn"])
def test_mhc_fast_vs_wrap(mesh_device, which):
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    wrap, fast, (fn, base, scale) = _build(mesh_device, cfg, raw, which)
    assert isinstance(wrap, TtMHCWrap) and isinstance(fast, TtMHCFast)
    n, C = kcfg.n, kcfg.dim
    g = torch.Generator().manual_seed(0)
    for T in SHAPES:
        x = _streams(T, offset=100)
        logger.info(f"hc_{which} T={T}: |x| max {x.abs().max():.3f} rms {x.pow(2).mean().sqrt():.4f}")
        r_mixes, r_h, r_post, r_comb = torch_reference(x, fn, base, scale, kcfg)
        xd = _up(mesh_device, x.reshape(1, 1, T, n * C))

        # project
        m_w = _down(wrap.project(xd))[0, 0]  # [T, 24]
        m_f = _down(fast.project(xd))[0, 0]  # decode: [T, 32] with zero columns past 24; prefill: [T, 24]
        assert m_f.shape[0] == T and m_f.shape[1] in (24, 32), m_f.shape
        assert m_f.shape[1] == 24 or float(m_f[:, 24:].abs().max()) == 0.0
        _cmp("mixes fast vs wrap", m_w, m_f[:, :24], PCC_BAR)
        _cmp("mixes fast vs torch", r_mixes, m_f[:, :24], PCC_REF)

        # hc_pre
        h_w, post_w, comb_w = wrap.hc_pre(xd)
        h_f, post_f, comb_f = fast.hc_pre(xd)
        assert tuple(h_f.shape) == (1, 1, T, C) and tuple(post_f.shape) == (1, 1, T, n)
        assert tuple(comb_f.shape) == (1, 1, T, n * n)
        h_w_t, h_f_t = _down(h_w)[0, 0], _down(h_f)[0, 0]
        _cmp("h fast vs wrap", h_w_t, h_f_t, PCC_BAR)
        _cmp("h fast vs torch", r_h, h_f_t, PCC_REF)
        post_w_t, post_f_t = _down(post_w)[0, 0], _down(post_f)[0, 0]
        comb_w_t, comb_f_t = _down(comb_w)[0, 0], _down(comb_f)[0, 0]
        _cmp("post fast vs wrap", post_w_t, post_f_t, PCC_BAR)
        _cmp("comb fast vs wrap", comb_w_t, comb_f_t, PCC_BAR)
        _cmp("post fast vs torch", r_post, post_f_t, PCC_REF)
        _cmp("comb fast vs torch", r_comb.reshape(T, -1), comb_f_t, PCC_REF)
        # a doubly-stochastic comb: columns sum to 1 (the Sinkhorn ends on a column normalisation; the
        # fused kernel's tile matmuls leave ~1e-3, the same for both implementations)
        col_sums = comb_f_t.reshape(T, n, n).sum(1)
        assert (col_sums - 1).abs().max() < 5e-3, col_sums

        # hc_post on the same inputs: y = a bf16-rounded sublayer-like output (the collapsed stream
        # with a random per-column gain, so it has the sublayer's magnitude and is not a stream)
        y = (h_w_t * (0.5 + torch.rand(1, C, generator=g))).to(torch.bfloat16).float()
        yd = _up(mesh_device, y.reshape(1, 1, T, C))
        o_w = _down(wrap.hc_post(yd, xd, post_w, comb_w))[0, 0]
        o_f = _down(fast.hc_post(yd, xd, post_w, comb_w))[0, 0]  # the same (wrap) coefficients: isolates the mixing
        assert tuple(o_f.shape) == (T, n * C)
        _cmp("hc_post fast vs wrap (same coefficients)", o_w, o_f, PCC_BAR)
        r_out = torch_post(y, x, post_w_t, comb_w_t.reshape(T, n, n), n)
        _cmp("hc_post fast vs torch (same coefficients)", r_out, o_f, PCC_REF)

        # whole wrap through a bf16 sublayer (a fixed per-column gain, like the real path's typecasts)
        gain = _up(mesh_device, (0.5 + torch.rand(1, 1, 1, C, generator=g)))
        gain_bf = ttnn.typecast(gain, ttnn.bfloat16)

        def sublayer(t):
            return ttnn.multiply(t, gain_bf)

        f_w = _down(wrap.forward(xd, sublayer, sublayer_dtype=ttnn.bfloat16))[0, 0]
        f_f = _down(fast.forward(xd, sublayer, sublayer_dtype=ttnn.bfloat16))[0, 0]
        _cmp("forward fast vs wrap", f_w, f_f, PCC_BAR)
        r_fwd = torch_post(
            (r_h.to(torch.bfloat16).float() * _down(gain_bf).reshape(1, C)).to(torch.bfloat16).float(),
            x,
            r_post,
            r_comb,
            n,
        )
        _cmp("forward fast vs torch", r_fwd, f_f, PCC_REF)

        # the decoder's call: the sublayer's input RMSNorm fused into the pre-block (bf16 out) vs the
        # composite path (typecast + interleaved rms_norm inside the sublayer closure)
        w_norm = ttnn.from_torch(
            (0.8 + 0.4 * torch.rand(1, 1, 1, C, generator=g)),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
        )
        eps = float(cfg.rms_norm_eps)
        hs = {}

        def normed_sublayer(t):
            hs["wrap"] = _down(t)
            hn = ttnn.rms_norm(t, epsilon=eps, weight=w_norm, compute_kernel_config=wrap.ckc)
            hs["wrap_n"] = _down(hn)
            return ttnn.multiply(hn, gain_bf)

        def fused_sublayer(t):
            hs["fast_n"] = _down(t)
            return ttnn.multiply(t, gain_bf)

        f_w = _down(wrap.forward(xd, normed_sublayer, sublayer_dtype=ttnn.bfloat16))[0, 0]
        f_f = _down(fast.forward(xd, fused_sublayer, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps)))[0, 0]
        _cmp("normed hidden fast vs wrap", hs["wrap_n"][0, 0], hs["fast_n"][0, 0], PCC_BAR)
        r_hn = r_h * torch.rsqrt(r_h.pow(2).mean(-1, keepdim=True) + eps) * _down(w_norm).reshape(1, C)
        _cmp("normed hidden wrap vs torch", r_hn, hs["wrap_n"][0, 0], PCC_REF)
        _cmp("normed hidden fast vs torch", r_hn, hs["fast_n"][0, 0], PCC_REF)
        _cmp("forward+norm fast vs wrap", f_w, f_f, PCC_BAR)
        if T <= 32:
            # collapse mode: the head's unweighted mean over the n new streams, straight from the post-block
            c_f = _down(
                fast.forward(xd, fused_sublayer, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, eps), collapse=True)
            )
            assert tuple(c_f.shape) == (1, 1, T, C), c_f.shape
            _cmp("collapse fast vs mean of the wrap's streams", f_w.reshape(T, n, C).mean(1), c_f[0, 0], PCC_BAR)
        ttnn.deallocate(xd)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mhc_fast_pad_rows(mesh_device):
    """At T < 32 the fast path widens the logical rows to the tile and its block-diagonal matmuls
    contract over the pad rows too, so the contract is: finite pad rows in -> the real rows do not
    depend on them (bit-exact) and every output row is finite."""
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    _, fast, _ = _build(mesh_device, cfg, raw, "attn")
    n, C = cfg.hc_mult, cfg.hidden_size
    g = torch.Generator().manual_seed(1)
    w_norm = ttnn.from_torch(torch.ones(1, 1, 1, C), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    for T in (1, 4):
        x = _streams(T, offset=500)
        xd = _up(mesh_device, x.reshape(1, 1, T, n * C))
        x32 = torch.zeros(32, n * C)
        x32[:T] = x
        x32[T:] = torch.randn(32 - T, n * C, generator=g) * 3.0  # finite garbage in the pad rows
        xg = ttnn.reshape(
            _up(mesh_device, x32.reshape(1, 1, 32, n * C)), ttnn.Shape([1, 1, T, n * C]), ttnn.Shape([1, 1, 32, n * C])
        )
        seen = []

        def sublayer(t):  # the whole 32-row tile of the hidden, incl. its pad rows
            seen.append(
                ttnn.to_torch(ttnn.reshape(t, ttnn.Shape([1, 1, 32, C]), ttnn.Shape([1, 1, 32, C])))[0, 0].float()
            )
            return ttnn.multiply(t, 1.0)

        full = lambda t: ttnn.to_torch(ttnn.reshape(t, ttnn.Shape([1, 1, 32, n * C]), ttnn.Shape([1, 1, 32, n * C])))[
            0, 0
        ].float()
        o = full(fast.forward(xd, sublayer, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, float(cfg.rms_norm_eps))))
        og = full(fast.forward(xg, sublayer, sublayer_dtype=ttnn.bfloat16, norm=(w_norm, float(cfg.rms_norm_eps))))
        for name, t in (
            ("hidden (clean)", seen[0]),
            ("hidden (garbage pad)", seen[1]),
            ("out", o),
            ("out (garbage pad)", og),
        ):
            assert torch.isfinite(t).all(), f"{name}: non-finite rows"
        logger.info(
            f"T={T}: clean vs garbage-pad output rows equal: {torch.equal(o[:T], og[:T])}; "
            f"hidden rows equal: {torch.equal(seen[0][:T], seen[1][:T])}; all rows finite"
        )
        assert torch.equal(seen[0][:T], seen[1][:T]), "garbage pad rows leaked into the real hidden rows"
        assert torch.equal(o[:T], og[:T]), "garbage pad rows leaked into the real output rows"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_decoder_collapse(mesh_device):
    """``FunctionalDecoder.decode_forward(collapse=True)`` (the last layer feeding the LM head) returns
    the mean over the 4 new streams of the plain decode step: layer 4, B=2, fresh KDA state for each
    call, real token rows; the head's own ``collapse`` of the streams is the reference."""
    from models.demos.glm53_flash.tt.functional_decoder import FunctionalDecoder, host_streams_to_device
    from models.demos.glm53_flash.tt.model_head import TtModelHead

    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    B = 2
    dec = FunctionalDecoder.from_state_dict(
        raw, hf_config=cfg, layer_idx=LAYER, mesh_device=mesh_device, max_batch=B, max_seq_len=256
    )
    assert dec.layer.ffn_hc.supports_collapse
    x_in = torch.load(DUMP)["x_in"][:, 200 : 200 + B]  # [1, B, 4, 4096]
    dp = dec.make_decode_positions([100] * B)
    streams = dec.decode_forward(host_streams_to_device(x_in, mesh_device, pad_to=B), positions=dp)
    ref = TtModelHead.collapse_streams(streams, cfg.hc_mult)  # the head's 8-op mean over the streams
    ref_h = _down(ref)[0, 0]
    dec.kda_batched = dec.layer.attn.allocate_state(B)  # the step above advanced the state: start again
    h = dec.decode_forward(host_streams_to_device(x_in, mesh_device, pad_to=B), positions=dp, collapse=True)
    assert tuple(h.shape) == (1, 1, B, cfg.hidden_size), h.shape
    _cmp("decode_forward(collapse=True) vs head.collapse(streams)", ref_h, _down(h)[0, 0], PCC_BAR)


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_mhc_exact_residual(mesh_device):
    """``GLM53_MHC_EXACT`` (TtMHCFast(exact=)): the residual update ``new_j = post_j y + sum_i comb[i,j] x_i``
    on fp32 streams that are NOT bf16-exact (the dump rows perturbed by 1e-3 relative noise, like the
    residual after layer 0) vs the composite path's exact fp32 chain: exact=0 (one HiFi4 matmul) ~1e-3
    rel rms, exact=2 (2-way bf16 split, 3 products) ~3e-6, exact=3 (3-way, 6 products) <= 1e-6."""
    snapshot = default_snapshot_dir()
    cfg = GLM53FlashConfig.from_hf_config_json(snapshot / "config.json")
    kcfg = mhc_kernel_config(cfg)
    raw = load_layer_raw(snapshot, LAYER, shard_headers(snapshot))
    fn, base, scale = raw["hc_ffn_fn"].float(), raw["hc_ffn_base"].float(), raw["hc_ffn_scale"].float()
    wrap = TtMHCWrap(mesh_device, kcfg, fn, base, scale)
    fasts = {e: TtMHCFast(mesh_device, kcfg, fn, base, scale, exact=e) for e in (0, 2, 3)}
    n, C = kcfg.n, kcfg.dim
    g = torch.Generator().manual_seed(2)
    bars = {0: 5e-3, 2: 1e-5, 3: 1e-6}
    for T in (4, 32):
        x = _streams(T, offset=300)
        x = x + torch.randn(x.shape, generator=g) * 1e-3 * x.abs()  # fp32, not bf16-exact
        xd = _up(mesh_device, x.reshape(1, 1, T, n * C))
        _, post_w, comb_w = wrap.hc_pre(xd)
        y = (_streams(T, offset=900)[:, :C] * 0.5).to(torch.bfloat16).float()
        yd = _up(mesh_device, y.reshape(1, 1, T, C))
        o_w = _down(wrap.hc_post(yd, xd, post_w, comb_w))[0, 0]
        c_w = o_w.reshape(T, n, C).mean(1)
        for e, fast in fasts.items():
            o_f = _down(fast.hc_post(yd, xd, post_w, comb_w))[0, 0]
            d = (o_w.double() - o_f.double()).norm() / o_w.double().norm()
            logger.info(f"T={T} exact={e}: residual update vs composite rel rms {d:.3e}")
            assert d <= bars[e], (e, d)
            c_f = _down(fast.hc_post(yd, xd, post_w, comb_w, collapse=True))[0, 0]
            dc = (c_w.double() - c_f.double()).norm() / c_w.double().norm()
            logger.info(f"T={T} exact={e}: collapse vs composite mean rel rms {dc:.3e}")
            assert dc <= bars[e], (e, dc)
