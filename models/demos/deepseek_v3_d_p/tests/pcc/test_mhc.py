# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: device mHC (models/demos/deepseek_v3_d_p/tt/mhc/tt_mhc.py) vs the pure-torch ground
truth (models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py).

Each piece is checked independently so a failure localises immediately:
    expand | project | hc_pre | hc_post | head | the wrapped sublayer, incl. real F.

Precision note. Everything mHC-specific -- the Sinkhorn parametrization and the stream mixing
-- matches fp32 torch to ~1e-7 PCC. The one lossy step is the projection matmul
mixes = RMSNorm(X) @ P: at the model width n*C = 28672 the TT fp32 matmul tops out at ~0.9989
PCC vs fp32 torch (HiFi4 + fp32 accumulation is already max fidelity -- this is the hardware,
not this code). In the real model that error is squashed by the sigmoid and the small init
a_res~0.01, so end-to-end pieces are checked at C7168 with that init and at C256 with an
aggressive a_res=1.0, isolating the projection-matmul variable from the mHC math. The
Sinkhorn's own robustness to large logits is covered by the op's nightly test.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import (
    MHCConfig,
    MHCHead,
    MHCWrap,
    mhc_expand,
    sinkhorn_knopp,
)
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import TtMHCHead, TtMHCWrap
from models.demos.deepseek_v3_d_p.tt.mhc.tt_mhc import mhc_expand as tt_mhc_expand

PCC = 0.999
PCC_PROJ = 0.998  # TT fp32-matmul ceiling at reduction depth n*C=28672 (see module docstring)

_E2E = [(256, 1.0), (7168, 0.01)]
_E2E_IDS = ["C256-s1.0", "C7168-s0.01"]


def _check(name, ref, dev, pcc=PCC):
    ref = ref.float().flatten()
    dev = dev.float().flatten()
    md = (ref - dev).abs().max().item()
    passed, val = comp_pcc(ref, dev, pcc)
    logger.info(f"{name}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"{name}: pcc={val} | max|Δ|={md:.2e} (threshold {pcc})"


def _params(cfg, scale_val, seed):
    """Shared trainable params for reference and device."""
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02
    base = torch.randn(cfg.mix_hc, generator=g)  # non-zero to exercise the biases
    scale = torch.full((3,), float(scale_val))
    return fn, base, scale


def _ref_wrap(cfg, fn, base, scale):
    wrap = MHCWrap(cfg, constraint="sinkhorn")
    wrap.fn.data, wrap.base.data, wrap.scale.data = fn, base, scale
    return wrap


def _up(device, t):
    return ttnn.from_torch(t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _up_x(device, x):
    """Reference stream tensor [1,T,n,C] -> the device packing [1,1,T,n*C].

    Row-major identical, so every output below still compares element for element flattened.
    """
    _, T, n, C = x.shape
    return _up(device, x.reshape(1, 1, T, n * C))


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
def test_mhc_expand(device, T):
    torch.manual_seed(0)
    C, n = 256, 4
    h = torch.randn(1, 1, T, C)
    r_out = mhc_expand(h.reshape(1, T, C), n)  # [1,T,n,C]
    d_out = ttnn.to_torch(tt_mhc_expand(_up(device, h), n))  # [1,1,T,n*C], same order flattened
    _check("expand", r_out, d_out)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, 7168], ids=["C256", "C7168"])
def test_project(device, T, C):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, 1.0, seed=2)
    x = torch.randn(1, T, cfg.n, C)

    xf = x.reshape(T, cfg.n * C).float()
    rsqrt = torch.rsqrt(xf.square().mean(-1, keepdim=True) + cfg.norm_eps)
    r_mixes = (F.linear(xf, fn) * rsqrt).reshape(1, 1, T, cfg.mix_hc)

    d_mixes = TtMHCWrap(device, cfg, fn, base, scale).project(_up_x(device, x))
    _check("mixes", r_mixes, ttnn.to_torch(d_mixes), PCC_PROJ)


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
def test_hc_pre(device, T, C, scale_val):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, scale_val, seed=3)
    x = torch.randn(1, T, cfg.n, C)

    r_y, r_post, r_comb = _ref_wrap(cfg, fn, base, scale).hc_pre(x)  # [1,T,C], [1,T,n], [1,T,n,n]

    d_y, d_post, d_comb = TtMHCWrap(device, cfg, fn, base, scale).hc_pre(_up_x(device, x))
    _check("y", r_y, ttnn.to_torch(d_y))
    _check("post", r_post, ttnn.to_torch(d_post))
    _check("comb", r_comb, ttnn.to_torch(d_comb))


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, 7168], ids=["C256", "C7168"])
def test_hc_post(device, T, C):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, 1.0, seed=4)
    x = torch.randn(1, T, cfg.n, C)
    f_out = torch.randn(1, T, C)
    post = 2 * torch.sigmoid(torch.randn(1, T, cfg.n))
    comb = sinkhorn_knopp(torch.randn(1, T, cfg.n, cfg.n), cfg.sinkhorn_iters, cfg.eps)

    r_out = _ref_wrap(cfg, fn, base, scale).hc_post(f_out, x, post, comb)  # [1,T,n,C]

    d_out = TtMHCWrap(device, cfg, fn, base, scale).hc_post(
        _up(device, f_out.reshape(1, 1, T, C)),
        _up_x(device, x),
        _up(device, post.reshape(1, 1, T, cfg.n)),
        _up(device, comb.reshape(1, 1, T, cfg.n * cfg.n)),
    )
    _check("out", r_out, ttnn.to_torch(d_out))


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C", [256, 7168], ids=["C256", "C7168"])
def test_hc_head(device, T, C):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    g = torch.Generator().manual_seed(5)
    fn = torch.randn(cfg.n, cfg.n * C, generator=g) * 0.02
    base = torch.randn(cfg.n, generator=g)
    scale = torch.full((1,), 0.01)
    x = torch.randn(1, T, cfg.n, C)

    head = MHCHead(cfg)
    head.fn.data, head.base.data, head.scale.data = fn, base, scale
    r_y = head(x)  # [1,T,C]

    d_y = TtMHCHead(device, cfg, fn, base, scale)(_up_x(device, x))
    _check("head_y", r_y, ttnn.to_torch(d_y))


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
@pytest.mark.parametrize("f_kind", ["identity", "linear"])
def test_mhc_wrap(device, T, C, scale_val, f_kind):
    """expand -> hc_pre -> F -> hc_post, device vs reference."""
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, scale_val, seed=7)
    x = torch.randn(1, T, cfg.n, C)

    if f_kind == "identity":  # isolates the mHC wiring from F's own arithmetic
        ref_f = dev_f = lambda z: z
    else:
        w = torch.randn(C, C, generator=torch.Generator().manual_seed(8)) * 0.02
        ref_f = lambda z: F.linear(z, w)  # z @ w.T
        w_t_tt = _up(device, w.t())
        dev_f = lambda z: ttnn.matmul(z, w_t_tt)

    ref_out = _ref_wrap(cfg, fn, base, scale)(x, ref_f)  # [1,T,n,C]
    d_out = ttnn.to_torch(TtMHCWrap(device, cfg, fn, base, scale)(_up_x(device, x), dev_f))
    _check(f"wrap[{f_kind}] C={C} T={T}", ref_out, d_out)


# ---- real sublayers F (issue #40726 acceptance: "Attention, MLP") ----
# Small C so F's own matmuls stay fp32-precise, isolating the mHC wiring.
def _mlp_fns(device, C, H, seed):
    """SwiGLU FFN: down(silu(gate(z)) * up(z))."""
    g = torch.Generator().manual_seed(seed)
    Wg, Wu, Wd = (
        torch.randn(C, H, generator=g) * 0.03,
        torch.randn(C, H, generator=g) * 0.03,
        torch.randn(H, C, generator=g) * 0.03,
    )
    ref = lambda z: (F.silu(z @ Wg) * (z @ Wu)) @ Wd
    Wg_t, Wu_t, Wd_t = _up(device, Wg), _up(device, Wu), _up(device, Wd)
    dev = lambda z: ttnn.matmul(ttnn.mul(ttnn.silu(ttnn.matmul(z, Wg_t)), ttnn.matmul(z, Wu_t)), Wd_t)
    return ref, dev


def _attn_fns(device, C, seed):
    """Single-head non-causal self-attention over the sequence."""
    g = torch.Generator().manual_seed(seed)
    Wq, Wk, Wv, Wo = (torch.randn(C, C, generator=g) * 0.03 for _ in range(4))
    s = 1.0 / (C**0.5)

    def ref(z):  # [1,T,C]
        q, k, v = z @ Wq, z @ Wk, z @ Wv
        return (torch.softmax((q @ k.transpose(-2, -1)) * s, dim=-1) @ v) @ Wo

    Wq_t, Wk_t, Wv_t, Wo_t = _up(device, Wq), _up(device, Wk), _up(device, Wv), _up(device, Wo)

    def dev(z):  # [1,1,T,C] -- the packed layout hands F the shape it already wants
        q, k, v = ttnn.matmul(z, Wq_t), ttnn.matmul(z, Wk_t), ttnn.matmul(z, Wv_t)
        scores = ttnn.softmax(ttnn.mul(ttnn.matmul(q, ttnn.transpose(k, -2, -1)), s), dim=-1)
        return ttnn.matmul(ttnn.matmul(scores, v), Wo_t)

    return ref, dev


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("f_kind", ["mlp", "attn"])
def test_mhc_wrap_real_sublayer(device, T, f_kind):
    torch.manual_seed(0)
    C = 256
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, 0.5, seed=7)
    x = torch.randn(1, T, cfg.n, C)

    if f_kind == "mlp":
        ref_f, dev_f = _mlp_fns(device, C, 2 * C, seed=8)
    else:
        if T == 1:
            pytest.skip("attention over a length-1 sequence is trivial")
        ref_f, dev_f = _attn_fns(device, C, seed=8)

    ref_out = _ref_wrap(cfg, fn, base, scale)(x, ref_f)
    d_out = ttnn.to_torch(TtMHCWrap(device, cfg, fn, base, scale)(_up_x(device, x), dev_f))
    _check(f"wrap-real[{f_kind}] T={T}", ref_out, d_out)
