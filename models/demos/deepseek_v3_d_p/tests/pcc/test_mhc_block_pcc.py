# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: mHC-wrapped block (kernel parametrization + composite computation, tt/mhc_block.py)
vs reference MHCWrap.forward (models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py).

Exercises the full X' = H_res@X + H_post.T@F(H_pre@X) around an arbitrary sublayer F.
"""

import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.reference.mhc.mhc_reference import MHCConfig, MHCWrap
from models.demos.deepseek_v3_d_p.tt.mhc.mhc_block import TtMHCBlock

PCC = 0.999

# C256 at aggressive a_res=1.0 (projection precise); C7168 at init a_res~0.01 -- isolates the
# projection-matmul precision from the mHC math (same rationale as the composite tests).
_E2E = [(256, 1.0), (7168, 0.01)]
_E2E_IDS = ["C256-s1.0", "C7168-s0.01"]


def _params(cfg, scale_val, seed):
    g = torch.Generator().manual_seed(seed)
    fn = torch.randn(cfg.mix_hc, cfg.n * cfg.dim, generator=g) * 0.02
    base = torch.randn(cfg.mix_hc, generator=g)
    scale = torch.full((3,), float(scale_val))
    return fn, base, scale


def _ref_wrap(cfg, fn, base, scale):
    wrap = MHCWrap(cfg, constraint="sinkhorn")
    wrap.fn.data, wrap.base.data, wrap.scale.data = fn, base, scale
    return wrap


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("C, scale_val", _E2E, ids=_E2E_IDS)
@pytest.mark.parametrize("f_kind", ["identity", "linear"])
def test_mhc_block(device, T, C, scale_val, f_kind):
    torch.manual_seed(0)
    cfg = MHCConfig(dim=C, n=4)
    fn, base, scale = _params(cfg, scale_val, seed=7)
    x = torch.randn(1, T, cfg.n, C)

    if f_kind == "identity":
        ref_f = lambda z: z
        dev_f = lambda z: z
    else:
        w = torch.randn(C, C, generator=torch.Generator().manual_seed(8)) * 0.02
        ref_f = lambda z: F.linear(z, w)  # z @ w.T
        w_t_tt = ttnn.from_torch(w.t().contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
        dev_f = lambda z: ttnn.matmul(z, w_t_tt)

    ref_out = _ref_wrap(cfg, fn, base, scale)(x, ref_f)  # [1,T,n,C]

    block = TtMHCBlock(device, cfg, fn, base, scale)
    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    d_out = ttnn.to_torch(block(x_tt, dev_f))

    ref = ref_out.float().flatten()
    dev = d_out.float().flatten()
    md = (ref - dev).abs().max().item()
    passed, val = comp_pcc(ref, dev, PCC)
    logger.info(f"block[{f_kind}] C={C} T={T}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"block[{f_kind}]: pcc={val} | max|Δ|={md:.2e}"


# ---- real sublayers F (issue #40726 acceptance: "Attention, MLP") ----
# Small C so F's own matmuls stay fp32-precise, isolating the mHC wiring.
def _up(device, w):
    return ttnn.from_torch(w.contiguous(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)


def _to_seq(z):  # [1,T,1,C] -> [1,1,T,C] so F can contract over the sequence
    T, C = z.shape[1], z.shape[3]
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(ttnn.reshape(z, [1, 1, T, C]), ttnn.TILE_LAYOUT)


def _from_seq(z, T):  # [1,1,T,C] -> [1,T,1,C]
    C = z.shape[3]
    z = ttnn.to_layout(z, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.to_layout(ttnn.reshape(z, [1, T, 1, C]), ttnn.TILE_LAYOUT)


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

    def dev(z):  # [1,T,1,C]
        T = z.shape[1]
        z2 = _to_seq(z)
        q, k, v = ttnn.matmul(z2, Wq_t), ttnn.matmul(z2, Wk_t), ttnn.matmul(z2, Wv_t)
        scores = ttnn.softmax(ttnn.mul(ttnn.matmul(q, ttnn.transpose(k, -2, -1)), s), dim=-1)
        return _from_seq(ttnn.matmul(ttnn.matmul(scores, v), Wo_t), T)

    return ref, dev


@pytest.mark.parametrize("T", [1, 32], ids=["T1", "T32"])
@pytest.mark.parametrize("f_kind", ["mlp", "attn"])
def test_mhc_block_real_sublayer(device, T, f_kind):
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

    block = TtMHCBlock(device, cfg, fn, base, scale)
    x_tt = ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.float32)
    d_out = ttnn.to_torch(block(x_tt, dev_f))

    ref = ref_out.float().flatten()
    dev = d_out.float().flatten()
    md = (ref - dev).abs().max().item()
    passed, val = comp_pcc(ref, dev, PCC)
    logger.info(f"block-real[{f_kind}] T={T}: pcc={val} | max|Δ|={md:.2e}")
    assert passed, f"block-real[{f_kind}] T={T}: pcc={val} | max|Δ|={md:.2e}"
