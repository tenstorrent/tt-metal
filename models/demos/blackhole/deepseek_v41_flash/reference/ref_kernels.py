# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pure-torch stand-ins for DeepSeek-V4.1-Flash's `inference/kernel.py` (tilelang, CUDA only).

Injected as the `kernel` module before importing the checkpoint's own `model.py`, so the
reference runs the model code unchanged. Signatures and rounding follow kernel.py:

  * act_quant: per-block FP8 E4M3, power-of-two (UE8M0) scales when scale_fmt is set
  * fp4_act_quant: per-block FP4 E2M1, E8M0 scales (indexer) or E4M3 scales (compressed KV)
  * fp8_gemm / fp4_gemm: dequantise both operands, accumulate in fp32
  * sparse_attn: gathers `topk_idxs` KV rows, softmax with a per-head sink logit
  * hc_split_sinkhorn: mHC coefficient split and Sinkhorn normalisation

Set ``FAKE_QUANT = False`` to skip activation rounding (unquantised reference).
"""

import torch

FAKE_QUANT = True

FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
# Midpoints between adjacent grid values; ties go to the even code, i.e. round up at 0.75, 1.75, 3.5.
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])
_FP4_TIE_UP = torch.tensor([False, True, False, True, False, True, False])
FP4_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32
)


def _pow2_ceil_scale(v: torch.Tensor) -> torch.Tensor:
    """2 ** ceil(log2(v)) computed from the exponent, like fast_round_scale (no log/ceil error)."""
    m, e = torch.frexp(v)
    return torch.ldexp(torch.ones_like(v), e - (m == 0.5).to(e.dtype))


def _round_fp4(a: torch.Tensor) -> torch.Tensor:
    """Round |a| <= 6 to the E2M1 grid, ties to even code, keep sign."""
    mag = a.abs()
    mids, tie_up, grid = _FP4_MIDS.to(a.device), _FP4_TIE_UP.to(a.device), FP4_GRID.to(a.device)
    mag = mag.unsqueeze(-1)
    idx = ((mag > mids) | ((mag == mids) & tie_up)).sum(-1)
    return torch.copysign(grid[idx], a)


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    n = x.size(-1)
    assert n % block_size == 0
    if not FAKE_QUANT:  # unquantised variant: pass values through, unit scales
        if inplace:
            return x
        ones = torch.ones(*x.shape[:-1], n // block_size, dtype=torch.float32)
        return x.float(), ones.to(scale_dtype)
    z = x.float().unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True).clamp(min=1e-4)
    s = _pow2_ceil_scale(amax / 448.0) if scale_fmt is not None else amax / 448.0
    q = (z / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    if inplace:
        deq = (q.float() * s).flatten(-2).to(x.dtype)
        x.copy_(deq)
        return x
    return q.flatten(-2), s.squeeze(-1).to(scale_dtype)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    n = x.size(-1)
    assert n % block_size == 0
    if not FAKE_QUANT:
        return x
    z = x.float().unflatten(-1, (-1, block_size))
    amax = z.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp(min=6 * 2**-9)
        s = (amax / 6.0).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp(min=6 * 2**-126)
        s = _pow2_ceil_scale(amax / 6.0)
    q = _round_fp4((z / s).clamp(-6.0, 6.0))
    assert inplace, "only the in-place form is used by model.py"
    x.copy_((q * s).flatten(-2).to(x.dtype))
    return x


def dequant_fp8_weight(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP8 weight [N, K] with per-(block x block) scales [N/block, K/block] -> fp32 [N, K]."""
    n, k = w.shape
    wf = w.float().view(n // block, block, k // block, block)
    return (wf * s.float()[:, None, :, None]).reshape(n, k)


def dequant_fp4_weight(w: torch.Tensor, s: torch.Tensor, block: int = 32) -> torch.Tensor:
    """FP4 weight packed [N, K/2] (low nibble first) with per-32 scales [N, K/32] -> fp32 [N, K]."""
    b = w.view(torch.uint8)
    pair = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    n = b.size(0)
    wf = pair.reshape(n, -1)
    return (wf.view(n, -1, block) * s.float().unsqueeze(-1)).reshape(n, -1)


def _deq_act(a, a_s, block):
    k = a.size(-1)
    af = a.float().reshape(-1, k // block, block)
    return (af * a_s.float().reshape(-1, k // block, 1)).reshape(-1, k)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    k = a.size(-1)
    out = _deq_act(a, a_s, block_size) @ dequant_fp8_weight(b, b_s, block_size).T
    return out.reshape(*a.size()[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    out = _deq_act(a, a_s, act_block_size) @ dequant_fp4_weight(b, b_s).T
    return out.reshape(*a.size()[:-1], b.size(0)).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b,m,h,d], kv [b,n,d], attn_sink [h], topk_idxs [b,m,topk] (-1 = unused) -> [b,m,h,d]."""
    b, m, h, d = q.shape
    idx = topk_idxs.long()
    valid = idx >= 0
    bi = torch.arange(b, device=q.device)[:, None, None]
    kvg = kv[bi, idx.clamp(min=0)].float()  # [b,m,topk,d]
    scores = torch.einsum("bmhd,bmkd->bmhk", q.float(), kvg) * softmax_scale
    scores = scores.masked_fill(~valid[:, :, None, :], float("-inf"))
    mx = scores.amax(-1).clamp(min=-1e30)  # kernel starts its running max at -1e30
    p = torch.exp(scores - mx.unsqueeze(-1))
    denom = p.sum(-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx)
    o = torch.einsum("bmhk,bmkd->bmhd", p, kvg) / denom.unsqueeze(-1)
    return o.to(q.dtype)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).unflatten(-1, (hc, hc))
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
