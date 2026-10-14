# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU port of DeepSeek-V4.1's GPU kernels (checkpoint ``inference/kernel.py``, tilelang), same signatures.

The HF oracle installs this module as ``kernel`` under DeepSeek's own ``inference/model.py``; the CPU reference calls
the same functions, so both quantize activations exactly as the GPU kernels do:

* ``act_quant``: per 32 activations, amax (at least 1e-4), scale = 2^ceil(log2(amax * fp32(1/448))) (ue8m0, the
  kernel's bit-level ``fast_round_scale``), value = e4m3(clamp(x / scale, +-448)) (round to nearest even);
  ``inplace`` writes value * scale back (quantize-dequantize).
* ``fp4_act_quant`` (in place only, all the model uses): per 16 or 32 values, e2m1 (round to nearest even on the
  grid 0, .5, 1, 1.5, 2, 3, 4, 6) times either a ue8m0 scale 2^ceil(log2(amax * fp32(1/6))), amax >= 6 * 2^-126
  (indexer q / k), or an e4m3 scale e4m3(amax / 6), amax >= 6 * 2^-9 (compressed KV, groups of 16).
* ``fp8_gemm`` / ``fp4_gemm``: both operands dequantized (exact in fp32: e4m3 or e2m1 times a power of two), fp32
  GEMMs in fixed tiles (see "GEMM tiling"), output in torch's default dtype (the kernels accumulate in fp32 too; only
  the summation order differs).
* ``sparse_attn``: per query, softmax over its selected KV rows (index -1 = none) plus a per-head sink logit that
  only adds to the denominator; KV rows are both key and value. With bf16 inputs P is rounded to bf16 before P @ V,
  as the kernel does; with fp32 inputs everything stays fp32.
* ``hc_split_sinkhorn``: mHC pre / post / comb (row softmax + eps, column normalization, iters - 1 row / column
  rounds), fp32.

The arithmetic runs in fp32 throughout (the kernels' compute dtype); a bf16 input is widened first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP8_MAX = 448.0
FP4_MAX = 6.0
_INV_FP8_MAX = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)  # the kernels' fp32 constant 1/448
_INV_FP4_MAX = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)
_FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
_FP4_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


# --------------------------------------------------------------------------------------------- rounding helpers
def pow2_ceil(t: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(t)) for positive normal fp32 t, from the exponent bits (the kernel's fast_log2_ceil + fast_pow2)."""
    bits = t.contiguous().view(torch.int32)
    exp = (bits >> 23) & 0xFF
    k = exp - 127 + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return ((k + 127) << 23).view(torch.float32)


def round_fp4(v: torch.Tensor) -> torch.Tensor:
    """fp32 in [-6, 6] -> nearest e2m1 value, ties to the even code (cudaRoundNearest), sign kept."""
    a = v.abs()
    idx = torch.bucketize(a, _FP4_MIDS)  # a == mids[j] -> j (the lower neighbour)
    tie = (idx < len(_FP4_MIDS)) & (a == _FP4_MIDS[idx.clamp(max=len(_FP4_MIDS) - 1)])
    idx = torch.where(tie & (idx % 2 == 1), idx + 1, idx)  # even grid index = even mantissa
    return torch.copysign(_FP4_GRID[idx], v)


def round_e4m3(v: torch.Tensor) -> torch.Tensor:
    """fp32 -> e4m3fn value (round to nearest even, saturating at +-448, as __NV_SATFINITE), as fp32."""
    return v.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float()


def _groups(x: torch.Tensor, block: int) -> torch.Tensor:
    n = x.size(-1)
    assert n % block == 0, f"last dim {n} is not a multiple of {block}"
    return x.float().reshape(*x.shape[:-1], n // block, block)


# --------------------------------------------------------------------------------------------- activation quant
def act_quant_scale(xg: torch.Tensor) -> torch.Tensor:
    """ue8m0 scale per group (last dim) of fp32 values xg [..., G, block] -> [..., G, 1]."""
    amax = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    return pow2_ceil(amax * _INV_FP8_MAX)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Block-wise fp8 quantization of the last dim. inplace: quantize-dequantize into x (returns x); else returns
    (e4m3 values, scales [..., N / block_size] in scale_dtype)."""
    xg = _groups(x, block_size)
    if scale_fmt is not None:
        s = act_quant_scale(xg)
    else:
        s = xg.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4) * _INV_FP8_MAX
    y = (xg / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).reshape(x.shape).to(x.dtype))
        return x
    return y.reshape(x.shape), s.squeeze(-1).to(scale_dtype)


def act_quant_dequant(x: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """What act_quant(..., 'ue8m0') followed by a dequantizing GEMM sees: fp32 values e4m3(x / s) * s."""
    xg = _groups(x, block_size)
    s = act_quant_scale(xg)
    return ((xg / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).float() * s).reshape(x.shape)


def fp4_quant_dequant(x: torch.Tensor, block_size: int = 32, e4m3_scale: bool = False) -> torch.Tensor:
    """fp32 values e2m1(clamp(x / s, +-6)) * s per group of block_size (see fp4_act_quant)."""
    xg = _groups(x, block_size)
    amax = xg.abs().amax(dim=-1, keepdim=True)
    if e4m3_scale:
        s = round_e4m3(amax.clamp_min(FP4_MAX * 2.0**-9) / FP4_MAX)
    else:
        s = pow2_ceil(amax.clamp_min(FP4_MAX * 2.0**-126) * _INV_FP4_MAX)
    return (round_fp4((xg / s).clamp(-FP4_MAX, FP4_MAX)) * s).reshape(x.shape)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """fp4 quantize-dequantize in place (e8m0 scales: indexer; e4m3 scales: compressed KV)."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    if not inplace:
        raise NotImplementedError("the model only uses fp4_act_quant in place")
    x.copy_(fp4_quant_dequant(x, block_size, scale_dtype == torch.float8_e4m3fn).to(x.dtype))
    return x


# --------------------------------------------------------------------------------------------- GEMM tiling
# Every GEMM of the model runs in fixed tiles, in the oracle and in the reference alike: a row's result then depends
# only on that row (never on how many rows a call has, i.e. on the chunk), and the two agree to the last bit. That
# matters here because the model's fp8 / fp4 quantizers turn a last-bit difference into a whole quantization step.
# (MKL's per-row sgemm results were measured independent of M for M >= 32 on the expert shapes; full tiles keep M
# fixed anyway, and no call runs below ROW_PAD rows.)
ROW_BLOCK = 256  # rows per dense GEMM call (a shorter last block zero-padded to a multiple of ROW_PAD)
ROW_PAD = (
    32  # a GEMM never runs with fewer rows than this or a non-multiple of it (K-24: MKL kernels depend on M below)
)
EXPERT_ROW_BLOCK = 32  # routed-expert token groups are padded to a multiple of this (K-24) and run in one call
INDEX_Q_BLOCK = 64  # queries per index-score GEMM (zero-padded)
INDEX_K_BLOCK = 512  # keys per index-score GEMM (zero-padded)


def _pad_rows(x: torch.Tensor, n: int) -> torch.Tensor:
    return x if x.shape[0] == n else torch.cat([x, x.new_zeros(n - x.shape[0], *x.shape[1:])])


def _padded(n: int) -> int:
    return n + (-n % ROW_PAD)


def linear(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """F.linear in ROW_BLOCK-row GEMMs over the flattened leading dims (the last block padded to ROW_PAD rows)."""
    assert bias is None
    lead = x.shape[:-1]
    x2 = x.reshape(-1, x.shape[-1])
    s = x2.shape[0]
    out = x2.new_empty(s, w.shape[0], dtype=torch.promote_types(x.dtype, w.dtype))
    for r in range(0, s, ROW_BLOCK):
        xb = x2[r : r + ROW_BLOCK]
        n = xb.shape[0]
        out[r : r + n] = F.linear(_pad_rows(xb, _padded(n)), w)[:n]
    return out.reshape(*lead, w.shape[0])


def expert_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """F.linear of one expert's token group, padded to a multiple of EXPERT_ROW_BLOCK rows, one call."""
    n = x.shape[0]
    return F.linear(_pad_rows(x, n + (-n % EXPERT_ROW_BLOCK)), w)[:n]


def grouped_out(o: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """einsum('sgd,grd->sgr') (the grouped wo_a) in ROW_BLOCK-row calls as linear(); o [S, g, d], w [g, r, d]."""
    s = o.shape[0]
    out = o.new_empty(s, w.shape[0], w.shape[1], dtype=torch.promote_types(o.dtype, w.dtype))
    for r in range(0, s, ROW_BLOCK):
        ob = o[r : r + ROW_BLOCK]
        n = ob.shape[0]
        out[r : r + n] = torch.einsum("sgd,grd->sgr", _pad_rows(ob, _padded(n)), w)[:n]
    return out


def index_score_tile(qb: torch.Tensor, keys: torch.Tensor, k0: int) -> torch.Tensor:
    """q.k for one tile: qb [INDEX_Q_BLOCK, h, d] (padded), keys[k0 : k0 + INDEX_K_BLOCK] -> [Q, h, nk]."""
    kb = keys[k0 : k0 + INDEX_K_BLOCK]
    nk = kb.shape[0]
    return torch.matmul(qb, _pad_rows(kb, INDEX_K_BLOCK).t())[..., :nk]


def index_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """einsum('bshd,btd->bsht') in fixed (INDEX_Q_BLOCK x INDEX_K_BLOCK) tiles."""
    b, s, h, d = q.shape
    t = k.shape[1]
    out = q.new_empty(b, s, h, t, dtype=torch.promote_types(q.dtype, k.dtype))
    for bi in range(b):
        for q0 in range(0, s, INDEX_Q_BLOCK):
            qb = q[bi, q0 : q0 + INDEX_Q_BLOCK]
            n = qb.shape[0]
            qb = _pad_rows(qb, INDEX_Q_BLOCK)
            for k0 in range(0, t, INDEX_K_BLOCK):
                tile = index_score_tile(qb, k[bi], k0)
                out[bi, q0 : q0 + n, :, k0 : k0 + tile.shape[-1]] = tile[:n]
    return out


def einsum(eq: str, *ops):
    """torch.einsum with the model's two GEMM-like contractions tiled (installed as model.py's torch.einsum)."""
    if eq == "bsgd,grd->bsgr":
        o, w = ops
        return torch.stack([grouped_out(ob, w) for ob in o])
    if eq == "bshd,btd->bsht":
        return index_scores(*ops)
    return torch.einsum(eq, *ops)


# --------------------------------------------------------------------------------------------- GEMMs
def _dequant_act(a: torch.Tensor, a_s: torch.Tensor, block: int) -> torch.Tensor:
    k = a.size(-1)
    return (a.float().reshape(-1, k // block, block) * a_s.float().reshape(-1, k // block, 1)).reshape(-1, k)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size: int = 128) -> torch.Tensor:
    """C[M, N] = A[M, K] @ B[N, K]^T with per-block fp8 scales (activation per row and K block, weight per block)."""
    from models.demos.deepseek_v41_flash_d_p.reference.weights import fp8_block_dequant

    assert block_size == 32, "the checkpoint uses 32 x 32 blocks"
    w = fp8_block_dequant(b, b_s, torch.float32)
    c = linear(_dequant_act(a, a_s, block_size), w)
    return c.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size: int = 128) -> torch.Tensor:
    """C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T; B stored [N, K / 2] (2 e2m1 per byte, low nibble first) with an e8m0
    scale per 32 along K."""
    from models.demos.deepseek_v41_flash_d_p.reference.weights import fp4_dequant

    w = getattr(b, "dequant", None)  # dequantized ahead by weights.ExpertPrefetcher (same values)
    if w is None:
        w = fp4_dequant(b.view(torch.uint8), b_s, torch.float32)
    c = expert_linear(_dequant_act(a, a_s, act_block_size), w)
    return c.reshape(*a.shape[:-1], b.size(0)).to(torch.get_default_dtype())


# --------------------------------------------------------------------------------------------- attention
def attend(q: torch.Tensor, kv: torch.Tensor, valid: torch.Tensor, sink: torch.Tensor, scale: float) -> torch.Tensor:
    """q [B, H, D], kv [B, T, D] (key = value), valid [B, T] bool, sink [H] fp32 -> [B, H, D] fp32.

    The kernel's softmax: m = max(-1e30, valid scores), p = exp(s - m), out = (P @ V) / (sum p + exp(sink - m)),
    P rounded to bf16 before P @ V when the inputs are bf16."""
    s = torch.matmul(q.float(), kv.float().transpose(-1, -2)) * torch.tensor(scale, dtype=torch.float32)
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    m = s.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    p = torch.exp(s - m)
    denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink.float()[None, :, None] - m)
    if q.dtype == torch.bfloat16:
        p = p.to(torch.bfloat16).float()
    return torch.matmul(p, kv.float()) / denom


SPARSE_Q_BLOCK = 64


def sparse_attn(q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale):
    """q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, topk] int32 (-1 = none) -> [b, m, h, d] in q's dtype."""
    b, m, h, d = q.shape
    out = torch.empty_like(q)
    for bi in range(b):
        for t0 in range(0, m, SPARSE_Q_BLOCK):
            t1 = min(m, t0 + SPARSE_Q_BLOCK)
            idx = topk_idxs[bi, t0:t1].long()
            valid = idx >= 0
            rows = kv[bi][idx.clamp_min(0)]  # [B, topk, d]
            out[bi, t0:t1] = attend(q[bi, t0:t1], rows, valid, attn_sink, softmax_scale).to(q.dtype)
    return out


# --------------------------------------------------------------------------------------------- mHC
def sinkhorn_split(mixes, hc_scale, hc_base, hc: int = 4, iters: int = 20, eps: float = 1e-6):
    """mixes [..., (2 + hc) * hc] fp32 -> pre [..., hc], post [..., hc], comb [..., hc, hc] (rows: input stream j,
    columns: output stream k), exactly the kernel's sequence of fp32 operations."""
    mixes = mixes.float()
    sc, base = hc_scale.float(), hc_base.float()
    pre = torch.sigmoid(mixes[..., :hc] * sc[0] + base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc : 2 * hc] * sc[1] + base[hc : 2 * hc])
    comb = mixes[..., 2 * hc :].unflatten(-1, (hc, hc)) * sc[2] + base[2 * hc :].view(hc, hc)
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6):
    """Kernel signature: mixes [b, s, (2 + hc) * hc] -> (pre [b, s, hc], post [b, s, hc], comb [b, s, hc, hc])."""
    return sinkhorn_split(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
