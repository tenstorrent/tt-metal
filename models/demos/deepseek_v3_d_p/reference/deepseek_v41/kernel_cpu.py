# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Torch (CPU) ports of the DeepSeek-V4.1-Flash tilelang kernels.

Drop-in replacement for upstream ``inference/kernel.py`` (HF ``deepseek-ai/DeepSeek-V4.1-Flash`` at
revision dba1be0a40aa45a94ad051997016db3960a90277): same function names, signatures, return types and
numerics. Each function documents the kernel lines it mirrors (``K41:<line>`` = upstream kernel.py).

Exactness:

* ``act_quant`` / ``fp4_act_quant`` are elementwise over fixed groups, so they are bit-exact ports
  (scale rounding reproduces ``fast_round_scale``'s IEEE-754 bit manipulation; FP8/FP4 casts are
  round-to-nearest-even with saturation, like the CUDA ``satfinite`` conversions).
* ``hc_split_sinkhorn`` performs the same fp32 operations in the same order; only ``exp``/``sigmoid``
  transcendental implementations may differ from CUDA (last-ulp fp32).
* ``fp8_gemm`` / ``fp4_gemm`` accumulate each K block in fp32 (block products are exact because
  FP8 x FP8 and FP4 x FP8 products fit in fp32) and add the scaled block partials sequentially in K
  order, as the kernel does. Only the summation order inside one K block differs from the tensor core,
  which can change the fp32 result by a few ulp, i.e. at most 1 bf16 ulp of the output, rarely.
* ``sparse_attn`` reproduces the online softmax over blocks of 64 indices, including the bf16 rounding
  of the probabilities relative to the running (not final) max; remaining differences come from
  GEMM summation order and ``exp`` implementation, expected within ~1 bf16 ulp of the output.
"""

from typing import Optional, Tuple

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
# The kernels multiply by the reciprocal as a float32 constant (K41:47,138: fp8_max_inv / fp4_max_inv).
_FP8_MAX_INV = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)
_FP4_MAX_INV = torch.tensor(1.0 / FP4_MAX, dtype=torch.float32)

# E2M1 magnitudes by 3-bit code, and the sign-extended 4-bit table (same as upstream convert.py FP4_TABLE).
_E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
FP4_TABLE = torch.cat([_E2M1_GRID, -_E2M1_GRID])
# Round-to-nearest-even boundaries between consecutive codes; at an exact tie the result goes up only
# when the upper code is even, i.e. for boundaries 1, 3, 5 (0.75 -> 1.0, 1.75 -> 2.0, 3.5 -> 4.0).
_E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
_E2M1_TIE_UP = torch.tensor([False, True, False, True, False, True, False])

SPARSE_ATTN_BLOCK = 64  # K41:328 `block`


def fast_round_scale(amax: torch.Tensor, max_inv: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(amax * max_inv)) computed exactly like K41:22-38 (exponent bits + mantissa != 0)."""
    x = (amax.float() * max_inv).contiguous()
    bits = x.view(torch.int32)
    exp = (bits >> 23) & 0xFF
    man = bits & ((1 << 23) - 1)
    log2_ceil = exp - 127 + (man != 0).to(torch.int32)
    return ((log2_ceil + 127) << 23).view(torch.float32)


def _to_e4m3_satfinite(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> float8_e4m3fn, round-to-nearest-even, saturating at +-448 (torch alone maps overflow to NaN)."""
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


def _e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """fp32 in [-6, 6] -> 4-bit E2M1 codes (bit 3 = sign), round-to-nearest-even."""
    mag = x.abs().unsqueeze(-1)
    code = (mag > _E2M1_MIDS).sum(-1) + ((mag == _E2M1_MIDS) & _E2M1_TIE_UP).sum(-1)
    return code.to(torch.uint8) | (torch.signbit(x).to(torch.uint8) << 3)


def _e2m1_round(x: torch.Tensor) -> torch.Tensor:
    """fp32 in [-6, 6] -> nearest E2M1 value as fp32 (sign of zero preserved)."""
    return FP4_TABLE[_e2m1_codes(x).long()]


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    """float4_e2m1fn_x2 [..., K//2] -> fp32 [..., K]; the low nibble holds the even element."""
    b = x.view(torch.uint8)
    vals = torch.stack([FP4_TABLE[(b & 0x0F).long()], FP4_TABLE[(b >> 4).long()]], dim=-1)
    return vals.flatten(-2)


def pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """4-bit codes uint8 [..., K] -> float4_e2m1fn_x2 [..., K//2] (even element in the low nibble)."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().view(torch.float4_e2m1fn_x2)


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: Optional[str] = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
) -> torch.Tensor:
    """Block-wise FP8 quantization (K41:41-113). inplace=True does fused quant+dequant back to BF16.
    When scale_fmt is set, scales are rounded to power-of-2 (MXFP)."""
    N = x.size(-1)
    assert N % block_size == 0
    assert x.dtype == torch.bfloat16, "act_quant_kernel is compiled for bf16 input (K41:41 in_dtype=BF16)"
    xg = x.float().unflatten(-1, (-1, block_size))
    amax = xg.abs().amax(dim=-1).clamp_min(1e-4)  # K41:76-77
    if scale_fmt is not None:
        s = fast_round_scale(amax, _FP8_MAX_INV)
    else:
        s = amax * _FP8_MAX_INV
    q = _to_e4m3_satfinite(xg / s.unsqueeze(-1))  # K41:81-89: clamp to +-448, cast to e4m3
    if inplace:
        y = (q.float() * s.unsqueeze(-1)).flatten(-2).to(x.dtype)
        x.copy_(y)
        return x
    return q.flatten(-2), s.to(scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype = torch.float8_e8m0fnu,
) -> torch.Tensor:
    """FP4 with E8M0 scales for the indexer or E4M3 scales for compressed KV (K41:116-199).
    inplace=True writes the dequantized values back to x."""
    assert scale_dtype in (torch.float8_e8m0fnu, torch.float8_e4m3fn)
    N = x.size(-1)
    assert N % block_size == 0
    assert x.dtype == torch.bfloat16, "fp4_quant_kernel is compiled for bf16 input (K41:116 in_dtype=BF16)"
    xg = x.float().unflatten(-1, (-1, block_size))
    amax = xg.abs().amax(dim=-1)
    if scale_dtype == torch.float8_e4m3fn:
        # K41:158-161: keep even an all-zero group's scale nonzero; scale is amax/6 rounded to E4M3
        amax = amax.clamp_min(6 * (2**-9))
        s = _to_e4m3_satfinite(amax / FP4_MAX).float()
    else:
        amax = amax.clamp_min(6 * (2**-126))  # K41:163-164
        s = fast_round_scale(amax, _FP4_MAX_INV)
    v = (xg / s.unsqueeze(-1)).clamp(-FP4_MAX, FP4_MAX)
    if inplace:
        y = (_e2m1_round(v) * s.unsqueeze(-1)).flatten(-2).to(x.dtype)  # K41:165-170
        x.copy_(y)
        return x
    return pack_fp4(_e2m1_codes(v).flatten(-2)), s.to(scale_dtype)


def _block_scaled_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor, block_k: int):
    """sum_k (A_k @ B_k^T) * a_s[:, k] * b_s[:, k], accumulated in fp32 sequentially over K blocks.

    a: [M, K] fp32 values, a_s: [M, K//block_k] fp32, b: [N, K] fp32 values, b_s: [N, K//block_k] fp32.
    """
    M, K = a.shape
    N = b.size(0)
    acc = torch.zeros(M, N, dtype=torch.float32)
    for k in range(K // block_k):
        sl = slice(k * block_k, (k + 1) * block_k)
        partial = a[:, sl] @ b[:, sl].T
        acc += partial * a_s[:, k : k + 1] * b_s[:, k].unsqueeze(0)
    return acc


def fp8_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A[M,K] @ B[N,K]^T with per-block FP8 scaling (K41:202-293)."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    assert block_size in (32, 128)
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float8_e4m3fn
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert K % block_size == 0
    assert a_s.numel() == M * (K // block_size)
    assert b_s.shape == (
        (N + block_size - 1) // block_size,
        K // block_size,
    )
    # weight scales are per (N block, K block): expand to per output row
    b_s_rows = b_s.float().repeat_interleave(block_size, dim=0)[:N]
    c = _block_scaled_gemm(a.view(M, K).float(), a_s.view(M, -1).float(), b.float(), b_s_rows, block_size)
    return c.to(torch.get_default_dtype()).view(*a.size()[:-1], N)


def fp4_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
    act_block_size: int = 128,
) -> torch.Tensor:
    """C[M,N] = A_fp8[M,K] @ B_fp4[N,K]^T (K41:461-591).
    A has per-32 or per-128 activation scale; B has per-32 E8M0 weight scale.
    B is stored as [N, K//2] in float4_e2m1fn_x2 (2 FP4 values per byte, packed along K)."""
    assert a.is_contiguous() and b.is_contiguous(), "Input tensors must be contiguous"
    assert a_s.is_contiguous() and b_s.is_contiguous(), "Scaling factor tensors must be contiguous"
    assert a.dtype == torch.float8_e4m3fn and b.dtype == torch.float4_e2m1fn_x2
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    assert act_block_size in (32, 128)
    assert K % act_block_size == 0
    assert a_s.numel() == M * (K // act_block_size)
    assert b_s.shape == (N, K // 32)
    # the kernel walks K in blocks of 32 (the weight group); the act scale index is k // (act_block/32)
    a_s_32 = a_s.view(M, -1).float().repeat_interleave(act_block_size // 32, dim=1)
    c = _block_scaled_gemm(a.view(M, K).float(), a_s_32, unpack_fp4(b), b_s.float(), 32)
    return c.to(torch.get_default_dtype()).view(*a.size()[:-1], N)


def sparse_attn(
    q: torch.Tensor, kv: torch.Tensor, attn_sink: torch.Tensor, topk_idxs: torch.Tensor, softmax_scale: float
) -> torch.Tensor:
    """Sparse single-KV-head attention over gathered indices with an attention sink (K41:296-403).

    q [b, m, h, d] bf16, kv [b, n, d] bf16 (K = V), attn_sink [h] fp32, topk_idxs [b, m, topk] int32.
    Index -1 contributes nothing; a row with no valid index yields zeros. Output bf16 [b, m, h, d].
    """
    b, m, h, d = q.size()
    topk = topk_idxs.size(-1)
    scale = torch.tensor(softmax_scale, dtype=torch.float32)
    qf = q.float()
    kvf = kv.float()
    acc_o = torch.zeros(b, m, h, d, dtype=torch.float32)
    sum_exp = torch.zeros(b, m, h, dtype=torch.float32)
    # finite lower bound (K41:352-355): all -1 rows give exp(-inf - (-1e30)) = 0, not NaN
    scores_max = torch.full((b, m, h), -1e30, dtype=torch.float32)
    batch = torch.arange(b).view(b, 1, 1)
    for t in range(0, topk, SPARSE_ATTN_BLOCK):
        idxs = topk_idxs[:, :, t : t + SPARSE_ATTN_BLOCK].long()
        valid = idxs != -1
        kv_blk = kvf[batch, idxs.clamp_min(0)] * valid.unsqueeze(-1)  # [b, m, blk, d], invalid rows zeroed
        acc_s = torch.einsum("bmhd,bmkd->bmhk", qf, kv_blk)
        acc_s = torch.where(valid.unsqueeze(2), acc_s, -torch.inf) * scale  # K41:365-368
        scores_max_prev = scores_max
        scores_max = torch.maximum(scores_max_prev, acc_s.amax(dim=-1))
        scores_scale = torch.exp(scores_max_prev - scores_max)
        p = torch.exp(acc_s - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * scores_scale + p.sum(dim=-1)  # fp32 probabilities in the denominator
        p_bf16 = p.to(torch.bfloat16)  # K41:377: probabilities cast to bf16 before PV
        acc_o = acc_o * scores_scale.unsqueeze(-1) + torch.einsum("bmhk,bmkd->bmhd", p_bf16.float(), kv_blk)
    sum_exp = sum_exp + torch.exp(attn_sink.float() - scores_max)  # K41:382-383: sink not scaled
    return (acc_o / sum_exp.unsqueeze(-1)).to(torch.bfloat16)


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split hc mixes into pre / post / Sinkhorn-normalized comb (K41:406-458), all fp32."""
    hc = hc_mult
    m = mixes.float()
    hc_scale = hc_scale.float()
    hc_base = hc_base.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc])
    comb = m[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb.unflatten(-1, (hc, hc))
    # comb = comb.softmax(-1) + eps
    comb = torch.exp(comb - comb.amax(dim=-1, keepdim=True))
    comb = comb / comb.sum(dim=-1, keepdim=True) + eps
    # comb = comb / (comb.sum(-2) + eps)
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb
