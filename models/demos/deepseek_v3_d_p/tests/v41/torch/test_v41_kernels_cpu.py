# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU contract tests for the torch ports of the DeepSeek-V4.1 tilelang kernels (host only, no device).

Each port is checked against hand-computed values and an independent formulation (explicit dequantize +
matmul, dense masked softmax with a sink column, the PyTorch Sinkhorn formulas) rather than against a
copy of its own loop structure.
"""

import math

import torch

from models.demos.deepseek_v3_d_p.reference.deepseek_v41.kernel_cpu import (
    act_quant,
    fast_round_scale,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    sparse_attn,
    unpack_fp4,
)
from models.demos.deepseek_v3_d_p.reference.deepseek_v41.testing import quantize_fp8_blocks

INV448 = torch.tensor(1 / 448.0)
BF16_EPS = 2.0**-8  # bf16 has 8 significand bits: one ulp at 1.0 is 2^-7, half-ulp relative rounding 2^-8


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _bf16_randn(*shape: int, seed: int, scale: float = 1.0) -> torch.Tensor:
    return (torch.randn(*shape, generator=_gen(seed)) * scale).to(torch.bfloat16)


# ---------------------------------------------------------------- scale rounding / act_quant (FP8)


def test_fast_round_scale_hand_values():
    amax = torch.tensor([448.0, 449.0, 224.0, 300.0, 1e-4, 3.0])
    expected = torch.tensor([1.0, 2.0, 0.5, 1.0, 2.0**-22, 2.0**-7])  # 2^ceil(log2(amax / 448))
    assert torch.equal(fast_round_scale(amax, INV448), expected)


def test_act_quant_hand_case_ue8m0():
    x = torch.zeros(1, 32)
    x[0, :4] = torch.tensor([3.0, 0.01, -1.0, 0.3])
    x = x.to(torch.bfloat16)
    y, s = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    # amax 3 -> scale 2^ceil(log2(3/448)) = 2^-7; x/s = 384, 1.28 -> 1.25 (e4m3 step 1/8 in [1,2)), -128, 38.4 -> 40
    assert s.dtype == torch.float8_e8m0fnu and s.float().item() == 2.0**-7
    assert y.dtype == torch.float8_e4m3fn
    assert y.float()[0, :4].tolist() == [384.0, 1.25, -128.0, 40.0]
    assert torch.all(y.float()[0, 4:] == 0)
    xq = x.clone()
    assert act_quant(xq, 32, "ue8m0", torch.float8_e8m0fnu, inplace=True) is xq
    assert xq.float()[0, :4].tolist() == [3.0, 1.25 / 128, -1.0, 40.0 / 128]


def test_act_quant_hand_case_fp32_scale_and_floor():
    x = torch.zeros(2, 32, dtype=torch.bfloat16)
    x[0, 0] = 3.0
    y, s = act_quant(x, 32)  # no scale_fmt: s = amax * fp32(1/448), stored fp32
    assert s.dtype == torch.float32
    assert s[0, 0].item() == (torch.tensor(3.0) * INV448).item()
    assert y.float()[0, 0].item() == 448.0
    # all-zero block: amax floored at 1e-4, output zero
    assert s[1, 0].item() == (torch.tensor(1e-4) * INV448).item() and torch.all(y.float()[1] == 0)
    _, s_pow2 = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
    assert s_pow2.float()[1, 0].item() == 2.0**-22


def test_act_quant_round_trip_properties():
    x = _bf16_randn(3, 5, 128, seed=0, scale=4.0)
    x[0, 0, :32] *= 1e-3  # a block with a very different magnitude
    for scale_fmt, scale_dtype in ((None, torch.float32), ("ue8m0", torch.float8_e8m0fnu)):
        for block in (32, 128):
            y, s = act_quant(x, block, scale_fmt, scale_dtype)
            assert y.shape == x.shape and s.shape == (3, 5, 128 // block)
            deq = (y.float().unflatten(-1, (-1, block)) * s.float().unsqueeze(-1)).flatten(-2)
            xq = x.clone()
            act_quant(xq, block, scale_fmt, scale_dtype, inplace=True)
            assert torch.equal(xq, deq.to(torch.bfloat16))  # inplace QDQ == quantize then dequantize
            # |error| <= half an e4m3 step (2^-4 relative) of each value, plus the subnormal floor
            amax = x.float().abs().unflatten(-1, (-1, block)).amax(-1, keepdim=True)
            bound = x.float().abs().unflatten(-1, (-1, block)) * 2**-4 + amax * 2**-9 * 2**-5
            assert torch.all((deq.float() - x.float()).abs().unflatten(-1, (-1, block)) <= bound + 1e-12)
            assert torch.all(y.float().abs() <= 448)
            if scale_fmt is not None:
                again = xq.clone()
                act_quant(again, block, scale_fmt, scale_dtype, inplace=True)
                assert torch.equal(again, xq)  # QDQ is idempotent with power-of-2 scales


# ---------------------------------------------------------------- fp4_act_quant


def test_fp4_act_quant_e2m1_rne_hand_case():
    vals = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0, -0.3, -2.9, 0.1, -0.74, 4.4, -6.0, 1.0, 0.0]
    rne = [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0, -0.5, -3.0, 0.0, -0.5, 4.0, -6.0, 1.0, 0.0]
    x = torch.tensor([vals + [0.0] * 16], dtype=torch.bfloat16)  # amax 6 -> both scale kinds are exactly 1
    for block, scale_dtype in ((32, torch.float8_e8m0fnu), (16, torch.float8_e4m3fn)):
        q, s = fp4_act_quant(x, block, scale_dtype=scale_dtype)
        assert q.dtype == torch.float4_e2m1fn_x2 and q.shape == (1, 16)
        assert s.dtype == scale_dtype and s.float()[0, 0].item() == 1.0
        assert unpack_fp4(q)[0, :16].tolist() == rne
        xq = x.clone()
        fp4_act_quant(xq, block, True, scale_dtype)
        assert xq.float()[0, :16].tolist() == rne


def test_fp4_act_quant_e4m3_scale_is_not_power_of_two():
    x = torch.zeros(1, 16, dtype=torch.bfloat16)
    x[0, :3] = torch.tensor([7.0, 1.0, -3.5])
    q, s = fp4_act_quant(x, 16, scale_dtype=torch.float8_e4m3fn)
    # scale = e4m3(7/6 = 1.1667) = 1.125; 7/1.125 = 6.22 -> clamped 6; 0.889 -> 1.0; -3.11 -> -3
    assert s.float().item() == 1.125
    assert unpack_fp4(q)[0, :3].tolist() == [6.0, 1.0, -3.0]
    # ue8m0 path on the same values: scale = 2^ceil(log2(7/6)) = 2; 3.5, 0.5, -1.75 -> 4, 0.5, -2
    x32 = torch.cat([x, torch.zeros_like(x)], dim=-1)
    q, s = fp4_act_quant(x32, 32)
    assert s.float().item() == 2.0 and unpack_fp4(q)[0, :3].tolist() == [4.0, 0.5, -2.0]


def test_fp4_act_quant_zero_group_floors():
    x = torch.zeros(1, 32, dtype=torch.bfloat16)
    _, s = fp4_act_quant(x, 16, scale_dtype=torch.float8_e4m3fn)
    assert torch.all(s.float() == 2.0**-9)  # amax floor 6 * 2^-9, scale = floor / 6
    _, s = fp4_act_quant(x, 32)
    assert s.float().item() == 2.0**-126


def test_fp4_act_quant_pack_round_trip():
    x = _bf16_randn(4, 7, 64, seed=1, scale=3.0)
    for block, scale_dtype in ((32, torch.float8_e8m0fnu), (16, torch.float8_e4m3fn)):
        q, s = fp4_act_quant(x, block, scale_dtype=scale_dtype)
        deq = (unpack_fp4(q).unflatten(-1, (-1, block)) * s.float().unsqueeze(-1)).flatten(-2)
        xq = x.clone()
        fp4_act_quant(xq, block, True, scale_dtype)
        assert torch.equal(xq, deq.to(torch.bfloat16))
        # every dequantized value is an E2M1 grid point times its group scale
        grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
        ratio = (deq.unflatten(-1, (-1, block)) / s.float().unsqueeze(-1)).abs()
        assert torch.all((ratio.unsqueeze(-1) == grid).any(-1))


# ---------------------------------------------------------------- GEMMs


def _assert_bf16_close(out: torch.Tensor, ref64: torch.Tensor, min_exact: float):
    """out (bf16) vs a float64 reference: within 1 bf16 ulp everywhere, bit-equal to the rounded
    reference for at least `min_exact` of the elements (only in-block fp32 summation order differs)."""
    assert out.dtype == torch.bfloat16
    ulp = ref64.abs().clamp_min(1e-30) * 2 * BF16_EPS
    assert torch.all((out.double() - ref64).abs() <= ulp + 1e-6), (out.double() - ref64).abs().max()
    exact = (out == ref64.to(torch.bfloat16)).double().mean().item()
    assert exact >= min_exact, exact


def test_fp8_gemm_matches_dequantized_matmul():
    torch.set_default_dtype(torch.bfloat16)  # the model runs with bf16 default dtype (output dtype)
    try:
        for block in (32, 128):
            M, K, N = 37, 256, 80  # N not a block multiple: last weight scale row is partial
            a = _bf16_randn(M, K, seed=2, scale=2.0)
            a_q, a_s = act_quant(a, block, "ue8m0", torch.float8_e8m0fnu)
            w = torch.randn(N, K, generator=_gen(3)) * K**-0.5
            b_q, b_s = quantize_fp8_blocks(w, block)
            out = fp8_gemm(a_q, a_s, b_q, b_s, torch.float8_e8m0fnu, block)
            a_deq = a_q.double() * a_s.double().repeat_interleave(block, dim=-1)
            b_deq = b_q.double() * b_s.double().repeat_interleave(block, 0)[:N].repeat_interleave(block, 1)
            _assert_bf16_close(out, a_deq @ b_deq.T, min_exact=0.99)
            # leading batch dims are preserved
            out3 = fp8_gemm(a_q.view(1, M, K), a_s.view(1, M, -1), b_q, b_s, torch.float8_e8m0fnu, block)
            assert out3.shape == (1, M, N) and torch.equal(out3[0], out)
    finally:
        torch.set_default_dtype(torch.float32)


def test_fp4_gemm_matches_dequantized_matmul():
    torch.set_default_dtype(torch.bfloat16)
    try:
        for act_block in (32, 128):
            M, K, N = 19, 256, 48
            a = _bf16_randn(M, K, seed=4)
            a_q, a_s = act_quant(a, act_block, "ue8m0", torch.float8_e8m0fnu)
            b_q, b_s = fp4_act_quant(_bf16_randn(N, K, seed=5, scale=K**-0.5), 32)
            out = fp4_gemm(a_q, a_s, b_q, b_s, torch.float8_e8m0fnu, act_block)
            a_deq = a_q.double() * a_s.double().repeat_interleave(act_block, dim=-1)
            b_deq = unpack_fp4(b_q).double() * b_s.double().repeat_interleave(32, dim=-1)
            _assert_bf16_close(out, a_deq @ b_deq.T, min_exact=0.99)
    finally:
        torch.set_default_dtype(torch.float32)


def test_fp4_unpack_nibble_order():
    raw = torch.tensor([[0x21, 0xF7]], dtype=torch.uint8).view(torch.float4_e2m1fn_x2)
    # low nibble first: 0x1 -> 0.5, 0x2 -> 1.0, 0x7 -> 6.0, 0xF -> -6.0 (convert.py FP4_TABLE order)
    assert unpack_fp4(raw).tolist() == [[0.5, 1.0, 6.0, -6.0]]


# ---------------------------------------------------------------- sparse_attn


def _dense_sparse_attn(q, kv, sink, idxs, scale):
    """Independent formulation: gather the selected rows, dense softmax in float64 over
    [logits, sink] (sink unscaled), drop the sink column, weighted sum of the same rows."""
    b, m, h, d = q.shape
    out = torch.zeros(b, m, h, d, dtype=torch.float64)
    for bi in range(b):
        for t in range(m):
            sel = idxs[bi, t][idxs[bi, t] >= 0].long()
            rows = kv[bi, sel].double()  # [k, d], duplicates kept
            logits = q[bi, t].double() @ rows.T * scale  # [h, k]
            full = torch.cat([logits, sink.double().unsqueeze(-1)], dim=-1)
            p = full.softmax(dim=-1)[:, :-1]
            out[bi, t] = p @ rows
    return out


def test_sparse_attn_matches_dense_softmax_with_sink():
    b, m, h, d, n, topk = 2, 5, 3, 64, 200, 150  # topk > 64: several online-softmax blocks
    q = _bf16_randn(b, m, h, d, seed=6)
    kv = _bf16_randn(b, n, d, seed=7)
    sink = torch.randn(h, generator=_gen(8))
    idxs = torch.randint(0, n, (b, m, topk), generator=_gen(9), dtype=torch.int32)
    idxs[:, :, ::3] = -1  # invalid slots interleaved
    idxs[0, 1, 70:] = -1  # a whole trailing block invalid
    idxs[1, 2, :5] = 17  # duplicates count every time
    scale = d**-0.5
    out = sparse_attn(q, kv, sink, idxs, scale)
    assert out.dtype == torch.bfloat16 and out.shape == q.shape
    ref = _dense_sparse_attn(q, kv, sink, idxs, scale)
    # bf16 output (|out| <~ 1.2: half ulp 2^-9) plus bf16-rounded probabilities: observed max 3.0e-3.
    # Scaling the sink by softmax_scale would give 8.5e-3 here, so the bound is sensitive to that error.
    torch.testing.assert_close(out.double(), ref, atol=4e-3, rtol=0)


def test_sparse_attn_hand_cases():
    d = 32
    q = torch.zeros(1, 2, 2, d, dtype=torch.bfloat16)
    kv = torch.arange(3 * d, dtype=torch.float32).view(1, 3, d).to(torch.bfloat16)
    sink = torch.tensor([0.0, math.log(3.0)])
    idxs = torch.tensor([[[1, -1, -1], [-1, -1, -1]]], dtype=torch.int32)
    out = sparse_attn(q, kv, sink, idxs, 0.125)
    # one valid row with logit 0: weight 1 / (1 + e^sink) -> 1/2 for head 0, 1/4 for head 1
    torch.testing.assert_close(out[0, 0, 0].float(), kv[0, 1].float() / 2, atol=0, rtol=2**-8)
    torch.testing.assert_close(out[0, 0, 1].float(), kv[0, 1].float() / 4, atol=0, rtol=2**-8)
    # no valid index at all: finite running-max floor gives exact zeros, not NaN
    assert torch.equal(out[0, 1], torch.zeros(2, d, dtype=torch.bfloat16))
    # an empty top-k list behaves the same
    empty = sparse_attn(q, kv, sink, idxs[..., :0], 0.125)
    assert torch.equal(empty, torch.zeros_like(q))


# ---------------------------------------------------------------- hc_split_sinkhorn


def test_hc_split_sinkhorn_hand_case():
    # hc=2, eps=0: comb logits [[0, ln3], [ln3, 0]] softmax to [[1/4, 3/4], [3/4, 1/4]], already doubly
    # stochastic, so Sinkhorn leaves it; zero pre/post logits give sigmoid 1/2 -> pre 0.5, post 1.0
    mixes = torch.zeros(1, 1, 8)
    base = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, math.log(3.0), math.log(3.0), 0.0])
    pre, post, comb = hc_split_sinkhorn(mixes, torch.ones(3), base, hc_mult=2, sinkhorn_iters=20, eps=0.0)
    assert pre.tolist() == [[[0.5, 0.5]]] and post.tolist() == [[[1.0, 1.0]]]
    torch.testing.assert_close(comb, torch.tensor([[[[0.25, 0.75], [0.75, 0.25]]]]), atol=1e-6, rtol=0)
    # scale applies per section: mixes * hc_scale[i] + base
    mixes = torch.tensor([[[1.0, -1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    pre, post, _ = hc_split_sinkhorn(mixes, torch.tensor([0.0, 0.5, 1.0]), torch.zeros(8), 2, 20, 0.0)
    assert pre.tolist() == [[[0.5, 0.5]]]
    torch.testing.assert_close(post, 2 * torch.sigmoid(torch.tensor([[[1.0, 0.0]]])))


def test_hc_split_sinkhorn_properties_and_formula():
    hc, iters, eps = 4, 20, 1e-6
    mixes = torch.randn(2, 7, (2 + hc) * hc, generator=_gen(10)) * 3
    scale = torch.tensor([0.7, 1.3, 2.0])
    base = torch.randn((2 + hc) * hc, generator=_gen(11))
    pre, post, comb = hc_split_sinkhorn(mixes, scale, base, hc, iters, eps)
    assert pre.shape == (2, 7, hc) and post.shape == (2, 7, hc) and comb.shape == (2, 7, hc, hc)
    assert pre.dtype == post.dtype == comb.dtype == torch.float32
    assert torch.all((pre > eps) & (pre < 1 + eps)) and torch.all((post > 0) & (post < 2))
    torch.testing.assert_close(comb.sum(-2), torch.ones(2, 7, hc), atol=1e-5, rtol=0)  # last step: columns
    # rows converge only approximately in 20 iterations; for peaked inputs like these (|row err| ~2e-2)
    # that is the algorithm, not the port. Check rows on moderate mixes instead.
    _, _, comb_moderate = hc_split_sinkhorn(mixes / 3, torch.tensor([0.7, 1.3, 1.0]), base, hc, iters, eps)
    torch.testing.assert_close(comb_moderate.sum(-1), torch.ones(2, 7, hc), atol=1e-3, rtol=0)
    # the PyTorch formulation the kernel comments describe, in float64
    m, s, bs = mixes.double(), scale.double(), base.double()
    c = (m[..., 2 * hc :] * s[2] + bs[2 * hc :]).unflatten(-1, (hc, hc)).softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    torch.testing.assert_close(comb.double(), c, atol=1e-6, rtol=0)
    torch.testing.assert_close(pre.double(), torch.sigmoid(m[..., :hc] * s[0] + bs[:hc]) + eps, atol=1e-6, rtol=0)
