# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for mhc_post (immutable spec — the implementer must not edit it).

Phase 0 cells (plain mode, float32 streams + float32 sublayer, separate
coefficients, any T) are always exercised. Cells that later refinements add
(bf16 streams / sublayer, packed mixes, stats, collapse) are exercised as
soon as the op's SUPPORTED declares them, and skipped until then.

Reference: pure torch (float64), DeepSeek-V4 Block.hc_post / hc step.
Tolerances: PCC floors per dtype (float32 0.999, bfloat16 0.995) plus the
golden suite's relative-RMS and signed-bias gates (eval/golden_tests/mhc_post/
helpers.py) — the same numbers, not stricter.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.mhc_post import mhc_post, SUPPORTED, default_compute_kernel_config
from ttnn.operations._op_contract import UnsupportedAxisValue

N = 4
NG = N * (N + 2)

PCC = {ttnn.float32: 0.999, ttnn.bfloat16: 0.995}
# Golden-suite gates (helpers.py TOLERANCES / SIGNED_BIAS_FLOOR / BIAS_SIGMAS).
REL_RMS = {ttnn.float32: 2e-6, ttnn.bfloat16: 4e-3}
BIAS_FLOOR = {ttnn.float32: 1e-6, ttnn.bfloat16: 1e-5}
BIAS_SIGMAS = 6.0
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}


# ---------------------------------------------------------------------------
# Reference (pure torch, float64)
# ---------------------------------------------------------------------------


def ref_mix(f, x, post, comb):
    """X'_j = post_j * F + sum_i comb[i][j] * X_i  (comb applied transposed)."""
    lead, C = tuple(f.shape[:-1]), f.shape[-1]
    ff = f.double().reshape(-1, 1, C)
    xx = x.double().reshape(-1, N, C)
    p = post.double().reshape(-1, N, 1)
    m = comb.double().reshape(-1, N, N)
    return (p * ff + torch.einsum("tij,tic->tjc", m, xx)).reshape(*lead, N * C)


def ref_hc(row, *, scale, base, norm_width, norm_eps, hc_eps, iters):
    """DeepSeek-V4 hc step (hc_split_sinkhorn_kernel after the projection), float64."""
    row = row.double()
    mix, ss = row[..., :NG], row[..., NG : NG + 1]
    b = torch.tensor(base, dtype=torch.float64)
    a_pre, a_post, a_res = scale
    z = mix * torch.rsqrt(ss / norm_width + norm_eps)
    pre = torch.sigmoid(z[..., :N] * a_pre + b[:N]) + hc_eps
    post = 2 * torch.sigmoid(z[..., N : 2 * N] * a_post + b[N : 2 * N])
    m = (z[..., 2 * N :] * a_res + b[2 * N :]).reshape(*z.shape[:-1], N, N)
    m = torch.softmax(m, dim=-1) + hc_eps
    m = m / (m.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + hc_eps)
        m = m / (m.sum(dim=-2, keepdim=True) + hc_eps)
    return torch.cat([pre, post, m.reshape(*m.shape[:-2], N * N)], dim=-1)


def sinkhorn(logits, iters=20, eps=1e-6):
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


# ---------------------------------------------------------------------------
# Inputs / checks
# ---------------------------------------------------------------------------


def make_inputs(x_shape, *, dtype, sublayer_dtype, comb_mode="sinkhorn"):
    torch.manual_seed(42)
    lead, nc = tuple(x_shape[:-1]), x_shape[-1]
    C = nc // N
    f = torch.randn(*lead, C).to(TORCH_DTYPE[sublayer_dtype])
    x = torch.randn(*lead, nc).to(TORCH_DTYPE[dtype])
    post = (2.0 * torch.sigmoid(torch.randn(*lead, N))).float()
    if comb_mode == "sinkhorn":
        comb = sinkhorn(torch.randn(*lead, N, N))
    elif comb_mode == "cyclic":  # comb[i][(i+1)%n] = 1 -> X'_j carries X_{j-1}
        comb = torch.roll(torch.eye(N), shifts=1, dims=1).expand(*lead, N, N)
    else:
        raise ValueError(comb_mode)
    comb = comb.reshape(*lead, N * N).float().contiguous()
    return f, x, post, comb


def dev(t, device, dtype=ttnn.float32):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def cfg(fidelity=ttnn.MathFidelity.HiFi4):
    return ttnn.ComputeConfigDescriptor(math_fidelity=fidelity, fp32_dest_acc_en=True, math_approx_mode=False)


def pcc(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    if torch.equal(a, b):
        return 1.0
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def check_residual(out, expected64, *, dtype, shape):
    assert list(out.shape) == list(shape), f"shape {list(out.shape)} != {list(shape)}"
    assert out.dtype == dtype, f"dtype {out.dtype} != {dtype}"
    assert out.layout == ttnn.TILE_LAYOUT
    got = ttnn.to_torch(out).double()
    exp_rounded = expected64.to(TORCH_DTYPE[dtype]).double()
    p = pcc(got, exp_rounded)
    assert p >= PCC[dtype], f"PCC {p} < {PCC[dtype]}"
    rel = (got - exp_rounded).pow(2).mean().sqrt().item() / max(exp_rounded.std().item(), 1e-30)
    assert rel <= REL_RMS[dtype], f"relative RMS {rel:.3g} > {REL_RMS[dtype]}"
    d = ((got - expected64) * expected64.sign()).flatten()
    scale = expected64.abs().mean().clamp_min(1e-30)
    bias = (d.mean() / scale).item()
    se = (d.std() / d.numel() ** 0.5 / scale).item()
    lim = BIAS_FLOOR[dtype] + BIAS_SIGMAS * se
    assert abs(bias) <= lim, f"signed relative bias {bias:+.3g} > {lim:.3g} (shrinks over depth)"
    return got


def need(**axes):
    for axis, value in axes.items():
        if value not in SUPPORTED.get(axis, []):
            pytest.skip(f"{axis}={value} not in SUPPORTED yet")


# (F, X) shapes by X's shape (last dim n*C).
SHAPES = [
    (32, 4 * 32),  # single tile per stream, rank 2
    (64, 4 * 256),  # multi-tile, rank 2
    (1, 128, 4 * 1024),  # non-square, rank 3
    (1, 2, 256, 4 * 2048),  # multi-batch, rank 4
    (1, 1, 160, 4 * 5120),  # DeepSeek-V4.1 Galaxy rows (5 token-tile rows): 2-D split
    (1, 1, 640, 4 * 1792),  # DeepSeek-V4 TP4
    (17, 4 * 128),  # T not a multiple of 32
    (1, 1, 150, 4 * 5120),  # ragged, wide
    (1, 4 * 7168),  # T = 1 decode token: column-only split
    (1, 1, 1024, 4 * 64),  # narrow C, many rows: row-dominant split
]
IDS = ["x".join(map(str, s)) for s in SHAPES]


# ---------------------------------------------------------------------------
# Phase 0: plain, float32 / float32, separate coefficients
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("x_shape", SHAPES, ids=IDS)
def test_plain_fp32(x_shape, device):
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    out = mhc_post(dev(f, device), dev(x, device), dev(post, device), dev(comb, device))
    check_residual(out, ref_mix(f, x, post, comb), dtype=ttnn.float32, shape=x_shape)


def test_default_config_is_hifi4_fp32_dest():
    c = default_compute_kernel_config()
    assert c.fp32_dest_acc_en is True
    assert c.math_fidelity == ttnn.MathFidelity.HiFi4
    assert c.math_approx_mode is False


@pytest.mark.parametrize("x_shape", [(1, 1, 64, 4 * 256), (1, 1, 150, 4 * 640)], ids=["T64", "T150"])
def test_cyclic_comb_orientation(x_shape, device):
    """comb[i][(i+1)%n] = 1: X'_j = post_j F + X_{j-1}. Applying comb instead of comb^T fails."""
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32, comb_mode="cyclic")
    out = mhc_post(dev(f, device), dev(x, device), dev(post, device), dev(comb, device), compute_kernel_config=cfg())
    check_residual(out, ref_mix(f, x, post, comb), dtype=ttnn.float32, shape=x_shape)


def test_column_offsets_separate(device):
    """Engram-add form: post read at column 12 of a 32-wide row, comb at column 16 of
    another 32-wide row (identity comb). Offsets are read in-kernel."""
    x_shape = (1, 1, 96, 4 * 512)
    f, x, post, _ = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    lead = x_shape[:-1]
    eye = torch.eye(N).expand(*lead, N, N).reshape(*lead, N * N).float()
    post_row = torch.randn(*lead, 32) * 10
    post_row[..., 12:16] = post
    comb_row = torch.randn(*lead, 32) * 10
    comb_row[..., 16:32] = eye
    out = mhc_post(
        dev(f, device),
        dev(x, device),
        dev(post_row.float(), device),
        dev(comb_row.float(), device),
        post_col=12,
        comb_col=16,
    )
    check_residual(out, ref_mix(f, x, post, eye), dtype=ttnn.float32, shape=x_shape)


def test_max_grid_same_bits(device):
    """Every core inside (0,0)..(2,1); X' bit-identical to the whole-grid run."""
    x_shape = (1, 1, 160, 4 * 1024)
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    args = (dev(f, device), dev(x, device), dev(post, device), dev(comb, device))
    whole = ttnn.to_torch(mhc_post(*args))
    capped = mhc_post(*args, max_grid=ttnn.CoreCoord(3, 2))
    check_residual(capped, ref_mix(f, x, post, comb), dtype=ttnn.float32, shape=x_shape)
    assert torch.equal(ttnn.to_torch(capped).view(torch.int32), whole.view(torch.int32))


def test_output_tensor_and_determinism(device):
    """Result lands in the caller's buffer; two calls give identical bits."""
    x_shape = (1, 1, 128, 4 * 768)
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    args = (dev(f, device), dev(x, device), dev(post, device), dev(comb, device))
    buf = dev(torch.full(x_shape, 1.0e4), device)
    out = mhc_post(*args, output_tensor=buf)
    assert out.buffer_address() == buf.buffer_address()
    check_residual(buf, ref_mix(f, x, post, comb), dtype=ttnn.float32, shape=x_shape)
    again = mhc_post(*args)
    assert torch.equal(ttnn.to_torch(again).view(torch.int32), ttnn.to_torch(buf).view(torch.int32))


def test_refuses_fp32_dest_off(device, expect_error):
    x_shape = (32, 4 * 32)
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    bad = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, math_approx_mode=False
    )
    with expect_error(UnsupportedAxisValue, ""):
        mhc_post(dev(f, device), dev(x, device), dev(post, device), dev(comb, device), compute_kernel_config=bad)


def test_collapse_pre_without_weight_errors(device, expect_error):
    x_shape = (32, 4 * 32)
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    with expect_error(ValueError, ""):
        mhc_post(dev(f, device), dev(x, device), dev(post, device), dev(comb, device), collapse_pre=dev(post, device))


def test_caller_buffer_wrong_shape_refused(device, expect_error):
    x_shape = (32, 4 * 64)
    f, x, post, comb = make_inputs(x_shape, dtype=ttnn.float32, sublayer_dtype=ttnn.float32)
    with expect_error(Exception, ""):
        mhc_post(
            dev(f, device),
            dev(x, device),
            dev(post, device),
            dev(comb, device),
            output_tensor=dev(torch.zeros(32, 4 * 32), device),
        )


# ---------------------------------------------------------------------------
# Refinement cells — active once SUPPORTED declares them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype,sublayer_dtype",
    [(ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32)],
    ids=["fp32X_bf16F", "bf16X_bf16F", "bf16X_fp32F"],
)
@pytest.mark.parametrize("x_shape", [(1, 1, 160, 4 * 5120), (1, 77, 4 * 2560)], ids=["T160", "T77"])
def test_plain_mixed_dtypes(x_shape, dtype, sublayer_dtype, device):
    need(dtype=dtype, sublayer_dtype=sublayer_dtype)
    f, x, post, comb = make_inputs(x_shape, dtype=dtype, sublayer_dtype=sublayer_dtype)
    out = mhc_post(dev(f, device, sublayer_dtype), dev(x, device, dtype), dev(post, device), dev(comb, device))
    check_residual(out, ref_mix(f, x, post, comb), dtype=dtype, shape=x_shape)


def _packed(x_shape, dtype):
    f, x, post, comb = make_inputs(x_shape, dtype=dtype, sublayer_dtype=dtype)
    lead = x_shape[:-1]
    g = torch.Generator().manual_seed(7)
    pre = (torch.sigmoid(torch.randn(*lead, N, generator=g)) + 1e-6).float()
    mixes = torch.cat([pre, post, comb], dim=-1).contiguous()
    return f, x, post, comb, pre, mixes


def test_plain_packed(device):
    need(coef_source="packed")
    x_shape = (1, 1, 150, 4 * 1024)
    dtype = ttnn.bfloat16 if ttnn.float32 not in SUPPORTED["dtype"] else ttnn.float32
    f, x, post, comb, _, mixes = _packed(x_shape, dtype)
    m = dev(mixes, device)
    out = mhc_post(dev(f, device, dtype), dev(x, device, dtype), m, m, post_col=N, comb_col=2 * N)
    check_residual(out, ref_mix(f, x, post, comb), dtype=dtype, shape=x_shape)


def _weight(nc):
    g = torch.Generator().manual_seed(11)
    w = torch.zeros(nc, 32)
    raw = torch.randn(nc, NG, generator=g) * nc**-0.5
    i = raw.contiguous().view(torch.int32)  # round to 9 mantissa bits (exact on the FPU)
    sh = 23 - 9
    i = ((i + (1 << (sh - 1)) - 1 + ((i >> sh) & 1)) >> sh) << sh
    w[:, :NG] = i.view(torch.float32)
    return w


def _hc_scalars(nc):
    g = torch.Generator().manual_seed(13)
    return dict(
        scale=[float(v) for v in torch.rand(3, generator=g) * 0.5 + 0.1],
        base=[float(v) for v in torch.randn(NG, generator=g) * 0.5],
        norm_width=float(nc),
        norm_eps=1e-6,
        hc_eps=1e-6,
        iters=20,
    )


def _rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("mode", ["stats", "collapse"])
@pytest.mark.parametrize("x_shape", [(1, 1, 160, 4 * 5120), (1, 1, 150, 4 * 640)], ids=["T160", "T150"])
def test_fused_modes(x_shape, mode, device):
    need(dtype=ttnn.bfloat16, sublayer_dtype=ttnn.bfloat16, mode=mode, coef_source="packed")
    dtype = ttnn.bfloat16
    f, x, post, comb, pre, mixes = _packed(x_shape, dtype)
    nc = x_shape[-1]
    w = _weight(nc)
    hc = _hc_scalars(nc)
    m = dev(mixes, device)
    kw = dict(
        post_col=N,
        comb_col=2 * N,
        stats_weight=dev(w.reshape(1, 1, nc, 32), device),
        compute_kernel_config=cfg(ttnn.MathFidelity.HiFi4),
    )
    if mode == "collapse":
        kw.update(
            collapse_pre=m,
            hc_scale=hc["scale"],
            hc_base=hc["base"],
            hc_norm_width=hc["norm_width"],
            hc_norm_eps=hc["norm_eps"],
            hc_eps=hc["hc_eps"],
            hc_sinkhorn_iters=hc["iters"],
        )
    outs = mhc_post(dev(f, device, dtype), dev(x, device, dtype), m, m, **kw)
    assert len(outs) == (2 if mode == "stats" else 4)
    mix = ref_mix(f, x, post, comb)
    xo = check_residual(outs[0], mix, dtype=dtype, shape=x_shape)
    # stats: projection on the op's own X' (HiFi4 -> exact products), ss of the unrounded mix.
    st = ttnn.to_torch(outs[1]).double()
    assert list(outs[1].shape) == list(x_shape[:-1]) + [32] and outs[1].dtype == ttnn.float32
    proj = xo @ w.double()
    assert _rel(st[..., :NG], proj[..., :NG]) <= 1e-5
    assert _rel(st[..., NG], mix.square().sum(-1)) <= 1e-6
    assert torch.equal(st[..., NG + 1 :], proj[..., NG + 1 :])
    if mode == "stats":
        return
    got_hc = ttnn.to_torch(outs[2]).double()
    assert list(outs[2].shape) == list(x_shape[:-1]) + [NG] and outs[2].dtype == ttnn.float32
    own = ref_hc(
        st,
        scale=hc["scale"],
        base=hc["base"],
        norm_width=hc["norm_width"],
        norm_eps=hc["norm_eps"],
        hc_eps=hc["hc_eps"],
        iters=hc["iters"],
    )
    for sl in (slice(0, N), slice(N, 2 * N), slice(2 * N, NG)):
        assert _rel(got_hc[..., sl], own[..., sl]) <= 1e-5
    colsum = got_hc[..., 2 * N :].reshape(*got_hc.shape[:-1], N, N).sum(-2)
    assert (colsum - 1).abs().max().item() <= 1e-5
    # y = sum_i pre_i X'_i on the op's own bf16 X', one bf16 rounding.
    C = nc // N
    y_ref = (pre.double().unsqueeze(-1) * xo.reshape(*x_shape[:-1], N, C)).sum(-2)
    assert outs[3].dtype == ttnn.bfloat16
    check_residual(outs[3], y_ref, dtype=ttnn.bfloat16, shape=list(x_shape[:-1]) + [C])
