# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared native-TTNN primitives for the voxtral-tts-full port.

Every helper here is pure ttnn on the forward path -- `models/common/native_probe.py`
graduates a stub only when its forward runs ZERO torch ops, and `ttnn.from_torch` itself
counts (it surfaces as `__dlpack__`).  So the split is strict:

  * `build_*` / `stage_*` helpers run on the host and may use torch freely (build() is not
    probed);
  * `tt_*` helpers are called from forward and touch nothing but ttnn.

Numerics, measured on this Blackhole p150b against a float64 reference:

  * `ttnn.rms_norm` (fused) carries 1.5e-3 relative error, `x * rsqrt(mean(x*x)+eps) * w`
    carries 8.8e-8 -- the loss is in the fused op's reduction, so norms are composed by hand.
    Same story for `ttnn.softmax`; `ttnn.sum` is the loose primitive and is never used
    (`mean * n` is exact to 2e-7).
  * matmul with FLOAT32 activations is 4.9e-4 relative whether the weight is staged fp32 or
    bfloat16, but 2.4e-3 once the ACTIVATION is bf16.  So activations stay fp32 everywhere and
    weights are staged at whatever dtype is lossless for them.
  * `fp32_dest_acc_en` is the load-bearing compute-config flag (worth ~5x); MathFidelity
    alone does nothing without it.
"""

from __future__ import annotations

import torch

import ttnn

# `ttnn.BlackholeComputeKernelConfig` is an alias of the Wormhole one (ttnn/ttnn/types.py),
# so the name is not a bug on a Blackhole board.
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

NEG_INF = -1.0e30  # finite stand-in for -inf: exp() underflows to exactly 0, NaN-free


# =============================================================================== host side
def is_exactly_bf16(t: torch.Tensor) -> bool:
    """True when every value of `t` survives a bfloat16 round-trip, i.e. staging it as bf16
    loses NOTHING.  The released checkpoint is bf16, so this is usually true and halves the
    device footprint of the 3B backbone for free."""
    return bool(torch.equal(t.float(), t.float().to(torch.bfloat16).float()))


def stage_weight(t: torch.Tensor, device, *, transpose: bool = True, force_bf16: bool = False):
    """Stage one `F.linear` weight for `ttnn.matmul`.

    `F.linear(x, W)` is `x @ W.T`, so the transpose is folded in here once at build time
    rather than per call."""
    w = t.detach().float()
    if transpose:
        w = w.t().contiguous()
    dtype = ttnn.bfloat16 if (force_bf16 or is_exactly_bf16(w)) else ttnn.float32
    return ttnn.from_torch(w, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def stage(t: torch.Tensor, device, *, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t.detach().float(), dtype=dtype, layout=layout, device=device)


class SplitWeight:
    """A linear weight carried as a hi/lo bfloat16 PAIR, for `tt_linear_hp`.

    A plain fp32 matmul bottoms out at 1.2e-3 relative on this board (measured, K=3072); adding
    the cross terms of a hi/lo split takes it to 3.1e-4 and no further.  `lo` is None when the
    weight is exactly representable in bfloat16 (which the released checkpoint's are), so that
    term is skipped rather than multiplied by zero."""

    def __init__(self, hi, lo):
        self.hi = hi
        self.lo = lo


def stage_weight_split(t: torch.Tensor, device, *, transpose: bool = True) -> SplitWeight:
    w = t.detach().float()
    if transpose:
        w = w.t().contiguous()
    hi = w.to(torch.bfloat16)
    lo = w - hi.float()
    return SplitWeight(
        ttnn.from_torch(hi, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        None if bool((lo == 0).all()) else ttnn.from_torch(
            lo.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def interleaved_to_halves(w: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Fold the Mistral-native INTERLEAVED RoPE pairing into a q/k projection's rows.

    `voxtral_common_ref.apply_rope` rotates adjacent pairs (x0,x1), (x2,x3), ...
    (`view_as_complex`), while the cheap ttnn form -- slice the head in halves, negate one,
    concat -- is HF's `rotate_half`.  Permuting the projection ROWS

        interleaved row 2i   -> half-split row i
        interleaved row 2i+1 -> half-split row i + head_dim/2

    makes the two conventions identical.  Apply to wq and wk ONLY: the permutation is the
    same on both sides of q.k^T so it cancels in the scores, and v / wo stay untouched, which
    keeps the value path bit-for-bit the reference's."""
    out_features = w.shape[0]
    assert out_features == n_heads * head_dim, (out_features, n_heads, head_dim)
    return (
        w.reshape(n_heads, head_dim // 2, 2, -1).transpose(1, 2).reshape(out_features, -1).contiguous()
    )


def rope_tables(max_seq: int, head_dim: int, theta: float):
    """Half-split cos/sin tables [1, 1, max_seq, head_dim] matching `rope_cis`'s angles.

    A probed forward cannot build these (that is torch), so they are precomputed for
    `max_position_embeddings` at build time and `ttnn.slice`d per call."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_seq).float(), freqs)  # [S, head_dim/2]
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1)
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1)
    return cos.view(1, 1, max_seq, head_dim), sin.view(1, 1, max_seq, head_dim)


def causal_bias_table(max_seq: int) -> torch.Tensor:
    """Additive causal mask [1, 1, max_seq, max_seq] with a finite -1e30 instead of -inf.

    Every entry depends only on `rel = j - i`, so the length-S mask is the top-left corner of
    the max-length one and one table serves every call."""
    m = torch.full((max_seq, max_seq), NEG_INF)
    return torch.triu(m, diagonal=1).view(1, 1, max_seq, max_seq)


def verify_rope_permutation(w: torch.Tensor, n_heads: int, head_dim: int, theta: float, seq: int = 8):
    """Assert the permuted-weight + rotate_half path equals the reference's interleaved
    `apply_rope` on this very weight.  Costs microseconds at build time and turns a silent
    ~0.30-PCC accuracy loss into a build error."""
    x = torch.randn(1, seq, w.shape[1])
    cis = torch.polar(
        torch.ones(seq, head_dim // 2),
        torch.outer(
            torch.arange(seq).float(),
            1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)),
        ),
    )

    def split(t, n):
        b, s, _ = t.shape
        return t.view(b, s, n, head_dim).permute(0, 2, 1, 3)

    ref = split(torch.nn.functional.linear(x, w), n_heads)
    ref = torch.view_as_real(
        torch.view_as_complex(ref.reshape(1, n_heads, seq, head_dim // 2, 2)) * cis.view(1, 1, seq, -1)
    ).reshape(1, n_heads, seq, head_dim)

    got = split(torch.nn.functional.linear(x, interleaved_to_halves(w, n_heads, head_dim)), n_heads)
    cos, sin = rope_tables(seq, head_dim, theta)
    half = head_dim // 2
    rot = torch.cat([-got[..., half:], got[..., :half]], dim=-1)
    got = got * cos.view(1, 1, seq, head_dim) + rot * sin.view(1, 1, seq, head_dim)

    # `ref` is in interleaved row order, `got` in half-split order: compare like for like.
    ref = ref.reshape(1, n_heads, seq, half, 2)
    ref = torch.cat([ref[..., 0], ref[..., 1]], dim=-1)
    err = (ref - got).abs().max()
    assert err < 1e-4, f"RoPE weight permutation is wrong for this checkpoint (max abs {err})"


# =============================================================================== device side
def tt_linear(x, w_t):
    return ttnn.matmul(x, w_t, compute_kernel_config=COMPUTE_CONFIG)


def tt_split_hi_lo(x):
    """A tensor as a bfloat16 hi part (widened back to fp32) and its exact fp32 remainder."""
    hi = ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)
    return hi, ttnn.sub(x, hi)


def tt_linear_hp(x, sw: "SplitWeight"):
    """Higher-precision matmul: split the ACTIVATION into a bfloat16 hi part and its fp32
    remainder and sum the cross terms against the weight's own hi/lo pair.  ~4x tighter than a
    plain fp32 matmul (3.1e-4 vs 1.2e-3 at K=3072) for 2-3 dispatches instead of 1."""
    x_hi, x_lo = tt_split_hi_lo(x)
    out = ttnn.matmul(x_hi, sw.hi, compute_kernel_config=COMPUTE_CONFIG)
    if sw.lo is not None:
        out = ttnn.add(out, ttnn.matmul(x_hi, sw.lo, compute_kernel_config=COMPUTE_CONFIG))
    return ttnn.add(out, ttnn.matmul(x_lo, sw.hi, compute_kernel_config=COMPUTE_CONFIG))


def tt_matmul_hp(a, b):
    """The same split for an ACTIVATION x ACTIVATION product, where neither side is a staged
    weight -- i.e. the two matmuls inside attention (q.k^T and probs.v).

    Measured against a float64 reference at K = 3, 128 and 3072 alike: plain 1.1-1.8e-3, this
    3.1e-4.  A THREE-term split measures 3.1e-4 as well, so 3.1e-4 is this board's floor for a
    matmul and it is the accumulator, not the operands -- there is nothing further to buy here.

    It is worth buying at all because Block 2's output is QUANTISED: its 7-step ODE integrates
    these attention outputs, and a dimension that lands within ~1e-2 FSQ code units of a boundary
    flips a code.  With the plain form the flow block carried 7.9e-4 into the ODE and the rollout
    flipped codes from frame 1 on; with this it carries ~3e-4."""
    a_hi, a_lo = tt_split_hi_lo(a)
    b_hi, b_lo = tt_split_hi_lo(b)
    out = ttnn.matmul(a_hi, b_hi, compute_kernel_config=COMPUTE_CONFIG)
    out = ttnn.add(out, ttnn.matmul(a_hi, b_lo, compute_kernel_config=COMPUTE_CONFIG))
    return ttnn.add(out, ttnn.matmul(a_lo, b_hi, compute_kernel_config=COMPUTE_CONFIG))


def tt_rms_norm(x, w, eps: float):
    """`x * rsqrt(mean(x^2) + eps) * w`, composed -- 8.8e-8 relative vs 1.5e-3 for the fused op."""
    ms = ttnn.mean(ttnn.mul(x, x), dim=-1, keepdim=True)
    return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(ms, eps))), w)


def tt_softmax_lastdim(x, width: int):
    """Composed softmax over the last (logical) dim.

    `ttnn.softmax` is built on the same loose reduction as `ttnn.rms_norm` (1.8e-3); this form
    is ~1e-7.  `width` is the LOGICAL width, so `mean * width` is the exact row sum even when
    the tensor is tile-padded."""
    m = ttnn.max(x, dim=-1, keepdim=True)
    e = ttnn.exp(ttnn.sub(x, m))
    den = ttnn.mul(ttnn.mean(e, dim=-1, keepdim=True), float(width))
    return ttnn.mul(e, ttnn.reciprocal(den))


def tt_split_heads(x, n_heads: int, head_dim: int):
    """[B, S, n*d] -> [B, n, S, d]."""
    b, s, _ = x.shape
    return ttnn.permute(ttnn.reshape(x, (b, s, n_heads, head_dim)), (0, 2, 1, 3))


def tt_merge_heads(x):
    """[B, n, S, d] -> [B, S, n*d]."""
    b, n, s, d = x.shape
    return ttnn.reshape(ttnn.permute(x, (0, 2, 1, 3)), (b, s, n * d))


def tt_apply_rope(x, cos, sin):
    """Half-split RoPE; correct for this checkpoint ONLY with `interleaved_to_halves` folded
    into the projection that produced `x`."""
    b, n, s, d = x.shape
    half = d // 2
    x1 = ttnn.slice(x, [0, 0, 0, 0], [b, n, s, half])
    x2 = ttnn.slice(x, [0, 0, 0, half], [b, n, s, d])
    rot = ttnn.concat([ttnn.neg(x2), x1], dim=-1)
    return ttnn.add(ttnn.mul(x, cos), ttnn.mul(rot, sin))


def tt_swiglu(x, w1_t, w2_t, w3_t):
    """`w2(silu(w1 x) * w3 x)` -- the reference's bias-free FeedForward."""
    return tt_linear(ttnn.mul(ttnn.silu(tt_linear(x, w1_t)), tt_linear(x, w3_t)), w2_t)


def tt_gqa_attention(q, k, v, bias, n_heads: int, n_kv_heads: int, head_dim: int, seq: int):
    """[B,n,S,d] x [B,n_kv,S,d] -> [B,n,S,d], scores scaled by 1/sqrt(d), `bias` added pre-softmax.

    KV is expanded with `repeat_interleave`, which reproduces the reference's
    `unsqueeze(2).expand(...).reshape(...)` (query head j reads KV head j // repeats) -- a
    plain repeat would pair the wrong heads.  The explicit q@k^T -> softmax -> @v chain is
    used rather than fused SDPA: the fused kernel works the TILE-PADDED shape and silently
    loses accuracy below 32 rows (0.83 PCC at S=16), which this model's codec stack hits.

    Both products go through `tt_matmul_hp`.  These are the only ACTIVATION x ACTIVATION
    matmuls in the model, so they were the last places still carrying the plain form's 1.2e-3,
    and they sit directly upstream of a quantiser (see `tt_matmul_hp`)."""
    repeats = n_heads // n_kv_heads
    if repeats > 1:
        k = ttnn.repeat_interleave(k, repeats, dim=1)
        v = ttnn.repeat_interleave(v, repeats, dim=1)
    scores = ttnn.mul(tt_matmul_hp(q, ttnn.permute(k, (0, 1, 3, 2))), 1.0 / (head_dim ** 0.5))
    if bias is not None:
        scores = ttnn.add(scores, bias)
    return tt_matmul_hp(tt_softmax_lastdim(scores, seq), v)
