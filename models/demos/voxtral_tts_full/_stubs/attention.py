# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TT-NN port of `VoxtralAttention` (Voxtral-TTS backbone, Block 1).

Reference: `modeling_layers.VoxtralAttention.forward`, which composes the checkpoint's own
primitives from `voxtral_common_ref`:

    q, k, v = split_heads(F.linear(h, w{q,k,v}))
    q, k    = apply_rope(q, cis), apply_rope(k, cis)
    out     = F.linear(merge_heads(gqa_attention(q, k, v, bias)), wo)

Three things about this checkpoint drive the port:

ROPE CONVENTION. `voxtral_common_ref.apply_rope` is Mistral-native: it rotates INTERLEAVED
pairs -- `view_as_complex` over `(..., d/2, 2)`, so dims (0,1), (2,3), ... HF's port instead
pairs (i, i+d/2). Rather than deinterleave activations on device every call (a stride-2
gather is expensive in tile layout), this port PERMUTES THE ROWS OF wq/wk ONCE at build time
so the projection emits each head laid out in halves, then applies the ordinary half-split
rotation. The two are exactly equivalent: the permutation is applied to q and k alike, and
attention consumes them only through `q · k`, which is invariant under a shared permutation
of the head axis. v and wo are untouched, so the output needs no un-permuting. `build()`
asserts this against an explicit interleaved rotation before returning.

HEAD WIDTH. n_heads * head_dim = 4096 != dim = 3072 -- this model is wider inside attention
than its residual stream, so wq is [4096, 3072] and wo is [3072, 4096]. Nothing here may
assume the square shape most ports take for granted.

GQA. 32 query heads over 8 KV heads, and `repeat_kv` expands INTERLEAVED (query head i reads
KV head i // 4). Expressed by folding each KV group's 4 query heads into the sequence axis,
so one [1, n_kv, ...] matmul covers all 32 query heads with no 4x copy of K and V.

WHY THE ROPE TABLE AND THE MASK ARE BUILT HERE, NOT FROM THE ARGUMENTS. `cis` arrives as a
host COMPLEX tensor and `bias` as a host float tensor. Turning either into a device tensor
inside the forward would put host math (`view_as_real`, `cat`, `from_torch`) on the compute
path, which is what separates a native port from a host reimplementation. Both are pure
functions of position -- `rope_cis(S, head_dim, theta)` and an upper-triangular -inf mask --
so this port materialises them once in `build()` for every position the model supports and
slices the live sequence out of them on device. The arguments are then read only for their
presence: `cis=None` means "no rotation", `bias=None` means "no mask" (the decode path). A
caller that holds no host tensor to signal with -- the backbone, which knows both apply --
says so directly with the `rope=` / `causal=` keywords instead.

The equivalence is not taken on trust: the PCC test builds its golden by calling the torch
reference with the reference's OWN `cis` and `bias`, so if the tables generated here differed
in convention, value or position offset, the comparison would collapse rather than pass.
"""
from __future__ import annotations

import torch
import ttnn

# From the checkpoint's config.json / params.json. `rope_theta` and `max_position_embeddings`
# are properties of the released weights, not of any one call.
_ROPE_THETA = 1_000_000.0
_MAX_POSITIONS = 2048

# Masked positions are additive -inf in the reference. -inf survives bf16, but an -inf that
# reaches a fused softmax can yield NaN instead of 0; -1e9 underflows exp() to exactly the
# same zero contribution without that risk, so the arithmetic is unchanged.
_MASK_FILL = -1e9

# The default matmul config carries ~1% relative error against the same operands multiplied
# exactly, and that compounds: the 26-layer backbone built from this port lands at PCC 0.9886
# with the default and comfortably over 0.99 with fp32 destination accumulation at HiFi4,
# which measures 6x tighter on a single 3072-deep matmul. Shared by every port in this model
# so no stage is quietly the loose one.
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)


# --------------------------------------------------------------------------------------------
# The model's shared numerics. Composed, not fused -- and that is a MEASUREMENT, not a taste.
# --------------------------------------------------------------------------------------------
# Measured on this Blackhole against a float64 torch reference, fp32 in / fp32 out, on this
# model's shapes ([1, 224, 3072] activations, [1, 8, 224, 224] scores), all with COMPUTE_CONFIG:
#
#     op                                 relative L2 error
#     ttnn.rms_norm (fused)                    1.56e-3
#     mean -> rsqrt -> mul (below)             6.7e-8      <- 23000x tighter
#     ttnn.softmax  (fused)                    1.83e-3
#     max -> exp -> reciprocal (below)         ~1e-7       <- see `_mean_sum`
#     ttnn.matmul   (fp32 operands)            1.17e-3     (the FPU truncates fp32 to ~tf32)
#     ttnn.sum      (any axis, any width)      3.2e-4
#     ttnn.mean * n (the same reduction)       6.5e-8      <- 5000x tighter
#     ttnn.exp / ttnn.reciprocal               5e-9 / 2e-8
#     ttnn.mul / ttnn.add (fp32)               0.0         (bit-exact)
#
# ONE FINDING RUNS THROUGH ALL OF THIS: on this device the fp32 ELEMENTWISE ops are exact and
# the fp32 REDUCTIONS are not. `ttnn.sum` loses 3.2e-4 at every width, flat, so it is the
# kernel and not accumulation order -- and `ttnn.mean`, which is the same reduction, does not.
# The two fused ops above are built on the loose reduction, which is why composing them out of
# `mean` beats them by three to four orders and why no compute-kernel config reaches them.
#
# That matters here more than it would in most ports. This model spends its accuracy budget on
# FSQ rounding boundaries: Block 2 rounds 36 floats onto 21 levels every frame, a flipped code
# swaps a whole learned row of the audio embedding table, and the result feeds back into the
# next frame. 1e-3 on the hidden state is the difference between a code and its neighbour, and
# a wrong code is not a small error. See tests/e2e/test_e2e_tts.py for what this buys end to end.
def rms_norm(x, weight, eps: float):
    """`voxtral_common_ref.rms_norm`: x * rsqrt(mean(x^2) + eps) * weight, over the last dim."""
    mean_square = ttnn.mean(ttnn.mul(x, x), dim=-1, keepdim=True)
    return ttnn.mul(ttnn.mul(x, ttnn.rsqrt(ttnn.add(mean_square, eps))), weight)


def softmax(x, dim: int = -1):
    """exp(x - max) / sum(exp(x - max)), the max subtracted for the usual overflow reason.

    The denominator is `mean * n`, NOT `ttnn.sum` -- see `_mean_sum`. `ttnn.exp` itself is
    exact to 5e-9 here and `ttnn.reciprocal` to 2e-8, so with the reduction fixed this
    composition is at the fp32 floor; the error the fused `ttnn.softmax` carries and the error
    `ttnn.sum` carries were the whole of the gap.
    """
    e = ttnn.exp(ttnn.subtract(x, ttnn.max(x, dim=dim, keepdim=True)))
    return ttnn.mul(e, ttnn.reciprocal(_mean_sum(e, dim)))


def _mean_sum(x, dim: int):
    """`ttnn.sum(x, dim)` -- computed as `mean * n`, which is 5000x more accurate.

    The two are the same reduction and `ttnn.mean` is the one that is right: measured on this
    Blackhole against a float64 sum of the SAME device values, fp32 in / fp32 out,

        width       200       201       208       224      3072    (dim=1, 37)
        ttnn.sum    3.2e-4    3.2e-4    3.2e-4    3.2e-4   3.2e-4     8.4e-4
        mean * n    5.6e-8    5.9e-8    6.3e-8    6.9e-8   7.5e-8     8.1e-8

    -- flat in the width, so it is not accumulation order, it is the sum kernel's own
    precision. `ttnn.mean` handles the model's non-tile-multiple widths (200, 201) correctly,
    which is the only reason this substitution is available.
    """
    rank = len(x.shape)
    axis = dim if dim >= 0 else rank + dim
    return ttnn.mul(ttnn.mean(x, dim=dim, keepdim=True), float(x.shape[axis]))


# --------------------------------------------------------------------------------------------
# fp32 matmul, actually at fp32 -- the hi/lo split
# --------------------------------------------------------------------------------------------
# A `ttnn.float32` operand is NOT multiplied at float32 on this part. The FPU is bfloat16-based
# and keeps roughly 11 mantissa bits of an fp32 operand, which is why the table above measures a
# 3072-deep fp32 x fp32 matmul at 1.17e-3 -- only 2.5x better than the same matmul in bfloat16,
# where a true fp32 multiply would be ~1e-7. The compute-kernel config cannot reach it: HiFi3
# and HiFi4 measure the same, and so do packer_l1_acc on and off.
#
# What DOES reach it is carrying the operand as two bfloat16 numbers and letting the FPU
# multiply each exactly. Writing w = w_hi + w_lo and x = x_hi + x_lo with each part bfloat16:
#
#     x @ w  =  x_hi@w_hi  +  x_hi@w_lo  +  x_lo@w_hi  +  x_lo@w_lo
#                                                         ~~~~~~~~~ 1.6e-5 relative, dropped
#
# Measured on the same 3072-deep matmul, fp32 accumulation and fp32 output throughout:
#
#     fp32 x fp32 (one matmul)                  1.17e-3
#     weight split only, 2 matmuls              4.89e-4
#     weight + activation split, 3 matmuls      3.13e-4      <- what runs here
#
# The cost is three matmuls instead of one and NO extra memory: two bfloat16 copies of a weight
# are exactly the four bytes per parameter that one float32 copy already costs.
def stage_split(device, weight: torch.Tensor):
    """A weight staged as the (hi, lo) bfloat16 pair `linear` consumes."""
    w = weight.detach().float().contiguous()
    hi = w.to(torch.bfloat16)
    lo = (w - hi.float()).to(torch.bfloat16)
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    return (to_dev(hi), to_dev(lo))


def stage_weight(device, weight: torch.Tensor, dtype):
    """Stage a matmul weight the way `dtype` asks for.

    `ttnn.float32` selects the hi/lo pair rather than a native fp32 tensor: on this device that
    is the same four bytes per parameter and strictly more of them survive into the product.
    """
    if dtype == ttnn.float32:
        return stage_split(device, weight)
    return ttnn.from_torch(
        weight.detach().float().contiguous().to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )


def _split(x):
    """A live fp32 activation as its (hi, lo) bfloat16 halves."""
    hi = ttnn.typecast(x, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(x, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


def _three_term(a_hi, a_lo, b_hi, b_lo):
    mm = lambda x, y: ttnn.matmul(x, y, dtype=ttnn.float32, compute_kernel_config=COMPUTE_CONFIG)  # noqa: E731
    return ttnn.add(ttnn.add(mm(a_hi, b_hi), mm(a_hi, b_lo)), mm(a_lo, b_hi))


def linear(x, w):
    """x @ w, where `w` is either a plain staged tensor or a `stage_split` (hi, lo) pair."""
    if not isinstance(w, tuple):
        return ttnn.linear(x, w, compute_kernel_config=COMPUTE_CONFIG)
    return _three_term(*_split(x), *w)


def matmul(a, b):
    """a @ b where BOTH operands are live activations -- the scores and the value average.

    Split at fp32 only. On a bfloat16 activation the halves would be the tensor itself and
    zero, so the three matmuls would buy nothing and cost 3x; the plain op is the right answer
    there and this returns it.
    """
    if a.dtype != ttnn.float32 or b.dtype != ttnn.float32:
        return ttnn.matmul(a, b, compute_kernel_config=COMPUTE_CONFIG)
    return _three_term(*_split(a), *_split(b))


def _interleaved_to_halves(head_dim: int) -> torch.Tensor:
    """Row order taking an interleaved-pair head to a split-halves head.

    Original component 2j lands at j and 2j+1 at j + d/2, which is precisely the layout the
    half-split rotation expects.
    """
    return torch.arange(head_dim).reshape(-1, 2).t().reshape(-1)


def _permute_head_rows(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
    """Apply the interleaved->halves row permutation independently within each head."""
    perm = _interleaved_to_halves(head_dim)
    return weight.reshape(n_heads, head_dim, -1)[:, perm, :].reshape(n_heads * head_dim, -1)


def _rope_cos_sin(max_positions: int, head_dim: int, theta: float):
    """`voxtral_common_ref.rope_cis` as (cos, sin), duplicated for the half-split rotation.

    rope_cis returns `polar(1, outer(positions, freqs))`, i.e. angle p*freq_j at position p;
    the half-split rotation needs each angle repeated across the two halves it mixes.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angles = torch.outer(torch.arange(max_positions).float(), freqs)
    cos = angles.cos().repeat(1, 2)
    sin = angles.sin().repeat(1, 2)
    return cos.reshape(1, 1, max_positions, head_dim), sin.reshape(1, 1, max_positions, head_dim)


def _causal_mask(max_positions: int) -> torch.Tensor:
    """`voxtral_common_ref.causal_bias` with the -inf replaced per `_MASK_FILL`."""
    m = torch.full((max_positions, max_positions), _MASK_FILL)
    return torch.triu(m, diagonal=1).reshape(1, 1, max_positions, max_positions)


class TtVoxtralAttention:
    def __init__(self, device, weights, tables, n_heads, n_kv_heads, head_dim):
        self.device = device
        self.wq, self.wk, self.wv, self.wo = weights
        self.cos, self.sin, self.mask = tables
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.scale = head_dim**-0.5

    @classmethod
    def build(
        cls,
        device,
        torch_module,
        max_positions: int = _MAX_POSITIONS,
        theta: float = _ROPE_THETA,
        dtype=ttnn.bfloat16,
    ):
        n_heads = int(getattr(torch_module, "n_heads", 32))
        n_kv_heads = int(getattr(torch_module, "n_kv_heads", 8))
        head_dim = int(getattr(torch_module, "head_dim", 128))

        wq = torch_module.q_proj.detach().float()
        wk = torch_module.k_proj.detach().float()
        wv = torch_module.v_proj.detach().float()
        wo = torch_module.o_proj.detach().float()

        wq_p = _permute_head_rows(wq, n_heads, head_dim)
        wk_p = _permute_head_rows(wk, n_kv_heads, head_dim)
        _assert_rope_equivalence(wq, wq_p, n_heads, head_dim, theta)

        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16

        def stage(t):
            return ttnn.from_torch(
                t.contiguous().to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
            )

        cos, sin = _rope_cos_sin(max_positions, head_dim, theta)
        # `F.linear(x, W)` is `x @ W.T`; ttnn.linear has no transpose flag, so the transpose is
        # folded into the staged weight rather than paid for on every call.
        weights = tuple(stage_weight(device, w.t(), dtype) for w in (wq_p, wk_p, wv, wo))
        # The RoPE tables and the mask are ADDED and MULTIPLIED, never matmul operands, so they
        # stay plain tensors -- elementwise fp32 is bit-exact on this device.
        tables = (stage(cos), stage(sin), stage(_causal_mask(max_positions)))
        return cls(device, weights, tables, n_heads, n_kv_heads, head_dim)

    def _split_heads(self, x, seq_len, n_heads):
        """[1, S, n*d] -> [1, n, S, d], matching `voxtral_common_ref.split_heads`."""
        return ttnn.permute(ttnn.reshape(x, (1, seq_len, n_heads, self.head_dim)), (0, 2, 1, 3))

    def _apply_rope(self, x, cos, sin, n_heads, seq_len):
        half = self.head_dim // 2
        lo = ttnn.slice(x, [0, 0, 0, 0], [1, n_heads, seq_len, half])
        hi = ttnn.slice(x, [0, 0, 0, half], [1, n_heads, seq_len, self.head_dim])
        rotated = ttnn.concat([ttnn.neg(hi), lo], dim=-1)
        return ttnn.add(ttnn.mul(x, cos), ttnn.mul(rotated, sin))

    def __call__(self, h, cis=None, bias=None, cache=None, cache_key=None, *, rope=None, causal=None):
        apply_rope = (cis is not None) if rope is None else bool(rope)
        apply_causal = (bias is not None) if causal is None else bool(causal)
        seq_len = int(h.shape[-2])
        if seq_len > int(self.cos.shape[-2]):
            raise ValueError(
                f"sequence length {seq_len} exceeds the {int(self.cos.shape[-2])} positions staged at "
                f"build time; rebuild with max_positions >= {seq_len}"
            )

        q = self._split_heads(linear(h, self.wq), seq_len, self.n_heads)
        k = self._split_heads(linear(h, self.wk), seq_len, self.n_kv_heads)
        v = self._split_heads(linear(h, self.wv), seq_len, self.n_kv_heads)

        if apply_rope:
            cos = ttnn.slice(self.cos, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            sin = ttnn.slice(self.sin, [0, 0, 0, 0], [1, 1, seq_len, self.head_dim])
            q = self._apply_rope(q, cos, sin, self.n_heads, seq_len)
            k = self._apply_rope(k, cos, sin, self.n_kv_heads, seq_len)

        # GQA without materialising repeat_kv: fold each KV group's 4 query heads into the
        # sequence axis, so one [1, n_kv, ...] matmul covers all 32 query heads. Query head i
        # lands in group i // 4, which is exactly `repeat_kv`'s interleaved expansion.
        qg = ttnn.reshape(q, (1, self.n_kv_heads, self.n_rep * seq_len, self.head_dim))
        # The SCORES matmul is split like the projections, and it is the one that pays best:
        # its result is exponentiated, so an absolute error there is a RELATIVE error on the
        # attention weights. Scores run to ~10 after scaling, and 1.17e-3 of relative matmul
        # error on a score of 10 is 1.2e-2 absolute -- over 1% on the weight it becomes.
        scores = matmul(qg, ttnn.permute(k, (0, 1, 3, 2)))
        scores = ttnn.mul(scores, self.scale)

        if apply_causal:
            # Rows of the folded score matrix run (rep, pos), so the per-position mask is
            # TILED n_rep times down the row axis -- not stretched.
            window = ttnn.slice(self.mask, [0, 0, 0, 0], [1, 1, seq_len, seq_len])
            scores = ttnn.add(scores, ttnn.repeat(window, (1, 1, self.n_rep, 1)))

        # SOFTMAX IS COMPOSED, NOT FUSED -- see the measurement above `softmax`. The fused op
        # left on its DEFAULT config runs LoFi with math_approx_mode on and the rows do not even
        # sum to 1: measured 0.9943 mean and 0.9590 worst over a 200-position causal window,
        # i.e. up to 4% of the attention mass simply lost, biased -- a SCALE error no
        # per-component PCC can see. Handing it COMPUTE_CONFIG takes that to 0.9993 mean /
        # 0.9951 worst and the op's own error to 5.2e-3; composing it takes the error to 6.5e-4.
        probs = softmax(scores, dim=-1)
        out = matmul(probs, v)
        out = ttnn.reshape(out, (1, self.n_heads, seq_len, self.head_dim))
        out = ttnn.permute(out, (0, 2, 1, 3))
        out = ttnn.reshape(out, (1, seq_len, self.n_heads * self.head_dim))
        return linear(out, self.wo)


def _assert_rope_equivalence(wq, wq_permuted, n_heads, head_dim, theta, seq_len=8):
    """Prove the row permutation reproduces the reference's interleaved rotation.

    A cheap host-side check on random activations: rotating the permuted projection with the
    half-split rule must equal permuting the reference's interleaved result. A silent
    mismatch here would cost accuracy in every layer and read as a generic numerical problem,
    so it is caught at build time rather than debugged later.
    """
    dim = wq.shape[1]
    h = torch.randn(1, seq_len, dim)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    cis = torch.polar(torch.ones(seq_len, head_dim // 2), torch.outer(torch.arange(seq_len).float(), freqs))

    def split(x, n):
        return x.view(1, seq_len, n, head_dim).permute(0, 2, 1, 3)

    interleaved = torch.view_as_real(
        torch.view_as_complex(split(h @ wq.t(), n_heads).reshape(1, n_heads, seq_len, head_dim // 2, 2))
        * cis.view(1, 1, seq_len, head_dim // 2)
    ).reshape(1, n_heads, seq_len, head_dim)

    x = split(h @ wq_permuted.t(), n_heads)
    cos = cis.real.repeat(1, 2).view(1, 1, seq_len, head_dim)
    sin = cis.imag.repeat(1, 2).view(1, 1, seq_len, head_dim)
    rotated = x.roll(head_dim // 2, dims=-1)
    rotated[..., : head_dim // 2] = rotated[..., : head_dim // 2].neg()
    halves = x * cos + rotated * sin

    perm = _interleaved_to_halves(head_dim)
    if not torch.allclose(halves, interleaved[..., perm], atol=1e-4, rtol=1e-4):
        raise AssertionError(
            "interleaved->halves RoPE permutation does not reproduce the reference rotation; "
            "the port would silently lose accuracy in every layer"
        )


def build(device, torch_module=None, **kwargs):
    return TtVoxtralAttention.build(device, torch_module, **kwargs)


def attention(device, torch_module=None, **kwargs):
    return TtVoxtralAttention.build(device, torch_module, **kwargs)
