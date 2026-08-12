# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The reference ladder, on CPU. No device, no PCC.

Every device gate in this directory is measured against `reference/attn_res/attn_res.py`,
which is not the definition — it is the definition with two algebraic shortcuts already
taken: the two weight vectors folded into one query, and `rsqrt` pulled out of the dot.
A device test cannot see an error in either, because it compares against the shortcut.
So the ladder is pinned here instead, one rung at a time, and each rung is what makes
the rung above it worth trusting:

  * `hf_attn_res` is upstream's own `_apply_attn_res`, vendored byte-identical. It is
    the only rung not written here, and the only evidence that the equation the naive
    form transcribes from the published definition is the equation upstream runs.
  * `attn_res_reference.read` is that definition spelled out, materializing the
    normalized keys and keeping the two weight vectors as separate factors.
  * `attn_res_inter_block` + `attn_res_merge` are the split the device op is structured
    around. Splitting is a reassociation of the same softmax, so it has to reproduce the
    direct form too.
  * `attn_res_stack` is the walk the device's 186-read gate calls. Which layers seal and
    which reads see how many candidates is scheduling, not algebra, and no read-level
    rung above can reach it — so it is pinned against the naive walk directly.

All of it runs in milliseconds at a `d` far below production's. What is under test is
algebra and scheduling, neither of which depends on the shape — the shapes production
runs are what the device suites exist for.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.attn_res import attn_res_reference as ref
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import (
    EPS,
    attn_res,
    attn_res_inter_block,
    attn_res_merge,
    attn_res_stack,
    fold_query,
)
from models.demos.deepseek_v3_d_p.reference.attn_res.hf_attn_res import hf_attn_res

# fp32, the widest the device ever computes in, and the width upstream forces anyway by
# widening with `.float()`. Every rung here is an exact rewrite, so the only difference
# left is the forms' multiply order over `d`; measured, that is under 1.2e-6 across all
# four. Absolute, not relative: the read is a convex combination, so an output row that
# happens to cancel to near zero carries a large relative error at fp32 rounding.
DTYPE = torch.float32
TOL = 1e-5

NUM_TOKENS = 64
HIDDEN_SIZE = 256
READ_SITES = 3
PROJ_STD = 0.02

# Enough layers to cross a block boundary more than once, which is the whole content of
# the walk: seals land on layers 0, 2 and 4, and the pre-attention read is skipped only
# before the first of them.
NUM_LAYERS = 5
BLOCK_SIZE = 2


def _case(num_sealed, seed=0):
    """One read's inputs, with the query still in its two unfolded factors."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=DTYPE)
    return (
        randn(NUM_TOKENS, HIDDEN_SIZE),
        randn(NUM_TOKENS, num_sealed, HIDDEN_SIZE),
        1.0 + 0.1 * randn(HIDDEN_SIZE),
        PROJ_STD * randn(1, HIDDEN_SIZE),
    )


def _max_abs(got, want):
    return (got - want).abs().max().item()


@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_naive_matches_upstream(num_sealed):
    """The naive form computes the equation upstream computes.

    Everything else in this file compares one of our transcriptions against another of
    ours, which cannot catch a definition read wrong in the first place. This is the one
    rung that can, and it is the reason `hf_attn_res.py` is vendored rather than
    paraphrased: the two differ in how they spell the mixture — an explicit weighted sum
    here, a `matmul` against a softmax upstream — and agreeing anyway is the claim.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)

    want = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS, dtype=DTYPE)
    got = hf_attn_res(prefix_sum, block_residual, norm_weight, proj_weight, EPS)

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    delta = _max_abs(got, want)
    assert delta <= TOL, f"S={num_sealed}: the naive form differs from upstream by {delta:.3e}"


@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_folded_matches_naive(num_sealed):
    """The folded query and the hoisted `rsqrt` are exact rewrites, not approximations.

    `res_norm` scales `v` by `norm_weight` after normalizing and `res_proj` contracts the
    result with `proj_weight`, so the two weights only ever meet as a product against `v`
    — folding them is associativity. `rsqrt(mean(v²) + eps)` is a per-(token, candidate)
    scalar, so pulling it out of the dot is distributivity. Neither changes the value, so
    what is left for the gate to measure is the rounding on a reordered multiply.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)

    want = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS, dtype=DTYPE)
    got = attn_res(prefix_sum, block_residual, fold_query(norm_weight, proj_weight), EPS)

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    delta = _max_abs(got, want)
    assert delta <= TOL, f"S={num_sealed}: folded form differs from the naive one by {delta:.3e}"


@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_split_matches_direct(num_sealed):
    """The online-softmax split reproduces the one-shot softmax.

    `inter_block` scores and mixes the sealed set against its own running maximum, and
    `merge` rescales that partial when the live stream's score exceeds it. The device op
    implements exactly this, so an error in the shift/mass convention here would be
    invisible to every device gate — they all compare against the direct form.

    `S = 0` is the case the convention is built for: an empty mixture carries a `-inf`
    shift, whose rescale factor is exactly zero, and the read collapses to the live
    stream with no branch anywhere.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)
    q = fold_query(norm_weight, proj_weight)
    q_batch = torch.stack([q] * READ_SITES)

    want = attn_res(prefix_sum, block_residual, q, EPS)
    partials, shifts, masses = attn_res_inter_block(block_residual, q_batch, EPS)

    # Every site got the same query, so every site has to land on the same read.
    for site in range(READ_SITES):
        got = attn_res_merge(partials[site], shifts[site], masses[site], prefix_sum, q, EPS)
        delta = _max_abs(got, want)
        assert delta <= TOL, f"S={num_sealed} site {site}: split differs from the direct read by {delta:.3e}"


def _stack_case(seed=0):
    """A whole stack's inputs: two queries per layer, one model-level, and the modules.

    The modules only have to be deterministic and to mix `d`, since what is under test is
    where the reads and seals land rather than what the layers compute.
    """
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=DTYPE)
    query = lambda: (1.0 + 0.1 * randn(HIDDEN_SIZE), PROJ_STD * randn(1, HIDDEN_SIZE))

    q_pre = [query() for _ in range(NUM_LAYERS)]
    q_post = [query() for _ in range(NUM_LAYERS)]
    q_out = query()
    weights = [randn(HIDDEN_SIZE, HIDDEN_SIZE) * HIDDEN_SIZE**-0.5 for _ in range(2 * NUM_LAYERS)]
    module_fns = [(lambda h, w=w: h @ w) for w in weights]

    return randn(NUM_TOKENS, HIDDEN_SIZE), q_pre, q_post, q_out, module_fns[:NUM_LAYERS], module_fns[NUM_LAYERS:]


def test_stack_matches_naive():
    """The two walk drivers place the same reads and seals.

    `attn_res_stack` is what the device's 186-read gate scores against, and it carries
    the part of AttnRes that is not algebra: a layer seals on a block boundary, the
    live stream is `None` until the next accumulate, and the pre-attention read is
    skipped only while nothing is sealed. Get any of that wrong in both the driver and
    the device and every read still matches its reference — the walk is self-consistent
    and wrong. Pinning it against the naive walk, whose bookkeeping was written out
    separately, is what closes that.
    """
    hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns = _stack_case()

    want = ref.stack(
        hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, eps=EPS, dtype=DTYPE
    )
    got = attn_res_stack(
        hidden_states,
        [fold_query(*q) for q in q_pre],
        [fold_query(*q) for q in q_post],
        fold_query(*q_out),
        attn_fns,
        mlp_fns,
        block_size=BLOCK_SIZE,
        eps=EPS,
    )

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    delta = _max_abs(got, want)
    assert delta <= TOL, f"the folded walk differs from the naive one by {delta:.3e}"
