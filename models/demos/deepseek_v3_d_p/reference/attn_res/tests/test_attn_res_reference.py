# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the device is scored against, checked on CPU. No device, no PCC.

Every device gate is measured against `reference/attn_res/attn_res.py`,
which is not the definition — it is the definition with two algebraic shortcuts already
taken: the two weight vectors folded into one query, and `rsqrt` pulled out of the dot.
A device test cannot see an error in either, because it compares against the shortcut.
So each shortcut is checked here instead, against something that does not share it:

  * the folded query, against `hf_attn_res` — the HuggingFace `_apply_attn_res`, vendored
    byte-identical, which applies the two weight vectors as separate factors.
  * `attn_res_inter_block` + `attn_res_merge`, the split the device op is structured
    around, against the one-shot read. Splitting reassociates the same softmax, so it has
    to reproduce the direct form. HuggingFace computes one softmax over the whole
    candidate set and exposes no seam, so this is the only place the split can be checked.
  * `attn_res_stack`, the walk the device's 186-read gate calls, against `hf_walk.hf_stack`,
    which places the same seals and reads but drives the vendored read with the two weights
    unfolded. Which layers seal and which reads see how many candidates is scheduling rather
    than algebra, and no read-level check can reach it.

All of it runs in milliseconds at a `d` far below production's. What is under test is
algebra and scheduling, neither of which depends on the shape — the shapes production
runs are what the device suites exist for.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import (
    EPS,
    attn_res,
    attn_res_inter_block,
    attn_res_merge,
    attn_res_stack,
    fold_query,
)
from models.demos.deepseek_v3_d_p.reference.attn_res.hf_attn_res import hf_attn_res
from models.demos.deepseek_v3_d_p.reference.attn_res.hf_walk import hf_stack

# fp32, the widest the device ever computes in, and the width `hf_attn_res` forces anyway
# by widening with `.float()`. Every form checked here is an exact rewrite, so the only
# difference left is their multiply order over `d`; measured, that is under 1.2e-6.
# Absolute, not relative: the read is a convex combination, so an output row that happens
# to cancel to near zero carries a large relative error at fp32 rounding.
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
def test_folded_matches_huggingface(num_sealed):
    """The folded query and the hoisted `rsqrt` are exact rewrites, not approximations.

    `res_norm` scales `v` by `norm_weight` after normalizing and `res_proj` contracts the
    result with `proj_weight`, so the two weights only ever meet as a product against `v`
    — folding them is associativity. `rsqrt(mean(v²) + eps)` is a per-(token, candidate)
    scalar, so pulling it out of the dot is distributivity. Neither changes the value, so
    what is left for the gate to measure is the rounding on a reordered multiply.

    `hf_attn_res` keeps the two weights apart and spells the mixture as a `matmul` against
    a softmax, so agreeing with it also says the definition was read correctly in the
    first place — which no comparison between two forms written here could establish.
    """
    running_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)

    want = hf_attn_res(running_sum, block_residual, norm_weight, proj_weight, EPS)
    got = attn_res(running_sum, block_residual, fold_query(norm_weight, proj_weight), EPS)

    assert got.shape == want.shape, f"{got.shape} != {want.shape}"
    delta = _max_abs(got, want)
    assert delta <= TOL, f"S={num_sealed}: folded form differs from HuggingFace by {delta:.3e}"


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
    running_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)
    q = fold_query(norm_weight, proj_weight)
    q_batch = torch.stack([q] * READ_SITES)

    want = attn_res(running_sum, block_residual, q, EPS)
    partials, shifts, masses = attn_res_inter_block(block_residual, q_batch, EPS)

    # Every site got the same query, so every site has to land on the same read.
    for site in range(READ_SITES):
        got = attn_res_merge(partials[site], shifts[site], masses[site], running_sum, q, EPS)
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


def test_stack_matches_hf_walk():
    """The two walk drivers place the same reads and seals.

    `attn_res_stack` is what the device's 186-read gate scores against, and it carries
    the part of AttnRes that is not algebra: a layer seals on a block boundary, the
    live stream is `None` until the next accumulate, and the pre-attention read is
    skipped only while nothing is sealed. `hf_stack` places all of that identically and
    differs only in the read it calls and the query form it takes, so what this gate
    holds is the folded query and the split read at every site of a whole stack at once,
    rather than the single call the two gates above cover.
    """
    hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns = _stack_case()

    want = hf_stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, block_size=BLOCK_SIZE, eps=EPS)
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
    assert delta <= TOL, f"the folded walk differs from the HuggingFace-driven one by {delta:.3e}"
