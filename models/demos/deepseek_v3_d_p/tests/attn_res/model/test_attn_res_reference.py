# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The torch reference against itself, in fp64. No device, no PCC.

Every device gate in this directory is measured against `reference/attn_res/attn_res.py`,
which is not the definition — it is the definition with two algebraic shortcuts already
taken: the two weight vectors folded into one query, and `rsqrt` pulled out of the dot.
A device test cannot see an error in either, because it compares against the shortcut.

So the reference is pinned here instead, one rung at a time:

  * `attn_res_reference.read` is the naive fp64 form, written from the published
    definition and materializing the normalized keys. Folding is exact, so the folded
    form has to reproduce it to fp64 and not to a tolerance.
  * `attn_res_inter_block` + `attn_res_merge` are the split the device op is structured
    around. Splitting is a reassociation of the same softmax, so it has to reproduce the
    direct form too.

Both run on CPU in milliseconds at a `d` far below production's. What is under test is
algebra, and algebra does not depend on the shape — the shapes production runs are what
the device suites exist for.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.attn_res import attn_res_reference as ref
from models.demos.deepseek_v3_d_p.reference.attn_res.attn_res import (
    EPS,
    attn_res,
    attn_res_inter_block,
    attn_res_merge,
    fold_query,
)

# fp64 throughout, so what survives is the algebra rather than the rounding. The residual
# slack is the two forms' different multiply order over `d`, which lands a few ULPs apart.
TOL = 1e-12

NUM_TOKENS = 64
HIDDEN_SIZE = 256
READ_SITES = 3
PROJ_STD = 0.02


def _case(num_sealed, seed=0):
    """One read's inputs in fp64, with the query still in its two unfolded factors."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.float64)
    return (
        randn(NUM_TOKENS, HIDDEN_SIZE),
        randn(NUM_TOKENS, num_sealed, HIDDEN_SIZE),
        1.0 + 0.1 * randn(HIDDEN_SIZE),
        PROJ_STD * randn(1, HIDDEN_SIZE),
    )


def _max_abs(got, want):
    return (got - want).abs().max().item()


@pytest.mark.parametrize("num_sealed", [0, 1, 8])
def test_folded_matches_naive(num_sealed):
    """The folded query and the hoisted `rsqrt` are exact rewrites, not approximations.

    `res_norm` scales `v` by `norm_weight` after normalizing and `res_proj` contracts the
    result with `proj_weight`, so the two weights only ever meet as a product against `v`
    — folding them is associativity. `rsqrt(mean(v²) + eps)` is a per-(token, candidate)
    scalar, so pulling it out of the dot is distributivity. Neither is a tolerance claim.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _case(num_sealed)

    want = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
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
