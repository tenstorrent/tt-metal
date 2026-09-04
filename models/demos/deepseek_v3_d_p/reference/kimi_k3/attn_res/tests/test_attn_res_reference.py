# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""What the device is scored against, checked on CPU. No device, no PCC.

Every device gate is measured against `reference/kimi_k3/attn_res/attn_res.py`,
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
  * the schedule itself, against a written-down trace. Both walks were transcribed from the
    same reading of the model's layer loop, so comparing them cannot expose a misreading —
    only a later change that moves a boundary in one of them. The trace is what a reviewer
    checks against the model source by eye.

All of it runs in milliseconds at a `d` far below production's. What is under test is
algebra and scheduling, neither of which depends on the shape — the shapes production
runs are what the device suites exist for.
"""

import pytest
import torch

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import BLOCK_SIZE as PRODUCTION_BLOCK_SIZE
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import EPS
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import NUM_LAYERS as PRODUCTION_LAYERS
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.attn_res import (
    AttnResStream,
    attn_res,
    attn_res_inter_block,
    attn_res_merge,
    attn_res_stack,
    fold_query,
)
from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.hf_walk import hf_attn_res, hf_stack

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
    holds is the folded query at every site of a whole stack at once, rather than at the
    single call the two gates above cover.

    It does not hold the schedule itself. Both walks transcribe the same reading of the
    model's layer loop, so a misreading agrees with itself here; the two tests below pin
    the schedule to a written-down trace instead.
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


def _record_reads(monkeypatch):
    """Candidate counts, in issue order, for every read a walk performs.

    `attn_res_stack` builds its own stream, so there is no seam to inject a recorder at.
    """
    original = AttnResStream.read
    candidates = []

    def recording_read(stream, q):
        candidates.append(stream.num_sealed + 1)
        return original(stream, q)

    monkeypatch.setattr(AttnResStream, "read", recording_read)
    return candidates


def _tiny_stack(num_layers, hidden_size=8, num_tokens=2):
    """A stack whose modules and shapes are as small as the schedule allows.

    The schedule does not depend on `d`, so pricing these at production width would buy
    nothing and make the 93-layer case slow enough to skip.
    """
    generator = torch.Generator().manual_seed(0)
    zeros = torch.zeros(hidden_size, dtype=DTYPE)
    query = (1.0 + zeros, zeros.reshape(1, -1))
    return (
        torch.randn(num_tokens, hidden_size, generator=generator, dtype=DTYPE),
        [fold_query(*query)] * num_layers,
        [fold_query(*query)] * num_layers,
        fold_query(*query),
        [torch.zeros_like] * num_layers,
        [torch.zeros_like] * num_layers,
    )


def test_stack_seals_on_block_boundaries(monkeypatch):
    """The seal schedule, against a trace rather than against a second walk.

    A layer seals when its index is a multiple of the block size, before its attention
    output is accumulated, so the sealed snapshot holds the previous block and nothing
    of this one. Layer 0 seals the token embedding, which no block contributes to.

    This does not establish that the schedule matches the model — the expected trace was
    written from the same reading of the layer loop as the walks. What it establishes is
    that the schedule is stated somewhere a reviewer can check against the model source,
    and that a later change to a boundary fails here instead of moving both walks together.
    """
    hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns = _tiny_stack(NUM_LAYERS)

    sealed_after_layer = []
    candidates = _record_reads(monkeypatch)
    attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        attn_fns,
        mlp_fns,
        block_size=BLOCK_SIZE,
        eps=EPS,
        hook=lambda layer_idx, stream: sealed_after_layer.append(stream.num_sealed),
    )

    # Five layers at block size two: seals at 0, 2 and 4.
    assert sealed_after_layer == [1, 1, 2, 2, 3]

    # Two reads per layer, less layer 0's skipped pre-attention read, plus the model-level
    # read after the last layer. Each entry is the sealed count at that read, plus the live
    # stream.
    assert candidates == [2, 2, 2, 2, 3, 3, 3, 3, 4, 4]


def test_production_schedule_performs_186_reads(monkeypatch):
    """The counts the device modules and their PCC gates are sized against.

    93 layers at block size 12 seal on layers 0, 12, …, 84 — eight snapshots, the first
    being the embedding and the last nine layers still live at the end. So the final read
    mixes nine candidates, and the stack performs 186 reads.
    """
    hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns = _tiny_stack(PRODUCTION_LAYERS)

    sealed_after_layer = []
    candidates = _record_reads(monkeypatch)
    attn_res_stack(
        hidden_states,
        q_pre,
        q_post,
        q_out,
        attn_fns,
        mlp_fns,
        block_size=PRODUCTION_BLOCK_SIZE,
        eps=EPS,
        hook=lambda layer_idx, stream: sealed_after_layer.append(stream.num_sealed),
    )

    assert sealed_after_layer[-1] == 8
    assert len(candidates) == 186
    assert candidates[-1] == 9


@pytest.mark.parametrize("shortened", ["q_pre", "q_post", "attn_fns", "mlp_fns"])
def test_stack_rejects_mismatched_sequences(shortened, expect_error):
    """A walk shorter than the caller asked for must raise, not return a plausible tensor.

    Both walks are handed the same sequences by `test_stack_matches_hf_walk`, so a silent
    truncation would shorten both and agree with itself.
    """
    hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns = _tiny_stack(NUM_LAYERS)
    sequences = {"q_pre": q_pre, "q_post": q_post, "attn_fns": attn_fns, "mlp_fns": mlp_fns}
    sequences[shortened] = sequences[shortened][:-1]

    with expect_error(AssertionError, "lengths"):
        attn_res_stack(
            hidden_states,
            sequences["q_pre"],
            sequences["q_post"],
            q_out,
            sequences["attn_fns"],
            sequences["mlp_fns"],
            block_size=BLOCK_SIZE,
            eps=EPS,
        )
