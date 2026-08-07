# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gates on the ground-truth reference itself.

Everything else in this module measures against `reference/attn_res_reference.py`,
so it cannot be gated by comparison with another implementation without making
the whole ladder circular. What pins it instead are closed forms and structural
properties of the definition:

  * Two limits where the answer is known outright — a zero query gives the plain
    arithmetic mean of the candidates, and a saturated query selects exactly one.
  * Scale invariance of the scores, which is what RMS normalization *means*.
  * Sensitivity of the output to a candidate's scale, which is what distinguishes
    mixing `v` from mixing the normalized keys.
  * A convex-combination bracket, which any row-stochastic mixture must satisfy.

The limits are the load-bearing ones: they fail for any error in the score
pipeline or the mixture that the property tests could absorb.
"""

import pytest
import torch

from models.experimental.kimi_k3_attn_res.reference import attn_res_reference as ref

EPS = 1e-5
BLOCK_SIZE = 12
NUM_LAYERS = 93

# fp64 summation-order noise over d = 7168. A real algebra error clears it by
# orders of magnitude.
FP64_TOL = 1e-12

PROJ_STD = 0.02


def _rel_err(actual, expected):
    scale = expected.double().abs().max().clamp_min(1e-300)
    return ((actual.double() - expected.double()).abs().max() / scale).item()


def _make_case(num_tokens, hidden_size, num_sealed, score_scale=1.0, seed=0):
    """Weights are built the way the model stores them — an RMSNorm gain near one
    and a small linear projection — because score magnitude decides whether the
    softmax is in a useful regime at all."""
    gen = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)

    prefix_sum = randn(num_tokens, hidden_size)
    block_residual = randn(num_tokens, num_sealed, hidden_size)
    norm_weight = torch.ones(hidden_size, dtype=torch.float64) + 0.02 * randn(hidden_size)
    proj_weight = score_scale * PROJ_STD * randn(1, hidden_size)
    return prefix_sum, block_residual, norm_weight, proj_weight


SHAPES = [(64, 256), (64, 7168)]
SEALED = [0, 1, 4, 8]


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
def test_zero_query_gives_the_plain_mean(num_tokens, hidden_size, num_sealed):
    """First closed form: a zero projection makes every score zero, so the softmax
    is uniform and the read is the arithmetic mean of the candidates."""
    prefix_sum, block_residual, norm_weight, _ = _make_case(num_tokens, hidden_size, num_sealed)
    proj_weight = torch.zeros(1, hidden_size, dtype=torch.float64)

    actual = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    expected = ref.candidates(prefix_sum, block_residual).mean(1)

    assert _rel_err(actual, expected) <= FP64_TOL


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", [1, 4, 8], ids=["S1", "S4", "S8"])
def test_saturated_query_selects_one_candidate(num_tokens, hidden_size, num_sealed):
    """Second closed form: scale the projection until the score gaps exceed the
    fp64 exp range, and the mixture collapses onto the top-scoring candidate.

    This is the arm that catches a mixture over the wrong tensor: the winner is
    chosen by score but what comes out must be the raw candidate.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(
        num_tokens, hidden_size, num_sealed, score_scale=1e6
    )

    v = ref.candidates(prefix_sum, block_residual)
    row_scores = ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    probs = ref.softmax_rows(row_scores)
    assert (probs == 1.0).sum() == num_tokens, "scores not saturated enough to make this a limit test"

    winner = v[torch.arange(num_tokens), row_scores.argmax(-1)]
    actual = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)

    assert torch.equal(actual, winner)


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
def test_constant_candidates_give_closed_form_scores(num_tokens, hidden_size):
    """Third closed form, and the one that pins the softmax temperature.

    A candidate that is constant along the hidden dim has `mean(v²) = a²` outright,
    so the score collapses to `(a / √(a² + eps)) · Σⱼ gainⱼ · projⱼ` exactly. Two
    errors are otherwise detectable only by agreeing with an implementation that
    got them right: reducing with `sum` instead of `mean`, which divides every
    score by `√d`, and dropping the `res_norm` gain — so the gain here is a real
    vector rather than ones.
    """
    amplitudes = torch.tensor([0.5, -2.0, 1.0, 7.0, -0.25], dtype=torch.float64)
    num_sealed = amplitudes.numel() - 1
    v = amplitudes.reshape(1, -1, 1).expand(num_tokens, amplitudes.numel(), hidden_size).contiguous()
    prefix_sum, block_residual = v[:, -1], v[:, :-1]

    gen = torch.Generator().manual_seed(1)
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
    norm_weight = torch.ones(hidden_size, dtype=torch.float64) + 0.02 * randn(hidden_size)
    proj_weight = 0.02 * randn(1, hidden_size)

    actual = ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    weight_sum = (norm_weight * proj_weight.reshape(-1)).sum()
    expected = (amplitudes / (amplitudes.pow(2) + EPS).sqrt()).reshape(1, -1) * weight_sum

    assert actual.shape == (num_tokens, num_sealed + 1)
    assert _rel_err(actual, expected.expand_as(actual)) <= FP64_TOL


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
def test_empty_snapshot_set_is_identity(num_tokens, hidden_size):
    """S == 0 is a one-candidate softmax, so the read must return the stream."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(num_tokens, hidden_size, 0)

    actual = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)

    assert actual.shape == prefix_sum.shape
    assert torch.equal(actual, prefix_sum)


@pytest.mark.parametrize("num_sealed", [1, 4, 8], ids=["S1", "S4", "S8"])
def test_scores_are_scale_invariant(num_sealed):
    """`rsqrt(mean((cv)²)) · ⟨w, cv⟩` cancels `c`, so scaling one candidate must
    not move its score. This is the sharpest available gate on the RMS
    denominator — a wrong reduction axis or a missing `mean` breaks it outright.

    Exact only at `eps == 0`: `c²·mean(v²) + eps` is not `c²(mean(v²) + eps)`, so a
    finite `eps` skews the score by `≈ (eps / 2·mean(v²))·(1 − c⁻²)`. At `eps=1e-5`,
    `mean(v²) ≈ 1` and `c = 3` that is ~4e-6, which is why the exact arm has to run
    at `eps = 0` rather than at a loosened tolerance.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, num_sealed)
    scale = 3.0

    scaled = block_residual.clone()
    scaled[:, 0] *= scale

    for eps, tol in ((0.0, FP64_TOL), (EPS, 1e-5)):
        baseline = ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, eps)
        perturbed = ref.scores(prefix_sum, scaled, norm_weight, proj_weight, eps)
        assert _rel_err(perturbed, baseline) <= tol, (eps, _rel_err(perturbed, baseline))

    # The finite-eps skew is real but bounded by the analytic O(eps) term, so the
    # loose arm above is not hiding a defect that would grow with the scale.
    mean_square = ref.candidates(prefix_sum, block_residual).pow(2).mean(-1).min().item()
    predicted = (EPS / (2.0 * mean_square)) * (1.0 - scale**-2)
    measured = _rel_err(
        ref.scores(prefix_sum, scaled, norm_weight, proj_weight, EPS),
        ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, EPS),
    )
    assert measured <= 2.0 * predicted, (measured, predicted)


def test_mixture_is_over_raw_values():
    """The porting bug this op invites: mixing the normalized keys.

    Scores are scale-invariant, so if the mixture were over `k` too, rescaling one
    candidate would change nothing whatsoever. It must change the output.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, 4)

    scaled = block_residual.clone()
    scaled[:, 0] *= 3.0

    baseline = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    perturbed = ref.read(prefix_sum, scaled, norm_weight, proj_weight, EPS)

    assert _rel_err(perturbed, baseline) > 0.01


@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
def test_probs_are_row_stochastic(num_sealed):
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, num_sealed)

    probs = ref.softmax_rows(ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, EPS))

    assert probs.shape == (64, num_sealed + 1)
    assert (probs >= 0).all()
    assert _rel_err(probs.sum(-1), torch.ones(64, dtype=torch.float64)) <= FP64_TOL


@pytest.mark.parametrize("num_sealed", [1, 4, 8], ids=["S1", "S4", "S8"])
def test_output_is_a_convex_combination(num_sealed):
    """A row-stochastic mixture cannot leave the per-coordinate bracket of its
    inputs. Catches a scale defect the PCC-style gates elsewhere would miss."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, num_sealed)

    v = ref.candidates(prefix_sum, block_residual)
    actual = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    slack = FP64_TOL * (v.amax(1) - v.amin(1))

    assert (actual <= v.amax(1) + slack).all()
    assert (actual >= v.amin(1) - slack).all()


@pytest.mark.parametrize("num_sealed", [4, 8], ids=["S4", "S8"])
def test_sealed_order_does_not_change_the_read(num_sealed):
    """The sealed set is unordered — no positional term anywhere in the read. The
    live stream stays last, so only the snapshots are permuted."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, num_sealed)
    permutation = torch.randperm(num_sealed, generator=torch.Generator().manual_seed(5))

    baseline = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    permuted = ref.read(prefix_sum, block_residual[:, permutation], norm_weight, proj_weight, EPS)

    assert _rel_err(permuted, baseline) <= FP64_TOL


def test_internal_precision_never_narrows():
    """`dtype` raises precision, it never lowers it. A reference that quietly ran
    in fp32 when handed fp64 could not measure an fp32 implementation — which is
    exactly how the upstream `.float()` spelling behaves."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 256, 4)

    asked_for_fp32 = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS, dtype=torch.float32)
    fp64 = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS, dtype=torch.float64)

    assert asked_for_fp32.dtype == torch.float64
    assert torch.equal(asked_for_fp32, fp64)


def test_narrow_inputs_are_widened_and_the_output_is_narrowed_back():
    """bf16 in, bf16 out, fp64 in between — the shape the device tests need."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 256, 4)
    cast = lambda t: t.to(torch.bfloat16)

    narrow = ref.read(cast(prefix_sum), cast(block_residual), cast(norm_weight), cast(proj_weight), EPS)
    wide = ref.read(
        cast(prefix_sum).double(),
        cast(block_residual).double(),
        cast(norm_weight).double(),
        cast(proj_weight).double(),
        EPS,
    )

    assert narrow.dtype == torch.bfloat16
    assert torch.equal(narrow, wide.to(torch.bfloat16))


def test_saturated_scores_do_not_overflow():
    """The max shift is what keeps this finite; assert it rather than trusting it."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, 8, score_scale=1e12)

    probs = ref.softmax_rows(ref.scores(prefix_sum, block_residual, norm_weight, proj_weight, EPS))
    actual = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)

    assert torch.isfinite(probs).all()
    assert torch.isfinite(actual).all()
    assert _rel_err(probs.sum(-1), torch.ones(64, dtype=torch.float64)) <= FP64_TOL


def test_lifecycle_seal_schedule():
    """Seals, snapshot growth and read count over a full stack."""
    hidden_size = 64
    gen = torch.Generator().manual_seed(3)
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
    hidden_states = randn(8, hidden_size)
    queries = [
        (torch.ones(hidden_size, dtype=torch.float64), PROJ_STD * randn(1, hidden_size)) for _ in range(2 * NUM_LAYERS)
    ]

    stream = ref.Stream(hidden_states, block_size=BLOCK_SIZE, eps=EPS)
    reads = []
    seal_layers = []

    unwrapped_read = stream.read
    stream.read = lambda q: (reads.append(stream.num_sealed), unwrapped_read(q))[1]

    for layer_idx in range(NUM_LAYERS):
        before = stream.num_sealed
        ref.layer(
            stream, layer_idx, queries[2 * layer_idx], queries[2 * layer_idx + 1], lambda h: 0.1 * h, lambda h: 0.1 * h
        )
        if stream.num_sealed > before:
            seal_layers.append(layer_idx)

    assert seal_layers == [0, 12, 24, 36, 48, 60, 72, 84]
    assert stream.num_sealed == 8
    # 92 pre-attention reads (layer 0 skips it at S == 0) + 93 pre-MLP.
    assert len(reads) == 2 * NUM_LAYERS - 1
    assert min(reads) == 1


def test_stack_matches_layerwise_walk():
    """`stack` is the same walk as driving `layer` by hand."""
    num_layers, hidden_size = 25, 64
    gen = torch.Generator().manual_seed(11)
    randn = lambda *shape: torch.randn(*shape, generator=gen, dtype=torch.float64)
    hidden_states = randn(8, hidden_size)
    make_query = lambda: (torch.ones(hidden_size, dtype=torch.float64), PROJ_STD * randn(1, hidden_size))
    q_pre = [make_query() for _ in range(num_layers)]
    q_post = [make_query() for _ in range(num_layers)]
    q_out = make_query()
    attn_fns = [lambda h, s=0.1 + 0.001 * i: s * h for i in range(num_layers)]
    mlp_fns = [lambda h, s=0.05 + 0.001 * i: s * h for i in range(num_layers)]

    stacked = ref.stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns, BLOCK_SIZE, EPS)

    stream = ref.Stream(hidden_states, block_size=BLOCK_SIZE, eps=EPS)
    for layer_idx in range(num_layers):
        ref.layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fns[layer_idx], mlp_fns[layer_idx])
    manual = stream.read(q_out)

    assert torch.equal(stacked, manual)
