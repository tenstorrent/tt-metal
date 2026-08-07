# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Phase-4 numeric ladder for the AttnRes torch reference.

Three implementations meet here, and each answers a question the others cannot:

  * `reference/attn_res_reference.py` — unfolded, fp64, ours. The precision root.
    Pinned by closed forms in `test_attn_res_reference.py`, not by agreement.
  * `reference/hf_attn_res.py` — the vendored upstream read. The external anchor:
    the only evidence that upstream computes the equation we believe it does. It is
    fp32-locked, so it can anchor the algebra but never the precision.
  * `torch_functional/attn_res.py` — the folded form, what the device op mirrors.

Rung 0b crosses the anchor against the root. Rung 1 proves the fold against the
root, at fp64 as well as fp32: the fp32 arm alone cannot separate an algebra error
near the rounding floor from rounding itself, while at fp64 the two forms must agree
to ~1e-14, so a real reassociation error has nowhere to hide. Rung 2 anchors the
inter-block/merge split to the direct form. Rung 3 pins the `block_residual`
lifecycle. See `API_SPEC.md` §6.

No rung above 1 is bit-exact: folding `res_norm.weight * res_proj.weight` reassociates
the score from `Σ_j (v_j · rms_inv · w_j)` to `rms_inv · Σ_j (v_j · q_j)`, and the
online-softmax split reassociates the mixture. Both change rounding. The fp32
tolerance is the dot-product noise floor, `√d · ε_fp32` ≈ 1e-5 at d=7168, which a
real algebra error clears by orders of magnitude.
"""

import pytest
import torch

from models.experimental.kimi_k3_attn_res.reference import attn_res_reference as ref
from models.experimental.kimi_k3_attn_res.reference.hf_attn_res import hf_attn_res
from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import (
    BLOCK_SIZE,
    EPS,
    NUM_LAYERS,
    AttnResStream,
    attn_res,
    attn_res_inter_block,
    attn_res_layer,
    attn_res_merge,
    attn_res_scores,
    attn_res_stack,
    fold_query,
)

FP32_DOT_TOL = 1e-5

# fp64 summation-order noise over d = 7168, the floor the fold has to clear when
# rounding is taken off the table.
FP64_DOT_TOL = 1e-13

# Queries are built the way the model does — an RMSNorm gain near one times a
# small linear projection — because score magnitude decides whether the softmax
# is in a useful regime at all. A unit-variance query gives ⟨q, v⟩ ~ ±√d, which
# saturates the softmax to one-hot and makes every gate below vacuous.
PROJ_STD = 0.02


def _pcc(actual, expected):
    """PCC in fp64. In fp32, `corrcoef` over 458k elements caps near 0.99999988
    even for bit-identical inputs, which would make the gate measure the metric."""
    a = actual.double().reshape(-1)
    b = expected.double().reshape(-1)
    return torch.corrcoef(torch.stack((a, b)))[0, 1].item()


def _rel_err(actual, expected):
    scale = expected.double().abs().max().clamp_min(1e-300)
    return ((actual.double() - expected.double()).abs().max() / scale).item()


def _make_case(num_tokens, hidden_size, num_sealed, score_scale=1.0, dtype=torch.float32, seed=0):
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32)

    prefix_sum = randn(num_tokens, hidden_size).to(dtype)
    block_residual = randn(num_tokens, num_sealed, hidden_size).to(dtype)
    norm_weight = (torch.ones(hidden_size) + 0.02 * randn(hidden_size)).to(dtype)
    proj_weight = (score_scale * PROJ_STD * randn(1, hidden_size)).to(dtype)
    return prefix_sum, block_residual, norm_weight, proj_weight


SHAPES = [(64, 256), (64, 7168)]
SEALED = [0, 1, 4, 8]


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
@pytest.mark.parametrize("score_scale", [1.0, 8.0], ids=["scores-o1", "scores-saturated"])
def test_reference_matches_the_upstream_read(num_tokens, hidden_size, num_sealed, score_scale):
    """Rung 0b — our unfolded reference computes the same read upstream does.

    This is the only rung that reaches outside the module. Every other gate
    compares two things we wrote, so all of them are consistent with the whole
    ladder having the wrong equation; this one is not.

    It cannot run at fp64: the vendored function spells `.float()`, so it computes
    in fp32 whatever it is handed. So the gate is upstream-at-fp32 against
    ours-at-fp64, and the residual is fp32 rounding — which is why the direction of
    evidence only ever runs one way. This rung says the equation is right. It says
    nothing about precision, and `attn_res_reference.py` is the root for that.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(num_tokens, hidden_size, num_sealed, score_scale)

    expected = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    actual = hf_attn_res(prefix_sum, block_residual, norm_weight, proj_weight, EPS)

    assert actual.shape == expected.shape
    assert actual.dtype == prefix_sum.dtype
    assert _rel_err(actual, expected) <= FP32_DOT_TOL
    assert _pcc(actual, expected) >= 1.0 - 1e-9


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
@pytest.mark.parametrize("score_scale", [1.0, 8.0], ids=["scores-o1", "scores-saturated"])
def test_folded_matches_reference_exactly_in_fp64(num_tokens, hidden_size, num_sealed, score_scale):
    """Rung 1, algebra arm — the fold and the rsqrt pull-out are exact.

    Both sides run in fp64, so the only difference left is summation order. This
    is what proves the reassociation rather than merely failing to detect it.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(
        num_tokens, hidden_size, num_sealed, score_scale, dtype=torch.float64
    )

    expected = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    actual = attn_res(prefix_sum, block_residual, fold_query(norm_weight, proj_weight), EPS)

    assert actual.shape == expected.shape
    assert actual.dtype == torch.float64
    assert _rel_err(actual, expected) <= FP64_DOT_TOL


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
@pytest.mark.parametrize("score_scale", [1.0, 8.0], ids=["scores-o1", "scores-saturated"])
def test_folded_matches_reference_in_fp32(num_tokens, hidden_size, num_sealed, score_scale):
    """Rung 1, precision arm — at fp32 the fold stays at the dot-product floor."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(num_tokens, hidden_size, num_sealed, score_scale)

    expected = ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    actual = attn_res(prefix_sum, block_residual, fold_query(norm_weight, proj_weight), EPS)

    assert _rel_err(actual, expected) <= FP32_DOT_TOL
    assert _pcc(actual, expected) >= 1.0 - 1e-9


@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
def test_fold_does_not_degrade_accuracy(num_sealed):
    """The fold is a load-time transform, so it must not cost accuracy.

    Both forms run in fp32 against the same fp64 ground truth: the unfolded form
    is the reference held down to fp32, the folded form is what ships.
    Equal-or-better is the claim D5 makes; the 4x slack absorbs which one happens
    to round luckier, and the noise-floor arm keeps the ratio meaningful when the
    unfolded form lands exactly on the correctly-rounded fp32 answer.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, num_sealed)

    exact = ref.read(prefix_sum.double(), block_residual.double(), norm_weight.double(), proj_weight.double(), EPS)
    unfolded_err = _rel_err(
        ref.read(prefix_sum, block_residual, norm_weight, proj_weight, EPS, dtype=torch.float32), exact
    )
    folded_err = _rel_err(attn_res(prefix_sum, block_residual, fold_query(norm_weight, proj_weight), EPS), exact)

    assert folded_err <= max(4.0 * unfolded_err, FP32_DOT_TOL), (unfolded_err, folded_err)


@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES, ids=["d256", "d7168"])
@pytest.mark.parametrize("num_sealed", SEALED, ids=[f"S{s}" for s in SEALED])
@pytest.mark.parametrize("num_reads", [1, 24], ids=["R1", "R24"])
def test_block_split_matches_direct_form(num_tokens, hidden_size, num_sealed, num_reads):
    """Rung 2 — inter_block + merge equals the direct read, for every read site."""
    prefix_sum, block_residual, norm_weight, _ = _make_case(num_tokens, hidden_size, num_sealed)
    gen = torch.Generator().manual_seed(7)
    projections = PROJ_STD * torch.randn(num_reads, 1, hidden_size, generator=gen)
    queries = torch.stack([fold_query(norm_weight, p) for p in projections])

    partials, shifts, masses = attn_res_inter_block(block_residual, queries, EPS)
    assert partials.shape == (num_reads, num_tokens, hidden_size)

    for read_idx in range(num_reads):
        merged = attn_res_merge(
            partials[read_idx],
            shifts[read_idx],
            masses[read_idx],
            prefix_sum,
            queries[read_idx],
            EPS,
        )
        direct = attn_res(prefix_sum, block_residual, queries[read_idx], EPS)
        assert _rel_err(merged, direct) <= FP32_DOT_TOL


def test_empty_snapshot_set_is_identity():
    """S == 0 is a one-candidate softmax, so both forms must return the stream."""
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, 0)
    query = fold_query(norm_weight, proj_weight)

    assert _rel_err(attn_res(prefix_sum, block_residual, query, EPS), prefix_sum) <= FP32_DOT_TOL

    partials, shifts, masses = attn_res_inter_block(block_residual, query.unsqueeze(0), EPS)
    assert torch.isneginf(shifts).all()
    assert (masses == 0).all()
    merged = attn_res_merge(partials[0], shifts[0], masses[0], prefix_sum, query, EPS)
    assert torch.isfinite(merged).all()
    assert _rel_err(merged, prefix_sum) <= FP32_DOT_TOL


def test_mixture_weights_are_row_stochastic():
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, 8)
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    probs = attn_res_scores(v, fold_query(norm_weight, proj_weight), EPS).softmax(-1)

    assert probs.shape == (64, 9)
    assert torch.allclose(probs.sum(-1), torch.ones(64), atol=1e-6)
    assert (probs >= 0).all()


def test_values_are_not_normalized():
    """The porting bug this op invites: mixing `k` instead of `v`.

    A per-candidate score is scale-invariant — `rsqrt(mean((cv)²)) · ⟨q, cv⟩`
    cancels `c`. So if the mixture were over normalized values too, rescaling one
    candidate would change nothing whatsoever. It must change the output.
    """
    prefix_sum, block_residual, norm_weight, proj_weight = _make_case(64, 7168, 4)
    query = fold_query(norm_weight, proj_weight)

    scaled = block_residual.clone()
    scaled[:, 0] *= 3.0

    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    v_scaled = torch.cat((scaled, prefix_sum.unsqueeze(1)), dim=1)
    assert _rel_err(attn_res_scores(v_scaled, query, EPS), attn_res_scores(v, query, EPS)) <= 1e-5

    baseline = attn_res(prefix_sum, block_residual, query, EPS)
    perturbed = attn_res(prefix_sum, scaled, query, EPS)
    assert _rel_err(perturbed, baseline) > 0.01


def test_lifecycle_seal_schedule():
    """Rung 3 — seals, snapshot growth and read count over a full stack."""
    hidden_size = 64
    gen = torch.Generator().manual_seed(3)
    hidden_states = torch.randn(8, hidden_size, generator=gen)
    queries = [PROJ_STD * torch.randn(hidden_size, generator=gen) for _ in range(2 * NUM_LAYERS)]

    stream = AttnResStream(hidden_states, block_size=BLOCK_SIZE)
    reads = []
    seal_layers = []
    sealed_before_layer = []

    unwrapped_read = stream.read
    stream.read = lambda q: (reads.append(stream.num_sealed), unwrapped_read(q))[1]

    for layer_idx in range(NUM_LAYERS):
        sealed_before_layer.append(stream.num_sealed)
        before = stream.num_sealed
        attn_res_layer(
            stream,
            layer_idx,
            queries[2 * layer_idx],
            queries[2 * layer_idx + 1],
            lambda h: 0.1 * h,
            lambda h: 0.1 * h,
        )
        if stream.num_sealed > before:
            seal_layers.append(layer_idx)

    assert seal_layers == list(range(0, NUM_LAYERS, BLOCK_SIZE))
    assert seal_layers == [0, 12, 24, 36, 48, 60, 72, 84]
    assert stream.num_sealed == 8
    assert sealed_before_layer[0] == 0
    assert sealed_before_layer[-1] == 8

    # 92 pre-attention reads (layer 0 skips it at S == 0) + 93 pre-MLP + 1 output.
    assert len(reads) == 2 * NUM_LAYERS - 1
    assert min(reads) == 1


def test_stack_matches_layerwise_walk():
    """`attn_res_stack` is the same walk as driving `attn_res_layer` by hand."""
    num_layers = 25
    hidden_size = 64
    gen = torch.Generator().manual_seed(11)
    hidden_states = torch.randn(8, hidden_size, generator=gen)
    q_pre = [PROJ_STD * torch.randn(hidden_size, generator=gen) for _ in range(num_layers)]
    q_post = [PROJ_STD * torch.randn(hidden_size, generator=gen) for _ in range(num_layers)]
    q_out = PROJ_STD * torch.randn(hidden_size, generator=gen)
    attn_fns = [lambda h, s=0.1 + 0.001 * i: s * h for i in range(num_layers)]
    mlp_fns = [lambda h, s=0.05 + 0.001 * i: s * h for i in range(num_layers)]

    stacked = attn_res_stack(hidden_states, q_pre, q_post, q_out, attn_fns, mlp_fns)

    stream = AttnResStream(hidden_states, block_size=BLOCK_SIZE)
    for layer_idx in range(num_layers):
        attn_res_layer(stream, layer_idx, q_pre[layer_idx], q_post[layer_idx], attn_fns[layer_idx], mlp_fns[layer_idx])
    manual = stream.read(q_out)

    assert torch.equal(stacked, manual)
