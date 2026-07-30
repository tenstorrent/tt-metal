# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rung 4 of the AttnRes numeric ladder: `tt/` against `torch_functional/`.

The torch reference is the oracle here, not a second opinion — rungs 1–3 already
tied it to the vendored upstream function. PCC ≥ 0.9999 is the inherited op-level
gate; the depth-compounding gate is relative and lives in the Phase-6 harness.
"""

import pytest
import torch
import ttnn

from models.experimental.kimi_k3_attn_res.torch_functional.attn_res import (
    EPS,
    attn_res,
    attn_res_inter_block,
    attn_res_scores,
)
from models.experimental.kimi_k3_attn_res.tt.attn_res import TtAttnRes

PCC_GATE = 0.9999
PROJ_STD = 0.02

# bf16 storage is 2**-8 = 3.9e-3 relative per element; the `d`-length reductions
# and the softmax carry that to a measured 7e-3 on the `inter_block` statistics.
# An algebra error is O(1), so this gate still fails one by orders of magnitude.
STAT_REL_TOL = 2e-2

# `PCC_GATE` is calibrated for scores of order 1, which is where the folded query
# puts them. The saturated-score test drives them to 120 on purpose and gates
# relatively instead: the device trails torch-bf16 by ~1.3e-4 there versus ~1.5e-5
# at order-1 scores, so an absolute gate would be measuring bf16, not our kernels.
SATURATED_PCC_SLACK = 1e-3

SHAPES = [(64, 256), (64, 7168)]
SEALED = [0, 1, 4, 8]


def _pcc(a, b):
    """Correlation in fp64. In fp32 `corrcoef` caps near 0.99999988 on ~458 k
    elements even for bit-identical inputs, which reads as a dimension-dependent
    op bug."""
    stacked = torch.stack((a.double().reshape(-1), b.double().reshape(-1)))
    return torch.corrcoef(stacked)[0, 1].item()


def _rel_err(got, want):
    """Max absolute deviation over the reference's peak magnitude. PCC cannot
    gate these: at `S == 1` the reference mass is exactly 1.0 for every token,
    and the correlation of a constant vector is nan. `shift` passes through zero,
    so normalize by the peak rather than elementwise."""
    got, want = got.double(), want.double()
    return ((got - want).abs().max() / want.abs().max()).item()


def _make_case(num_tokens, hidden_size, num_sealed, seed=0):
    """Queries are built the way the model builds them — an RMSNorm gain near 1
    times a `std = 0.02` projection — so scores land at order 1. A unit-variance
    query gives `<q, v> ~ ±sqrt(d)`, which saturates the softmax to one-hot and
    makes every gate downstream vacuous."""
    generator = torch.Generator().manual_seed(seed)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    return (
        randn(num_tokens, hidden_size),
        randn(num_tokens, num_sealed, hidden_size),
        (1.0 + 0.1 * randn(hidden_size)) * (PROJ_STD * randn(hidden_size)),
    )


def _to_device(mesh_device, prefix_sum, block_residual, query):
    """torch `[N, d]` / `[N, S, d]` -> ttnn `[1, 1, N, d]` / `[1, S, N, d]`.

    Candidates move to dim 1; `block_residual` with no snapshots becomes None,
    since ttnn has no zero-extent dimension."""
    to_tt = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
        to_tt(query.reshape(1, 1, 1, -1)),
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES)
@pytest.mark.parametrize("num_sealed", SEALED)
def test_forward_matches_torch_reference(mesh_device, num_tokens, hidden_size, num_sealed):
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, num_sealed)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)

    tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, block_residual, query)
    got = ttnn.to_torch(op.forward(tt_prefix, tt_block, tt_query)).reshape(num_tokens, hidden_size)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    case = f"S={num_sealed} d={hidden_size}"
    pcc = _pcc(got, want)
    assert pcc >= PCC_GATE, f"{case}: PCC {pcc:.7f} < {PCC_GATE}"

    # PCC is scale-invariant, so it cannot see lost softmax mass: weights summing
    # to 0.96 scale the output by 0.96 and PCC stays above 0.9999. That is not
    # hypothetical — it is what `ttnn.softmax(dim=1)` does. Gate the magnitude too.
    rel_err = _rel_err(got, want)
    assert rel_err <= STAT_REL_TOL, f"{case}: rel err {rel_err:.3e} > {STAT_REL_TOL}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES)
@pytest.mark.parametrize("num_sealed", [1, 4, 8])
@pytest.mark.parametrize("num_reads", [1, 24])
def test_split_matches_forward_on_device(mesh_device, num_tokens, hidden_size, num_sealed, num_reads):
    """`inter_block` + `merge` must equal `forward` for every read site in a
    block. This is the only independent check on the online-softmax merge
    algebra, so unlike KDA's composed op it is not disposable.

    `num_reads = 24` is the real per-block count: 12 layers x 2 read sites."""
    prefix_sum, block_residual, _ = _make_case(num_tokens, hidden_size, num_sealed)
    queries = [_make_case(num_tokens, hidden_size, 0, seed=100 + r)[2] for r in range(num_reads)]
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, torch_queries=queries)

    tt_prefix, tt_block, _ = _to_device(mesh_device, prefix_sum, block_residual, queries[0])
    partials, shifts, masses = op.inter_block(tt_block, op.queries)
    assert len(partials) == num_reads

    for r, query in enumerate(queries):
        merged = op.merge(partials[r], shifts[r], masses[r], tt_prefix, op.queries[r])
        got = ttnn.to_torch(merged).reshape(num_tokens, hidden_size)
        want = attn_res(prefix_sum, block_residual, query, EPS)
        pcc = _pcc(got, want)
        assert pcc >= PCC_GATE, f"read {r} S={num_sealed} d={hidden_size}: PCC {pcc:.7f} < {PCC_GATE}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_sealed", [1, 8])
def test_split_statistics_match_torch(mesh_device, num_sealed):
    """Gate the `inter_block` intermediates directly. `merge` can absorb a wrong
    shift into a compensating rescale and still land on the right output, so
    checking only the merged result would not pin down `m` and `Z`."""
    num_tokens, hidden_size = 64, 256
    _, block_residual, query = _make_case(num_tokens, hidden_size, num_sealed)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size, torch_queries=[query])

    tt_block = ttnn.from_torch(
        block_residual.permute(1, 0, 2).unsqueeze(0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    partials, shifts, masses = op.inter_block(tt_block, op.queries)
    want_partial, want_shift, want_mass = attn_res_inter_block(block_residual, query.reshape(1, -1), EPS)

    for name, got_tt, want in (
        ("partial", partials[0], want_partial[0]),
        ("shift", shifts[0], want_shift[0]),
        ("mass", masses[0], want_mass[0]),
    ):
        got = ttnn.to_torch(got_tt).reshape(want.shape)
        rel_err = _rel_err(got, want)
        assert rel_err <= STAT_REL_TOL, f"{name} S={num_sealed}: rel err {rel_err:.3e} > {STAT_REL_TOL}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_candidates", [1, 5, 9])
def test_mixture_weights_are_row_stochastic(mesh_device, num_candidates):
    """`sum_i alpha_i = 1` per token. The primary gate on the mixture weights —
    PCC is scale-invariant and cannot see lost mass."""
    num_tokens = 64
    scores = torch.randn(1, num_candidates, num_tokens, 1, generator=torch.Generator().manual_seed(7))
    op = TtAttnRes(mesh_device, hidden_size=256)

    tt_scores = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    weights = ttnn.to_torch(op._softmax_over_candidates(tt_scores)).double()

    mass_error = (weights.sum(dim=1) - 1.0).abs().max().item()
    assert mass_error <= 1e-2, f"C={num_candidates}: weights sum to 1 +/- {mass_error:.3e}"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_hand_rolled_softmax_beats_fused(mesh_device):
    """`forward` hand-rolls the candidate softmax instead of calling
    `ttnn.softmax(dim=1)`. Hold that choice to its justification.

    `ttnn.softmax` reaches its attention-optimized kernel only when reducing the
    last dim; the dim-1 fallback loses ~4% of the mass even in fp32. Asserting
    the hand-rolled path is no worse means this fails exactly when the fused path
    becomes the better option — the signal to switch — and never merely because
    upstream improved."""
    num_candidates, num_tokens = 9, 64
    scores = torch.randn(1, num_candidates, num_tokens, 1, generator=torch.Generator().manual_seed(7))
    want = scores.double().softmax(dim=1)
    op = TtAttnRes(mesh_device, hidden_size=256)

    tt_scores = ttnn.from_torch(scores, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    rolled = _rel_err(ttnn.to_torch(op._softmax_over_candidates(tt_scores)), want)
    fused = _rel_err(ttnn.to_torch(ttnn.softmax(tt_scores, dim=1)), want)

    assert (
        rolled <= fused
    ), f"ttnn.softmax(dim=1) is now more accurate ({fused:.3e} vs {rolled:.3e}) — reconsider the fused path"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("num_tokens, hidden_size", SHAPES)
def test_saturated_scores_do_not_overflow(mesh_device, num_tokens, hidden_size):
    """Gate the max-subtraction in `_softmax_over_candidates`.

    Every other test here runs at scores of order 1, where dropping the shift
    changes nothing — `exp` simply does not overflow. The query is rescaled to put
    `max|score|` near 120, past `exp`'s finite range in both bf16 and fp32, so an
    unshifted softmax yields `inf/inf = nan` instead of a slightly worse number.

    Accuracy is gated against a torch-bf16 arm, not the absolute `PCC_GATE`. At
    `|score| = 120` bf16's mantissa step is 0.5, so candidate score *differences*
    quantize to 0.5 and `exp(-delta)` shifts by 1.65x. The device pays that on the
    bf16 `v*q` product while the torch reference computes scores in fp32 from bf16
    storage, and closing the gap would mean an fp32 `[1, C, N, d]` intermediate —
    a 2x cost on the op's largest tensor for a regime the folded query
    (`|score| ~ 5`) never enters."""
    target_score = 120.0
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 8)
    query = query * (target_score / attn_res_scores(block_residual.float(), query.float(), EPS).abs().max())
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)

    tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, block_residual, query)
    got = ttnn.to_torch(op.forward(tt_prefix, tt_block, tt_query)).reshape(num_tokens, hidden_size)

    assert torch.isfinite(got).all(), f"d={hidden_size}: saturated scores overflowed the softmax"

    want = attn_res(prefix_sum.float(), block_residual.float(), query.float(), EPS)
    analog = attn_res(prefix_sum.bfloat16(), block_residual.bfloat16(), query.bfloat16(), EPS).float()
    pcc, analog_pcc = _pcc(got, want), _pcc(analog, want)
    assert pcc >= analog_pcc - SATURATED_PCC_SLACK, (
        f"d={hidden_size}: saturated PCC {pcc:.7f} trails torch-bf16 {analog_pcc:.7f} "
        f"by more than {SATURATED_PCC_SLACK}"
    )


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_values_are_not_normalized(mesh_device):
    """The mixture is over raw `v`; only the key is normalized. Scaling one
    sealed snapshot must therefore change the output.

    Sharp because the score is scale-invariant —
    `rsqrt(mean((cv)^2)) * <q, cv>` cancels `c` — so the mixture weights are
    untouched and any output change comes purely from the raw value. Had the
    port normalized the values too, this test would see nothing move."""
    num_tokens, hidden_size = 64, 256
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 4)
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)

    scaled = block_residual.clone()
    scaled[:, 0, :] *= 4.0

    outputs = []
    for candidates in (block_residual, scaled):
        tt_prefix, tt_block, tt_query = _to_device(mesh_device, prefix_sum, candidates, query)
        outputs.append(ttnn.to_torch(op.forward(tt_prefix, tt_block, tt_query)).reshape(num_tokens, hidden_size))

    assert (
        outputs[0] - outputs[1]
    ).abs().max() > 1e-2, "output is invariant to candidate scale — values got normalized"


@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
def test_sharded_stream_is_rejected(mesh_device, expect_error):
    """Reductions run over `d`, so a stream sharded on it needs the statistics
    all-reduce Phase 8 adds. Fail loudly rather than reduce a shard locally."""
    op = TtAttnRes(mesh_device, hidden_size=7168)
    shard = ttnn.from_torch(
        torch.zeros(1, 1, 64, 1792), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device
    )
    with expect_error(AssertionError, "sharded on the reduction axis"):
        op.forward(shard, None, op.to_query(torch.zeros(7168)))
