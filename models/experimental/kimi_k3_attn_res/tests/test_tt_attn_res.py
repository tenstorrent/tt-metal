# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Rung 4 of the AttnRes numeric ladder: `tt/` against `torch_functional/`.

The torch reference is the oracle here, not a second opinion — rungs 0–3 already tied
it to the unfolded fp64 ground truth, and rung 0b tied that to upstream. Note what that
means for this rung: it shares the fold and the rsqrt pull-out with the device op, so it
gates numerics and plumbing, never the algebra. PCC >= 0.9999 is the inherited op-level
gate; the depth-compounding gate is relative and lives in the Phase-6 harness.

Every arm is sharded and every arm runs 640 rows per chip. Production splits `d` across
the TP axis, so `tp_factor == 1` is not a smaller version of the shipped op — it makes
`_reduce_stats` the identity and takes the score chain down a path the model never
executes. A green single-device arm certifies nothing about the op that ships.

`d` is walked here and pinned at 7168 in `test_tt_attn_res_distributed.py`, because on
a `(2, 4)` mesh `d = 256` shards to 64 — under `ONE_PASS_SQUARES_MAX_WIDTH`, so it
takes the one-pass statistics route that 1792 does not.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.kimi_k3_attn_res.reference.hf_attn_res import hf_attn_res
from models.experimental.kimi_k3_attn_res.tests.placements import PER_CHIP_TOKENS, on_placements
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

HIDDEN_SIZES = [256, 7168]
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


def _to_device(op, prefix_sum, block_residual, query):
    """torch `[N, d]` / `[N, S, d]` -> ttnn `[1, 1, N, d]` / `[1, S, N, d]`, in the
    op's own placement.

    Candidates move to dim 1; `block_residual` with no snapshots becomes None,
    since ttnn has no zero-extent dimension."""
    to_tt = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=op.mesh_device, mesh_mapper=op.stream_mapper
    )
    return (
        to_tt(prefix_sum.unsqueeze(0).unsqueeze(0)),
        to_tt(block_residual.permute(1, 0, 2).unsqueeze(0)) if block_residual.shape[1] else None,
        op.to_query(query),
    )


def _from_device(op, tensor, hidden_size):
    return ttnn.to_torch(tensor, mesh_composer=op.stream_composer).reshape(-1, hidden_size)


def _first_shard(tensor):
    """Device 0's shard as torch. It holds SP row 0 and TP column 0, so it is the
    leading `PER_CHIP_TOKENS` tokens over the leading `shard_width` of `d` — the
    slice a torch reference can be cut to without composing anything."""
    return ttnn.to_torch(ttnn.get_device_tensors(tensor.cpu())[0])


@on_placements
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_sealed", SEALED)
def test_forward_matches_torch_reference(mesh_device, hidden_size, num_sealed, device_params):
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    assert op.shard_width == hidden_size // op.tp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, num_sealed)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    got = _from_device(op, op.forward(tt_prefix, tt_block, tt_query), hidden_size)

    want = attn_res(prefix_sum, block_residual, query, EPS)
    case = f"S={num_sealed} d={hidden_size} shard={op.shard_width}"
    pcc = _pcc(got, want)
    assert pcc >= PCC_GATE, f"{case}: PCC {pcc:.7f} < {PCC_GATE}"

    # PCC is scale-invariant, so it cannot see lost softmax mass: weights summing
    # to 0.96 scale the output by 0.96 and PCC stays above 0.9999. That is not
    # hypothetical — it is what `ttnn.softmax(dim=1)` does. Gate the magnitude too.
    rel_err = _rel_err(got, want)
    assert rel_err <= STAT_REL_TOL, f"{case}: rel err {rel_err:.3e} > {STAT_REL_TOL}"


@on_placements
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_sealed", SEALED)
def test_forward_matches_upstream_reference(mesh_device, hidden_size, num_sealed, device_params):
    """The device op against upstream's own read, skipping every reference of ours.

    Everything else in this file gates against `torch_functional`, which shares the
    fold and the rsqrt pull-out with the op — so it cannot see an algebra error the
    two forms make together. This can. `hf_attn_res` needs the two weights apart,
    which is why the case is not built by `_make_case`.
    """
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    generator = torch.Generator().manual_seed(7)
    randn = lambda *shape: torch.randn(*shape, generator=generator)
    prefix_sum = randn(num_tokens, hidden_size)
    block_residual = randn(num_tokens, num_sealed, hidden_size)
    norm_weight = 1.0 + 0.1 * randn(hidden_size)
    proj_weight = PROJ_STD * randn(1, hidden_size)

    query = norm_weight * proj_weight.reshape(-1)
    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    got = _from_device(op, op.forward(tt_prefix, tt_block, tt_query), hidden_size)

    want = hf_attn_res(prefix_sum, block_residual, norm_weight, proj_weight, EPS)
    case = f"S={num_sealed} d={hidden_size}"
    pcc = _pcc(got, want)
    assert pcc >= PCC_GATE, f"{case}: PCC {pcc:.7f} < {PCC_GATE}"

    rel_err = _rel_err(got, want)
    assert rel_err <= STAT_REL_TOL, f"{case}: rel err {rel_err:.3e} > {STAT_REL_TOL}"


@on_placements
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("num_sealed", [1, 8])
@pytest.mark.parametrize("num_reads", [1, 24])
def test_split_matches_forward_on_device(mesh_device, hidden_size, num_sealed, num_reads, device_params):
    """`inter_block` + `merge` must equal `forward` for every read site in a
    block. This is the only independent check on the online-softmax merge
    algebra, so unlike KDA's composed op it is not disposable.

    `num_reads = 24` is the real per-block count: 12 layers x 2 read sites. `1` is
    the degenerate site axis, where the layout the sites ride on collapses."""
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    prefix_sum, block_residual, _ = _make_case(num_tokens, hidden_size, num_sealed)
    queries = [_make_case(num_tokens, hidden_size, 0, seed=100 + r)[2] for r in range(num_reads)]
    tt_queries = [op.to_query(q) for q in queries]

    tt_prefix, tt_block, _ = _to_device(op, prefix_sum, block_residual, queries[0])
    partials, shifts, masses = op.inter_block(tt_block, tt_queries)
    assert len(partials) == num_reads

    for r, query in enumerate(queries):
        merged = op.merge(partials[r], shifts[r], masses[r], tt_prefix, tt_queries[r])
        got = _from_device(op, merged, hidden_size)
        ttnn.deallocate(merged)
        want = attn_res(prefix_sum, block_residual, query, EPS)
        pcc = _pcc(got, want)
        assert pcc >= PCC_GATE, f"read {r} S={num_sealed} d={hidden_size}: PCC {pcc:.7f} < {PCC_GATE}"


@on_placements
@pytest.mark.parametrize("num_sealed", [1, 8])
def test_split_statistics_match_torch(mesh_device, num_sealed, device_params):
    """Gate the `inter_block` intermediates directly. `merge` can absorb a wrong
    shift into a compensating rescale and still land on the right output, so
    checking only the merged result would not pin down `m` and `Z`.

    Compared on device 0's shard rather than on a composed tensor: `partial` is
    sharded on `d` while `shift` and `mass` come back replicated across the TP axis,
    so one composer cannot read all three."""
    hidden_size = 256
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, num_sealed)

    _, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    partials, shifts, masses = op.inter_block(tt_block, [tt_query])
    want_partial, want_shift, want_mass = attn_res_inter_block(block_residual, query.reshape(1, -1), EPS)

    rows, cols = PER_CHIP_TOKENS, op.shard_width
    for name, got_tt, want in (
        ("partial", partials[0], want_partial[0][:rows, :cols]),
        ("shift", shifts[0], want_shift[0][:rows]),
        ("mass", masses[0], want_mass[0][:rows]),
    ):
        got = _first_shard(got_tt).reshape(want.shape)
        rel_err = _rel_err(got, want)
        assert rel_err <= STAT_REL_TOL, f"{name} S={num_sealed}: rel err {rel_err:.3e} > {STAT_REL_TOL}"


@on_placements
@pytest.mark.parametrize("num_candidates", [1, 5, 9])
def test_mixture_weights_are_row_stochastic(mesh_device, num_candidates, device_params):
    """`sum_i alpha_i = 1` per token. The primary gate on the mixture weights —
    PCC is scale-invariant and cannot see lost mass.

    Scores carry no `d`, so they replicate rather than shard and every rank must
    reach the same answer; device 0 is read as the representative."""
    scores = torch.randn(1, num_candidates, PER_CHIP_TOKENS, 1, generator=torch.Generator().manual_seed(7))
    op = TtAttnRes(mesh_device, hidden_size=256)

    tt_scores = ttnn.from_torch(scores, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    weights = _first_shard(op._softmax_over_candidates(tt_scores)).double()

    mass_error = (weights.sum(dim=1) - 1.0).abs().max().item()
    assert mass_error <= 1e-2, f"C={num_candidates}: weights sum to 1 +/- {mass_error:.3e}"


@on_placements
def test_hand_rolled_softmax_beats_fused(mesh_device, device_params):
    """`forward` hand-rolls the candidate softmax instead of calling
    `ttnn.softmax(dim=1)`. Hold that choice to its justification.

    `ttnn.softmax` reaches its attention-optimized kernel only when reducing the
    last dim; the dim-1 fallback loses ~4% of the mass even in fp32. Asserting
    the hand-rolled path is no worse means this fails exactly when the fused path
    becomes the better option — the signal to switch — and never merely because
    upstream improved."""
    num_candidates = 9
    scores = torch.randn(1, num_candidates, PER_CHIP_TOKENS, 1, generator=torch.Generator().manual_seed(7))
    want = scores.double().softmax(dim=1)
    op = TtAttnRes(mesh_device, hidden_size=256)

    tt_scores = ttnn.from_torch(scores, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh_device)
    rolled = _rel_err(_first_shard(op._softmax_over_candidates(tt_scores)), want)
    fused = _rel_err(_first_shard(ttnn.softmax(tt_scores, dim=1)), want)

    assert (
        rolled <= fused
    ), f"ttnn.softmax(dim=1) is now more accurate ({fused:.3e} vs {rolled:.3e}) — reconsider the fused path"


@on_placements
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
def test_saturated_scores_do_not_overflow(mesh_device, hidden_size, device_params):
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
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor

    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 8)
    query = query * (target_score / attn_res_scores(block_residual.float(), query.float(), EPS).abs().max())

    tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, block_residual, query)
    got = _from_device(op, op.forward(tt_prefix, tt_block, tt_query), hidden_size)

    assert torch.isfinite(got).all(), f"d={hidden_size}: saturated scores overflowed the softmax"

    want = attn_res(prefix_sum.float(), block_residual.float(), query.float(), EPS)
    analog = attn_res(prefix_sum.bfloat16(), block_residual.bfloat16(), query.bfloat16(), EPS).float()
    pcc, analog_pcc = _pcc(got, want), _pcc(analog, want)
    logger.info(f"d={hidden_size} saturated: device PCC {pcc:.7f}, torch-bf16 {analog_pcc:.7f}")
    assert pcc >= analog_pcc - SATURATED_PCC_SLACK, (
        f"d={hidden_size}: saturated PCC {pcc:.7f} trails torch-bf16 {analog_pcc:.7f} "
        f"by more than {SATURATED_PCC_SLACK}"
    )


@on_placements
def test_values_are_not_normalized(mesh_device, device_params):
    """The mixture is over raw `v`; only the key is normalized. Scaling one
    sealed snapshot must therefore change the output.

    Sharp because the score is scale-invariant —
    `rsqrt(mean((cv)^2)) * <q, cv>` cancels `c` — so the mixture weights are
    untouched and any output change comes purely from the raw value. Had the
    port normalized the values too, this test would see nothing move."""
    hidden_size = 256
    op = TtAttnRes(mesh_device, hidden_size=hidden_size)
    num_tokens = PER_CHIP_TOKENS * op.sp_factor
    prefix_sum, block_residual, query = _make_case(num_tokens, hidden_size, 4)

    scaled = block_residual.clone()
    scaled[:, 0, :] *= 4.0

    outputs = []
    for candidates in (block_residual, scaled):
        tt_prefix, tt_block, tt_query = _to_device(op, prefix_sum, candidates, query)
        outputs.append(_from_device(op, op.forward(tt_prefix, tt_block, tt_query), hidden_size))

    assert (
        outputs[0] - outputs[1]
    ).abs().max() > 1e-2, "output is invariant to candidate scale — values got normalized"


@on_placements
def test_unexpected_shard_width_is_rejected(mesh_device, expect_error, device_params):
    """Reductions run over `d` and the op cannot infer how `d` was split, so a
    stream whose last dim disagrees with `hidden_size / tp_factor` is a
    misconfiguration. Fail loudly rather than reduce the wrong width and divide by
    the wrong `d` — the sharded path's one silent-wrong-answer hazard."""
    op = TtAttnRes(mesh_device, hidden_size=7168)
    assert op.shard_width == 7168 // op.tp_factor

    # Replicated, so every rank sees a last dim of `shard_width / 2` and none of them
    # can reduce the width the op was configured for.
    shard = ttnn.from_torch(
        torch.zeros(1, 1, PER_CHIP_TOKENS, op.shard_width // 2),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
    )
    with expect_error(AssertionError, f"expected {op.shard_width}"):
        op.forward(shard, None, op.to_query(torch.zeros(7168)))
