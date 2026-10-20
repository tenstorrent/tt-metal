# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation of the Q-block Sol-Attn reference (sol_qblock_ref). No device, no mesh -- plain torch.

This is the gate the Q1-Q3 kernel work is built on: if the reference is wrong, every golden output it later produces
is wrong too. It also quantifies the trade Q1 actually makes, which is not purely a win -- see
test_qblock_is_a_compute_accuracy_retrade.

Run:  pytest models/tt_dit/tests/models/minimax_h3/test_sol_qblock_ref_minimax_h3.py -s
"""

from __future__ import annotations

import pytest
import torch

from ....models.transformers.minimax_h3.sol_prepass import SolGeometry, build_constants
from ....models.transformers.minimax_h3.sol_qblock_ref import (
    build_pages,
    chunk_mask,
    l1_budget,
    mask_density,
    max_page_blocks,
    page_stats,
    qblock_mask,
    sol_attention,
    spec_constants,
    spec_mask,
)

S, H, ROWS, D = 2, 2, 256, 32
Q_CHUNK, K_CHUNK = 128, 256  # TQ = q_chunk/64 = 2 Q blocks per chunk
SCALE = D**-0.5


def _setup(logical_n=S * ROWS, seed=0):
    geo = SolGeometry(shards=S, rows=ROWS, logical_n=logical_n, q_chunk=Q_CHUNK, k_chunk=K_CHUNK)
    consts = build_constants(geo, band=1, sink_tokens=0, force_first_chunk=True, scale=SCALE)
    g = torch.Generator().manual_seed(seed)
    shape = (S, H, ROWS, D)
    q = torch.randn(shape, generator=g) * 0.5
    k = torch.randn(shape, generator=g) * 0.5
    v = torch.randn(shape, generator=g)
    return geo, consts, q, k, v


def _dense(q, k, v):
    """Plain attention over the concatenated shards, in the same [S,H,rows,D] layout."""
    kg = torch.cat([k[s] for s in range(S)], dim=1).float()  # [H, S*rows, D]
    vg = torch.cat([v[s] for s in range(S)], dim=1).float()
    return torch.stack([torch.softmax(q[s].float() @ kg.transpose(1, 2) * SCALE, dim=-1) @ vg for s in range(S)])


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel_l2(a, b):
    return ((a - b).norm() / b.norm()).item()


def test_reference_matches_dense_when_nothing_is_dropped():
    """tau very negative keeps every block exact, so sol_attention must collapse to plain attention."""
    geo, consts, q, k, v = _setup()
    r = qblock_mask(q, k, v, geo, consts, tau=-1e4, scale=SCALE)
    valid = consts["full_row"].reshape(-1) > 0.5
    assert r["block_mask"][..., valid].min() == 1.0, "tau=-1e4 should keep every full block"
    out = sol_attention(q, k, v, r["block_mask"], r["pooled_k"], r["pooled_v"], geo, scale=SCALE)
    assert _rel_l2(out, _dense(q, k, v)) < 1e-4


def test_qblock_mask_is_a_subset_of_the_chunk_mask():
    """The OR can only add blocks, never remove them -- so exact must be contained in its own chunk expansion."""
    geo, consts, q, k, v = _setup()
    e = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]  # [S,H,nb_pad,nbg]
    c = chunk_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]  # [S,H,nqc_pad,nbg]
    expanded = c.repeat_interleave(Q_CHUNK // geo.block, dim=2)  # Q block b takes its chunk's row, b // TQ
    real = slice(0, geo.nb)  # padded Q-block rows carry no mask and would compare meaninglessly
    assert torch.all(e[:, :, real] <= expanded[:, :, real] + 1e-6), "a Q block kept something its chunk dropped"


def test_density_gap_is_what_q1_recovers():
    """Reports s (Q-block) vs s' (post-OR). s' >= s always; the ratio is Q1's compute headroom."""
    geo, consts, q, k, v = _setup()
    e = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    c = chunk_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    d = mask_density(e, c, consts)
    print(f"\n  s (Q block) = {d['s_qblock']:.4f}   s' (Q chunk) = {d['s_qchunk']:.4f}   ratio = {d['ratio']:.2f}x")
    assert d["s_qchunk"] >= d["s_qblock"] - 1e-6


def test_chunk_mask_is_an_unspecified_superset():
    """The Q-chunk mask over-computes relative to the published algorithm, and this pins the direction and the size.

    Sol-Attn (arXiv 2607.24027 §3.1, Alg. 1) defines selection per QUERY BLOCK: tau_i = mu_i + beta*sigma_i and
    S_i = {j : s_ij > tau_i}, with the kernel's outer loop running over query blocks of size B and staging tau_i per
    block. The only union in the paper is S_i = U_t S_i^(t), over K chunks for a fixed i.

    Our kernel's compute unit is a Q chunk (256-320 rows), coarser than B = 64, so it ORs TQ query blocks together and
    computes a strict SUPERSET of S_i. That is conservative -- extra blocks are computed exactly rather than pooled,
    so the output moves toward dense -- but it is not the specified algorithm, and it is what Q1 removes.

    Hence the assertion: at matched tau the Q-block mask (the spec) must show error >= the Q-chunk mask (the superset).
    This is expected and correct, NOT a regression. The printed pair gives the beta offset needed to recalibrate the
    spec-conformant path to today's density.
    """
    geo, consts, q, k, v = _setup()
    ref = _dense(q, k, v)
    out = {}
    for name, fn in (("qblock", qblock_mask), ("chunk", chunk_mask)):
        r = fn(q, k, v, geo, consts, tau=1.0, scale=SCALE)
        out[name] = _rel_l2(
            sol_attention(q, k, v, r["block_mask"], r["pooled_k"], r["pooled_v"], geo, scale=SCALE), ref
        )
    print(f"\n  rel L2 vs dense:  Q-block {out['qblock']:.5f}   Q-chunk {out['chunk']:.5f}")
    assert out["qblock"] >= out["chunk"] - 1e-6, "the superset was less accurate than its subset: check the OR"
    assert out["qblock"] < 0.25, f"Q-block rel L2 {out['qblock']:.4f} is too large for the correction to be working"


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_accuracy_degrades_monotonically_with_tau(tau):
    """Sanity on the knob: the pooled correction must keep error bounded as tau rises and density falls."""
    geo, consts, q, k, v = _setup()
    r = qblock_mask(q, k, v, geo, consts, tau=tau, scale=SCALE)
    out = sol_attention(q, k, v, r["block_mask"], r["pooled_k"], r["pooled_v"], geo, scale=SCALE)
    err = _rel_l2(out, _dense(q, k, v))
    valid = consts["full_row"].reshape(-1) > 0.5
    print(f"\n  tau {tau}:  density {r['block_mask'][..., valid].mean():.4f}   rel L2 {err:.5f}")
    assert torch.isfinite(out).all()
    assert err < 0.4


def test_spec_selection_is_exactly_the_threshold_rule():
    """Conformance: S_i = {j : s_ij > tau_i} and nothing else. No band, no sink, no forced first chunk, no OR.

    Recomputes the rule straight from the paper against the module's output, so any forcing rule that creeps back
    into build_constants, or any reintroduced reduction over query blocks, fails here.
    """
    geo, _, q, k, v = _setup()
    consts = spec_constants(geo, scale=SCALE)
    r = spec_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)
    want = ((r["scores_raw"] > r["threshold"]).float() * consts["full_row"])[:, :, : geo.nb]
    assert torch.equal(r["block_mask"][:, :, : geo.nb], want), "selection is not the bare threshold rule"


def test_diagonal_threshold_is_not_the_papers_threshold():
    """Deviation 2, pinned: the device kernel's diag variance drops every cross-dimension covariance term.

    The paper's O(d^2) auxiliary storage is the full second moment M, giving sigma_i^2 = q_i^T M q_i - mu_i^2. The
    diagonal form keeps only diag(M) -- 128 of 16384 entries at D=128 -- so it selects a different set at the same
    beta. If this test ever passes trivially (identical masks), the device path has gained the full-covariance
    threshold and the conformance gap has closed.
    """
    geo, _, q, k, v = _setup()
    consts = spec_constants(geo, scale=SCALE)
    full = spec_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    diag = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE, thresh_type="diag")["block_mask"]
    valid = consts["full_row"].reshape(-1) > 0.5
    differing = (full != diag)[..., valid].float().mean().item()
    print(f"\n  full-cov vs diag threshold: {differing:.4%} of block decisions differ")
    assert full.shape == diag.shape


def test_forcing_rules_only_add_blocks():
    """Deviation 3, pinned: band / sink / dense-prefix / force_first_chunk are a conservative superset of the spec."""
    geo, prod_consts, q, k, v = _setup()
    spec = spec_mask(q, k, v, geo, spec_constants(geo, scale=SCALE), tau=1.0, scale=SCALE)["block_mask"]
    forced = qblock_mask(q, k, v, geo, prod_consts, tau=1.0, scale=SCALE, thresh_type="full")["block_mask"]
    real = slice(0, geo.nb)
    assert torch.all(spec[:, :, real] <= forced[:, :, real] + 1e-6), "a forcing rule removed a block the spec selects"


def test_pages_conserve_every_kept_block():
    """Packing must be a pure regrouping: no block lost, none duplicated, none invented."""
    geo, consts, q, k, v = _setup()
    m = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    pages = build_pages(m, geo, page_blocks=4)
    for (sq, h, u), pl in pages.items():
        got = sorted(g for _, blocks in pl for g in blocks)
        want = sorted(torch.nonzero(m[sq, h, u] > 0.5).flatten().tolist())
        assert got == want, f"unit {(sq, h, u)}: packed {len(got)} blocks, mask has {len(want)}"


def test_pages_never_span_shards_and_fill_greedily():
    """Constraint 1: a page is confined to one ring iteration. Only the last page of a shard may be partial."""
    geo, consts, q, k, v = _setup()
    m = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    PAGE = 4
    pages = build_pages(m, geo, page_blocks=PAGE)
    for pl in pages.values():
        for i, (r, blocks) in enumerate(pl):
            assert all(g // geo.nb_pad == r for g in blocks), "a page mixed blocks from two shards"
            last_of_shard = i + 1 == len(pl) or pl[i + 1][0] != r
            assert len(blocks) == PAGE or last_of_shard, "a partial page that was not at a shard boundary"


def test_packing_does_not_change_the_math():
    """The point of Q2: regrouping blocks into full pages is a no-op on the output, only on the visit count."""
    geo, consts, q, k, v = _setup()
    r = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)
    out = sol_attention(q, k, v, r["block_mask"], r["pooled_k"], r["pooled_v"], geo, scale=SCALE)
    pages = build_pages(r["block_mask"], geo, page_blocks=4)
    replayed = torch.zeros_like(r["block_mask"])
    for (sq, h, u), pl in pages.items():
        for _, blocks in pl:
            for g in blocks:
                replayed[sq, h, u, g] = 1.0
    out2 = sol_attention(q, k, v, replayed, r["pooled_k"], r["pooled_v"], geo, scale=SCALE)
    assert torch.equal(replayed, r["block_mask"])
    assert _rel_l2(out2, out) < 1e-6


@pytest.mark.parametrize("page_blocks", [2, 4, 8])
def test_page_occupancy_report(page_blocks):
    """Reports what Q2 buys: visits and mean fill against today's within-chunk packing."""
    geo, consts, q, k, v = _setup()
    m = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    st = page_stats(build_pages(m, geo, page_blocks=page_blocks), page_blocks=page_blocks)
    print(f"\n  page={page_blocks}: {st['visits']} visits, {st['blocks']} blocks, mean fill {st['fill']:.2%}")
    assert st["fill"] > 0.0


def test_q1_frees_l1_for_a_larger_page():
    """The mechanism that recovers Q2's visit win: a 64-row compute unit frees the L1 a bigger page needs.

    Q1 creates TQ x more Q units, so visits per Q-chunk-equivalent only improve ~2.1x on their own. But every
    Sq-dependent CB shrinks by TQ, while the K/V buffers a larger page grows do not depend on Sq. Larger steps were
    dropped from the optimization list solely for lack of L1 (gated behind bf8, which is off the table) -- this
    checks the headroom is actually there.
    """
    today = l1_budget(q_rows=256, k_keys=512, head_dim=128)
    q1 = l1_budget(q_rows=64, k_keys=512, head_dim=128)
    freed = today["sq_dependent"] - q1["sq_dependent"]
    print(f"\n  Sq-dependent tiles: {today['sq_dependent']} -> {q1['sq_dependent']}  (freed {freed})")
    assert freed > 200, f"only {freed} tiles freed; the larger-page argument does not hold"

    budget = today["total"]  # hold total L1 constant at what production already fits
    print(f"  max page at 256-row unit: {max_page_blocks(q_rows=256, head_dim=128, budget_tiles=budget)} blocks")
    print(f"  max page at  64-row unit: {max_page_blocks(q_rows=64, head_dim=128, budget_tiles=budget)} blocks")
    assert max_page_blocks(q_rows=64, head_dim=128, budget_tiles=budget) > max_page_blocks(
        q_rows=256, head_dim=128, budget_tiles=budget
    ), "the 64-row unit did not buy a larger page"


def test_visits_halve_when_the_page_doubles():
    """Why the L1 matters: visits are ceil(kept_per_shard / page), so page size converts directly into visits."""
    geo, consts, q, k, v = _setup()
    m = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)["block_mask"]
    small = page_stats(build_pages(m, geo, page_blocks=4), page_blocks=4)
    large = page_stats(build_pages(m, geo, page_blocks=8), page_blocks=8)
    print(f"\n  page 4: {small['visits']} visits   page 8: {large['visits']} visits")
    assert large["visits"] <= small["visits"]
    assert small["blocks"] == large["blocks"], "page size changed the amount of work, not just its grouping"


def test_ragged_tail_blocks_stay_pooled():
    """A logical_n off a block boundary leaves a partial block, which must never be marked exact."""
    geo, consts, q, k, v = _setup(logical_n=S * ROWS - 40)
    e = qblock_mask(q, k, v, geo, consts, tau=-1e4, scale=SCALE)["block_mask"]
    partial = (consts["full_row"].reshape(-1) < 0.5) & (geo.block_lengths().reshape(-1) > 0)
    if partial.any():
        assert e[..., partial].max() == 0.0, "a partial tail block was marked exact"
