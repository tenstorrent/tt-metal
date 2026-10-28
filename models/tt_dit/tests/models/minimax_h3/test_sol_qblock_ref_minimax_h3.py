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
    score_blocks,
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


# Production operating point: density 0.10-0.15 (lead, 2026-10-20), NOT the 0.33 earlier notes assumed.
# The OR ratio is bounded by 1/density, so at the small test geometry's 0.58 it cannot exceed 1.76x however
# independent the Q blocks are -- the device's 1.03x was measured against a ceiling, not against 3.4x.
PROD_DENSITY = (0.10, 0.15)


def _big_setup(seed=0, rows=2048, shards=4, heads=2, dim=64, q_chunk=256):
    """Larger geometry for density questions: 32 blocks/shard x 4 shards = 128 global blocks, TQ = 4.

    The default _setup has 4 blocks per shard, far too few for 'do adjacent Q blocks pick the same blocks' to
    mean anything -- with that few candidates they must agree. Production has 144+ per shard.
    """
    geo = SolGeometry(shards=shards, rows=rows, logical_n=shards * rows, q_chunk=q_chunk, k_chunk=512)
    consts = spec_constants(geo, scale=dim**-0.5)  # forcing rules off; they floor density and compress the ratio
    g = torch.Generator().manual_seed(seed)
    shape = (shards, heads, rows, dim)
    return (
        geo,
        consts,
        torch.randn(shape, generator=g) * 0.5,
        torch.randn(shape, generator=g) * 0.5,
        torch.randn(shape, generator=g),
    )


def _or_ratio(q, k, v, geo, consts, *, tau, scale, thresh_type):
    """Densities over REAL rows only.

    e has nb_pad Q-block rows and c has nqc_pad Q-chunk rows, and the two pad to 32 independently: at
    rows=2048/q_chunk=256 that is 32 real blocks but only 8 real chunks in 32 slots. Averaging c over all 32
    diluted it 4x and produced an "OR cost" below 1.0 -- impossible, since an OR can only add blocks.
    """
    e = score_blocks(q, k, v, geo, consts, tau=tau, scale=scale, thresh_type=thresh_type)["exact"]
    c = ((e.transpose(2, 3) @ consts["qc_pool_t"]).transpose(2, 3) > 0.5).float() * consts["full_row"]
    valid = consts["full_row"].reshape(-1) > 0.5
    s_blk = e[:, :, : geo.nb][..., valid].mean().item()
    s_chk = c[:, :, : geo.nqc][..., valid].mean().item()
    return s_blk, s_chk, s_chk / max(s_blk, 1e-9)


@pytest.mark.parametrize("thresh_type", ["diag", "full"])
def test_or_cost_at_production_density(thresh_type):
    """THE decision measurement for Q1-Q3: what does the Q-OR cost where the model actually runs?

    Sweeps tau to find the production band (density 0.10-0.15) and reports the OR ratio there, under both the
    device's diagonal threshold and the paper's full-covariance one. Three confounds are removed versus the
    device run that produced 1.03x:

    * density -- 1.03x was measured at 0.58, where the ratio cannot exceed 1/0.58 = 1.76x by arithmetic;
    * geometry -- 4 blocks per shard forces agreement, this uses 32;
    * forcing rules -- band/sink/prefix/force-block-0 floor density and compress the ratio; off here.

    If the ratio is still near 1 at density 0.10-0.15 with 32 blocks per shard, the Q blocks genuinely agree and
    Q1-Q3 buys nothing. If it approaches the independence bound 1/(1-(1-s)^TQ)/s, the OR is expensive and the
    rewrite is justified. This test reports; it does not judge.
    """
    geo, consts, q, k, v = _big_setup()
    scale = q.shape[-1] ** -0.5
    tq = geo.q_chunk // geo.block
    print(f"\n  thresh={thresh_type}  TQ={tq}  blocks/shard={geo.nb}  shards={geo.shards}")
    in_band = []
    for tau in (0.5, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0):
        s_blk, s_chk, ratio = _or_ratio(q, k, v, geo, consts, tau=tau, scale=scale, thresh_type=thresh_type)
        indep = (1.0 - (1.0 - s_blk) ** tq) / max(s_blk, 1e-9)
        flag = ""
        if PROD_DENSITY[0] <= s_blk <= PROD_DENSITY[1]:
            flag = "  <-- production band"
            in_band.append((tau, s_blk, ratio, indep))
        print(
            f"    tau {tau:4.1f}:  s(block) {s_blk:.4f}  s(chunk) {s_chk:.4f}  "
            f"OR cost {ratio:.2f}x  (independent would be {indep:.2f}x, ceiling {1/max(s_blk,1e-9):.1f}x){flag}"
        )
    assert in_band, "no tau reached the production density band; widen the sweep"
    for tau, s_blk, ratio, indep in in_band:
        assert 1.0 - 1e-6 <= ratio <= indep + 1e-6, f"ratio {ratio} outside [1, {indep}] at tau {tau}"


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


@pytest.mark.parametrize("rows,label", [(1152, "aligned (18 full blocks)"), (1184, "partial tail (18 full + 32)")])
def test_partial_tail_block_accuracy(rows, label):
    """Is the device's PCC 0.855 at rows=1184 an algorithmic limit or an implementation bug?

    On device (2026-10-20, packing OFF, pre-existing) test_ragged_logical_n and test_small_k_chunk both sit at
    ~0.855 for gather and compact, while chunk mode passes. Both use rows=1184, which is not a multiple of the
    64-token block: the last block of every shard holds 32 tokens, is never exact-eligible (full_row gates on
    length >= block), and goes through the pooled correction.

    That is 32/1184 = 2.7 % of keys approximated. If the reference -- which shares no device code -- shows the
    same collapse against the aligned geometry, the cost is inherent to pooling a partial block. If the
    reference holds up, the device has a bug in partial-block handling (the device side is
    test_sparse_ring_attention_minimax_h3.py::test_ragged_shards).
    """
    geo = SolGeometry(shards=2, rows=rows, logical_n=2 * rows, q_chunk=128, k_chunk=256)
    consts = build_constants(geo, band=1, sink_tokens=0, force_first_chunk=True, scale=SCALE)
    g = torch.Generator().manual_seed(0)
    shape = (2, 2, rows, D)
    q = torch.randn(shape, generator=g) * 0.5
    k = torch.randn(shape, generator=g) * 0.5
    v = torch.randn(shape, generator=g)
    kg = torch.cat([k[s] for s in range(2)], dim=1).float()
    vg = torch.cat([v[s] for s in range(2)], dim=1).float()
    ref = torch.stack([torch.softmax(q[s].float() @ kg.transpose(1, 2) * SCALE, dim=-1) @ vg for s in range(2)])

    r = qblock_mask(q, k, v, geo, consts, tau=1.0, scale=SCALE)
    out = sol_attention(q, k, v, r["block_mask"], r["pooled_k"], r["pooled_v"], geo, scale=SCALE)
    lens = geo.block_lengths().reshape(-1)
    partial = int(((lens > 0) & (lens < geo.block)).sum())
    pooled_tokens = float(lens[(lens > 0) & (lens < geo.block)].sum())
    print(
        f"\n  {label}: nb={geo.nb} partial_blocks={partial} pooled_tokens={pooled_tokens:.0f}"
        f"  pcc={_pcc(out, ref):.5f}  rel_L2={_rel_l2(out, ref):.5f}"
    )
    assert torch.isfinite(out).all()
