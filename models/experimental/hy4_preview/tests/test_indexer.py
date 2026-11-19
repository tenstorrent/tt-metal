# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device indexer (tt/indexer.py) vs the pure-torch ground truth (reference/indexer.py),
tenstorrent/tt-metal#65887 and #65888.

Stage by stage, so a failure names where it is:
    project (wq_b, wk, weights_proj) | index keys (LayerNorm, RoPE) | queries (RoPE) |
    scores over the stripes | the selection of a block of queries | of single decode queries,
then end to end from the hidden state to the selection. On one chip first (the arithmetic), then
on the 8x4 Galaxy with the keys striped block-cyclic over the 32 chips: at a small width and at
Hy4's own, at 4K context, with layer 0 / 1 / 5 indexer weights from the checkpoint, and with
synthetic keys at 128K, 200K and 1M context, where the decode step is also timed, stage by stage.
The long contexts and one-chip scoring run with both ways of summing the heads (TtHy4Indexer
``head_sum``), and 128K also with the key stripes in BFP8, the cache's default format.

Each stage is fed the reference's input for that stage. A selection (the picks' valid positions,
identical on every chip) is compared as a SET, twice:

* with the top-k of the scores the device itself computed (read back from every chip). This is the
  selection logic on its own (mask, local top-k, gather, merge, the bf16 refinement passes) and
  owes nothing to the matmul: the only disagreements allowed are scores tied at the resolution of
  the refinement passes, 2^-14 of the cut.
* with the reference's selection. The device's scores differ from torch's by the matmul's error,
  so two scores closer than that can land on either side of the cut: a disagreement on a position
  whose reference score is within 2x the measured score error of the reference's cut is counted as
  a near-tie, anything else as a real error. Real errors must be zero and near-ties stay under 1%
  of the picks (or two swaps per query, where the top-k is small). The measured score error is
  itself bounded (SCORE_ERROR_BUDGET), so that a wrong score cannot widen its own tolerance.

Every query must select exactly min(top-k, positions it sees) distinct positions, none of them after
the query or in an unfilled slot. Contexts and blocks come both tile- and block-aligned (4096 / 128)
and ragged (3000 / 37 / 1: a part-filled block, chips that hold nothing, a single new token).
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.attention import rms_norm, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import (
    NEG,
    Hy4Indexer,
    IndexerConfig,
    causal_allowed,
    compare_selection,
    full_layers,
    selection_mask,
)
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
)
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.indexer import TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

HY4 = IndexerConfig()
# Hy4's structure at 1/8 of the hidden widths; still splits 4 ways (hidden) and 8 ways (heads, key dims).
SMALL = IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8, head_dim=64, qk_rope_head_dim=32)
# The same with a top-k below a stripe's 128 slots at 4K context, so the local top-k runs there too.
SMALL_TOP64 = IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8, head_dim=64, qk_rope_head_dim=32, topk=64)
CONFIGS = {"small": SMALL, "small-top64": SMALL_TOP64, "hy4": HY4}

# What the Galaxy measured against the thresholds below (2026-11-19, one chip and 8x4, random and
# checkpoint weights, 4K to 1M context; more than 170 selections, none with a real error):
# * every stage PCC is at least 0.999995, and 0.999971 for keys held in BFP8;
# * near-ties are at most 0.29% of the picks at top-k 2048 (0.20% to 0.23% end to end at 4K, where
#   the keys also go through the device; one to three swapped positions per decode query at 128K to
#   1M) and at most 0.85% at top-k 64;
# * the score error is at most 0.34% of the largest sum of |terms| (0.19% when the keys and
#   queries are the reference's, 0.03% with head_sum="fp32").
PCC_STAGE = 0.999
# Near-ties allowed per selection, as a share of the reference's picks, and per query. A swap is two
# near-ties (one position missing, one extra). How many positions sit within the score error of the
# cut does not shrink with the top-k, so on the small configurations 1% of the picks is less than
# one swap: the floor of two swaps per query keeps the budget meaningful there. On the Galaxy a
# top-64 block of 128 queries had 8 swaps, about one query in 16, and a single-query selection
# with one swap is 3% of its 64 picks; no measured selection has needed the floor so far.
NEAR_TIE_BUDGET = 0.01
NEAR_TIE_FLOOR = 4
# The largest score error allowed, as a share of the largest sum_h |w_h| relu(q_h . k) among the
# compared scores (the size of what the matmuls add up, which a signed sum can hide).
SCORE_ERROR_BUDGET = 0.02


def _near_tie_budget(total, queries):
    return max(NEAR_TIE_BUDGET * total, NEAR_TIE_FLOOR * queries)


def _bf16(t):
    """What a bf16 device tensor holds of ``t``."""
    return t.bfloat16().float()


def _reference(cfg, seed):
    """A reference indexer with random parameters at the scale a trained layer has: linear outputs
    of order one, the norm's weight and bias away from their trivial init."""
    ref = Hy4Indexer(cfg).eval()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in ref.named_parameters():
            if p.dim() == 2:
                p.copy_(_bf16(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5))
            else:
                p.copy_(_bf16((1.0 if name == "k_norm_weight" else 0.0) + 0.25 * torch.randn(p.shape, generator=g)))
    return ref


def _up(plan, t, dtype=ttnn.bfloat16, mesh_mapper=None):
    return ttnn.from_torch(
        t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=dtype, mesh_mapper=mesh_mapper
    )


def _rows(plan, t, dtype=ttnn.bfloat16, mesh_mapper=None):
    """[T, width] -> [1, 1, T, width]."""
    return _up(plan, t.reshape(1, 1, *t.shape), dtype, mesh_mapper)


def _down(t):
    """Chip 0's copy of a tensor that is whole on every chip, plus a check that the last chip agrees."""
    shards = ttnn.get_device_tensors(t)
    first, last = ttnn.to_torch(shards[0]).float(), ttnn.to_torch(shards[-1]).float()
    assert torch.equal(first, last), "a tensor that should be the same on every chip is not"
    return first


def _scores_down(tt, d_scores, num_positions):
    """Per-chip [1, 1, S, slots] scores -> [S, num_positions] in position order."""
    stripes = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(d_scores)])
    return tt.layout.merge(stripes, num_positions, dim=1)


def _score_error(ref_scores, dev_scores, q, w, keys, allowed, label):
    """max |device score - reference score| over the visible positions, checked against
    SCORE_ERROR_BUDGET. q, w, keys are the reference's operands for these scores."""
    eps = (ref_scores[allowed] - dev_scores[allowed]).abs().max().item()
    size = Hy4Indexer.scores(q, w.abs(), keys)[allowed].max().item()
    logger.info(f"SCORE ERROR {label}: max|Δ|={eps:.3e} = {eps / size:.2e} of the largest sum of |terms| ({size:.3f})")
    assert eps <= SCORE_ERROR_BUDGET * size, f"{label}: score error {eps:.3e} against terms of {size:.3f}"
    return eps


def _check_own_topk(tt, dev_masked, allowed, selected, label):
    """The device's selection against the top-k of the scores the device itself computed
    (``dev_masked``: [S, T] in position order, NEG where masked). Only scores tied at the resolution
    of the refinement passes may disagree."""
    cfg = tt.cfg
    own = dev_masked.masked_fill(~allowed, float("-inf"))
    top = own.topk(min(cfg.topk, own.shape[-1]), dim=-1)
    own_selected = selection_mask(top.indices, allowed)
    kth = top.values[:, -1]
    scale = kth[torch.isfinite(kth)].abs().max().item() if torch.isfinite(kth).any() else 0.0
    r = compare_selection(own, own_selected, selected, cfg.topk, tol=scale * 2.0 ** (-7 * tt.refine_passes))
    logger.info(
        f"OWN TOP-K {label}: picks={int(own_selected.sum())} missing={r['missing']} extra={r['extra']} "
        f"tied={r['near_tie']} real={r['real']} size_diff={r['size_diff']}"
    )
    assert r["real"] == 0 and r["size_diff"] == 0, f"{label}: not the top-k of the device's own scores: {r}"
    return r


def _check_selection(tt, q, w, keys, positions, stripes, label):
    """Select for the queries at ``positions`` on the device and compare with the reference.

    q: [S, heads, head_dim], w: [S, heads], keys: [T, head_dim], all as the device holds them;
    ``stripes`` already holds ``keys``. Returns the comparison counts.
    """
    cfg = tt.cfg
    T, S = keys.shape[0], len(positions)
    positions = torch.as_tensor(positions)
    allowed = causal_allowed(positions, T)
    ref_scores = Hy4Indexer.scores(q, w, keys)
    ref_masked = ref_scores.masked_fill(~allowed, float("-inf"))
    ref_picks = ref_masked.topk(min(cfg.topk, T), dim=-1).indices
    ref_selected = selection_mask(ref_picks, allowed)

    d_q = _up(tt.plan, q.reshape(1, S, cfg.num_heads, cfg.head_dim))
    d_w = _rows(tt.plan, w, ttnn.float32)
    d_pos = tt.query_positions(positions)

    # the scores every chip computed for its own stripe, and the mask
    dev_masked = _scores_down(tt, tt.masked_scores(d_q, d_w, d_pos, stripes), T)
    assert torch.equal(dev_masked == NEG, ~allowed), f"{label}: the visibility mask is wrong"
    if int(allowed.sum()) > 2:  # a correlation needs more than two points (the first tokens of a context)
        check_pcc(f"{label} scores", ref_scores[allowed], dev_masked[allowed], PCC_STAGE)
    eps = _score_error(ref_scores, dev_masked, q, w, keys, allowed, label)

    picks = tt.select(d_q, d_w, d_pos, stripes)
    assert picks.width == min(cfg.topk, tt.num_chips * stripes.shape[-2]) and picks.num_queries == S
    for name, t in (("positions", picks.positions), ("valid", picks.valid)):
        shards = [ttnn.to_torch(shard) for shard in ttnn.get_device_tensors(t)]
        assert len(shards) == tt.num_chips
        differ = [chip for chip, shard in enumerate(shards) if not torch.equal(shard, shards[0])]
        assert not differ, f"{label}: {name} on chips {differ} differ from chip 0's"
    selected = tt.selection_mask(picks, T)
    picks.deallocate()
    assert not (selected & ~allowed).any(), f"{label}: a position the query may not see was selected"
    assert torch.equal(selected.sum(-1), ref_selected.sum(-1)), f"{label}: a query did not select min(top-k, visible)"
    _check_own_topk(tt, dev_masked, allowed, selected, label)

    r = compare_selection(ref_masked, ref_selected, selected, cfg.topk, tol=2 * eps)
    total = int(ref_selected.sum())
    kth = ref_masked.topk(min(cfg.topk, T), dim=-1).values[:, -1]
    scale = kth[torch.isfinite(kth)].abs().mean().item() if torch.isfinite(kth).any() else float("nan")
    logger.info(
        f"SELECT {label}: queries={S} picks={total} missing={r['missing']} extra={r['extra']} "
        f"near_tie={r['near_tie']} real={r['real']} queries_off={r['queries']} size_diff={r['size_diff']} "
        f"| score max|Δ|={eps:.2e} cut~{scale:.3f}"
    )
    assert r["real"] == 0, f"{label}: {r}"
    assert r["near_tie"] <= _near_tie_budget(total, S), f"{label}: {r} of {total} picks"
    return r


def _check_layer(
    plan, cfg, ref, context, block, decode_positions, seed, label, layout=None, q_resid_fn=None, **indexer_kwargs
):
    """One full layer at ``context`` tokens: the last ``block`` tokens are the new ones."""
    g = torch.Generator().manual_seed(seed)
    T, B = context, block
    x = _bf16(torch.randn(T, cfg.hidden_size, generator=g))
    if q_resid_fn is None:
        q_resid = _bf16(torch.randn(T, cfg.q_lora_rank, generator=g))
    else:
        q_resid = _bf16(q_resid_fn(x))
    positions = torch.arange(T)
    cos, sin = rope_cos_sin(positions, cfg)
    new = slice(T - B, T)

    with torch.no_grad():
        r_q_raw = torch.nn.functional.linear(q_resid, ref.wq_b).view(T, cfg.num_heads, cfg.head_dim)
        r_k_raw = torch.nn.functional.linear(x, ref.wk)
        # The device stores keys and queries in the cache's frame (tt/frames.py): key dims reordered to
        # [rope, interleaved | pass], then the Hadamard rotation. Scores are the same in either frame.
        rope, perm = cfg.qk_rope_head_dim, frames.index_key_perm(cfg.head_dim, cfg.qk_rope_head_dim)
        r_q_raw, r_k_raw = r_q_raw[..., perm], r_k_raw[..., perm]
        r_keys = frames.index_to_device(ref.keys(x, cos, sin), rope)
        r_q = frames.index_to_device(ref.queries(q_resid, cos, sin), rope)
        r_w = ref.head_weights(x)
        r_scores = ref.scores(r_q[new], r_w[new], r_keys)  # the pure reference for the new tokens, [B, T]

    tt = TtHy4Indexer(plan, cfg, ref.weights(), layout=layout, **indexer_kwargs)
    d_x = _rows(plan, x[new], mesh_mapper=plan.shard_hidden(3))
    d_q_resid = _rows(plan, q_resid[new])
    d_cos, d_sin = (_rows(plan, t[new], ttnn.float32) for t in frames.rope_tables(cos, sin))

    # project: the three projections of the new tokens, whole on every chip
    d_q_raw, d_k_raw, d_w = tt.project(d_x, d_q_resid)
    assert tuple(d_q_raw.shape) == (1, B, cfg.num_heads, cfg.head_dim)
    check_pcc(f"{label} wq_b", r_q_raw[new], _down(d_q_raw), PCC_STAGE)
    check_pcc(f"{label} wk", r_k_raw[new], _down(d_k_raw), PCC_STAGE)
    check_pcc(f"{label} head weights", r_w[new], _down(d_w), PCC_STAGE)

    # index keys and queries, on the reference's projections
    d_keys = tt.keys(_rows(plan, r_k_raw[new], ttnn.float32), d_cos, d_sin)
    assert d_keys.dtype == ttnn.bfloat16 and tuple(d_keys.shape) == (1, 1, B, cfg.head_dim)
    check_pcc(f"{label} index keys", r_keys[new], _down(d_keys), PCC_STAGE)
    d_q = tt.queries(_up(plan, r_q_raw[new].reshape(1, B, cfg.num_heads, cfg.head_dim), ttnn.float32), d_cos, d_sin)
    check_pcc(f"{label} queries", r_q[new], _down(d_q), PCC_STAGE)

    # selection, on the reference's keys / queries / weights as the device holds them (keys and q in bf16)
    keys16, q16 = _bf16(r_keys), _bf16(r_q)
    stripes = tt.stripes_from_host(keys16)
    assert tuple(stripes.shape) == (1, 1, tt.layout.slots(T), cfg.head_dim)
    _check_selection(tt, q16[new], r_w[new], keys16, positions[new], stripes, f"{label} block")
    _check_selection(
        tt, q16[decode_positions], r_w[decode_positions], keys16, decode_positions, stripes, f"{label} mixed"
    )
    for p in decode_positions:
        _check_selection(tt, q16[p : p + 1], r_w[p : p + 1], keys16, [p], stripes, f"{label} decode")

    # end to end: hidden state -> keys into the cache -> selection, against the pure reference
    e_keys, e_q, e_w = tt(d_x, d_q_resid, d_cos, d_sin)
    check_pcc(f"{label} e2e index keys", r_keys[new], _down(e_keys), PCC_STAGE)
    cache = torch.cat([keys16[: T - B], _down(e_keys)[0, 0]], dim=0)  # the new tokens' keys join the cache
    e_stripes = tt.stripes_from_host(cache)
    picks = tt.select(e_q, e_w, positions[new], e_stripes)
    selected = tt.selection_mask(picks, T)
    allowed = causal_allowed(positions[new], T)
    ref_masked = r_scores.masked_fill(~allowed, float("-inf"))
    ref_selected = selection_mask(ref_masked.topk(min(cfg.topk, T), dim=-1).indices, allowed)
    dev_masked = _scores_down(tt, tt.masked_scores(e_q, e_w, tt.query_positions(positions[new]), e_stripes), T)
    if int(allowed.sum()) > 2:
        check_pcc(f"{label} e2e scores", ref_masked[allowed], dev_masked[allowed], PCC_STAGE)
    eps = _score_error(r_scores, dev_masked, r_q[new], r_w[new], r_keys, allowed, f"{label} e2e")
    assert not (selected & ~allowed).any(), f"{label} e2e: a position the query may not see was selected"
    assert torch.equal(selected.sum(-1), ref_selected.sum(-1)), f"{label} e2e: not min(top-k, visible) picks"
    _check_own_topk(tt, dev_masked, allowed, selected, f"{label} e2e")
    r = compare_selection(ref_masked, ref_selected, selected, cfg.topk, tol=2 * eps)
    total = int(ref_selected.sum())
    logger.info(
        f"SELECT {label} e2e: queries={B} picks={total} missing={r['missing']} extra={r['extra']} "
        f"near_tie={r['near_tie']} real={r['real']} size_diff={r['size_diff']} | score max|Δ|={eps:.2e}"
    )
    assert r["real"] == 0, f"{label} e2e: {r}"
    assert r["near_tie"] <= _near_tie_budget(total, B), f"{label} e2e: {r} of {total} picks"


# ---------------------------------------------------------------------------------------------
# What ttnn.topk supports at the sizes the selection needs
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
@pytest.mark.parametrize("rows", [1, 32], ids=["rows1", "rows32"])
@pytest.mark.parametrize("width", [4096, 31360, 65536], ids=lambda w: f"w{w}")
def test_topk_2048_envelope(device, dtype, rows, width):
    """ttnn.topk at k = 2048 over a stripe at 128K / 1M context and over the 32 x 2048 gathered
    candidates. The selection relies on the bf16 rows of this table; the fp32 rows record whether
    the fp32 top-k has become available (it would make the bf16 refinement passes unnecessary)."""
    k = 2048
    x = torch.randn(1, 1, rows, width, generator=torch.Generator().manual_seed(width + rows))
    if dtype == "bf16":
        x = _bf16(x)
    d = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16 if dtype == "bf16" else ttnn.float32
    )
    try:
        values, indices = ttnn.topk(d, k=k, dim=-1, largest=True, sorted=True)
    except RuntimeError as e:
        reason = next((line for line in str(e).splitlines() if "TopK" in line or "info:" in line), str(e)[:160])
        logger.info(f"TOPK {dtype} rows={rows} width={width} k={k}: not supported ({reason.strip()})")
        assert dtype == "fp32", "the selection needs the bf16 top-k"
        return
    v, i = ttnn.to_torch(values).float(), ttnn.to_torch(indices).long()
    values_ok = torch.equal(v, x.topk(k, dim=-1).values)
    indices_ok = int(i.max()) < width and torch.equal(x.gather(-1, i.clamp(max=width - 1)), v)
    distinct = all(row.unique().numel() == k for row in i[0, 0])
    logger.info(
        f"TOPK {dtype} rows={rows} width={width} k={k}: values_ok={values_ok} indices_ok={indices_ok} "
        f"distinct={distinct} index dtype={indices.dtype}"
    )
    assert values_ok and indices_ok and distinct


@pytest.mark.parametrize("width", [2176, 6272, 65536], ids=lambda w: f"w{w}")
def test_topk_2048_masked_rows(device, width):
    """The rows the selection really hands ttnn.topk, at stripe widths that are not powers of two
    (2176 and 6272 slots: contexts of about 70K and 200K) and at the gathered candidates' width:
    mostly NEG with fewer than k real scores, real scores plus NEG, and the second refinement pass
    of such rows (2^100 where a score was real, 0 where it was masked). Every real entry must come
    back, and every index must be in range and distinct, or a gather on it reads another slot."""
    k = 2048
    g = torch.Generator().manual_seed(width)
    few, most = 100, width - 37
    x = torch.full((1, 1, 4, width), NEG)
    x[0, 0, 0, torch.randperm(width, generator=g)[:few]] = _bf16(torch.randn(few, generator=g))
    x[0, 0, 1, torch.randperm(width, generator=g)[:most]] = _bf16(torch.randn(most, generator=g))
    x[0, 0, 2] = torch.where(x[0, 0, 0] > 0.5 * NEG, -NEG, 0.0)  # row 0 re-centred on a cut of NEG
    x[0, 0, 3] = NEG  # a chip that holds nothing the query may see
    d = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    values, indices = ttnn.topk(d, k=k, dim=-1, largest=True, sorted=True)
    v, i = ttnn.to_torch(values).float()[0, 0], ttnn.to_torch(indices).long()[0, 0]
    assert int(i.max()) < width, f"an index past the row: {int(i.max())}"
    for row in range(4):
        assert i[row].unique().numel() == k, f"row {row}: {k - i[row].unique().numel()} repeated indices"
        assert torch.equal(x[0, 0, row].gather(-1, i[row]), v[row]), f"row {row}: values do not match their indices"
        assert torch.equal(v[row], x[0, 0, row].topk(k).values), f"row {row}: not the top-k values"
    logger.info(f"TOPK masked rows width={width} k={k}: ok, index dtype={indices.dtype}")


# ---------------------------------------------------------------------------------------------
# One chip: the arithmetic, one stripe, no collectives
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "context, block, decode_positions",
    [(512, 64, [3, 63, 64, 200, 511]), (500, 37, [0, 64, 499]), (500, 1, [499])],
    ids=["512x64", "500x37", "500x1"],
)
@pytest.mark.parametrize("config", ["small", "small-top64"])
def test_indexer_one_chip(device, config, context, block, decode_positions):
    cfg = CONFIGS[config]
    plan = MeshPlan(device, hidden_axis=None)
    _check_layer(
        plan,
        cfg,
        _reference(cfg, seed=1),
        context,
        block,
        decode_positions,
        2,
        f"one chip {config} {context}x{block}",
        BlockCyclicLayout(1),
    )


@pytest.mark.parametrize("queries", [1, 5], ids=["S1", "S5"])
def test_head_sum_precision(device, queries):
    """The two ways of adding up the per-head terms (TtHy4Indexer ``head_sum``) on one chip's stripe
    of 4096 slots at Hy4's width. Both must give the reference's scores within the budget and pass
    the selection checks; their score errors are logged side by side."""
    cfg = HY4
    plan = MeshPlan(device, hidden_axis=None)
    weights = _reference(cfg, seed=1).weights()
    g = torch.Generator().manual_seed(7)
    T, S = 4096, queries
    keys = _bf16(torch.randn(T, cfg.head_dim, generator=g))
    q = _bf16(torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g))
    w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale
    positions = [T - 1] if S == 1 else [100, 2047, 2048, 3000, T - 1]
    ref_scores = Hy4Indexer.scores(q, w, keys)

    errors = {}
    for head_sum in ("matmul", "fp32"):
        tt = TtHy4Indexer(plan, cfg, weights, layout=BlockCyclicLayout(1), head_sum=head_sum)
        stripes = tt.stripes_from_host(keys)
        d_scores = tt.scores(
            _up(plan, q.reshape(1, S, cfg.num_heads, cfg.head_dim)), _rows(plan, w, ttnn.float32), stripes
        )
        assert tuple(d_scores.shape) == (1, 1, S, T) and d_scores.dtype == ttnn.float32
        errors[head_sum] = (ref_scores - _scores_down(tt, d_scores, T)).abs().max().item()
        _check_selection(tt, q, w, keys, positions, stripes, f"head sum {head_sum} S{S}")
    logger.info(
        f"HEAD SUM S={S}: score max|Δ| matmul={errors['matmul']:.3e} fp32={errors['fp32']:.3e} "
        f"(scores up to {ref_scores.abs().max().item():.3f}, std {ref_scores.std().item():.3f})"
    )


# ---------------------------------------------------------------------------------------------
# 8x4 Galaxy at 4K context: keys striped block-cyclic over the 32 chips
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("config", list(CONFIGS), ids=list(CONFIGS))
def test_indexer_sc1(mesh_device, config):
    cfg = CONFIGS[config]
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8)
    # decode queries below the top-k, either side of it, mid-block, and the newest token
    decode_positions = [5, 1000, 2047, 2048, 3001, 4095]
    _check_layer(plan, cfg, _reference(cfg, seed=1), 4096, 128, decode_positions, 2, f"sc1 {config}")


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize(
    "context, block, decode_positions",
    [(3000, 37, [0, 127, 2047, 2048, 2999]), (3000, 1, [2999]), (300, 1, [299]), (1, 1, [0])],
    ids=["3000x37", "3000x1", "300x1", "1x1"],
)
@pytest.mark.parametrize("config", ["small-top64", "hy4"])
def test_indexer_sc1_ragged(mesh_device, config, context, block, decode_positions):
    """Contexts that end mid-block and new-token counts that are not tile multiples.

    3000 positions are 23 whole blocks and 56 positions: chip 23's block is part filled and chips
    24 to 31 hold nothing. 300 positions leave 29 chips empty and every query below the top-k. A
    block of 37 is a prefill chunk that is not a multiple of 32; a block of 1 is the decode step
    (one new token through forward(), its key into the cache, select); 1x1 is the first token."""
    cfg = CONFIGS[config]
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    label = f"sc1 {config} {context}x{block}"
    _check_layer(plan, cfg, _reference(cfg, seed=1), context, block, decode_positions, 2, label)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("context, block", [(4096, 128), (3000, 37), (3000, 1)], ids=["4096x128", "3000x37", "3000x1"])
def test_indexer_sc1_fp32_head_sum(mesh_device, context, block):
    """The whole layer check at Hy4's width with ``head_sum="fp32"`` (the batched scoring matmul and
    the elementwise weighted sum), on an aligned and on a ragged context."""
    cfg = HY4
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    decode_positions = [5, 2047, 2048, context - 1]
    label = f"sc1 hy4 fp32-sum {context}x{block}"
    _check_layer(plan, cfg, _reference(cfg, seed=1), context, block, decode_positions, 2, label, head_sum="fp32")


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_prefill_chunks(mesh_device):
    """select() on more queries than one pass may score: the chunks (16 + 16 + 8 queries here) must
    give what one pass over the same queries gives."""
    cfg = SMALL
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    weights = _reference(cfg, seed=1).weights()
    g = torch.Generator().manual_seed(5)
    T, S = 3000, 40
    keys = _bf16(torch.randn(T, cfg.head_dim, generator=g))
    q = _bf16(torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g))
    w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale
    positions = torch.arange(T - S, T)

    chunked = TtHy4Indexer(plan, cfg, weights, max_per_head_elements=16 * 128 * cfg.num_heads)
    whole = TtHy4Indexer(plan, cfg, weights)
    stripes = chunked.stripes_from_host(keys)
    assert stripes.shape[-2] == 128
    _check_selection(chunked, q, w, keys, positions, stripes, "chunked")

    # chunked against one pass, on the scores the device computed: the two may differ only on
    # scores tied at the resolution of the refinement passes (the order of exact ties is the top-k's)
    d_q = _up(plan, q.reshape(1, S, cfg.num_heads, cfg.head_dim))
    d_w = _rows(plan, w, ttnn.float32)
    allowed = causal_allowed(positions, T)
    own = _scores_down(whole, whole.masked_scores(d_q, d_w, whole.query_positions(positions), stripes), T)
    a, b = chunked.select(d_q, d_w, positions, stripes), whole.select(d_q, d_w, positions, stripes)
    assert a.num_queries == b.num_queries == S and a.width == b.width == cfg.topk
    sel_a, sel_b = chunked.selection_mask(a, T), whole.selection_mask(b, T)
    _check_own_topk(whole, own, allowed, sel_b, "one pass")
    kth = own.masked_fill(~allowed, float("-inf")).topk(cfg.topk, dim=-1).values[:, -1]
    tol = kth.abs().max().item() * 2.0 ** (-7 * whole.refine_passes)
    r = compare_selection(own.masked_fill(~allowed, float("-inf")), sel_b, sel_a, cfg.topk, tol=tol)
    logger.info(f"CHUNKED vs one pass: {int((sel_a ^ sel_b).sum())} positions differ of {int(sel_b.sum())}, {r}")
    assert r["real"] == 0 and r["size_diff"] == 0, r


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_select_over_bfp8_stripes(mesh_device):
    """tt/kv_cache.py stores its stripes in BFP8 unless told otherwise. select() must work on such a
    stripe: here it is checked against the reference run on the values the BFP8 stripes hold (read
    back from the device). How far those values move the selection from the bf16 keys' is logged."""
    cfg = HY4
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    tt = TtHy4Indexer(plan, cfg, _reference(cfg, seed=1).weights())
    g = torch.Generator().manual_seed(6)
    T = 4096
    keys = _bf16(torch.randn(T, cfg.head_dim, generator=g))
    positions = [2047, 3000, 4095]
    S = len(positions)
    q = _bf16(torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g))
    w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale

    stripes = tt._on_chips(tt.layout.split(keys), ttnn.bfloat8_b)
    held = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(stripes)])
    held = tt.layout.merge(held, T)
    check_pcc("bfp8 keys as held", keys, held, PCC_STAGE)
    _check_selection(tt, q, w, held, positions, stripes, "bfp8 stripes")

    allowed = causal_allowed(torch.as_tensor(positions), T)
    sets = []
    for k in (keys, held):
        masked = Hy4Indexer.scores(q, w, k).masked_fill(~allowed, float("-inf"))
        sets.append(selection_mask(masked.topk(cfg.topk, dim=-1).indices, allowed))
    moved = int((sets[0] & ~sets[1]).sum())
    logger.info(
        f"BFP8 KEYS at {T} positions: {moved} of {int(sets[0].sum())} picks differ from the bf16 keys' "
        f"({moved / int(sets[0].sum()):.2%}), key max|Δ|={(keys - held).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("layer", [0, 1, 5], ids=lambda layer: f"layer{layer}")
def test_indexer_sc1_real_weights(mesh_device, layer):
    """The first three full layers with the checkpoint's indexer weights at 4K context. The hidden
    states are random (the layer's real input needs the layers below it, #65893); q_resid is the
    attention sublayer's own RMSNorm(x W_qa) with the checkpoint's q_a weights."""
    cfg = HY4
    prefix = f"model.layers.{layer}.self_attn."
    names = list(Hy4Indexer.CHECKPOINT_NAMES.values())
    tensors = hy4_tensors(
        [prefix + "indexer." + n for n in names] + [prefix + "q_a_proj.weight", prefix + "q_a_layernorm.weight"]
    )
    ref = Hy4Indexer(cfg).eval().load_weights(dict(zip(names, tensors)))
    q_a_proj, q_a_norm = tensors[-2], tensors[-1]

    def q_resid_fn(x):
        return rms_norm(torch.nn.functional.linear(x, q_a_proj), q_a_norm, 1e-6)

    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    _check_layer(plan, cfg, ref, 4096, 128, [5, 1000, 2048, 4095], 2, f"sc1 real L{layer}", q_resid_fn=q_resid_fn)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_masked_positions_are_never_selected(mesh_device):
    """Keys after the query and garbage in unfilled slots score far above every visible key; none
    of them may be selected, and the selection must still be the reference's."""
    cfg = SMALL_TOP64
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    tt = TtHy4Indexer(plan, cfg, _reference(cfg, seed=1).weights())
    g = torch.Generator().manual_seed(3)
    T, filled, p = 8000, 6000, 4500  # stripes sized for T, filled to 6000, query at 4500
    keys = _bf16(torch.randn(filled, cfg.head_dim, generator=g))
    keys[p + 1 :] *= 50.0
    q = _bf16(torch.randn(1, cfg.num_heads, cfg.head_dim, generator=g))
    w = torch.rand(1, cfg.num_heads, generator=g) * cfg.weight_scale  # positive: a larger key scores higher
    slots = tt.layout.slots(T)
    host = tt.layout.split(keys, slots=slots)
    host[~tt.layout.filled(filled, slots=slots)] = 80.0  # what an unfilled slot might hold
    stripes = tt._on_chips(host, ttnn.bfloat16)
    picks = tt.select(
        _up(plan, q.reshape(1, 1, cfg.num_heads, cfg.head_dim)), _rows(plan, w, ttnn.float32), [p], stripes
    )
    pairs = tt.locations(picks)[0]
    position = tt.layout.position(pairs[:, 0], pairs[:, 1])
    assert pairs.shape == (cfg.topk, 2) and position.unique().numel() == cfg.topk
    assert int(position.max()) <= p, "a position after the query, or an unfilled slot, was selected"
    _check_selection(tt, q, w, keys, [p], stripes, "masked")


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_selection_with_every_pick_in_one_stripe(mesh_device):
    """The worst skew for the distributed top-k: the whole global top-2048 sits in one chip's stripe
    (4096 slots at 128K), so that chip's local top-k is the selection, cut and all, and the other
    31 chips' candidates must all lose the merge."""
    cfg = HY4
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    tt = TtHy4Indexer(plan, cfg, _reference(cfg, seed=1).weights())
    g = torch.Generator().manual_seed(9)
    T, chip = 131072, 3
    keys = torch.randn(T, cfg.head_dim, generator=g)
    keys[tt.layout.chip(torch.arange(T)) == chip] *= 3.0
    keys = _bf16(keys)
    q = _bf16(torch.randn(1, cfg.num_heads, cfg.head_dim, generator=g))
    w = torch.rand(1, cfg.num_heads, generator=g) * cfg.weight_scale  # positive: a larger key scores higher
    ref_picks = Hy4Indexer.scores(q, w, keys).topk(cfg.topk, dim=-1).indices
    assert bool((tt.layout.chip(ref_picks) == chip).all()), "the keys do not put every pick on one chip"
    _check_selection(tt, q, w, keys, [T - 1], tt.stripes_from_host(keys), "one chip holds every pick")


# ---------------------------------------------------------------------------------------------
# Long context, synthetic keys: the selected set and the per-step latency of the decode selection
# ---------------------------------------------------------------------------------------------
def _long_context(mesh_device, context, head_sum, stripe_dtype=ttnn.bfloat16):
    """The selection of decode queries and of a small block over ``context`` synthetic keys, then
    the per-step cost of one layer's indexer at that context. ``stripe_dtype`` is the format the
    key stripes are held in; for a packed format the reference runs on the values the stripes
    hold, read back from the device."""
    cfg = HY4
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    tt = TtHy4Indexer(plan, cfg, _reference(cfg, seed=1).weights(), head_sum=head_sum)
    g = torch.Generator().manual_seed(context)
    T = context
    keys = _bf16(torch.randn(T, cfg.head_dim, generator=g))
    exact, moved, compared = None, 0, 0
    if stripe_dtype == ttnn.bfloat16:
        stripes = tt.stripes_from_host(keys)
    else:
        exact = keys
        stripes = tt._on_chips(tt.layout.split(keys), stripe_dtype)
        held = torch.stack([ttnn.to_torch(t).float()[0, 0] for t in ttnn.get_device_tensors(stripes)])
        keys = tt.layout.merge(held, T)
        head_sum = f"{head_sum} {stripe_dtype.name.lower()}-stripes"
    logger.info(f"LONG {T}: stripes {tuple(stripes.shape)} {stripes.dtype} per chip")

    # the newest token, a query with a quarter of the stripes after it, and a block of prefill queries
    # (200,000 positions: stripes of 6272 slots, not a power of two, the last block part filled)
    for name, positions in (("last", [T - 1]), ("mid", [3 * T // 4 - 77]), ("block8", list(range(T - 8, T)))):
        S = len(positions)
        q = _bf16(torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g))
        w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale
        _check_selection(tt, q, w, keys, positions, stripes, f"long {T} {head_sum} {name}")
        if exact is not None:  # how far the packed keys move the reference's own selection
            allowed = causal_allowed(torch.as_tensor(positions), T)
            sets = []
            for k in (exact, keys):
                masked = Hy4Indexer.scores(q, w, k).masked_fill(~allowed, float("-inf"))
                sets.append(selection_mask(masked.topk(cfg.topk, dim=-1).indices, allowed))
            moved, compared = moved + int((sets[0] & ~sets[1]).sum()), compared + int(sets[0].sum())
    if exact is not None:
        logger.info(
            f"PACKED KEYS at {T} positions: {moved} of {compared} picks differ from the bf16 keys' "
            f"({moved / compared:.2%}), key max|Δ|={(exact - keys).abs().max().item():.2e}"
        )

    # Per-step cost of one full layer's indexer at decode, one user: the new token through forward()
    # (index key, query heads, head weights), then the selection (scores + mask + local top-k +
    # gather + merge). Each is timed twice: with the steps queued back to back and one synchronize
    # at the end (what a decode loop that never reads the picks back pays), and with a synchronize
    # after every step (the latency of one step on its own).
    d_q = _up(plan, q[:1].reshape(1, 1, cfg.num_heads, cfg.head_dim))
    d_w = _rows(plan, w[:1], ttnn.float32)
    d_pos = tt.query_positions([T - 1])
    d_x = _rows(plan, _bf16(torch.randn(1, cfg.hidden_size, generator=g)), mesh_mapper=plan.shard_hidden(3))
    d_q_resid = _rows(plan, _bf16(torch.randn(1, cfg.q_lora_rank, generator=g)))
    d_cos, d_sin = (_rows(plan, t, ttnn.float32) for t in frames.rope_tables(*rope_cos_sin(torch.tensor([T - 1]), cfg)))
    masked = tt.masked_scores(d_q, d_w, d_pos, stripes)
    # stand-ins with the shapes the merge handles: one chip's (score | position) candidates, and
    # every chip's candidate scores once gathered
    K = cfg.topk
    pairs = _up(plan, torch.randn(1, 1, 1, 2 * K, generator=g), ttnn.float32)
    gathered = _up(plan, torch.randn(1, 1, 1, tt.num_chips * K, generator=g), ttnn.float32)
    local_index = tt._top(masked, K, tt.refine_passes)[2]
    merge_index = tt._top(gathered, K, tt.refine_passes)[2]

    def timed(fn, blocking, steps=20):
        for _ in range(2):
            fn()
        ttnn.synchronize_device(mesh_device)
        start = time.perf_counter()
        for _ in range(steps):
            fn()
            if blocking:
                ttnn.synchronize_device(mesh_device)
        ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - start) / steps * 1e3

    def forward_step():
        for t in tt(d_x, d_q_resid, d_cos, d_sin):
            t.deallocate(True)

    def local_top_step():
        tt._top(masked, cfg.topk, tt.refine_passes)[2].deallocate(True)

    steps = {
        "select": lambda: tt.select(d_q, d_w, d_pos, stripes).deallocate(),
        "scoring and masking": lambda: tt.masked_scores(d_q, d_w, d_pos, stripes).deallocate(True),
        f"local top-k ({tt.refine_passes} passes)": local_top_step,
        "one ttnn.gather over a stripe's scores": lambda: ttnn.gather(masked, dim=-1, index=local_index).deallocate(
            True
        ),
        "all-gather of the candidates": lambda: tt._gather_all(pairs).deallocate(True),
        f"merge top-k ({tt.refine_passes} passes)": lambda: tt._top(gathered, K, tt.refine_passes)[2].deallocate(True),
        "one ttnn.gather over the candidates": lambda: ttnn.gather(gathered, dim=-1, index=merge_index).deallocate(
            True
        ),
        "forward of the new token": forward_step,
    }
    ms = {name: (timed(fn, blocking=False), timed(fn, blocking=True)) for name, fn in steps.items()}
    logger.info(
        f"LATENCY decode selection ({head_sum} head sum) at {T} positions ({tuple(stripes.shape)[2]} slots per chip), "
        f"ms per step queued / synchronized every step: "
        + "; ".join(f"{name} {queued:.2f} / {blocking:.2f}" for name, (queued, blocking) in ms.items())
        + " (untraced, 32 chips, one layer, one user)"
    )


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("head_sum", ["matmul", "fp32"])
@pytest.mark.parametrize("context", [131072, 200000, 1048576], ids=["128K", "200K", "1M"])
def test_decode_selection_long_context(mesh_device, context, head_sum):
    _long_context(mesh_device, context, head_sum)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("head_sum", ["matmul", "fp32"])
def test_decode_selection_over_bfp8_stripes_128k(mesh_device, head_sum):
    """The long-context check and the per-step cost with the key stripes in BFP8, the format
    tt/kv_cache.py holds its stripes in by default, at 128K."""
    _long_context(mesh_device, 131072, head_sum, stripe_dtype=ttnn.bfloat8_b)


# ---------------------------------------------------------------------------------------------
# Full / shared: a full layer's picks are handed to the shared layers above it
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_shared_layers_reuse_picks(mesh_device):
    cfg = SMALL
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    types = full_layers(6)  # full full shared shared shared full
    cache = TtHy4IndexCache(types)
    assert [cache.is_full(layer) for layer in range(6)] == [True, True, False, False, False, True]

    g = torch.Generator().manual_seed(4)
    T, p = 4096, 4095
    reference, device, stripes, expected, tolerance = {}, {}, {}, {}, {}
    for layer in (0, 1, 5):
        tt = TtHy4Indexer(plan, cfg, _reference(cfg, seed=10 + layer).weights())
        keys = _bf16(torch.randn(T, cfg.head_dim, generator=g))  # every full layer has its own key cache
        q = _bf16(torch.randn(1, cfg.num_heads, cfg.head_dim, generator=g))
        w = torch.randn(1, cfg.num_heads, generator=g) * cfg.weight_scale
        device[layer] = (tt, _up(plan, q.reshape(1, 1, cfg.num_heads, cfg.head_dim)), _rows(plan, w, ttnn.float32))
        stripes[layer] = tt.stripes_from_host(keys)
        scores = Hy4Indexer.scores(q, w, keys)
        expected[layer] = selection_mask(scores.topk(cfg.topk, dim=-1).indices, torch.ones(1, T, dtype=torch.bool))
        reference[layer] = scores
        # near-ties are judged against this layer's measured score error, as in _check_selection
        d_scores = tt.masked_scores(device[layer][1], device[layer][2], tt.query_positions([p]), stripes[layer])
        tolerance[layer] = 2 * (scores - _scores_down(tt, d_scores, T)).abs().max().item()
    assert not torch.equal(expected[0], expected[1])

    def run_full(layer):
        tt, d_q, d_w = device[layer]
        cache.store(layer, tt.select(d_q, d_w, [p], stripes[layer]))

    def agrees(layer, source):
        """The picks layer ``layer`` gets are the source layer's selection, still intact on the device."""
        picks = cache.picks(layer)
        selected = device[source][0].selection_mask(picks, T)
        r = compare_selection(reference[source], expected[source], selected, cfg.topk, tol=tolerance[source])
        assert r["real"] == 0 and r["near_tie"] <= _near_tie_budget(cfg.topk, 1), (layer, source, r)
        # another layer's picks would not pass for this one's
        other = next(full for full in (0, 1, 5) if full != source)
        wrong = compare_selection(reference[other], expected[other], selected, cfg.topk, tol=tolerance[other])
        assert wrong["real"] > 100, (layer, source, other, wrong)
        return picks

    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        cache.picks(0)  # nothing stored yet
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        cache.store(2, None)  # a shared layer has no indexer
    for step in range(2):  # two decode steps
        run_full(0)
        agrees(0, 0)
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            cache.picks(2)  # layer 2 attends over layer 1's picks, which are not there yet
        run_full(1)
        first = agrees(1, 1)
        for layer in (2, 3, 4):
            assert agrees(layer, 1) is first  # the same device tensors, not a copy
        with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
            cache.picks(5)
        run_full(5)
        agrees(5, 5)
    cache.clear()
