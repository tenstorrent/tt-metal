# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device prefill attention (tt/attention_prefill.py) vs the pure-torch ground truth
(reference/attention_prefill.py), tenstorrent/tt-metal#65887.

A full layer and the shared layer above it, on one prompt, stage by stage so that a failure names
where it is:

    stripes     the index keys dealt to the chips, against kv_layout's split (exact)
    selection   on the reference's index keys and queries as the device holds them (bf16): the set
                every chip ends up with, against the top-k of the scores the device itself computed
                and against the reference's selected set
    keys/values with the sink's zero rows appended, on the reference's kv_b output
    core        the chunked dense core (the sink as a key) under the reference's selection, on the
                reference's q / keys / values
    end to end  the full layer: its selection against the pure reference's, the output, and what it
                wrote to the cache (latent and index key of every token, read back in position order)
                the shared layer: the output with the full layer's selection, and its cached latent

On one chip first, then on the Galaxy mesh (tests/common.py GALAXY_MESH, the hidden dimension on
SC1_HIDDEN_AXIS) at a small width and at Hy4's own, with prompts shorter
and longer than the top-k, with BF16 and BFP8 weights and cache, and with the checkpoint's weights.
Above the top-k the test also requires the device output of the queries the top-k cuts most to be
closer to the reference with the selection than to dense causal attention: a core that ignored the
selection cannot pass.

The end-to-end output is compared with the reference attending over the set the DEVICE selected,
and the device's set is checked against the reference's separately. The two cannot be folded into
one PCC against the pure reference: a position that swaps at the cut is a legitimate near-tie, but
with random weights attention spreads evenly over the picks, so m swaps out of k picks move a
query's output by about sqrt(2 m / k) of its size. PCC against the reference's own selection is
logged for every layer.

Frames (tt/frames.py). The device modules work in the frame the prefill stack stores its caches in:
interleaved RoPE pairs, and index keys and indexer queries as [rope interleaved | pass] in the
Hadamard basis. The references stay in the HF frame, so cos / sin go up through frames.rope_tables
and reference tensors fed to or compared with the device go through frames.*_to_device. Scores,
selections and the sublayer's output do not depend on the frame.

A selection is compared as a set. Two scores closer than the device's score error can land on
either side of the cut, so a disagreement on a position whose reference score is within the
tolerance of the reference's cut is a near-tie and anything else is a real error; real errors must
be zero. The device keeps every score it cannot tell from the cut, so its set may be larger than
top-k by such ties, never smaller.

The last tests measure: the selection of one chunk of queries at the end of a 128K context, and the
prefill time of a full and a shared layer at Hy4's width.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin
from models.experimental.hy4_preview.reference.attention_prefill import Hy4AttentionPrefill, additive_mask
from models.experimental.hy4_preview.reference.indexer import (
    NEG,
    Hy4Indexer,
    IndexerConfig,
    causal_allowed,
    compare_selection,
    selection_mask,
)
from models.experimental.hy4_preview.tests.common import (
    GALAXY_DEVICE_PARAMS,
    GALAXY_MESH,
    SC1_HIDDEN_AXIS,
    check_pcc,
    hy4_tensors,
    pcc,
)
from models.experimental.hy4_preview.tt import frames
from models.experimental.hy4_preview.tt.attention import TtHy4Attention
from models.experimental.hy4_preview.tt.attention_prefill import MASKED, SINK_ROWS, TtHy4AttentionPrefill
from models.experimental.hy4_preview.tt.common import MeshPlan
from models.experimental.hy4_preview.tt.indexer import TtHy4IndexCache, TtHy4Indexer
from models.experimental.hy4_preview.tt.kv_cache import INDEX_KEY, LATENT, KvCacheConfig, TtHy4KvCache
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

# (attention config, indexer config). "small" is Hy4's structure at about 1/8 of every hidden width;
# the index key keeps its real 128 dims and 64 RoPE dims, since attention and the indexer share one
# cos / sin. "small-top64" cuts at 64 picks, so the cut and the chips' own top-k run on short prompts.
HY4_ATTENTION = AttentionConfig()
SMALL_ATTENTION = AttentionConfig(
    hidden_size=768,
    num_heads=8,
    q_lora_rank=256,
    kv_lora_rank=128,
    qk_nope_head_dim=96,
    qk_rope_head_dim=64,
    v_head_dim=128,
)
CONFIGS = {
    "small": (SMALL_ATTENTION, IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8)),
    "small-top64": (SMALL_ATTENTION, IndexerConfig(hidden_size=768, q_lora_rank=256, num_heads=8, topk=64)),
    "hy4": (HY4_ATTENTION, IndexerConfig()),
}

# (attention weight dtype, cache dtype, per-stage PCC, end-to-end PCC, cache read-back PCC)
DTYPES = {
    "bf16": (ttnn.bfloat16, "bf16", 0.999, 0.998, 0.999),
    "bfp8": (ttnn.bfloat8_b, "bfp8", 0.998, 0.995, 0.998),
}

# Near-ties allowed in a selection, as a share of the reference's picks, with a floor of two swaps
# per query (the number of positions next to the cut does not shrink with the top-k).
NEAR_TIE_BUDGET = 0.01
NEAR_TIE_FLOOR = 4
# End to end the device's scores also carry the projections' error (bf16 weights and activations), which
# no stage of this test measures; a disagreement there is a near-tie within this share of the largest
# sum of |terms| of a score.
E2E_SCORE_TOLERANCE = 0.02
E2E_NEAR_TIE_BUDGET = 0.02
# The queries the "selection was applied" checks look at: the last ones, which the top-k cuts most.
LAST = 32


def _bf16(t):
    """What a bf16 device tensor holds of ``t``."""
    return t.bfloat16().float()


def _reference(cfg, icfg, seed):
    """A reference layer (full when ``icfg`` is given) with random parameters at the scale a trained
    layer has: linear outputs of order one, norm weights, biases and sinks away from their init."""
    g = torch.Generator().manual_seed(seed)
    attention = Hy4Attention(cfg).eval()
    with torch.no_grad():
        for name in Hy4Attention.PARAMETER_NAMES:
            p = getattr(attention, name)
            if p.dim() == 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5)
            elif name == "learnable_sink_param":
                p.copy_(torch.randn(p.shape, generator=g))
            else:
                p.copy_(1.0 + 0.25 * torch.randn(p.shape, generator=g))
        indexer = None
        if icfg is not None:
            indexer = Hy4Indexer(icfg).eval()
            for name, p in indexer.named_parameters():
                if p.dim() == 2:
                    p.copy_(_bf16(torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5))
                else:
                    p.copy_(_bf16((1.0 if name == "k_norm_weight" else 0.0) + 0.25 * torch.randn(p.shape, generator=g)))
    return Hy4AttentionPrefill(attention, indexer).eval()


def _up(plan, t, dtype=ttnn.bfloat16, mesh_mapper=None):
    return ttnn.from_torch(
        t.float().contiguous(), layout=ttnn.TILE_LAYOUT, device=plan.mesh_device, dtype=dtype, mesh_mapper=mesh_mapper
    )


def _rows(plan, t, dtype=ttnn.bfloat16, mesh_mapper=None):
    """[T, width] -> [1, 1, T, width]."""
    return _up(plan, t.reshape(1, 1, *t.shape), dtype, mesh_mapper)


def _rope_tables(plan, cos, sin):
    """The reference's cos / sin [T, rope] as the device modules take them: interleaved (tt/frames.py)."""
    return tuple(_rows(plan, t) for t in frames.rope_tables(cos, sin))


def _per_chip(t):
    return [ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(t)]


def _heads_down(plan, t):
    """Device [1, heads/ctp, T, width] per chip -> reference order [H, T, width]."""
    return plan.to_torch_head_concat(t, dim=1)[0].float()


def _hidden_down(plan, t):
    """Device [1, 1, T, hidden/htp] split over the hidden axis -> [T, hidden]."""
    return plan.to_torch_hidden_concat(t, dim=-1)[0, 0].float()


def _one_set(sets, label):
    """The selected set, after checking that every chip holds the same one."""
    for chip, other in enumerate(sets[1:], start=1):
        assert torch.equal(sets[0], other), f"{label}: chip {chip} holds another selection than chip 0"
    return sets[0]


def _check_selection_was_applied(label, want, dense, got):
    """Where the selection cuts, the device output must follow the reference with the selection, not
    dense causal attention. All three are [queries, width], the queries that see the most positions."""
    with_selection, without = pcc(want, got), pcc(dense, got)
    logger.info(
        f"{label}, last {want.shape[0]} queries: against the selection pcc={with_selection:.6f}, against dense "
        f"causal attention {without:.6f} (the two references agree to {pcc(want, dense):.6f})"
    )
    assert with_selection > without, f"{label}: closer to dense attention ({without:.6f}) than to the selection"


def _near_tie_budget(total, queries, share=NEAR_TIE_BUDGET):
    return max(share * total, NEAR_TIE_FLOOR * queries)


def _check_set(label, masked, selected, topk, tol, budget_share=NEAR_TIE_BUDGET):
    """A device selection (bool [S, T]) against the top-k of ``masked`` ([S, T], -inf where the query
    may not see the position), with ``tol`` the score distance from the cut that counts as a near-tie."""
    S, T = masked.shape
    allowed = torch.isfinite(masked)
    exact = selection_mask(masked.topk(min(topk, T), dim=-1).indices, allowed)
    assert not (selected & ~allowed).any(), f"{label}: a position the query may not see was selected"
    seen = allowed.sum(-1)
    count = selected.sum(-1)
    assert torch.equal(count[seen <= topk], seen[seen <= topk]), f"{label}: a query that sees <= top-k dropped one"
    assert (count[seen > topk] >= topk).all(), f"{label}: a query selected fewer than top-k"
    r = compare_selection(masked, exact, selected, topk, tol)
    total = int(exact.sum())
    logger.info(
        f"SELECT {label}: queries={S} picks={total} missing={r['missing']} extra={r['extra']} "
        f"near_tie={r['near_tie']} real={r['real']} queries_off={r['queries']} size_diff={r['size_diff']} tol={tol:.2e}"
    )
    assert r["real"] == 0, f"{label}: {r}"
    assert r["near_tie"] <= _near_tie_budget(total, S, budget_share), f"{label}: {r} of {total} picks"
    return r


def _device_scores(tt_indexer, d_q, d_w, stripes, rows, num_keys):
    """The masked scores the device computes for the queries ``rows`` (a range), [len(rows), num_keys]
    in position order, NEG where masked."""
    cfg = tt_indexer.cfg
    q = ttnn.slice(d_q, [0, rows.start, 0, 0], [1, rows.stop, cfg.num_heads, cfg.head_dim])
    w = ttnn.slice(d_w, [0, 0, rows.start, 0], [1, 1, rows.stop, cfg.num_heads])
    masked = tt_indexer.masked_scores(q, w, tt_indexer.query_positions(list(rows)), stripes)
    per_chip = torch.stack([t[0, 0] for t in _per_chip(masked)])
    return tt_indexer.layout.merge(per_chip, num_keys, dim=1)


def _check_selection_stage(prefill, keys16, q16, w, label):
    """stripes + select on the reference's index keys / queries / head weights as the device holds them."""
    plan, tt_indexer = prefill.plan, prefill.indexer
    icfg = tt_indexer.cfg
    T = keys16.shape[0]
    positions = torch.arange(T)
    allowed = causal_allowed(positions, T)

    stripes = prefill.stripes(_rows(plan, keys16))
    slots = prefill.layout.slots(T)
    assert tuple(stripes.shape) == (1, 1, slots, icfg.head_dim), stripes.shape
    want = prefill.layout.split(keys16, dim=0, slots=slots)
    for chip, got in enumerate(_per_chip(stripes)):
        assert torch.equal(got[0, 0], want[chip]), f"{label}: chip {chip}'s stripe is not kv_layout's"

    d_q = _up(plan, q16.reshape(1, T, icfg.num_heads, icfg.head_dim))
    d_w = _rows(plan, w, ttnn.float32)
    selection = prefill.select(d_q, d_w, stripes, T)
    assert (selection.num_queries, selection.num_keys) == (T, T)
    assert tuple(selection.mask.shape) == (1, 1, T, T + SINK_ROWS) and selection.mask.dtype == ttnn.bfloat16
    for chip, m in enumerate(_per_chip(selection.mask)):
        assert bool(((m == 0) | (m == MASKED)).all()), f"{label}: chip {chip}'s mask is not 0 / MASKED"
        assert bool((m[..., T:] == 0).all()), f"{label}: chip {chip}'s mask is not 0 over the sink's columns"
    selected = _one_set(prefill.selection_to_host(selection), label)
    selection.deallocate()

    ref_scores = Hy4Indexer.scores(q16, w, keys16)
    ref_masked = ref_scores.masked_fill(~allowed, float("-inf"))

    # the device's own scores for the first, a middle and the last chunk of queries: the score error,
    # and the selection logic on its own (cut, compare, reorder), which owes nothing to the matmul
    chunk = prefill.query_chunk
    starts = sorted({0, (T // 2) // chunk * chunk, (T - 1) // chunk * chunk})
    eps = 0.0
    for start in starts:
        rows = range(start, min(start + chunk, T))
        dev = _device_scores(tt_indexer, d_q, d_w, stripes, rows, T)
        seen = allowed[rows.start : rows.stop]
        assert torch.equal(dev == NEG, ~seen), f"{label}: the visibility mask is wrong"
        eps = max(eps, (dev[seen] - ref_scores[rows.start : rows.stop][seen]).abs().max().item())
        own = dev.masked_fill(~seen, float("-inf"))
        kth = own.topk(min(icfg.topk, T), dim=-1).values[:, -1]
        scale = kth[torch.isfinite(kth)].abs().max().item() if torch.isfinite(kth).any() else 0.0
        own_tol = scale * 2.0 ** (-7 * tt_indexer.refine_passes)
        r = _check_set(f"{label} own scores q{rows.start}", own, selected[rows.start : rows.stop], icfg.topk, own_tol)
        assert r["missing"] == 0, f"{label}: a position of the device's own top-k was dropped: {r}"
    size = Hy4Indexer.scores(q16, w.abs(), keys16)[allowed].max().item()
    logger.info(f"SCORE ERROR {label}: max|Δ|={eps:.3e} = {eps / size:.2e} of the largest sum of |terms| ({size:.3f})")

    _check_set(f"{label} vs reference", ref_masked, selected, icfg.topk, 2 * eps + size * 2.0**-14)
    return selected


def _check_layers(plan, ref_full, ref_shared, seq_len, dtype, seed, label, query_chunk=256, cache=True):
    """A full layer and the shared layer above it on one prompt of ``seq_len`` tokens."""
    cfg, icfg = ref_full.attention.cfg, ref_full.indexer.cfg
    weight_dtype, cache_dtype, pcc_stage, pcc_e2e, pcc_cache = DTYPES[dtype]
    T = seq_len
    g = torch.Generator().manual_seed(seed)
    x_full = torch.randn(T, cfg.hidden_size, generator=g)
    x_shared = torch.randn(T, cfg.hidden_size, generator=g)
    positions = torch.arange(T)
    cos, sin = rope_cos_sin(positions, cfg)
    allowed = causal_allowed(positions, T)
    cut = T > icfg.topk  # some query sees more than top-k positions

    with torch.no_grad():
        full = ref_full(x_full, cos, sin)
        shared = ref_shared(x_shared, cos, sin, prev_picks=full.picks)
        attention = ref_full.attention
        r_q_resid, r_k_pass, r_k_rope = attention.latent(x_full, cos, sin)
        r_q = attention.queries(r_q_resid, cos, sin)
        r_kv = torch.nn.functional.linear(r_k_pass, attention.kv_b_proj)  # [S, H * (nope + v)], head-major
        r_keys, r_values = attention.keys_values(r_k_pass, r_k_rope)
        r_core = attention.core(r_q, r_keys, r_values, additive_mask(full.selected))
        r_index_q = ref_full.indexer.queries(r_q_resid, cos, sin)
        r_index_w = ref_full.indexer.head_weights(x_full)
        r_scores = ref_full.indexer.scores(r_index_q, r_index_w, full.index_keys)
        r_core_dense = attention.core(r_q, r_keys, r_values, causal_mask(T))
        dense_full = attention(x_full, cos, sin, causal_mask(T))
        dense_shared = ref_shared.attention(x_shared, cos, sin, causal_mask(T))
        s_q_resid, s_k_pass, s_k_rope = ref_shared.attention.latent(x_shared, cos, sin)
    assert bool((full.selected.sum(-1) < allowed.sum(-1)).any()) == cut

    num_chips = plan.hidden_tp * plan.head_tp
    layout = BlockCyclicLayout(num_chips=num_chips)
    kv_cache = None
    if cache:
        kv_cache = TtHy4KvCache(
            plan.mesh_device,
            KvCacheConfig(
                num_layers=2,
                indexer_layers=(0,),
                latent_dim=cfg.latent_dim,
                index_key_dim=icfg.head_dim,
                max_context=T,
                dtype=cache_dtype,
                layout=layout,
            ),
        )
    index_cache = TtHy4IndexCache(["full", "shared"])
    tt_full = TtHy4AttentionPrefill(
        TtHy4Attention(plan, cfg, ref_full.attention.weights(), weight_dtype=weight_dtype),
        layer=0,
        index_cache=index_cache,
        indexer=TtHy4Indexer(plan, icfg, ref_full.indexer.weights(), layout=layout),
        kv_cache=kv_cache,
        query_chunk=query_chunk,
    )
    tt_shared = TtHy4AttentionPrefill(
        TtHy4Attention(plan, cfg, ref_shared.attention.weights(), weight_dtype=weight_dtype),
        layer=1,
        index_cache=index_cache,
        kv_cache=kv_cache,
        query_chunk=query_chunk,
    )
    assert tt_full.is_full and not tt_shared.is_full

    # stripes and selection, on the reference's index keys / queries as the device holds them: in the
    # cache's frame ([rope interleaved | pass], Hadamard basis; tt/frames.py), rounded to bf16. The
    # rotation is orthonormal, so the scores and the selection are the reference's.
    rope = cfg.qk_rope_head_dim
    _check_selection_stage(
        tt_full,
        _bf16(frames.index_to_device(full.index_keys, icfg.qk_rope_head_dim)),
        _bf16(frames.index_to_device(r_index_q, icfg.qk_rope_head_dim)),
        r_index_w,
        f"{label} selection",
    )

    # keys and values with the sink's rows appended, on the reference's kv_b output and k_rope
    d_keys, d_values = tt_full.keys_values(
        _rows(plan, r_kv, mesh_mapper=plan.shard(head_dim=3)), _rows(plan, frames.rope_to_device(r_k_rope))
    )
    for name, r, d in (("keys", frames.tail_to_device(r_keys, rope), d_keys), ("values", r_values, d_values)):
        got = _heads_down(plan, d)
        assert got.shape == (cfg.num_heads, T + SINK_ROWS, r.shape[-1]), got.shape
        check_pcc(f"{label} {name}", r, got[:, :T], pcc_stage)
        assert bool((got[:, T:] == 0).all()), f"{label}: the sink's {name} rows are not zero"

    # the chunked core under the reference's selection, on the reference's q / keys / values (the
    # core only takes q . k, which does not depend on the order of the rope dims)
    head_mapper = plan.shard(head_dim=1)
    given = tt_full.selection_from_host(full.selected)

    def with_sink_rows(t):  # [H, T, width] -> [1, H, T + SINK_ROWS, width]
        return torch.nn.functional.pad(t, (0, 0, 0, SINK_ROWS))[None]

    d_core = tt_full.attend(
        _up(plan, r_q[None], mesh_mapper=head_mapper),
        _up(plan, with_sink_rows(r_keys), mesh_mapper=head_mapper),
        _up(plan, with_sink_rows(r_values), mesh_mapper=head_mapper),
        given,
    )
    assert tuple(d_core.shape) == (1, cfg.num_heads // plan.head_tp, T, cfg.v_head_dim)
    core = _heads_down(plan, d_core)
    check_pcc(f"{label} core", r_core, core, pcc_stage)
    if cut:
        _check_selection_was_applied(f"{label} core", r_core[:, -LAST:], r_core_dense[:, -LAST:], core[:, -LAST:])
    given.deallocate()

    # the full layer end to end
    d_cos, d_sin = _rope_tables(plan, cos, sin)
    d_x = _rows(plan, x_full, mesh_mapper=plan.shard_hidden(3))
    d_out = tt_full(d_x, d_cos, d_sin)
    assert tuple(d_out.shape) == (1, 1, T, cfg.hidden_size // plan.hidden_tp)
    out = _hidden_down(plan, d_out)
    selected = _one_set(tt_full.selection_to_host(index_cache.picks(0)), f"{label} full layer")
    size = Hy4Indexer.scores(r_index_q, r_index_w.abs(), full.index_keys)[allowed].max().item()
    _check_set(
        f"{label} full layer vs reference",
        r_scores.masked_fill(~allowed, float("-inf")),
        selected,
        icfg.topk,
        E2E_SCORE_TOLERANCE * size,
        E2E_NEAR_TIE_BUDGET,
    )
    # The output is compared with the reference attending over the set the device selected: the
    # selection is checked above, and a near-tie that falls the other way is not an arithmetic error.
    with torch.no_grad():
        want = ref_full.attend(x_full, r_q_resid, r_k_pass, r_k_rope, cos, sin, additive_mask(selected))
    check_pcc(f"{label} full layer", want, out, pcc_e2e)
    logger.info(f"{label} full layer, against the reference's own selection: pcc={pcc(full.out, out):.6f}")
    if cut:
        _check_selection_was_applied(f"{label} full layer", want[-LAST:], dense_full[-LAST:], out[-LAST:])

    if kv_cache is not None:
        # what prefill left in the cache, in position order, against the reference and against what
        # the device computed (a bf16 cache holds exactly that)
        d_q_resid, d_k_pass, d_k_rope = tt_full.attention.latent(d_x, d_cos, d_sin)
        dev_latent = torch.cat([_per_chip(d_k_pass)[0], _per_chip(d_k_rope)[0]], dim=-1)[0, 0]
        dev_keys = _per_chip(tt_full.indexer(d_x, d_q_resid, d_cos, d_sin)[0])[0][0, 0]
        cached_latent = kv_cache.to_torch(0, 0, T, LATENT)
        cached_keys = kv_cache.to_torch(0, 0, T, INDEX_KEY)
        # the cache holds the device frame: the reference's entries are mapped into it, and the cached
        # rows are also mapped back and compared in the reference's own frame
        index_rope = icfg.qk_rope_head_dim
        check_pcc(f"{label} cached latent", frames.tail_to_device(full.latent, rope), cached_latent, pcc_cache)
        check_pcc(
            f"{label} cached index keys", frames.index_to_device(full.index_keys, index_rope), cached_keys, pcc_cache
        )
        check_pcc(
            f"{label} cached latent in the HF frame",
            full.latent,
            frames.tail_from_device(cached_latent, rope),
            pcc_cache,
        )
        check_pcc(
            f"{label} cached index keys in the HF frame",
            full.index_keys,
            frames.index_from_device(cached_keys, index_rope),
            pcc_cache,
        )
        check_pcc(f"{label} cached latent vs computed", dev_latent, cached_latent, 0.9995)
        check_pcc(f"{label} cached index keys vs computed", dev_keys, cached_keys, 0.9995)
        if cache_dtype == "bf16":
            assert torch.equal(cached_latent, dev_latent), f"{label}: the cached latent is not what the layer computed"
            assert torch.equal(cached_keys, dev_keys), f"{label}: the cached index keys are not what the layer computed"

    # the shared layer: the full layer's selection, another input, its own weights
    d_out = tt_shared(_rows(plan, x_shared, mesh_mapper=plan.shard_hidden(3)), d_cos, d_sin)
    out = _hidden_down(plan, d_out)
    assert index_cache.picks(1) is index_cache.picks(0)
    with torch.no_grad():
        want = ref_shared.attend(x_shared, s_q_resid, s_k_pass, s_k_rope, cos, sin, additive_mask(selected))
    check_pcc(f"{label} shared layer", want, out, pcc_e2e)
    logger.info(f"{label} shared layer, against the reference's own selection: pcc={pcc(shared.out, out):.6f}")
    if cut:
        _check_selection_was_applied(f"{label} shared layer", want[-LAST:], dense_shared[-LAST:], out[-LAST:])
    if kv_cache is not None:
        cached = kv_cache.to_torch(1, 0, T, LATENT)
        check_pcc(f"{label} shared cached latent", frames.tail_to_device(shared.latent, rope), cached, pcc_cache)
        assert INDEX_KEY not in kv_cache.cfg.kinds(1)
        kv_cache.deallocate()
    index_cache.clear()


def _pair(config, seed=1):
    cfg, icfg = CONFIGS[config]
    return _reference(cfg, icfg, seed), _reference(cfg, None, seed + 100)


# ---------------------------------------------------------------------------------------------
# One chip: the arithmetic. One stripe, so a cut only where the stripe is longer than the top-k
# ---------------------------------------------------------------------------------------------
def test_board_responds(device):
    """One add and a read-back on one chip: run alone, with a short job timeout, before anything else."""
    a = torch.arange(64, dtype=torch.float32).reshape(1, 1, 2, 32)
    d = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat16)
    assert torch.equal(ttnn.to_torch(ttnn.add(d, d)).float(), 2 * a)


@pytest.mark.parametrize(
    "config, seq_len, query_chunk",
    [("small", 96, 256), ("small", 96, 32), ("small-top64", 96, 64), ("small-top64", 320, 128)],
    ids=["no-cut-S96", "no-cut-S96-chunk32", "top64-S96", "top64-S320"],
)
def test_prefill_one_chip(device, config, seq_len, query_chunk):
    plan = MeshPlan(device, hidden_axis=None)
    ref_full, ref_shared = _pair(config)
    _check_layers(plan, ref_full, ref_shared, seq_len, "bf16", 2, f"one chip {config} S{seq_len}", query_chunk)


# ---------------------------------------------------------------------------------------------
# The Galaxy mesh: weights sharded once, index keys and cache striped block-cyclic over the 32 chips
# ---------------------------------------------------------------------------------------------
SC1_CASES = [
    # config, prompt length, query chunk, dtype
    ("small", 96, 256, "bf16"),  # below the top-k, one chunk
    ("small", 1536, 256, "bf16"),  # below the top-k, chunked
    ("small", 2304, 256, "bf16"),  # above the top-k, one round of stripes
    ("small", 2304, 256, "bfp8"),
    ("small", 4224, 512, "bf16"),  # above the top-k, two rounds (the reorder), a short last chunk
    ("small-top64", 96, 32, "bf16"),  # the chips' own top-k (a stripe is longer than the top-k)
    ("small-top64", 320, 128, "bfp8"),
    ("small-top64", 4224, 512, "bf16"),
    ("hy4", 96, 256, "bf16"),
    ("hy4", 2304, 256, "bf16"),
]


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize(
    "config, seq_len, query_chunk, dtype", SC1_CASES, ids=[f"{c}-S{s}-chunk{q}-{d}" for c, s, q, d in SC1_CASES]
)
def test_prefill_sc1(mesh_device, config, seq_len, query_chunk, dtype):
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    assert (plan.hidden_tp, plan.head_tp) == (4, 8), "SC1 splits the hidden dimension 4 ways and the heads 8 ways"
    ref_full, ref_shared = _pair(config)
    _check_layers(plan, ref_full, ref_shared, seq_len, dtype, 2, f"sc1 {config} S{seq_len} {dtype}", query_chunk)


def _checkpoint_layer(layer, full):
    """Reference layer ``layer`` with the checkpoint's attention (and indexer) weights."""
    prefix = f"model.layers.{layer}.self_attn."
    names = {n: prefix + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES}
    indexer_names = list(Hy4Indexer.CHECKPOINT_NAMES.values()) if full else []
    tensors = hy4_tensors(list(names.values()) + [prefix + "indexer." + n for n in indexer_names])
    attention = Hy4Attention(HY4_ATTENTION).eval()
    with torch.no_grad():
        for name, value in zip(names, tensors):
            getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
    indexer = None
    if full:
        indexer = Hy4Indexer(IndexerConfig()).eval().load_weights(dict(zip(indexer_names, tensors[len(names) :])))
    return Hy4AttentionPrefill(attention, indexer).eval()


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize(
    "layers, seq_len, dtype",
    [((1, 2), 96, "bf16"), ((1, 2), 2304, "bf16"), ((1, 2), 2304, "bfp8"), ((5, 6), 2304, "bf16")],
    ids=["L1-L2-S96-bf16", "L1-L2-S2304-bf16", "L1-L2-S2304-bfp8", "L5-L6-S2304-bf16"],
)
def test_prefill_sc1_real_weights(mesh_device, layers, seq_len, dtype):
    """A full layer and the shared layer above it with the checkpoint's weights (random hidden
    states: the layers' real inputs at this length need the whole stack below them, #65893)."""
    full_layer, shared_layer = layers
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    ref_full, ref_shared = _checkpoint_layer(full_layer, full=True), _checkpoint_layer(shared_layer, full=False)
    label = f"sc1 real L{full_layer}+L{shared_layer} S{seq_len} {dtype}"
    _check_layers(plan, ref_full, ref_shared, seq_len, dtype, 2, label)


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
def test_shared_layer_needs_the_full_layers_selection(mesh_device):
    """A shared layer cannot run before its full layer, nor on a prompt of another length; and a
    prompt is a whole number of tiles."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    cfg, icfg = CONFIGS["small"]
    ref_shared = _reference(cfg, None, 3)
    index_cache = TtHy4IndexCache(["full", "shared"])
    attention = TtHy4Attention(plan, cfg, ref_shared.attention.weights(), weight_dtype=ttnn.bfloat16)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        TtHy4AttentionPrefill(attention, layer=0, index_cache=index_cache)  # a full layer without its indexer
    tt_shared = TtHy4AttentionPrefill(attention, layer=1, index_cache=index_cache)
    T = 64
    x = torch.randn(T, cfg.hidden_size, generator=torch.Generator().manual_seed(4))
    cos, sin = rope_cos_sin(torch.arange(T), cfg)
    d_x, (d_cos, d_sin) = _rows(plan, x, mesh_mapper=plan.shard_hidden(3)), _rope_tables(plan, cos, sin)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        tt_shared(d_x, d_cos, d_sin)
    index_cache.store(0, tt_shared.selection_from_host(causal_allowed(torch.arange(32), 32)))
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        tt_shared(d_x, d_cos, d_sin)
    with pytest.raises(AssertionError, match="whole tiles"):  # allow-pytest.raises: host-side check
        tt_shared(_rows(plan, x[:50], mesh_mapper=plan.shard_hidden(3)), *_rope_tables(plan, cos[:50], sin[:50]))
    # with a selection of the right size it is plain masked attention
    index_cache.store(0, tt_shared.selection_from_host(causal_allowed(torch.arange(T), T)))
    with torch.no_grad():
        want = ref_shared.attention(x, cos, sin, causal_mask(T))
    check_pcc("shared layer, dense selection", want, _hidden_down(plan, tt_shared(d_x, d_cos, d_sin)), 0.998)
    index_cache.clear()


# ---------------------------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("context", [131072], ids=["128K"])
def test_selection_at_long_context(mesh_device, context):
    """The selection of one chunk of queries at the end of a long context, with synthetic keys: the
    stripes are longer than the top-k there, so every chip cuts its own candidates first, and the
    mask is reordered over 32 rounds. Correctness against the reference, and the time of a chunk."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    icfg = IndexerConfig(hidden_size=768, q_lora_rank=256)  # Hy4's index heads and keys on narrow projections
    ref = _reference(SMALL_ATTENTION, icfg, 5)
    index_cache = TtHy4IndexCache(["full"])
    tt_indexer = TtHy4Indexer(plan, icfg, ref.indexer.weights())
    attention = TtHy4Attention(plan, SMALL_ATTENTION, ref.attention.weights(), weight_dtype=ttnn.bfloat16)
    prefill = TtHy4AttentionPrefill(attention, layer=0, index_cache=index_cache, indexer=tt_indexer, query_chunk=64)

    S = prefill.query_chunk
    g = torch.Generator().manual_seed(6)
    keys = _bf16(torch.randn(context, icfg.head_dim, generator=g))
    q = _bf16(torch.randn(S, icfg.num_heads, icfg.head_dim, generator=g))
    w = torch.randn(S, icfg.num_heads, generator=g) * icfg.weight_scale
    positions = torch.arange(context - S, context)
    allowed = causal_allowed(positions, context)
    ref_scores = Hy4Indexer.scores(q, w, keys)

    stripes = tt_indexer.stripes_from_host(keys)
    assert stripes.shape[-2] == context // 32 > icfg.topk
    d_q = _up(plan, q.reshape(1, S, icfg.num_heads, icfg.head_dim))
    d_w = _rows(plan, w, ttnn.float32)
    seconds = []
    for _ in range(3):  # the first pass compiles the programs
        start = time.perf_counter()
        chosen = prefill.selected(d_q, d_w, positions.tolist(), stripes, context)
        ttnn.synchronize_device(mesh_device)
        seconds.append(time.perf_counter() - start)
    assert tuple(chosen.shape) == (1, 1, S, context)
    selected = _one_set([t[0, 0] > 0.5 for t in _per_chip(chosen)], "128K")

    masked = tt_indexer.masked_scores(d_q, d_w, tt_indexer.query_positions(positions), stripes)
    dev = tt_indexer.layout.merge(torch.stack([t[0, 0] for t in _per_chip(masked)]), context, dim=1)
    eps = (dev[allowed] - ref_scores[allowed]).abs().max().item()
    size = Hy4Indexer.scores(q, w.abs(), keys)[allowed].max().item()
    logger.info(f"SCORE ERROR {context}: max|Δ|={eps:.3e} = {eps / size:.2e} of the largest sum of |terms|")
    _check_set(
        f"context {context}",
        ref_scores.masked_fill(~allowed, float("-inf")),
        selected,
        icfg.topk,
        2 * eps + size * 2.0**-14,
    )
    best = min(seconds[1:])
    logger.info(
        f"PREFILL SELECTION context={context}: {S} queries in {best * 1e3:.1f} ms (first pass {seconds[0]:.1f} s) "
        f"= {S / best:.0f} queries/s for one full layer's selection at this context"
    )


@pytest.mark.parametrize("device_params", [GALAXY_DEVICE_PARAMS], indirect=True)
@pytest.mark.parametrize("mesh_device", [GALAXY_MESH], indirect=True)
@pytest.mark.parametrize("seq_len", [2048, 4096, 8192], ids=lambda s: f"S{s}")
def test_prefill_throughput(mesh_device, seq_len):
    """Prefill time of the attention sublayer of a full and of a shared layer at Hy4's width, random
    BFP8 weights, BFP8 cache, a prompt of ``seq_len`` tokens. The second run is the one timed (the
    first compiles). Logged as tokens per second per layer and for the attention of the whole stack
    (21 full + 57 shared layers); the MoE, the norms and iHC are not in this number."""
    plan = MeshPlan(mesh_device, hidden_axis=SC1_HIDDEN_AXIS)
    cfg, icfg = CONFIGS["hy4"]
    ref_full, ref_shared = _pair("hy4", seed=7)
    T = seq_len
    layout = BlockCyclicLayout(num_chips=32)
    kv_cache = TtHy4KvCache(
        mesh_device,
        KvCacheConfig(num_layers=2, indexer_layers=(0,), max_context=T, dtype="bfp8", layout=layout),
    )
    index_cache = TtHy4IndexCache(["full", "shared"])
    tt_full = TtHy4AttentionPrefill(
        TtHy4Attention(plan, cfg, ref_full.attention.weights()),
        layer=0,
        index_cache=index_cache,
        indexer=TtHy4Indexer(plan, icfg, ref_full.indexer.weights(), layout=layout),
        kv_cache=kv_cache,
    )
    tt_shared = TtHy4AttentionPrefill(
        TtHy4Attention(plan, cfg, ref_shared.attention.weights()), layer=1, index_cache=index_cache, kv_cache=kv_cache
    )
    g = torch.Generator().manual_seed(8)
    x = torch.randn(T, cfg.hidden_size, generator=g)
    cos, sin = rope_cos_sin(torch.arange(T), cfg)
    d_x, (d_cos, d_sin) = _rows(plan, x, mesh_mapper=plan.shard_hidden(3)), _rope_tables(plan, cos, sin)

    seconds = {"full": [], "shared": []}
    for _ in range(2):
        for kind, layer in (("full", tt_full), ("shared", tt_shared)):
            start = time.perf_counter()
            d_out = layer(d_x, d_cos, d_sin)
            ttnn.synchronize_device(mesh_device)
            seconds[kind].append(time.perf_counter() - start)
    out = _hidden_down(plan, d_out)
    assert torch.isfinite(out).all() and out.abs().max() > 0
    counts = tt_full.selection_to_host(index_cache.picks(0), chips=1)[0].sum(-1)
    assert torch.equal(counts[: icfg.topk], torch.arange(1, min(T, icfg.topk) + 1)), "a short query dropped a position"
    assert (counts[icfg.topk :] >= icfg.topk).all(), "a query selected fewer than top-k positions"
    logger.info(f"PREFILL THROUGHPUT S={T}: the largest selected set has {int(counts.max())} positions")
    assert int(counts.max()) <= icfg.topk + icfg.topk // 8, "far more than top-k positions were selected"

    full, shared = seconds["full"][-1], seconds["shared"][-1]
    stack = 21 * full + 57 * shared
    logger.info(
        f"PREFILL THROUGHPUT S={T}: full layer {full:.2f} s ({T / full:.0f} tok/s), shared layer {shared:.2f} s "
        f"({T / shared:.0f} tok/s), first runs {seconds['full'][0]:.1f} / {seconds['shared'][0]:.1f} s | attention of "
        f"21 full + 57 shared layers: {stack:.1f} s = {T / stack:.1f} tok/s"
    )
    kv_cache.deallocate()
    index_cache.clear()
