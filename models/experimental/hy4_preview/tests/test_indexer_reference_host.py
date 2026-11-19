# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the standalone indexer reference must track HF transformers' HYV4Indexer, and the
full / shared hand-over must be the one HYV4Model does.

reference/indexer.py is what the device tests validate against. This pins it to HF on the selected
SETS (a query's picks that it may see), at prompt lengths below and above the top-k, on a whole
prompt and on decode steps against HF's own cache; on the index key HF stores per token; and on
which layer's picks every layer attends over.

The second half pins the SC1 form of the selection (striped_select: per-chip stripes, local
candidates, one gather, a merge through a bf16-only top-k) to the plain top-k at 4K, 128K and 1M
context, and counts what each refinement pass leaves unresolved.

Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates ``hy_v4``.
"""

import inspect

import pytest
import torch

from models.experimental.hy4_preview.reference.attention import causal_mask, rope_cos_sin
from models.experimental.hy4_preview.reference.indexer import (
    Hy4Indexer,
    IndexerConfig,
    causal_allowed,
    compare_selection,
    full_layers,
    picked_mask,
    picks_source,
    selection_mask,
    striped_select,
    top_through_bf16,
)
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

SMALL = IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=4, head_dim=32, qk_rope_head_dim=8, topk=16)
# The real top-k and head count on narrow projections, so a prompt longer than 2048 stays cheap.
REAL_TOPK = IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=32, head_dim=32, qk_rope_head_dim=8)
CONFIGS = {"small": SMALL, "top2048": REAL_TOPK}


def _hf_config(cfg: IndexerConfig, num_layers=2) -> HYV4Config:
    return HYV4Config(
        vocab_size=64,
        hidden_size=cfg.hidden_size,
        intermediate_size=128,
        moe_intermediate_size=48,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=cfg.qk_rope_head_dim,
        num_hidden_layers=num_layers,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=32,
        qk_nope_head_dim=24,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=16,
        index_topk=cfg.topk,
        index_head_dim=cfg.head_dim,
        index_n_heads=cfg.num_heads,
        rms_norm_eps=cfg.norm_eps,
        max_position_embeddings=1 << 20,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )


def _randomize(module, seed):
    """Linear outputs of order one, and norm weights and biases away from their trivial init."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif "k_norm" in name:
                p.add_(0.25 * torch.randn(p.shape, generator=g))
    return module


def _pair(cfg, seed):
    hf_cfg = _hf_config(cfg)
    hf = _randomize(hy_v4.HYV4Indexer(hf_cfg, layer_idx=0).eval(), seed)
    ref = Hy4Indexer(cfg).eval().load_weights(hf.state_dict())
    return hf_cfg, hf, ref


class _KeyTap:
    """Stands in for the HF cache on a whole-prompt call: records the index keys the indexer hands it."""

    def update_indexer(self, keys, layer_idx):
        self.keys = keys
        return keys


def test_config_defaults_match_checkpoint_layout():
    assert full_layers(78).count("full") == 21
    assert full_layers(78) == HYV4Config(num_hidden_layers=78).indexer_types
    source = picks_source(full_layers(78))
    assert source[:7] == [0, 1, 1, 1, 1, 5, 5] and source[-1] == 77
    with pytest.raises(ValueError):  # allow-pytest.raises: host-side check
        picks_source(["shared", "full"])


@pytest.mark.parametrize(
    "config, seq_len",
    [("small", 12), ("small", 16), ("small", 40), ("top2048", 1536), ("top2048", 2304)],
    ids=["below-topk", "at-topk", "above-topk", "top2048-below", "top2048-above"],
)
@pytest.mark.parametrize("mask_kind", ["additive", "bool"])
def test_indexer_matches_hf(config, seq_len, mask_kind):
    cfg = CONFIGS[config]
    hf_cfg, hf, ref = _pair(cfg, seed=3)
    g = torch.Generator().manual_seed(4)
    x = torch.randn(seq_len, cfg.hidden_size, generator=g)
    q_resid = torch.randn(seq_len, cfg.q_lora_rank, generator=g)
    positions = torch.arange(seq_len) + 5  # a non-zero start, so RoPE is exercised off the origin
    allowed = causal_allowed(torch.arange(seq_len), seq_len)
    mask = allowed if mask_kind == "bool" else causal_mask(seq_len)

    tap = _KeyTap()
    with torch.no_grad():
        position_embeddings = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x[None], positions[None])
        # the cache goes in by position: its keyword was renamed between transformers releases
        hf_picks = hf(x[None], q_resid[None], position_embeddings, mask[None], positions[None], tap)[0]
        cos, sin = rope_cos_sin(positions, cfg)
        picks, keys = ref(x, q_resid, cos, sin, mask)
        scores = ref.scores(ref.queries(q_resid, cos, sin), ref.head_weights(x), keys)

    assert torch.equal(cos, position_embeddings[0][0]) and torch.equal(sin, position_embeddings[1][0])
    # the head weights carry HF's two scales (a positive factor does not move the selection, so the
    # selected sets below would not notice a wrong one)
    with torch.no_grad():
        hf_w = hf.weights_proj(x).float() * (hf.n_heads**-0.5) * hf.softmax_scale
        assert torch.allclose(ref.head_weights(x), hf_w, rtol=1e-6, atol=0)
    # the cache entry
    assert keys.shape == (seq_len, cfg.head_dim)
    max_diff = (keys - tap.keys[0]).abs().max().item()
    assert max_diff <= 1e-5 * tap.keys.abs().max().item(), f"index keys: max|Δ|={max_diff:.3e}"

    k = min(cfg.topk, seq_len)
    assert picks.shape == hf_picks.shape == (seq_len, k) and picks.dtype == hf_picks.dtype == torch.int32
    selected, hf_selected = selection_mask(picks, allowed), selection_mask(hf_picks, allowed)
    result = compare_selection(scores.masked_fill(~allowed, float("-inf")), hf_selected, selected, cfg.topk, tol=0.0)
    assert torch.equal(selected, hf_selected), result
    # a query that sees at most top-k positions selects all of them; one that sees more selects exactly top-k
    assert torch.equal(selected.sum(-1), torch.clamp(torch.arange(seq_len) + 1, max=cfg.topk))
    if seq_len > cfg.topk:
        assert (selected.sum(-1) < allowed.sum(-1)).any(), "no query was actually cut by the top-k"
    # The order inside a query's picks is not compared: nothing downstream reads it, and two scores an
    # ulp apart can swap between HF's batched matmul and the reference's.


def test_index_key_norm_eps():
    """k_norm's eps is the model's rms_norm_eps (1e-5). With hidden states of ordinary size the eps
    is lost next to the variance, so this uses small ones, where the key depends on it."""
    cfg = SMALL
    hf_cfg, hf, ref = _pair(cfg, seed=9)
    g = torch.Generator().manual_seed(10)
    S = 24
    x = 3e-3 * torch.randn(S, cfg.hidden_size, generator=g)
    q_resid = torch.randn(S, cfg.q_lora_rank, generator=g)
    positions = torch.arange(S)
    tap = _KeyTap()
    with torch.no_grad():
        position_embeddings = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x[None], positions[None])
        hf(x[None], q_resid[None], position_embeddings, causal_mask(S)[None], positions[None], tap)
        cos, sin = rope_cos_sin(positions, cfg)
        keys = ref.keys(x, cos, sin)
        other = Hy4Indexer(IndexerConfig(**{**cfg.__dict__, "norm_eps": 1e-6})).eval().load_weights(hf.state_dict())
        other_keys = other.keys(x, cos, sin)
    scale = tap.keys.abs().max().item()
    assert (keys - tap.keys[0]).abs().max().item() <= 1e-5 * scale
    assert (other_keys - tap.keys[0]).abs().max().item() > 1e-2 * scale, "the inputs do not exercise the eps"


def test_scores_formula():
    """The reference's score is sum_h w * relu(q . k) with a signed w, written out the slow way."""
    cfg = SMALL
    _, _, ref = _pair(cfg, seed=5)
    g = torch.Generator().manual_seed(6)
    S = 9
    x = torch.randn(S, cfg.hidden_size, generator=g)
    q_resid = torch.randn(S, cfg.q_lora_rank, generator=g)
    cos, sin = rope_cos_sin(torch.arange(S), cfg)
    with torch.no_grad():
        keys = ref.keys(x, cos, sin)
        q, w = ref.queries(q_resid, cos, sin), ref.head_weights(x)
        scores = ref.scores(q, w, keys)
        slow = torch.zeros(S, S)
        for s in range(S):
            for t in range(S):
                for h in range(cfg.num_heads):
                    slow[s, t] += w[s, h] * torch.relu(torch.dot(q[s, h], keys[t]))
    assert (w < 0).any() and (w > 0).any(), "head weights of one sign would not exercise the signed sum"
    assert (scores < 0).any()
    assert torch.allclose(scores, slow, atol=1e-5)
    # keys: RoPE only on the trailing dims, by position
    unrotated = ref.keys(x, torch.ones_like(cos), torch.zeros_like(sin))
    assert torch.equal(keys[:, : cfg.pass_dim], unrotated[:, : cfg.pass_dim])
    assert torch.equal(keys[0], unrotated[0]) and not torch.allclose(keys[1:], unrotated[1:])


def test_compare_selection_separates_near_ties():
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.999, 1.0, float("-inf")]])
    selected = torch.tensor([[True, True, True, False, False, False]])
    swapped = torch.tensor([[True, True, False, True, False, False]])  # took 2.999 instead of 3.0
    wrong = torch.tensor([[True, True, False, False, True, False]])  # took 1.0 instead of 3.0
    assert compare_selection(scores, selected, selected, 3, 0.01)["real"] == 0
    r = compare_selection(scores, selected, swapped, 3, 0.01)
    assert (r["missing"], r["extra"], r["near_tie"], r["real"]) == (1, 1, 2, 0)
    r = compare_selection(scores, selected, wrong, 3, 0.01)
    assert (r["missing"], r["extra"], r["near_tie"], r["real"]) == (1, 1, 1, 1)
    # no cut (the query sees no more than k positions): every disagreement is real
    r = compare_selection(scores, torch.isfinite(scores), wrong, 5, 0.01)
    assert r["near_tie"] == 0 and r["real"] == 2


# ---------------------------------------------------------------------------------------------
# The whole HF model with its cache: prefill, then decode steps, 6 layers (full full shared x3 full)
# ---------------------------------------------------------------------------------------------
def _taps(model):
    """Record, per layer and per forward call, what the indexer was given and what attention returned."""
    calls = {"indexer": {}, "picks": {}}
    for layer, block in enumerate(model.layers):
        attn = block.self_attn

        def attention_hook(module, args, kwargs, output, layer=layer):
            calls["picks"].setdefault(layer, []).append(output[2])

        attn.register_forward_hook(attention_hook, with_kwargs=True)
        if attn.indexer is not None:

            def indexer_hook(module, args, kwargs, output, layer=layer):
                x, q_resid, position_embeddings, mask = args[:4]
                calls["indexer"].setdefault(layer, []).append(
                    (x[0], q_resid[0], position_embeddings, mask[0], output[0])
                )

            attn.indexer.register_forward_hook(indexer_hook, with_kwargs=True)
    return calls


@pytest.mark.parametrize("prefill_len", [10, 24], ids=["prefill-below-topk", "prefill-above-topk"])
def test_decode_and_reuse_match_hf_model(prefill_len):
    cfg = SMALL
    num_layers, decode_steps = 6, 12  # the shorter prefill crosses the top-k while decoding
    hf_cfg = _hf_config(cfg, num_layers)
    assert hf_cfg.indexer_types == full_layers(num_layers) == ["full", "full", "shared", "shared", "shared", "full"]
    model = _randomize(hy_v4.HYV4Model(hf_cfg).eval(), seed=7)
    calls = _taps(model)
    g = torch.Generator().manual_seed(8)
    ids = torch.randint(3, hf_cfg.vocab_size, (1, prefill_len + decode_steps), generator=g)

    # the cache keyword and output field were renamed between transformers releases
    cache_name = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
    with torch.no_grad():
        out = model(input_ids=ids[:, :prefill_len], use_cache=True)
        for step in range(decode_steps):
            pos = prefill_len + step
            out = model(input_ids=ids[:, pos : pos + 1], use_cache=True, **{cache_name: getattr(out, cache_name)})
    cache = getattr(out, cache_name)

    # the hand-over: every layer attends over the picks of its source layer, the very same tensor
    source = picks_source(hf_cfg.indexer_types)
    assert source == [0, 1, 1, 1, 1, 5]
    for layer in range(num_layers):
        assert len(calls["picks"][layer]) == 1 + decode_steps
        for call, picks in enumerate(calls["picks"][layer]):
            assert picks is calls["picks"][source[layer]][call], (layer, call)
    assert sorted(calls["indexer"]) == [0, 1, 5]

    # every full layer, replayed through the reference with its own key cache
    crossed = False
    for layer in (0, 1, 5):
        ref = Hy4Indexer(cfg).eval().load_weights(model.layers[layer].self_attn.indexer.state_dict())
        cached = None
        for call, (x, q_resid, (hf_cos, hf_sin), mask, hf_picks) in enumerate(calls["indexer"][layer]):
            first = 0 if call == 0 else prefill_len + call - 1
            positions = torch.arange(first, first + x.shape[0])
            cos, sin = rope_cos_sin(positions, cfg)
            assert torch.equal(cos, hf_cos[0]) and torch.equal(sin, hf_sin[0])
            with torch.no_grad():
                picks, cached = ref(x, q_resid, cos, sin, mask, cached_keys=cached)
            num_keys = first + x.shape[0]
            assert cached.shape == (num_keys, cfg.head_dim) and mask.shape == (x.shape[0], num_keys)
            allowed = causal_allowed(positions, num_keys)
            selected, hf_selected = selection_mask(picks, allowed), selection_mask(hf_picks, allowed)
            assert torch.equal(selected, hf_selected), (layer, call)
            assert torch.equal(selected.sum(-1), torch.clamp(positions + 1, max=cfg.topk))
            crossed |= bool((selected.sum(-1) < allowed.sum(-1)).any())
        # the reference's key cache is the one HF kept
        hf_keys = cache.layers[layer].indexer_keys
        assert hf_keys.shape == (1, prefill_len + decode_steps, cfg.head_dim)
        assert (cached - hf_keys[0]).abs().max() <= 1e-5 * hf_keys.abs().max()
    assert crossed, "no decode step was cut by the top-k"


# ---------------------------------------------------------------------------------------------
# The SC1 selection (stripes, local candidates, gather, bf16-only top-k) against the plain top-k
# ---------------------------------------------------------------------------------------------
def _synthetic(cfg, context, positions, seed, weights="signed"):
    """Random bf16 keys for ``context`` positions and random queries at ``positions``:
    (scores [S, T] unmasked, masked with -inf, the exact selected set, allowed)."""
    g = torch.Generator().manual_seed(seed)
    S = len(positions)
    keys = torch.randn(context, cfg.head_dim, generator=g).bfloat16().float()
    q = torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g).bfloat16().float()
    w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale
    if weights == "negative":
        w = -w.abs()
    scores = Hy4Indexer.scores(q, w, keys)
    allowed = causal_allowed(positions, context)
    masked = scores.masked_fill(~allowed, float("-inf"))
    exact = selection_mask(masked.topk(min(cfg.topk, context), dim=-1).indices, allowed)
    return scores, masked, exact, allowed


def _unresolved(masked, k, passes):
    """The band around the cut that ``passes`` bf16 passes cannot order: a bf16 bin is at most 2^-7
    of the value wide, and every pass narrows the value being ranked to the previous bin."""
    kth = masked.topk(min(k, masked.shape[-1]), dim=-1).values[:, -1]
    kth = kth[torch.isfinite(kth)]  # empty when no query sees more than k positions: there is no cut
    return kth.abs().max().item() * 2.0 ** (-7 * passes) if kth.numel() else 0.0


@pytest.mark.parametrize("passes", [1, 2, 3], ids=lambda p: f"passes{p}")
@pytest.mark.parametrize("context", [4096, 131072, 1048576], ids=["4K", "128K", "1M"])
def test_striped_selection_matches_plain_topk(context, passes, record_property):
    cfg = IndexerConfig()
    layout = BlockCyclicLayout()
    T = context
    # the newest token, a query with a quarter of the context after it, mid-context, below the top-k
    positions = torch.tensor([T - 1, 3 * T // 4 - 77, T // 2, 1500])
    scores, masked, exact, allowed = _synthetic(cfg, T, positions, seed=T + passes)

    picked, valid = striped_select(scores, positions, layout, cfg.topk, refine_passes=passes)
    assert picked.shape == valid.shape == (4, cfg.topk)
    got = picked_mask(picked, valid, T)
    assert not (got & ~allowed).any(), "a position after the query was selected"
    assert torch.equal(got[3], allowed[3]), "a query below the top-k must select everything it sees"
    assert torch.equal(got.sum(-1), exact.sum(-1)), "every query selects min(top-k, positions it sees)"

    r = compare_selection(masked, exact, got, cfg.topk, tol=_unresolved(masked, cfg.topk, passes))
    print(f"\nSTRIPED context={T} passes={passes}: picks={int(exact.sum())} {r}")
    for key, value in r.items():
        record_property(key, value)
    assert r["real"] == 0, r
    if passes >= 2:
        assert r["missing"] <= 2 * len(positions), r


@pytest.mark.parametrize("context", [1, 129, 300, 2049, 3000, 70001, 200000])
def test_striped_selection_on_ragged_contexts(context):
    """Contexts that are not a whole number of blocks, or of rounds over the chips: a single
    position, chips that hold nothing, a part-filled last block, stripes whose length is not a power
    of two (70,001 positions: 2176 slots; 200,000: 6272). Queries at the first position, mid-context
    and at the newest token."""
    cfg = IndexerConfig()
    layout = BlockCyclicLayout()
    T = context
    positions = torch.tensor(sorted({0, T // 2, T - 1}))
    scores, masked, exact, allowed = _synthetic(cfg, T, positions, seed=T)
    picked, valid = striped_select(scores, positions, layout, cfg.topk)
    assert picked.shape == valid.shape == (len(positions), cfg.topk)
    assert torch.equal(valid.sum(-1), torch.clamp(positions + 1, max=cfg.topk))
    assert bool((picked < layout.num_chips * layout.slots(T)).all()), "a filler must still name a slot of the stripes"
    got = picked_mask(picked, valid, T)
    assert not (got & ~allowed).any()
    r = compare_selection(masked, exact, got, cfg.topk, tol=_unresolved(masked, cfg.topk, 2))
    assert r["real"] == 0 and r["missing"] <= 2 * len(positions) and r["size_diff"] == 0, r


@pytest.mark.parametrize("kind", ["all-equal", "integers", "zeros-at-the-cut"])
def test_striped_selection_with_exact_ties(kind):
    """Scores that tie exactly, in bulk: every position the same, a handful of distinct values, and
    relu-style scores where the cut falls among thousands of exact zeros. Which of the tied
    positions is taken is free; the selection must still hold exactly top-k distinct visible
    positions and differ from the plain top-k only on positions whose score equals the cut."""
    cfg = IndexerConfig()
    layout = BlockCyclicLayout()
    T = 200000
    g = torch.Generator().manual_seed(11)
    positions = torch.tensor([T - 1, 150000, 5000])
    if kind == "all-equal":
        scores = torch.full((3, T), 3.25)
    elif kind == "integers":
        scores = torch.randint(-3, 4, (3, T), generator=g).float()
    else:
        scores = torch.relu(torch.randn(3, T, generator=g) - 2.5)  # about 0.6% positive
    allowed = causal_allowed(positions, T)
    masked = scores.masked_fill(~allowed, float("-inf"))
    exact = selection_mask(masked.topk(cfg.topk, dim=-1).indices, allowed)
    picked, valid = striped_select(scores, positions, layout, cfg.topk)
    assert valid.all()
    got = picked_mask(picked, valid, T)  # raises on a repeated position
    assert not (got & ~allowed).any() and torch.equal(got.sum(-1), exact.sum(-1))
    r = compare_selection(masked, exact, got, cfg.topk, tol=0.0)
    assert r["real"] == 0 and r["near_tie"] > 0, r


def test_striped_selection_when_one_chip_holds_every_pick():
    """The worst skew: all of the global top-2048 sit in one chip's stripe, so that chip's local
    candidates are the whole selection and every other chip contributes nothing."""
    cfg = IndexerConfig()
    layout = BlockCyclicLayout()
    T = 1 << 18
    positions = torch.tensor([T - 1])
    scores, _, _, allowed = _synthetic(cfg, T, positions, seed=5)
    scores = scores + 10.0 * (layout.chip(torch.arange(T)) == 3)
    masked = scores.masked_fill(~allowed, float("-inf"))
    exact = selection_mask(masked.topk(cfg.topk, dim=-1).indices, allowed)
    assert bool((layout.chip(exact[0].nonzero()[:, 0]) == 3).all())
    picked, valid = striped_select(scores, positions, layout, cfg.topk)
    assert valid.all() and bool((layout.chip(picked) == 3).all())
    r = compare_selection(masked, exact, picked_mask(picked, valid, T), cfg.topk, tol=_unresolved(masked, cfg.topk, 2))
    assert r["real"] == 0 and r["missing"] <= 2 and r["size_diff"] == 0, r


def test_striped_selection_never_takes_masked_slots():
    """Every visible score is negative, so a slot that is not masked (later positions score higher,
    unfilled slots hold zero) would win a place."""
    cfg = IndexerConfig(topk=64)
    layout = BlockCyclicLayout()
    T, p = 6000, 4500
    positions = torch.tensor([p])
    scores, masked, exact, allowed = _synthetic(cfg, T, positions, seed=6, weights="negative")
    scores[:, p + 1 :] += 100.0
    assert (scores[:, : p + 1] <= 0).all()
    picked, valid = striped_select(scores, positions, layout, cfg.topk, slots=layout.slots(8000))
    got = picked_mask(picked, valid, T)
    assert not (got & ~allowed).any()
    r = compare_selection(masked, exact, got, cfg.topk, tol=_unresolved(masked, cfg.topk, 2))
    assert r["real"] == 0 and r["missing"] <= 1 and r["size_diff"] == 0, r


def test_striped_selection_on_one_chip():
    """One stripe: the chip's local top-k is the whole selection."""
    cfg = IndexerConfig(topk=256)
    layout = BlockCyclicLayout(num_chips=1)
    T = 8192
    positions = torch.tensor([T - 1, 5000, 100])
    scores, masked, exact, allowed = _synthetic(cfg, T, positions, seed=7)
    picked, valid = striped_select(scores, positions, layout, cfg.topk)
    assert picked.shape == (3, cfg.topk) and int(valid[2].sum()) == 101
    got = picked_mask(picked, valid, T)
    r = compare_selection(masked, exact, got, cfg.topk, tol=_unresolved(masked, cfg.topk, 2))
    assert r["real"] == 0 and r["missing"] <= 2 and r["size_diff"] == 0, r
    assert torch.equal(got[2], allowed[2])


def test_top_through_bf16_membership_rule():
    """The (centre, cut) a pass returns reproduces its own picks: bf16(score - centre) >= cut."""
    g = torch.Generator().manual_seed(8)
    scores = torch.randn(3, 5000, generator=g)
    for passes in (1, 2, 3):
        centre, cut, slots = top_through_bf16(scores, 300, passes)
        ranked = (scores if centre is None else scores - centre).bfloat16().float()
        member = ranked >= cut
        assert member.gather(-1, slots).all()
        assert (member.sum(-1) >= 300).all() and (member.sum(-1) <= (300 + (40 if passes == 1 else 3))).all()
