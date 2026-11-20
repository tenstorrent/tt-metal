# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the prefill attention reference must track HF transformers' HYV4Attention with its
indexer, on a full layer and on a shared one, below and above the top-k.

reference/attention_prefill.py is what the device prefill tests validate against. This pins it to
the HF module run with the eager core on a whole prompt:

* a "full" layer (HYV4Attention with its own HYV4Indexer): the output, the picks as selected sets,
  and what the layer hands the cache (the latent and the index key of every token);
* a "shared" layer (no indexer) fed the full layer's picks, as HYV4Model threads them upward;
* at a small top-k and at the real 2048, with prompts shorter and longer than the top-k. Above the
  top-k the test also requires the selection to change the output, so a reference that ignored it
  could not pass.

The second half pins threshold_select, the mask form of the selection the device prefill computes,
to the plain top-k: on one chip and on 32, at 4K with the real top-k and at 128K.

Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates ``hy_v4``.
"""

import inspect

import pytest
import torch

from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin
from models.experimental.hy4_preview.reference.attention_prefill import (
    Hy4AttentionPrefill,
    additive_mask,
    dsa_mask,
    threshold_select,
)
from models.experimental.hy4_preview.reference.indexer import (
    Hy4Indexer,
    IndexerConfig,
    causal_allowed,
    compare_selection,
    selection_mask,
)
from models.experimental.hy4_preview.tt.kv_layout import BlockCyclicLayout

hy_v4 = pytest.importorskip("transformers.models.hy_v4.modeling_hy_v4")
from transformers.models.hy_v4.configuration_hy_v4 import HYV4Config  # noqa: E402

# reference attribute -> HF module attribute
HF_NAMES = {
    "q_a_proj": "q_a_proj.weight",
    "q_a_layernorm": "q_a_layernorm.weight",
    "q_b_proj": "q_b_proj.weight",
    "kv_a_proj_with_mqa": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm": "kv_a_layernorm.weight",
    "kv_b_proj": "kv_b_proj.weight",
    "o_proj": "o_proj.weight",
    "linear_gate": "gate_proj.weight",
    "learnable_sink_param": "sinks",
}

ATTENTION = AttentionConfig(
    hidden_size=96,
    num_heads=4,
    q_lora_rank=48,
    kv_lora_rank=32,
    qk_nope_head_dim=24,
    qk_rope_head_dim=8,
    v_head_dim=16,
)
# (indexer config): a small top-k, and the real top-k and head count on narrow projections so that a
# prompt longer than 2048 stays cheap.
INDEXERS = {
    "top16": IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=4, head_dim=32, qk_rope_head_dim=8, topk=16),
    "top2048": IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=32, head_dim=32, qk_rope_head_dim=8),
}


def _hf_config(cfg: AttentionConfig, icfg: IndexerConfig) -> HYV4Config:
    return HYV4Config(
        hidden_size=cfg.hidden_size,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        head_dim=cfg.qk_rope_head_dim,
        num_hidden_layers=2,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        index_topk=icfg.topk,
        index_head_dim=icfg.head_dim,
        index_n_heads=icfg.num_heads,
        rms_norm_eps=icfg.norm_eps,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        max_position_embeddings=1 << 20,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        attn_implementation="eager",
    )


def _pair(cfg, icfg, layer, seed):
    """HF's attention layer ``layer`` (0: full, 1: shared) and the reference, holding the same random
    parameters: linear outputs of order one, norm weights, biases and sinks away from their init."""
    hf_cfg = _hf_config(cfg, icfg)
    hf = hy_v4.HYV4Attention(hf_cfg, layer_idx=layer).eval()
    assert (hf.indexer is not None) == (layer == 0)
    g = torch.Generator().manual_seed(seed)
    attention = Hy4Attention(cfg).eval()
    hf_state = dict(hf.named_parameters())
    with torch.no_grad():
        for name, hf_name in HF_NAMES.items():
            p = getattr(attention, name)
            if p.dim() == 2:
                value = torch.randn(p.shape, generator=g) * p.shape[1] ** -0.5
            else:
                value = 1.0 + 0.5 * torch.randn(p.shape, generator=g)
            p.copy_(value)
            assert hf_state[hf_name].shape == p.shape, (name, hf_state[hf_name].shape, p.shape)
            hf_state[hf_name].copy_(value)
        indexer = None
        if hf.indexer is not None:
            for name, p in hf.indexer.named_parameters():
                if p.dim() >= 2:
                    p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
                else:  # k_norm weight and bias
                    p.add_(0.25 * torch.randn(p.shape, generator=g))
            indexer = Hy4Indexer(icfg).eval().load_weights(hf.indexer.state_dict())
    return hf_cfg, hf, Hy4AttentionPrefill(attention, indexer).eval()


class _CacheTap:
    """Stands in for the HF cache on a whole-prompt call: records what the layer hands it."""

    def update(self, k_pass, k_rot, layer_idx, **kwargs):
        self.k_pass, self.k_rot = k_pass, k_rot
        return k_pass, k_rot

    def update_indexer(self, keys, layer_idx):
        self.index_keys = keys
        return keys


def _run_hf(hf, hf_cfg, x, positions, prev_picks=None, tap=None):
    seq_len = x.shape[0]
    # the cache keyword was renamed between transformers releases
    cache_kwarg = "cache" if "cache" in inspect.signature(hf.forward).parameters else "past_key_values"
    with torch.no_grad():
        position_embeddings = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x[None], positions[None])
        out, _, picks = hf(
            x[None],
            position_embeddings=position_embeddings,
            attention_mask=causal_mask(seq_len)[None, None],
            position_ids=positions[None],
            prev_topk_indices=None if prev_picks is None else prev_picks[None],
            **({} if tap is None else {cache_kwarg: tap}),
        )
    return out[0], picks[0], position_embeddings


def _close(name, a, b, rel=1e-5):
    max_diff = (a - b).abs().max().item()
    assert max_diff <= rel * b.abs().max().item(), f"{name}: max|Δ|={max_diff:.3e} vs scale {b.abs().max().item():.3e}"


CASES = [("top16", 12), ("top16", 16), ("top16", 40), ("top2048", 1536), ("top2048", 2304)]
CASE_IDS = ["top16-below", "top16-at", "top16-above", "top2048-below", "top2048-above"]


@pytest.mark.parametrize("indexer, seq_len", CASES, ids=CASE_IDS)
def test_full_layer_matches_hf(indexer, seq_len):
    cfg, icfg = ATTENTION, INDEXERS[indexer]
    hf_cfg, hf, ref = _pair(cfg, icfg, layer=0, seed=3)
    assert ref.is_full
    g = torch.Generator().manual_seed(4)
    x = torch.randn(seq_len, cfg.hidden_size, generator=g)
    positions = torch.arange(seq_len)

    tap = _CacheTap()
    hf_out, hf_picks, position_embeddings = _run_hf(hf, hf_cfg, x, positions, tap=tap)
    cos, sin = rope_cos_sin(positions, cfg)
    assert torch.equal(cos, position_embeddings[0][0]) and torch.equal(sin, position_embeddings[1][0])
    with torch.no_grad():
        got = ref(x, cos, sin)
        dense = ref.attention(x, cos, sin, causal_mask(seq_len))

    # the selection, as sets
    k = min(icfg.topk, seq_len)
    allowed = causal_allowed(positions, seq_len)
    assert got.picks.shape == hf_picks.shape == (seq_len, k) and got.picks.dtype == hf_picks.dtype == torch.int32
    assert torch.equal(got.selected, selection_mask(hf_picks, allowed))
    assert torch.equal(got.selected.sum(-1), torch.clamp(positions + 1, max=icfg.topk))
    # the output
    _close("output", got.out, hf_out)
    # what the layer leaves in the cache
    assert got.latent.shape == (seq_len, cfg.latent_dim) and got.index_keys.shape == (seq_len, icfg.head_dim)
    _close("latent", got.latent, torch.cat([tap.k_pass, tap.k_rot], dim=-1)[0, 0])
    _close("index keys", got.index_keys, tap.index_keys[0])

    scale = hf_out.abs().max().item()
    moved = (dense - hf_out).abs().max().item()
    if seq_len > icfg.topk:
        assert (got.selected.sum(-1) < allowed.sum(-1)).any(), "no query was cut by the top-k"
        assert moved > 1e-2 * scale, f"the selection moved the output by only {moved:.3e} of {scale:.3e}"
    else:  # every query picks everything it sees: plain causal attention
        assert moved <= 1e-5 * scale


@pytest.mark.parametrize("indexer, seq_len", CASES, ids=CASE_IDS)
def test_shared_layer_matches_hf(indexer, seq_len):
    """A shared layer attends over the picks of the full layer below it, which see other hidden
    states than its own: the picks come from HF's full layer run on its own input."""
    cfg, icfg = ATTENTION, INDEXERS[indexer]
    hf_cfg, hf_full, ref_full = _pair(cfg, icfg, layer=0, seed=3)
    _, hf_shared, ref_shared = _pair(cfg, icfg, layer=1, seed=5)
    assert not ref_shared.is_full
    g = torch.Generator().manual_seed(6)
    x_full = torch.randn(seq_len, cfg.hidden_size, generator=g)
    x_shared = torch.randn(seq_len, cfg.hidden_size, generator=g)
    positions = torch.arange(seq_len)
    cos, sin = rope_cos_sin(positions, cfg)

    _, hf_picks, _ = _run_hf(hf_full, hf_cfg, x_full, positions)
    tap = _CacheTap()
    hf_out, hf_passed_on, _ = _run_hf(hf_shared, hf_cfg, x_shared, positions, prev_picks=hf_picks, tap=tap)
    assert hf_passed_on is not None and torch.equal(hf_passed_on, hf_picks)  # handed on unchanged

    with torch.no_grad():
        full = ref_full(x_full, cos, sin)
        got = ref_shared(x_shared, cos, sin, prev_picks=full.picks)
        dense = ref_shared.attention(x_shared, cos, sin, causal_mask(seq_len))
    assert got.picks is full.picks and got.index_keys is None
    assert torch.equal(got.selected, full.selected)
    _close("output", got.out, hf_out)
    _close("latent", got.latent, torch.cat([tap.k_pass, tap.k_rot], dim=-1)[0, 0])
    assert not hasattr(tap, "index_keys"), "a shared layer caches no index key"

    scale = hf_out.abs().max().item()
    moved = (dense - hf_out).abs().max().item()
    if seq_len > icfg.topk:
        assert moved > 1e-2 * scale, f"the selection moved the output by only {moved:.3e} of {scale:.3e}"
    else:
        assert moved <= 1e-5 * scale

    with pytest.raises(ValueError):  # allow-pytest.raises: host-side check
        ref_shared(x_shared, cos, sin)


def test_mask_forms_agree():
    """dsa_mask (HF's construction from the picks) and additive_mask of the selected set are the same
    mask, also when a query's picks name positions after it."""
    cfg, icfg = ATTENTION, INDEXERS["top16"]
    _, _, ref = _pair(cfg, icfg, layer=0, seed=7)
    S = 40
    x = torch.randn(S, cfg.hidden_size, generator=torch.Generator().manual_seed(8))
    cos, sin = rope_cos_sin(torch.arange(S), cfg)
    with torch.no_grad():
        got = ref(x, cos, sin)
    allowed = causal_allowed(torch.arange(S), S)
    assert (~allowed.gather(-1, got.picks.long())).any(), "no pick names a position after its query"
    assert torch.equal(dsa_mask(got.picks, S), additive_mask(got.selected))


# ---------------------------------------------------------------------------------------------
# The selection in mask form (stripes, the cut through a bf16-only top-k, a compare) vs the plain top-k
# ---------------------------------------------------------------------------------------------
def _synthetic(cfg, context, positions, seed):
    """Random bf16 keys for ``context`` positions and random queries at ``positions``:
    (scores [S, T] unmasked, masked with -inf, the exact selected set)."""
    g = torch.Generator().manual_seed(seed)
    S = len(positions)
    keys = torch.randn(context, cfg.head_dim, generator=g).bfloat16().float()
    q = torch.randn(S, cfg.num_heads, cfg.head_dim, generator=g).bfloat16().float()
    w = torch.randn(S, cfg.num_heads, generator=g) * cfg.weight_scale
    scores = Hy4Indexer.scores(q, w, keys)
    allowed = causal_allowed(positions, context)
    masked = scores.masked_fill(~allowed, float("-inf"))
    exact = selection_mask(masked.topk(min(cfg.topk, context), dim=-1).indices, allowed)
    return scores, masked, exact


THRESHOLD_CASES = {
    # name: (indexer config, chips, context, query positions)
    "one-chip-no-cut": (INDEXERS["top2048"], 1, 96, list(range(96))),
    "one-chip-cut": (INDEXERS["top2048"], 1, 2304, [0, 1, 2046, 2047, 2048, 2049, 2200, 2303]),
    "one-chip-top16": (INDEXERS["top16"], 1, 200, list(range(0, 200, 7)) + [15, 16, 17, 199]),
    "mesh-top16": (INDEXERS["top16"], 32, 5000, [0, 15, 16, 17, 127, 128, 4095, 4096, 4999]),
    "mesh-below": (INDEXERS["top2048"], 32, 1536, [0, 1, 700, 1535]),
    "mesh-above": (INDEXERS["top2048"], 32, 2304, [0, 2047, 2048, 2049, 2303]),
    "mesh-4K": (INDEXERS["top2048"], 32, 4224, [0, 2047, 2048, 4095, 4096, 4223]),
    "mesh-128K": (INDEXERS["top2048"], 32, 131072, [5, 2047, 2048, 70000, 131071]),
}


@pytest.mark.parametrize("case", list(THRESHOLD_CASES))
def test_threshold_select_is_the_topk_set(case):
    cfg, chips, context, positions = THRESHOLD_CASES[case]
    positions = torch.tensor(positions)
    layout = BlockCyclicLayout(num_chips=chips)
    scores, masked, exact = _synthetic(cfg, context, positions, seed=11)
    selected = threshold_select(scores, positions, layout, cfg.topk)
    assert selected.shape == exact.shape

    # Two refinement passes resolve the cut to 2^-14 of its value; anything else is an error.
    visible = torch.isfinite(masked)
    kth = masked.topk(min(cfg.topk, context), dim=-1).values[:, -1:]
    tol = float(kth[torch.isfinite(kth)].abs().max()) * 2.0**-14 if torch.isfinite(kth).any() else 0.0
    result = compare_selection(masked, exact, selected, cfg.topk, tol)
    assert result["real"] == 0, result
    assert result["missing"] == 0, f"a top-k position was dropped: {result}"  # ties are kept, never dropped
    assert not (selected & ~visible).any(), "a position after the query was selected"
    seen = visible.sum(-1)
    assert torch.equal(selected.sum(-1)[seen <= cfg.topk], seen[seen <= cfg.topk])
    assert (selected.sum(-1)[seen > cfg.topk] >= cfg.topk).all()
    assert result["extra"] <= 2 * len(positions), result
    if context > cfg.topk:
        assert (selected.sum(-1) < seen).any(), "no query was cut"


def test_threshold_select_keeps_ties_at_the_cut():
    """Scores the bf16 passes cannot tell apart at the cut are all selected: the set grows, it does
    not lose a member. One pass (plain bf16 ranking) makes such ties common."""
    layout = BlockCyclicLayout(num_chips=4, block=4)
    g = torch.Generator().manual_seed(13)
    scores = 100.0 + 0.2 * torch.rand(3, 64, generator=g)  # bf16 steps by 0.5 here: all in one bin
    positions = torch.tensor([63, 40, 5])
    one_pass = threshold_select(scores, positions, layout, topk=8, refine_passes=1)
    assert torch.equal(one_pass.sum(-1), torch.tensor([64, 41, 6])), one_pass.sum(-1)
    three = threshold_select(scores, positions, layout, topk=8, refine_passes=3)
    allowed = causal_allowed(positions, 64)
    exact = selection_mask(scores.masked_fill(~allowed, float("-inf")).topk(8, dim=-1).indices, allowed)
    assert torch.equal(three, exact)
