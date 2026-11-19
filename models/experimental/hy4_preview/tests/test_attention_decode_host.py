# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Anchor: the decode-step reference (reference/attention_decode.py) must be HF transformers'
HYV4Attention for the newest token.

Three pins to HF, then the reference against itself:

* the step equals the LAST ROW of HF's full-sequence attention with the indexer's selection, on a
  full layer and on a shared layer that reuses the full layer's picks, below, at and above the
  top-k, with a small top-k and with the real 2048;
* the step equals HF's own cached decode: a 6-layer HYV4Model (full full shared shared shared
  full) prefilled and then stepped token by token with its DynamicCache, every layer's attention
  replayed through the reference, cache entries included;
* the same two pins with the checkpoint's weights: layers 0-5 against the recorded BF16 golden
  (128 tokens), and one full and one shared layer in fp32 at 2304 positions.

Then: the absorbed form and the log-sum-exp merge over context-parallel stripes equal the
expanded form, also when the selected rows sit on one chip or on a few; entries flagged not
valid are ignored; and a selection that is actually cut by the top-k changes the output, so none
of the above can pass with the selection wired to nothing.

Pure CPU (torch vs torch), no device. Skipped when the installed transformers predates ``hy_v4``.
"""

import copy
import inspect

import pytest
import torch

from models.experimental.hy4_preview.reference.attention import AttentionConfig, Hy4Attention, causal_mask, rope_cos_sin
from models.experimental.hy4_preview.reference.attention_decode import DecodeCache, Hy4AttentionDecode
from models.experimental.hy4_preview.reference.indexer import (
    Hy4Indexer,
    IndexerConfig,
    causal_allowed,
    picks_source,
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
# (indexer heads, index head dim, top-k): a small top-k that a short prompt crosses, and the real one
INDEXERS = {
    "top16": IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=4, head_dim=32, qk_rope_head_dim=8, topk=16),
    "top2048": IndexerConfig(hidden_size=96, q_lora_rank=48, num_heads=32, head_dim=32, qk_rope_head_dim=8),
}
RTOL = 1e-5


def _hf_config(cfg: AttentionConfig, icfg: IndexerConfig, num_layers: int = 2) -> HYV4Config:
    return HYV4Config(
        vocab_size=64,
        hidden_size=cfg.hidden_size,
        intermediate_size=128,
        moe_intermediate_size=48,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=cfg.num_heads,
        num_key_value_heads=cfg.num_heads,
        head_dim=cfg.qk_rope_head_dim,
        num_hidden_layers=num_layers,
        q_lora_rank=cfg.q_lora_rank,
        kv_lora_rank=cfg.kv_lora_rank,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        v_head_dim=cfg.v_head_dim,
        index_topk=icfg.topk,
        index_head_dim=icfg.head_dim,
        index_n_heads=icfg.num_heads,
        rms_norm_eps=icfg.norm_eps,
        max_position_embeddings=1 << 20,
        rope_parameters={"rope_type": "default", "rope_theta": cfg.rope_theta},
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attn_implementation="eager",
    )


def _randomize(module, seed):
    """Linear outputs of order one; norm weights, biases and sinks away from their trivial init."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) * p.shape[-1] ** -0.5)
            elif "hc_" not in name and "e_score_correction_bias" not in name:
                p.add_(0.25 * torch.randn(p.shape, generator=g))
    return module


def _reference(hf_attention, cfg: AttentionConfig, icfg: IndexerConfig) -> Hy4AttentionDecode:
    """The decode reference holding an HF attention layer's parameters (and its indexer's, if it has one)."""
    attention = Hy4Attention(cfg).eval()
    state = dict(hf_attention.named_parameters())
    with torch.no_grad():
        for name, hf_name in HF_NAMES.items():
            getattr(attention, name).copy_(state[hf_name].float().reshape(getattr(attention, name).shape))
    indexer = None
    if hf_attention.indexer is not None:
        indexer = (
            Hy4Indexer(icfg).eval().load_weights({k: v.float() for k, v in hf_attention.indexer.state_dict().items()})
        )
    return Hy4AttentionDecode(attention, indexer)


def _close(name, got, want, rtol=RTOL):
    max_diff = (got - want).abs().max().item()
    assert (
        max_diff <= rtol * want.abs().max().item()
    ), f"{name}: max|Δ|={max_diff:.3e} vs max|ref|={want.abs().max():.3e}"


def _hf_full_sequence(hf_cfg, hf, x, prev_topk_indices=None):
    """HF attention on a whole prompt, no cache: (output [T, hidden], picks [T, k])."""
    T = x.shape[0]
    positions = torch.arange(T)
    with torch.no_grad():
        position_embeddings = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x[None], positions[None])
        out, _, picks = hf(
            x[None],
            position_embeddings=position_embeddings,
            attention_mask=causal_mask(T)[None, None],
            prev_topk_indices=prev_topk_indices,
        )
    return out[0], picks


# ---------------------------------------------------------------------------------------------
# The step is the last row of HF's full-sequence attention
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "indexer, seq_len",
    [("top16", 1), ("top16", 10), ("top16", 16), ("top16", 17), ("top16", 40), ("top2048", 1536), ("top2048", 2304)],
    ids=["one-token", "below-topk", "at-topk", "one-above-topk", "above-topk", "top2048-below", "top2048-above"],
)
def test_decode_step_is_hf_last_row(indexer, seq_len):
    cfg, icfg = ATTENTION, INDEXERS[indexer]
    hf_cfg = _hf_config(cfg, icfg, num_layers=3)
    assert hf_cfg.indexer_types == ["full", "full", "shared"]
    hf_full = _randomize(hy_v4.HYV4Attention(hf_cfg, layer_idx=1).eval(), seed=3)
    hf_shared = _randomize(hy_v4.HYV4Attention(hf_cfg, layer_idx=2).eval(), seed=4)
    assert hf_full.indexer is not None and hf_shared.indexer is None
    full, shared = _reference(hf_full, cfg, icfg), _reference(hf_shared, cfg, icfg)

    g = torch.Generator().manual_seed(5)
    x_full = torch.randn(seq_len, cfg.hidden_size, generator=g)
    x_shared = torch.randn(seq_len, cfg.hidden_size, generator=g)  # a shared layer sees another hidden state
    hf_out_full, hf_picks = _hf_full_sequence(hf_cfg, hf_full, x_full)
    hf_out_shared, _ = _hf_full_sequence(hf_cfg, hf_shared, x_shared, hf_picks)

    last = seq_len - 1
    k = min(icfg.topk, seq_len)
    with torch.no_grad():
        cache = full.prefill_cache(x_full[:last])
        assert len(cache) == last and cache.index_keys.shape == (last, icfg.head_dim)
        step = full.step(x_full[last:], cache)
        shared_cache = shared.prefill_cache(x_shared[:last])
        assert shared_cache.index_keys is None
        shared_step = shared.step(x_shared[last:], shared_cache, picks=step.picks)

    # the selection: the same set HF picked for its last query
    assert step.picks.shape == (k,) and step.valid.all() and len(cache) == seq_len
    assert set(step.picks.tolist()) == set(hf_picks[0, last].tolist())
    assert step.picks.unique().numel() == k
    _close("full layer", step.out[0], hf_out_full[last])
    _close("shared layer", shared_step.out[0], hf_out_shared[last])

    if seq_len > icfg.topk:
        # the cut matters: attending over everything gives another answer, on both layers
        everything = torch.arange(seq_len)
        with torch.no_grad():
            dense = full.stages(x_full[last:], step.token, cache.latent)["out"]
            shared_dense = shared.stages(x_shared[last:], shared_step.token, shared_cache.latent)["out"]
        assert everything.numel() > k
        assert (dense - step.out).abs().max() > 1e-3 * step.out.abs().max()
        assert (shared_dense - shared_step.out).abs().max() > 1e-3 * shared_step.out.abs().max()


def test_decode_step_is_the_masked_reference_row():
    """The gather-then-attend step against reference/attention.py's own full-sequence forward with the
    selection as an additive mask: the form every other test of this model is built on."""
    cfg, icfg = ATTENTION, INDEXERS["top16"]
    hf_cfg = _hf_config(cfg, icfg)
    decode = _reference(_randomize(hy_v4.HYV4Attention(hf_cfg, layer_idx=0).eval(), seed=7), cfg, icfg)
    T = 50
    x = torch.randn(T, cfg.hidden_size, generator=torch.Generator().manual_seed(8))
    positions = torch.arange(T)
    cos, sin = rope_cos_sin(positions, cfg)
    with torch.no_grad():
        q_resid, _, _ = decode.attention.latent(x, cos, sin)
        allowed = causal_allowed(positions, T)
        picks, _ = decode.indexer(x, q_resid, cos, sin, allowed)
        keep = selection_mask(picks, allowed)
        mask = causal_mask(T).masked_fill(~keep, torch.finfo(torch.float32).min)
        whole = decode.attention(x, cos, sin, mask)
        cache = decode.prefill_cache(x[:0])
        assert len(cache) == 0
        for t in range(T):  # every row as a decode step, from an empty cache
            step = decode.step(x[t : t + 1], cache)
            assert set(step.picks.tolist()) == set(keep[t].nonzero().flatten().tolist()), t
            _close(f"row {t}", step.out[0], whole[t])


# ---------------------------------------------------------------------------------------------
# The step is HF's cached decode: the whole model with its DynamicCache
# ---------------------------------------------------------------------------------------------
def _attention_taps(model):
    """Per layer and per forward call: (attention input [S, hidden], output [S, hidden], picks [S, k])."""
    calls = {}
    for layer, block in enumerate(model.layers):

        def hook(module, args, kwargs, output, layer=layer):
            x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            calls.setdefault(layer, []).append((x[0].detach().clone(), output[0][0].detach().clone(), output[2][0]))

        block.self_attn.register_forward_hook(hook, with_kwargs=True)
    return calls


@pytest.mark.parametrize("prefill_len", [1, 10, 24], ids=["prefill-1", "prefill-below-topk", "prefill-above-topk"])
def test_decode_steps_are_hf_cached_decode(prefill_len):
    cfg, icfg = ATTENTION, INDEXERS["top16"]
    num_layers, decode_steps = 6, 20  # the shorter prefills cross the top-k while decoding
    hf_cfg = _hf_config(cfg, icfg, num_layers)
    assert hf_cfg.indexer_types == ["full", "full", "shared", "shared", "shared", "full"]
    source = picks_source(hf_cfg.indexer_types)
    model = _randomize(hy_v4.HYV4Model(hf_cfg).eval(), seed=9)
    calls = _attention_taps(model)
    ids = torch.randint(
        3, hf_cfg.vocab_size, (1, prefill_len + decode_steps), generator=torch.Generator().manual_seed(10)
    )

    # the cache keyword and output field were renamed between transformers releases
    cache_name = "cache" if "cache" in inspect.signature(model.forward).parameters else "past_key_values"
    with torch.no_grad():
        out = model(input_ids=ids[:, :prefill_len], use_cache=True)
        for step in range(decode_steps):
            pos = prefill_len + step
            out = model(input_ids=ids[:, pos : pos + 1], use_cache=True, **{cache_name: getattr(out, cache_name)})
    hf_cache = getattr(out, cache_name)

    layers = [_reference(block.self_attn, cfg, icfg) for block in model.layers]
    assert [layer.is_full for layer in layers] == [kind == "full" for kind in hf_cfg.indexer_types]
    with torch.no_grad():
        caches = [layer.prefill_cache(calls[n][0][0]) for n, layer in enumerate(layers)]
    crossed = False
    for step in range(decode_steps):
        position = prefill_len + step
        picks = {}
        for n, layer in enumerate(layers):
            x, hf_out, hf_picks = calls[n][1 + step]
            assert x.shape == (1, cfg.hidden_size)
            with torch.no_grad():
                result = layer.step(x, caches[n], picks=None if layer.is_full else picks[source[n]], position=position)
            picks[n] = result.picks
            assert set(result.picks.tolist()) == set(hf_picks[0].tolist()), (step, n)
            _close(f"step {step} layer {n}", result.out[0], hf_out[0])
            crossed |= result.picks.numel() < position + 1
    assert crossed, "no decode step was cut by the top-k"

    # the reference's caches are the ones HF kept
    total = prefill_len + decode_steps
    for n, cache in enumerate(caches):
        hf_layer = hf_cache.layers[n]
        hf_latent = torch.cat([hf_layer.keys[0, 0], hf_layer.values[0, 0]], dim=-1)
        assert cache.latent.shape == hf_latent.shape == (total, cfg.latent_dim)
        _close(f"layer {n} latent cache", cache.latent, hf_latent)
        if layers[n].is_full:
            _close(f"layer {n} index-key cache", cache.index_keys, hf_layer.indexer_keys[0])


# ---------------------------------------------------------------------------------------------
# The three forms of the step agree; flagged entries are ignored
# ---------------------------------------------------------------------------------------------
def _random_step(seed, context, indexer="top16"):
    cfg, icfg = ATTENTION, INDEXERS[indexer]
    hf_cfg = _hf_config(cfg, icfg)
    decode = _reference(_randomize(hy_v4.HYV4Attention(hf_cfg, layer_idx=0).eval(), seed=seed), cfg, icfg)
    x = torch.randn(context, cfg.hidden_size, generator=torch.Generator().manual_seed(seed + 1))
    with torch.no_grad():
        cache = decode.prefill_cache(x[:-1])
        token = decode.new_token(x[-1:], context - 1)
        cache.append(token.latent, token.index_key)
    return decode, x[-1:], token, cache


def test_absorbed_form_equals_expanded():
    decode, x, token, cache = _random_step(seed=11, context=70)
    cfg = decode.cfg
    with torch.no_grad():
        picks = decode.select(token, cache)
        selected = decode.gather(cache, picks)
        expanded = decode.stages(x, token, selected)
        absorbed = decode.absorbed_stages(x, token, selected)
        # the latent-space logits are the expanded q . key
        logits = torch.matmul(expanded["q"], expanded["keys"].transpose(-1, -2))[:, 0] * cfg.scaling
    _close("logits", absorbed["logits"], logits)
    _close("attn", absorbed["attn"], expanded["attn"])
    _close("out", absorbed["out"], expanded["out"])
    assert absorbed["probs"].sum(-1).max() < 1.0, "the sink takes a share of every head's probability"
    assert absorbed["q_lat"].shape == (cfg.num_heads, cfg.latent_dim)


def test_entries_flagged_not_valid_are_ignored():
    """A fixed-width selection pads a short context with entries that are not picks. Whatever
    position such an entry names, the step is the one over the valid entries alone."""
    decode, x, token, cache = _random_step(seed=13, context=9)
    g = torch.Generator().manual_seed(14)
    with torch.no_grad():
        picks = decode.select(token, cache)  # all 9 positions
        want = decode.stages(x, token, decode.gather(cache, picks))["out"]
        padded = torch.cat([picks, torch.randint(0, 9, (23,), generator=g)])
        valid = torch.arange(32) < 9
        order = torch.randperm(32, generator=g)  # the invalid entries anywhere in the list
        padded, valid = padded[order], valid[order]
        for form in (decode.stages, decode.absorbed_stages):
            got = form(x, token, decode.gather(cache, padded, valid), valid)["out"]
            _close(form.__name__, got, want)
            # and without the flag they would have counted
            wrong = form(x, token, decode.gather(cache, padded))["out"]
            assert (wrong - want).abs().max() > 1e-3 * want.abs().max()


SPREADS = ["indexer", "one-chip", "last-chip", "two-chips", "eight-chips", "short"]


@pytest.mark.parametrize("spread", SPREADS)
def test_lse_merge_over_stripes_equals_expanded(spread):
    """The design #65889 names (every chip attends over the selected rows it holds, partial
    outputs merged by log-sum-exp) gives the gather-then-attend answer, however the picks fall on
    the chips: all on one, on two, on eight of the 32, or wherever the indexer puts them."""
    layout = BlockCyclicLayout(num_chips=32, block=4)  # 4-token blocks: 1500 positions cover every chip 11 times
    context, k = 1500, 16
    decode, x, token, cache = _random_step(seed=15, context=context)
    positions = torch.arange(context)
    g = torch.Generator().manual_seed(16)

    def among(chips, count=k):
        pool = positions[torch.isin(layout.chip(positions), torch.tensor(chips))]
        return pool[torch.randperm(pool.numel(), generator=g)[:count]]

    valid = None
    with torch.no_grad():
        if spread == "indexer":
            picks = decode.select(token, cache)
        elif spread == "one-chip":
            picks = among([5])
        elif spread == "last-chip":
            picks = among([31])
        elif spread == "two-chips":
            picks = torch.cat([among([0], 15), among([31], 1)])
        elif spread == "eight-chips":
            picks = among(list(range(8, 16)))
        else:  # a short list: 3 real picks, the rest flagged
            picks = torch.cat([among([7], 3), torch.zeros(k - 3, dtype=torch.long)])
            valid = torch.arange(k) < 3
        want = decode.stages(x, token, decode.gather(cache, picks, valid), valid)["out"]
        got, rows = decode.lse_merge(x, token, cache, picks, valid, layout)
    assert rows.sum() == (k if valid is None else int(valid.sum()))
    if spread in ("one-chip", "last-chip"):
        assert rows.max() == k and (rows > 0).sum() == 1
    if spread == "two-chips":
        assert sorted(rows[rows > 0].tolist()) == [1, 15]
    if spread == "eight-chips":
        assert rows[:8].sum() == 0 and rows[16:].sum() == 0
    _close(spread, got, want)


# ---------------------------------------------------------------------------------------------
# The reference at the checkpoint's precision is HF run in bf16
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "indexer, seq_len",
    [("top16", 300), ("top2048", 2304), ("top2048", 4000)],
    ids=["top16-300", "top2048-2304", "top2048-4000"],
)
def test_bf16_reference_selects_what_hf_selects_in_bf16(indexer, seq_len):
    """``as_bf16()`` is the released precision: the copy's selection for the last token is the set
    HF's HYV4Attention picks when module and input are bf16, and its output is HF's to a bf16 ulp.

    The same HF layer in fp32 picks a different set once the top-k cuts deep enough: scores next
    to the cut change sides when the queries are rounded. That number is the floor the device tests
    hold a bf16 device selection against; here it only has to be non-zero where the context is long
    enough and every moved position has to be a near-tie of the fp32 scores.
    """
    cfg, icfg = ATTENTION, INDEXERS[indexer]
    hf_cfg = _hf_config(cfg, icfg, num_layers=3)
    hf = _randomize(hy_v4.HYV4Attention(hf_cfg, layer_idx=1).eval(), seed=3)
    full = _reference(hf, cfg, icfg)
    hf16 = copy.deepcopy(hf).to(torch.bfloat16)
    x16 = torch.randn(seq_len, cfg.hidden_size, generator=torch.Generator().manual_seed(5)).bfloat16()
    positions = torch.arange(seq_len)
    last = seq_len - 1
    k = min(icfg.topk, seq_len)
    with torch.no_grad():
        rope16 = hy_v4.HYV4RotaryEmbedding(hf_cfg)(x16[None], positions[None])
        assert rope16[0].dtype == torch.bfloat16
        hf_out16, _, hf_picks16 = hf16(
            x16[None], position_embeddings=rope16, attention_mask=causal_mask(seq_len, torch.bfloat16)[None, None]
        )
        _, hf_picks32 = _hf_full_sequence(hf_cfg, hf, x16.float())

        ref16 = full.as_bf16()
        cache = ref16.prefill_cache(x16[:last])
        step = ref16.step(x16[last:], cache)
        token = step.token
        assert token.q_resid.dtype == token.index_q.dtype == cache.index_keys.dtype == torch.bfloat16
        assert token.index_w.dtype == torch.float32  # HF sums the scores in fp32

        # the fp32 reference on the same input, for the size of the difference
        cache32 = full.prefill_cache(x16[:last].float())
        step32 = full.step(x16[last:].float(), cache32)
        scores32 = full.scores(step32.token, cache32)[None]

    picks16, hf16_set, hf32_set = (
        set(step.picks.tolist()),
        set(hf_picks16[0, last].tolist()),
        set(hf_picks32[0, last].tolist()),
    )
    assert len(picks16) == k and picks16 == hf16_set, f"{len(picks16 ^ hf16_set)} positions differ from HF in bf16"
    assert set(step32.picks.tolist()) == hf32_set
    _close("bf16 output", step.out[0].float(), hf_out16[0, last].float(), rtol=2e-2)

    moved = picks16 ^ hf32_set
    print(f"{indexer} at {seq_len}: bf16 moves {len(moved) // 2} of {k} picks against fp32")
    if moved:
        kth = scores32[0].topk(k).values[-1]
        furthest = (scores32[0, sorted(moved)] - kth).abs().max().item()
        spread = (scores32.max() - scores32.min()).item()
        assert furthest <= 0.02 * spread, f"a moved position is {furthest:.3e} from the cut (spread {spread:.3e})"
    if seq_len == 4000:
        assert moved, "bf16 and fp32 picked the same set: the floor this test documents is gone"


# ---------------------------------------------------------------------------------------------
# Checkpoint weights
# ---------------------------------------------------------------------------------------------
def _checkpoint_reference(layer: int, full: bool) -> Hy4AttentionDecode:
    from models.experimental.hy4_preview.reference import hf_harness

    prefix = f"model.layers.{layer}.self_attn."
    names = [prefix + n + ("" if n == "learnable_sink_param" else ".weight") for n in Hy4Attention.PARAMETER_NAMES]
    attention = Hy4Attention(AttentionConfig()).eval()
    with torch.no_grad():
        for name, value in zip(Hy4Attention.PARAMETER_NAMES, hf_harness.checkpoint_tensors(names)):
            getattr(attention, name).copy_(value.reshape(getattr(attention, name).shape))
    indexer = None
    if full:
        keys = list(Hy4Indexer.CHECKPOINT_NAMES.values())
        tensors = hf_harness.checkpoint_tensors([prefix + "indexer." + k for k in keys])
        indexer = Hy4Indexer(IndexerConfig()).eval().load_weights(dict(zip(keys, tensors)))
    return Hy4AttentionDecode(attention, indexer)


def _need_checkpoint():
    from models.experimental.hy4_preview.reference import hf_harness

    try:
        root = hf_harness.model_dir()
    except AssertionError:
        pytest.skip(f"{hf_harness.HY4_HF_MODEL_ENV} is not set (tencent/Hy4-preview checkpoint dir)")
    if not (root / "model.safetensors.index.json").is_file():
        pytest.skip(f"no checkpoint in {root}")
    return hf_harness


def test_checkpoint_decode_step_against_the_hf_golden():
    """Layers 0-5 with the checkpoint's weights and their real inputs: the last token of the recorded
    128-token prompt as a decode step on a cache of the first 127, against what HF (BF16) computed for
    that token in prefill. 128 < 2048, so the selection keeps everything; the reference runs in fp32,
    so this is a PCC, not an equality."""
    hf_harness = _need_checkpoint()
    golden = hf_harness.find_golden()
    if golden is None:
        pytest.skip(f"no golden in {hf_harness.GOLDEN_ROOT}")
    types = golden["meta"]["indexer_types"]
    source = picks_source(types)
    picks = {}
    for layer, rec in enumerate(golden["layers"]):
        decode = _checkpoint_reference(layer, types[layer] == "full")
        x = rec["attn_in"].float()
        last = x.shape[0] - 1
        with torch.no_grad():
            cache = decode.prefill_cache(x[:last])
            step = decode.step(x[last:], cache, picks=None if decode.is_full else picks[source[layer]])
        picks[layer] = step.picks
        assert rec["dsa_dropped"] == 0 and set(step.picks.tolist()) == set(rec["topk_indices"][last].tolist())
        want = rec["attn_out"][last].float()
        score = torch.corrcoef(torch.stack([step.out[0].double(), want.double()]))[0, 1].item()
        max_diff = (step.out[0] - want).abs().max().item()
        print(f"layer {layer} ({types[layer]}): pcc={score:.6f} max|Δ|={max_diff:.3e} max|ref|={want.abs().max():.3e}")
        assert score >= 0.9999, f"layer {layer}: pcc={score:.6f}"
        # the cache entry of the token is what HF computed for it
        _close(f"layer {layer} k_pass", cache.latent[last, :512], rec["k_pass"][last].float(), rtol=2e-2)


@pytest.mark.parametrize("seq_len", [2304], ids=["S2304"])
def test_checkpoint_decode_step_above_topk(seq_len):
    """Layer 1 (full) and layer 2 (shared, reusing layer 1's picks) with the checkpoint's weights in
    fp32, on 2304 positions: the step is the last row of HF's full-sequence attention, and the
    top-2048 really leaves 256 positions out. The inputs are random rows of unit RMS."""
    hf_harness = _need_checkpoint()
    hf_cfg = hf_harness.truncated_config(num_layers=3)
    # A bare HYV4Attention reads the implementation off the config; left unset it would skip the
    # sparse mask and attend over everything.
    hf_cfg._attn_implementation = "eager"
    assert hf_cfg.indexer_types == ["full", "full", "shared"] and hf_cfg.index_topk == 2048
    g = torch.Generator().manual_seed(21)
    last = seq_len - 1
    picks = hf_picks = None
    for layer, full in ((1, True), (2, False)):
        decode = _checkpoint_reference(layer, full)
        hf = hy_v4.HYV4Attention(hf_cfg, layer_idx=layer).eval().float()
        state = dict(hf.named_parameters())
        with torch.no_grad():
            for name, hf_name in HF_NAMES.items():
                state[hf_name].copy_(getattr(decode.attention, name).reshape(state[hf_name].shape))
            if full:
                hf.indexer.load_state_dict(decode.indexer.weights())
        x = torch.randn(seq_len, hf_cfg.hidden_size, generator=g)
        hf_out, hf_picks = _hf_full_sequence(hf_cfg, hf, x, None if full else hf_picks)
        with torch.no_grad():
            cache = decode.prefill_cache(x[:last])
            step = decode.step(x[last:], cache, picks=picks)
        picks = step.picks
        assert step.picks.numel() == 2048 and set(step.picks.tolist()) == set(hf_picks[0, last].tolist())
        _close(f"layer {layer}", step.out[0], hf_out[last])
        with torch.no_grad():
            dense = decode.stages(x[last:], step.token, cache.latent)["out"]
        change = ((dense - step.out).abs().max() / step.out.abs().max()).item()
        print(f"layer {layer}: attending over all {seq_len} positions instead moves the output by {change:.2e}")
        assert change > 1e-3, "the top-k cut did not matter: the test would pass with the selection ignored"
        del hf, decode


def test_decode_cache_bookkeeping():
    cache = DecodeCache(torch.zeros(3, 8), torch.zeros(3, 4))
    cache.append(torch.ones(1, 8), torch.ones(1, 4))
    assert len(cache) == 4 and cache.index_keys.shape == (4, 4)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check
        cache.append(torch.ones(1, 8))
    decode, x, token, cache = _random_step(seed=17, context=5)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check (the token must be the next position)
        decode.step(x, cache, position=3)
    shared = Hy4AttentionDecode(decode.attention)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-side check (a shared layer needs picks)
        shared.step(x, DecodeCache(cache.latent))
