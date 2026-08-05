# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The LLM stage -- `02_plan.md` P4. Text tokens in, semantic speech tokens out.

Structured like `test_estimator.py`: structure, then the graph on the host via
`tt/llm/reference.py`, then the device. That order matters more here than anywhere
else in this bring-up, because this stage has four places where the config implies
one thing and the code does another, and each produces a network that runs and is
silently wrong. The host tier pins all four.

Sampling is deliberately split out. RAS is a multinomial draw and cannot be
reproduced token-for-token on a different RNG, so what is asserted here is its
*deterministic* structure -- the candidate set, the repetition rule, the EOS
suppression -- and the device generation test uses greedy decoding.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from models.demos.cosyvoice.tt.common import GOLDEN_DIR, as_torch, load_golden, pcc
from models.demos.cosyvoice.tt.weights import default_weights_path

LLM_WEIGHTS = default_weights_path().replace("hift_", "llm_")

needs_l1_small = pytest.mark.parametrize(
    "device_params", [{"l1_small_size": 32768, "trace_region_size": 67108864}], indirect=True
)
needs_weights = pytest.mark.skipif(
    not os.path.exists(LLM_WEIGHTS), reason="run scripts/export_weights.py --module llm --fp16 first"
)
needs_golden = pytest.mark.skipif(
    not os.path.exists(os.path.join(GOLDEN_DIR, "llm.ar_forward_chunk.npz")), reason="generate goldens first"
)


def _weights():
    with np.load(LLM_WEIGHTS) as z:
        meta = json.loads(bytes(z["__meta__"]).decode())
        w = {k: torch.from_numpy(np.ascontiguousarray(z[k])).float() for k in z.files if k != "__meta__"}
    return w, meta


# --------------------------------------------------------------------------
# host tier -- the four traps
# --------------------------------------------------------------------------
@needs_weights
def test_the_two_stacks_differ_in_more_than_depth():
    """One checkpoint, two encoder stacks, three structural differences between them.

    Both are built from the same `cosyvoice.transformer.encoder` module family and
    the yaml sets almost nothing, so every one of these comes from a *default*.
    """
    _, meta = _weights()
    text, ar = meta["text_encoder"], meta["ar_decoder"]

    assert text["n_layers"] == 6 and ar["n_layers"] == 14
    # ConformerEncoder defaults activation_type="swish"; TransformerEncoder "relu"
    assert text["ffn_activation"] == "silu", text["ffn_activation"]
    assert ar["ffn_activation"] == "relu", ar["ffn_activation"]
    # input_layer 'linear' vs 'linear_legacy' -- only the latter appends a ReLU
    assert text["embed_has_relu"] is False
    assert ar["embed_has_relu"] is True
    # two epsilons, from two different source files
    assert text["layer_norm_eps"] == 1e-12 and text["embed_norm_eps"] == 1e-5


@needs_weights
def test_llm_weight_export_is_complete():
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta
    assert bag.tensor("text_embedding.weight").shape == (meta["text_token_size"], 512)
    assert bag.tensor("speech_embedding.weight").shape == (meta["speech_token_size"], 1024)
    assert bag.tensor("llm_embedding.weight").shape == (2, 1024)
    assert bag.sub("llm_decoder").tensor("weight").shape == (meta["speech_token_size"] + 1, 1024)
    assert meta["eos_token"] == meta["speech_token_size"], "EOS is the extra row on the head"
    for i in range(meta["ar_decoder"]["n_layers"]):
        layer = bag.sub(f"llm.encoders.{i}")
        for name in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            assert layer.sub(f"self_attn.{name}").has("weight"), (i, name)
        assert layer.sub("norm1").has("weight") and layer.sub("norm2").has("weight")


@needs_weights
@needs_golden
def test_text_encoder_is_causal():
    """`static_chunk_size: 1` -> `subsequent_chunk_mask` -> a plain causal mask.

    This is the single most consequential line in the LLM config and it is easy to
    read past: the flow encoder is the *same class* with `static_chunk_size` left at
    its default of 0, and attends fully. Running this stack unmasked gives PCC 0.78.
    """
    from models.demos.cosyvoice.tt.llm import reference as R

    w, meta = _weights()
    g = load_golden("llm.text_encoder")
    xs, want = as_torch(g["call0.in_xs"]), as_torch(g["call0.out_xs"])

    with torch.no_grad():
        causal = R.text_encoder(w, xs, meta["text_encoder"], causal=True)
        full = R.text_encoder(w, xs, meta["text_encoder"], causal=False)
    p_causal, p_full = pcc(causal, want), pcc(full, want)
    print(f"\n  text encoder causal {p_causal:.10f}   full-attention {p_full:.10f}")
    assert p_causal >= 0.9999, p_causal
    assert p_full < 0.99, "unmasked attention should NOT reproduce the golden"


@needs_weights
@needs_golden
def test_torch_reference_reproduces_prefill_and_decode():
    """Prefill, the KV cache it produces, and a decode step that consumes it."""
    from models.demos.cosyvoice.tt.llm import reference as R

    w, meta = _weights()
    g = load_golden("llm.ar_forward_chunk")

    with torch.no_grad():
        ys, caches = R.ar_forward_chunk(w, as_torch(g["call0.in_xs"]), meta["ar_decoder"])
    p = pcc(ys, as_torch(g["call0.out_ys"]))
    print(f"\n  prefill {ys.shape[1]} tokens: PCC {p:.10f}")
    assert p >= 0.9999, p

    packed = R.pack_cache(caches)
    want_cache = as_torch(g["call0.out_att_cache"])
    assert packed.shape == want_cache.shape, (packed.shape, want_cache.shape)
    assert pcc(packed, want_cache) >= 0.9999

    with torch.no_grad():
        ys1, caches1 = R.ar_forward_chunk(w, as_torch(g["call1.in_xs"]), meta["ar_decoder"], caches)
    p1 = pcc(ys1, as_torch(g["call1.out_ys"]))
    print(f"  decode step: PCC {p1:.10f}, cache {caches[0][0].shape[2]} -> {caches1[0][0].shape[2]}")
    assert p1 >= 0.9999, p1
    assert R.pack_cache(caches1).shape == as_torch(g["call1.out_att_cache"]).shape


@needs_weights
@needs_golden
def test_decoder_head_argmax_is_stable():
    """The head is 1024 -> 4097. Argmax agreement matters more than PCC here: the
    logits feed a sampler, and a token index is what actually leaves this stage."""
    w, _ = _weights()
    g = load_golden("llm.decoder_head")
    for i in range(10):
        x, want = as_torch(g[f"call{i}.in_x"]), as_torch(g[f"call{i}.out_logits"])
        got = torch.nn.functional.linear(x, w["llm_decoder.weight"], w["llm_decoder.bias"])
        assert pcc(got, want) >= 0.9999, i
        assert int(got.argmax()) == int(want.argmax()), i


# --------------------------------------------------------------------------
# host tier -- sampling
# --------------------------------------------------------------------------
def test_nucleus_filter_includes_the_token_that_crosses_top_p():
    """`cum_prob < top_p` is tested *before* adding, so the retained mass is >=
    top_p, not <=. Reimplementing it the other way silently narrows the candidate
    set, which biases every utterance toward its mode."""
    from models.demos.cosyvoice.tt.llm.sampling import nucleus_filter

    probs = torch.tensor([0.5, 0.35, 0.1, 0.05])
    value, idx = nucleus_filter(probs, top_p=0.8, top_k=25)
    assert idx.tolist() == [0, 1], idx.tolist()
    assert float(value.sum()) >= 0.8


def test_nucleus_filter_respects_top_k():
    from models.demos.cosyvoice.tt.llm.sampling import nucleus_filter

    probs = torch.full((100,), 0.01)
    value, idx = nucleus_filter(probs, top_p=0.99, top_k=25)
    assert len(idx) == 25


def test_one_repeat_in_the_window_triggers_resampling():
    """`rep_num >= win_size * tau_r` is `>= 1.0` with the shipped defaults, so a
    single occurrence in the last ten tokens is enough. Far more aggressive than
    "repetition aware" implies, and it fires constantly in real decoding."""
    from models.demos.cosyvoice.tt.llm.sampling import is_repetitive

    assert is_repetitive([5, 1, 2, 3], 5) is True
    assert is_repetitive([1, 2, 3], 5) is False
    # outside the 10-token window, so it does not count
    assert is_repetitive([5] + list(range(100, 111)), 5) is False


def test_ras_rejects_the_repeated_token():
    from models.demos.cosyvoice.tt.llm.sampling import ras_sampling

    torch.manual_seed(0)
    scores = torch.full((50,), -20.0)
    scores[7] = 10.0  # overwhelmingly the nucleus winner
    assert ras_sampling(scores.clone(), []) == 7
    # with 7 already in the window it must be rejected and something else drawn
    assert ras_sampling(scores.clone(), [7]) != 7


# --------------------------------------------------------------------------
# device tier
# --------------------------------------------------------------------------
@needs_weights
@needs_golden
@needs_l1_small
def test_device_text_encoder(device):
    """6 Conformer blocks, 16 heads, d=1024, causal."""
    import ttnn
    from models.demos.cosyvoice.tt.flow.encoder import TtConformerEncoder, espnet_rel_positional_encoding
    from models.demos.cosyvoice.tt.llm.decoder import causal_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("llm.text_encoder")
    xs, want = as_torch(g["call0.in_xs"]), as_torch(g["call0.out_xs"])
    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta["text_encoder"]
    enc = TtConformerEncoder(device, bag.sub("text_encoder"), meta)

    t = xs.shape[1]
    dev = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    got = ttnn.to_torch(
        enc(dev(xs), dev(espnet_rel_positional_encoding(t, meta["d_model"])), mask=dev(causal_bias(t)))
    ).float()

    p = pcc(got, want)
    print(f"\n  LLM text encoder, T={t}: PCC {p:.10f}  max|d| {(got - want).abs().max():.3e}")
    assert p >= 0.99, p


@needs_weights
@needs_golden
@needs_l1_small
def test_device_ar_prefill_and_decode(device):
    """The 14-block AR decoder: a 209-token causal prefill, then one cached step.

    The cache is the thing under test. Prefill and decode share every weight, so a
    decode step that matches while prefill does too means the concatenation, the
    positional window and the mask are all right together -- which is not implied
    by either passing alone.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, causal_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("llm.ar_forward_chunk")
    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])

    dev = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731
    xs0 = as_torch(g["call0.in_xs"])
    ys, caches = dec.forward_chunk(dev(xs0), caches=None, mask=dev(causal_bias(xs0.shape[1])))
    got0 = ttnn.to_torch(ys).float()
    ttnn.deallocate(ys)
    p0 = pcc(got0, as_torch(g["call0.out_ys"]))
    print(f"\n  AR prefill, {xs0.shape[1]} tokens: PCC {p0:.10f}")

    ys1, caches = dec.forward_chunk(dev(as_torch(g["call1.in_xs"])), caches=caches, mask=None)
    got1 = ttnn.to_torch(ys1).float()
    ttnn.deallocate(ys1)
    p1 = pcc(got1, as_torch(g["call1.out_ys"]))
    print(f"  AR decode step (cache {xs0.shape[1]} -> {caches[0][0].shape[2]}): PCC {p1:.10f}")
    TtARDecoder.free_caches(caches)

    assert p0 >= 0.99, p0
    assert p1 >= 0.99, p1


@needs_weights
@needs_golden
@needs_l1_small
def test_device_fixed_shape_cache_matches_the_growing_one(device):
    """The right-aligned fixed-width cache must give the *same answer* as the
    growing one -- it is a performance change, not a numerical one.

    It exists because a growing cache gives every token a new attention key size,
    and TTNN compiles per shape: 98.9% of cold decode time was JIT. Holding the
    key width fixed leaves exactly two shapes for the whole utterance.

    The alignment is the part that has to be right. `rel_shift` skews the score
    block assuming the queries are the **last** `t1` of the `K` key positions, so
    the live tokens go at the end of the buffer and the padding at the front.
    Left-aligning gives every query the relative geometry of a position it is not
    at -- which this test would catch and a shape check would not.
    """
    import ttnn
    from models.demos.cosyvoice.tt.llm.decoder import TtARDecoder, right_aligned_bias
    from models.demos.cosyvoice.tt.weights import WeightBag

    g = load_golden("llm.ar_forward_chunk")
    bag = WeightBag.load(LLM_WEIGHTS)
    dec = TtARDecoder(device, bag.sub("llm"), bag.meta["ar_decoder"])

    xs0 = as_torch(g["call0.in_xs"])
    prefix_len = xs0.shape[1]
    max_len = 256  # any width >= the sequence; two shapes get compiled, not 500

    dev = lambda v: ttnn.from_torch(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)  # noqa: E731

    caches = dec.empty_cache(max_len, prefix_len)
    ys, caches = dec.forward_chunk_fixed(
        dev(xs0),
        caches,
        max_len,
        valid=prefix_len,
        mask=dev(right_aligned_bias(max_len, prefix_len, prefix_len, causal=True)),
    )
    got0 = ttnn.to_torch(ys).float()
    ttnn.deallocate(ys)
    p0 = pcc(got0, as_torch(g["call0.out_ys"]))

    ys1, caches = dec.forward_chunk_fixed(
        dev(as_torch(g["call1.in_xs"])),
        caches,
        max_len,
        valid=prefix_len + 1,
        mask=dev(right_aligned_bias(max_len, prefix_len + 1, 1)),
    )
    got1 = ttnn.to_torch(ys1).float()
    ttnn.deallocate(ys1)
    p1 = pcc(got1, as_torch(g["call1.out_ys"]))
    TtARDecoder.free_caches(caches)

    print(f"\n  fixed-shape cache (max_len={max_len}): prefill {p0:.10f}, decode {p1:.10f}")
    assert p0 >= 0.99, p0
    assert p1 >= 0.99, p1


@needs_weights
@needs_golden
@needs_l1_small
def test_device_generates_tokens_greedily(device):
    """The whole LLM stage end to end, with greedy decoding so the stream is
    reproducible. Checks that it emits plausible speech tokens and terminates --
    the token *values* cannot be compared to the reference, which sampled."""
    import ttnn
    from models.demos.cosyvoice.tt.llm.model import TtTransformerLM
    from models.demos.cosyvoice.tt.weights import WeightBag

    bag = WeightBag.load(LLM_WEIGHTS)
    meta = bag.meta
    model = TtTransformerLM(device, bag, meta)

    # a short synthetic text token sequence; the golden's own text tokens are not
    # captured separately, and what is under test here is the loop, not the words
    text = torch.arange(10, 42, dtype=torch.int32).reshape(1, -1)
    spk = as_torch(load_golden("llm.spk_embed_affine")["call0.in_x"]).reshape(1, 1, -1)

    tokens = model.generate(
        ttnn.from_torch(text, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device),
        spk_emb=model.speaker_embedding(
            ttnn.from_torch(spk, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        ),
        sampler="greedy",
        max_tokens=24,
    )
    print(f"\n  generated {len(tokens)} tokens, first 8: {tokens[:8]}")
    assert len(tokens) > 0
    assert all(0 <= t < meta["speech_token_size"] for t in tokens), "EOS must never be emitted as a token"
