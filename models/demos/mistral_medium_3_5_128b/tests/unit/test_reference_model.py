# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M1 test 1 — the whole-model reference agrees with an inline torch golden, all layers.

Host-only; no device, no hardware. Widened from
``minimax_m3/tests/unit/test_reference_model.py``. The inline golden reuses the naive helpers from
:mod:`~.test_reference_modeling` — explicit loops, nothing imported from ``reference.modeling`` —
and adds the three things the decoder tests never exercised: the embedding lookup, the layer stack
(so a mis-ordered or weight-swapped layer shows up), and the final norm + lm head.

Everything runs on :func:`host_reduced_config`: 4 layers, hidden 512, vocab 2048. **A diagnostic
scale, never an acceptance result.** The real model is 128 B parameters; a host forward of it is not
something that happens, and its full-depth ground truth is the prepared golden trace instead —
see :mod:`~.test_golden_checkpoint_layer0` for the real-weights check and P1/P2 for full depth.
"""

import pytest
import torch

from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_3_5_128b.reference.golden import (
    GoldenCacheKey,
    load_golden,
    run_reference_model,
    save_golden,
)
from models.demos.mistral_medium_3_5_128b.reference.model_config import host_reduced_config
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    MistralYarnRotaryEmbedding,
    ModelWeights,
    build_model,
)
from models.demos.mistral_medium_3_5_128b.reference.regenerate import (
    HOST_TEST_TOKENS,
    WEIGHT_SEED,
    compute_model,
    model_key,
    random_token_ids,
)
from models.demos.mistral_medium_3_5_128b.tests.unit.test_reference_modeling import (
    inline_attention,
    inline_rmsnorm,
    inline_swiglu,
)

#: Layer 0 compares two implementations of the *same* arithmetic on the same input, so it is exact.
LAYER0_PCC = 0.999999
#: Deeper layers and the logits are not: the reference adds its residuals in a different order from
#: the inline golden, and in bf16 that difference compounds through the stack. Measured on this
#: fixture: layer 0 k/v 1.0000000, layer 1 0.9999883, layer 2 0.9999554, layer 3 0.9999196, logits
#: 0.9998863 — monotone in depth and ~4e-5 per layer, which is bf16 rounding and not a disagreement.
#: ``test_inline_gap_is_depth_accumulation`` holds that shape, so a real bug cannot hide in the
#: slack this threshold allows.
STACK_PCC = 0.9998


def inline_model(cfg, w: ModelWeights, ids, cos, sin):
    """The whole model written out: gather, then a plain layer loop, then norm and head."""
    x = w.embed_tokens[ids.reshape(-1)].reshape(*ids.shape, cfg.hidden_size)
    kvs = []
    for lw in w.layers:
        attn_out, k, v = inline_attention(inline_rmsnorm(x, lw.input_layernorm, cfg.rms_norm_eps), lw, cfg, cos, sin)
        x = x + attn_out
        normed = inline_rmsnorm(x, lw.post_attention_layernorm, cfg.rms_norm_eps)
        x = x + inline_swiglu(normed, lw.gate_proj, lw.up_proj, lw.down_proj)
        kvs.append((k, v))
    x = inline_rmsnorm(x, w.norm, cfg.rms_norm_eps)
    return x @ w.lm_head.T, kvs


@pytest.fixture(scope="module")
def cfg():
    return host_reduced_config()


@pytest.fixture(scope="module")
def weights(cfg):
    return ModelWeights.random(cfg, seed=WEIGHT_SEED)


@pytest.fixture(scope="module")
def ids(cfg):
    return random_token_ids(cfg, HOST_TEST_TOKENS)


@pytest.fixture(scope="module")
def reference(cfg, weights, ids):
    return run_reference_model(cfg, weights, ids)


@pytest.fixture(scope="module")
def inline_kv_pcc(cfg, weights, ids, reference):
    """``[(pcc_k, pcc_v)]`` per layer, inline golden vs reference. Computed once."""
    positions = torch.arange(ids.shape[1], dtype=torch.int64)[None]
    cos, sin = MistralYarnRotaryEmbedding(cfg, REF_DTYPE)(positions)
    want_logits, want_kv = inline_model(cfg, weights, ids, cos, sin)
    per_layer = [
        (comp_pcc(k, reference[f"k_{i}"], 0.0)[1], comp_pcc(v, reference[f"v_{i}"], 0.0)[1])
        for i, (k, v) in enumerate(want_kv)
    ]
    return comp_pcc(want_logits, reference["logits"], 0.0)[1], per_layer


def test_model_against_inline(cfg, inline_kv_pcc):
    """Logits and every layer's K/V against the inline golden."""
    logits_pcc, per_layer = inline_kv_pcc
    for i, (pk, pv) in enumerate(per_layer):
        bound = LAYER0_PCC if i == 0 else STACK_PCC
        assert pk >= bound, f"layer {i} k: {pk}"
        assert pv >= bound, f"layer {i} v: {pv}"
    assert logits_pcc >= STACK_PCC, f"logits: {logits_pcc}"


def test_inline_gap_is_depth_accumulation(cfg, inline_kv_pcc):
    """The inline gap must look like bf16 accumulation: monotone in depth and tiny per layer.

    Without this, ``STACK_PCC``'s slack would be somewhere a genuine defect could sit. A real
    disagreement — a wrong weight, a dropped residual, a mis-ordered layer — would not decay
    smoothly with depth from an exact layer 0.
    """
    ks = [pk for pk, _ in inline_kv_pcc[1]]
    assert ks == sorted(ks, reverse=True), f"per-layer PCC is not monotone in depth: {ks}"
    steps = [ks[i] - ks[i + 1] for i in range(len(ks) - 1)]
    assert max(steps) < 1e-3, f"per-layer PCC loss {max(steps):.2e} is too large to be bf16 rounding"


def test_every_layer_is_distinct(cfg, reference):
    """Adjacent layers must not produce identical K.

    ``ModelWeights.random`` seeds each layer separately, so identical K between two layers would
    mean the stack reused one weight set — the exact failure a single-layer test cannot see and the
    reason the recipe asks for a whole-model reference at all.
    """
    for i in range(cfg.num_hidden_layers - 1):
        assert not torch.equal(reference[f"k_{i}"], reference[f"k_{i + 1}"]), f"layers {i} and {i + 1} share weights"


def test_embedding_is_a_lookup(cfg, weights, ids):
    """The model's first act is a row gather, not a matmul against a transposed table."""
    model = build_model(cfg, weights)
    with torch.no_grad():
        got = model.embed_tokens(ids)
    want = weights.embed_tokens[ids.reshape(-1)].reshape(*ids.shape, cfg.hidden_size)
    torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_lm_head_is_untied(cfg, weights):
    """``tie_word_embeddings`` is False, so the head must not be the embedding table."""
    assert not cfg.tie_word_embeddings
    assert not torch.equal(weights.lm_head, weights.embed_tokens)


@pytest.mark.parametrize("chunk", [HOST_TEST_TOKENS // 4, HOST_TEST_TOKENS // 2])
def test_chunked_equals_one_shot(cfg, weights, ids, reference, chunk):
    """Chunked prefill must reproduce the one-shot result at every layer.

    This is the host-side statement of P2's acceptance property: chunk N attending the prefix left
    by chunks 0..N-1 is the same computation as processing the sequence at once. Proving it here
    means a P2 failure on device is a device problem, not an ambiguity in what chunking should do.
    """
    chunked = run_reference_model(cfg, weights, ids, chunk_size=chunk)
    ok, msg = comp_pcc(reference["logits"], chunked["logits"], STACK_PCC)
    assert ok, f"logits at chunk {chunk}: {msg}"
    for i in range(cfg.num_hidden_layers):
        for name in ("k", "v"):
            ok, msg = comp_pcc(reference[f"{name}_{i}"], chunked[f"{name}_{i}"], STACK_PCC)
            assert ok, f"layer {i} {name} at chunk {chunk}: {msg}"


def test_want_logits_false_skips_the_head(cfg, weights, ids, reference):
    """``want_logits=False`` drops the vocab matmul and changes nothing else.

    The head is ``hidden x vocab`` over every token — the most expensive op in a prefill and pure
    waste for the K/V comparisons P1 and P2 actually make.
    """
    out = run_reference_model(cfg, weights, ids, want_logits=False)
    assert "logits" not in out
    for i in range(cfg.num_hidden_layers):
        torch.testing.assert_close(out[f"k_{i}"], reference[f"k_{i}"], rtol=0, atol=0)


def test_golden_cache_round_trip(cfg, tmp_path, monkeypatch):
    """M1 test 3 — the model entry round-trips through the cache, and a key change misses.

    The decoder-layer entry is covered the same way in ``test_golden_cache.py``; this is the
    whole-model kind, whose payload is 2*num_layers + 2 tensors rather than four.
    """
    monkeypatch.setenv("MISTRAL_GOLDEN_CACHE", str(tmp_path))
    key = model_key(cfg)
    tensors = compute_model(cfg)
    save_golden(key, tensors)

    loaded = load_golden(key)
    assert set(loaded) == set(tensors)
    for name, t in tensors.items():
        torch.testing.assert_close(loaded[name], t, rtol=0, atol=0)

    # A changed field must miss rather than serve the entry above.
    with pytest.raises(FileNotFoundError, match="--regenerate"):  # allow-pytest.raises: host-side
        load_golden(
            GoldenCacheKey.for_config(
                cfg.reduced(hidden_size=1024),
                **{
                    "kind": "model",
                    "weight_source": f"random:{WEIGHT_SEED}",
                    "input_source": key.input_source,
                    "n_tokens": key.n_tokens,
                },
            )
        )
