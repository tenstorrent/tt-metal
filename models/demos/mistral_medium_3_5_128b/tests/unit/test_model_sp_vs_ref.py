# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M2/M3 row: the whole device model against the composed torch reference, at target SP x TP.

**Full width, reduced depth — a diagnostic, not acceptance.** Every dimension is the real one
(hidden 12288, 96/8 heads, intermediate 28672, vocab 131072, SP=8 x TP=4) except the layer count,
which ``MISTRAL_MODEL_LAYERS`` sets and which defaults to 2. The reason is that this row compares
against *random* weights forwarded on the **host**: ``ModelWeights.random`` at 88 layers is ~256 GB
of host bf16 and the forward would take hours. The full-depth statement is P1 and P2, which use the
real checkpoint and the prepared golden trace instead of a host forward, and the acceptance test
asserts the prepared dimensions so a reduced run cannot be mistaken for one.

What this row covers that no block row can:

* **The stack wiring.** Layer *i* must get layer *i*'s weights and write cache slot *i*. Both block
  rows run a single layer at ``layer_idx=0``, where a slicing or slot bug is invisible because
  every wrong answer is also the right one. :func:`test_model_kv_vs_ref` is the test — it reads
  back all N slots and PCCs each against the reference's own per-layer K/V, and the reference's
  layers have different random weights by construction (``seed + 1 + i``).
* **The two ends.** The embedding feeds the stack in the layout the stack expects, and the tail
  norm and head consume its output. The block rows check those against torch in isolation; here
  they are checked in place, which is where a layout mismatch at a seam would appear.
* **The tail norm instance.** The recipe's "same test, applied to the tail instance" is
  :func:`test_tail_norm_is_applied`: the model's output must be closer to the reference *after* the
  final norm than before it, which is what distinguishes a wired tail norm from a skipped one.

The chunked variant is the whole-model rehearsal for P2: the same model, run twice over halves of
the sequence with the cache carrying the first half, must reproduce the one-shot result.

The reference fixture walks the reference stack inline rather than calling
``MistralModel.forward``, because it needs the residual stream on both sides of the tail norm and
the forward only returns one of them. That the inline walk equals the forward is not assumed here —
it is ``test_reference_model.py::test_model_against_inline``, an M1 host row.
"""

import os

import pytest
import torch

import ttnn
from models.demos.mistral_medium_3_5_128b.reference.modeling import (
    REF_DTYPE,
    MistralYarnRotaryEmbedding,
    ModelWeights,
    build_model,
    causal_mask,
    hf_to_meta,
)
from models.demos.mistral_medium_3_5_128b.tests.device_utils import assert_pcc, from_mesh_2d, from_mesh_sp, read_kv_slot
from models.demos.mistral_medium_3_5_128b.tt.model import MistralModel, shard_tokens

#: Sequence length. A multiple of TILE_SIZE * sp = 256, and short enough that the host reference
#: forward at full width stays in the tens of seconds.
SEQ = 512
CHUNK = 256

#: Stack depth for this diagnostic. Full width, reduced depth — see the module docstring.
MODEL_LAYERS = int(os.getenv("MISTRAL_MODEL_LAYERS", "2"))


def _state_dict(w: ModelWeights) -> dict:
    """``ModelWeights`` -> the flat prefix-stripped dict :class:`MistralModel` takes."""
    sd = {
        "embed_tokens.weight": w.embed_tokens,
        "norm.weight": w.norm,
        "lm_head.weight": w.lm_head,
    }
    for i, layer in enumerate(w.layers):
        sd.update(
            {
                f"layers.{i}.input_layernorm.weight": layer.input_layernorm,
                f"layers.{i}.post_attention_layernorm.weight": layer.post_attention_layernorm,
                f"layers.{i}.self_attn.q_proj.weight": layer.q_proj,
                f"layers.{i}.self_attn.k_proj.weight": layer.k_proj,
                f"layers.{i}.self_attn.v_proj.weight": layer.v_proj,
                f"layers.{i}.self_attn.o_proj.weight": layer.o_proj,
                f"layers.{i}.mlp.gate_proj.weight": layer.gate_proj,
                f"layers.{i}.mlp.up_proj.weight": layer.up_proj,
                f"layers.{i}.mlp.down_proj.weight": layer.down_proj,
            }
        )
    return sd


def _pcc(a, b):
    """Plain correlation as a float, for the two comparisons that are ranked against each other."""
    x, y = a.double().flatten(), b.double().flatten()
    return torch.corrcoef(torch.stack([x, y]))[0, 1].item()


@pytest.fixture(scope="module")
def small_cfg(cfg):
    """Full width, ``MODEL_LAYERS`` deep."""
    return cfg.reduced(num_hidden_layers=MODEL_LAYERS)


@pytest.fixture(scope="module")
def weights(small_cfg):
    return ModelWeights.random(small_cfg, seed=11)


@pytest.fixture(scope="module")
def token_ids(small_cfg):
    g = torch.Generator().manual_seed(12)
    return torch.randint(0, small_cfg.vocab_size, (1, SEQ), generator=g, dtype=torch.int32)


@pytest.fixture(scope="module")
def reference(small_cfg, weights, token_ids):
    """``(pre_norm_hidden, post_norm_hidden, per_layer_kv)`` from the torch reference."""
    model = build_model(small_cfg, weights)
    ids = token_ids.long()
    with torch.no_grad():
        hidden = model.embed_tokens(ids).to(REF_DTYPE)
        cos, sin = MistralYarnRotaryEmbedding(small_cfg, REF_DTYPE)(torch.arange(SEQ, dtype=torch.int64)[None])
        mask = causal_mask(SEQ, SEQ, dtype=REF_DTYPE)
        kv = []
        for layer in model.layers:
            hidden, k, v = layer(hidden, cos, sin, mask)
            kv.append((k, v))
        return hidden, model.norm(hidden), kv


def _build(galaxy, small_cfg, mesh_config, ccl, weights, chunk_size, *, with_lm_head=False):
    return MistralModel(
        galaxy,
        small_cfg,
        _state_dict(weights),
        ccl,
        mesh_config,
        max_seq_len=SEQ,
        chunk_size=chunk_size,
        with_lm_head=with_lm_head,
    )


@pytest.fixture(scope="module")
def one_shot(galaxy, small_cfg, mesh_config, ccl, weights, token_ids):
    """One-shot device run: ``(post_norm_hidden, kv_cache)``. Built once and reused."""
    model = _build(galaxy, small_cfg, mesh_config, ccl, weights, SEQ)
    kv = model.allocate_cache()
    out = model(shard_tokens(galaxy, mesh_config, token_ids), kv_cache=kv, want_logits=False)
    ttnn.synchronize_device(galaxy)
    return from_mesh_sp(galaxy, out), kv


def test_model_hidden_vs_ref(reference, one_shot):
    """The whole model's output, embedding through tail norm, against the torch reference."""
    _, post_norm, _ = reference
    out, _ = one_shot
    assert_pcc("model_hidden", post_norm.unsqueeze(0), out)


def test_tail_norm_is_applied(reference, one_shot):
    """The output is closer to the reference *after* the tail norm than before it.

    An unwired tail norm is the failure this catches. The pre-norm and post-norm residual streams
    are strongly correlated, so a model that returned the pre-norm hidden would still PCC above the
    spec's lower bound against the post-norm reference; only the margin between the two comparisons
    distinguishes them.
    """
    pre_norm, post_norm, _ = reference
    out, _ = one_shot
    with_norm = _pcc(post_norm.unsqueeze(0), out)
    without_norm = _pcc(pre_norm.unsqueeze(0), out)
    print(f"model output vs post-norm {with_norm:.6f} / vs pre-norm {without_norm:.6f}")
    assert with_norm > without_norm, (
        "the model output is no closer to the post-norm reference than to the pre-norm one — "
        "the tail norm is probably not wired in"
    )


def test_model_kv_vs_ref(galaxy, reference, one_shot):
    """Every layer's cache slot holds that layer's K/V — the stack-wiring test.

    The reference's layers have distinct weights (``ModelWeights.random`` seeds per layer), so a
    stack that fed every layer layer-0's weights, or wrote every layer to slot 0, fails here and
    nowhere else in the suite.

    **K is permuted before comparison.** The device's Q/K projection rows are permuted HF
    half-split -> Meta interleaved at load (``tt/attention/weights.py``), so what lands in the cache
    is Meta-interleaved along ``head_dim`` while the reference produces HF half-split. The two are
    a permutation of each other within each head, which correlates at about ``1/head_dim`` — this
    test measured 0.0148 before the permutation went in, so the failure mode is unmistakable rather
    than subtle. ``GoldenTrace.layer_kv_meta`` applies the same permutation for P1 and P2; V is
    unrotated and layout-independent.
    """
    _, _, ref_kv = reference
    _, kv = one_shot
    seen = []
    for i in range(MODEL_LAYERS):
        k_ref, v_ref = ref_kv[i]
        k_ref = hf_to_meta(k_ref)
        slot = i  # user 0: slot = user_id * num_layers + layer_idx
        got_k = read_kv_slot(galaxy, kv.k, slot=slot, cache_global=kv.max_seq_len, chunk_size=SEQ, upto=SEQ)
        got_v = read_kv_slot(galaxy, kv.v, slot=slot, cache_global=kv.max_seq_len, chunk_size=SEQ, upto=SEQ)
        assert_pcc(f"model_kv_k[layer{i}]", k_ref, got_k)
        assert_pcc(f"model_kv_v[layer{i}]", v_ref, got_v)
        seen.append(got_k)
    for i in range(len(seen)):
        for j in range(i + 1, len(seen)):
            assert not torch.equal(seen[i], seen[j]), f"layers {i} and {j} wrote identical K — same slot?"


def test_model_chunked_vs_one_shot(galaxy, small_cfg, mesh_config, ccl, weights, token_ids, one_shot):
    """The same model over two chunks reproduces the one-shot run — the P2 rehearsal.

    Compared against the *device* one-shot result rather than against the reference, because the
    question here is whether chunking changes the answer, not whether the model is right; the
    reference comparison is :func:`test_model_hidden_vs_ref`. Both runs carry the same dataformat
    error, so any gap is chunking's alone.
    """
    ref_out, _ = one_shot
    model = _build(galaxy, small_cfg, mesh_config, ccl, weights, CHUNK)
    kv = model.allocate_cache()

    outs = []
    for start in range(0, SEQ, CHUNK):
        out = model(
            shard_tokens(galaxy, mesh_config, token_ids[:, start : start + CHUNK]),
            kv_cache=kv,
            cached_len=start,
            want_logits=False,
        )
        outs.append(from_mesh_sp(galaxy, out))
    ttnn.synchronize_device(galaxy)

    chunked = torch.cat(outs, dim=2)
    assert chunked.shape == ref_out.shape, f"{tuple(chunked.shape)} != {tuple(ref_out.shape)}"
    assert_pcc("model_chunked_vs_one_shot", ref_out, chunked)


def test_model_logits_vs_ref(galaxy, small_cfg, mesh_config, ccl, weights, token_ids, reference):
    """With the head attached, the model produces the reference's logits.

    Separate from the hidden-state tests because it is the only one that loads the 1.6 G-parameter
    head, and because ``want_logits=False`` — the mode the acceptance run uses — has to be shown to
    be a skip of this and not a different model. The reference side reuses the already-computed
    post-norm hidden rather than re-running the stack.
    """
    _, post_norm, _ = reference
    model = _build(galaxy, small_cfg, mesh_config, ccl, weights, SEQ, with_lm_head=True)
    kv = model.allocate_cache()
    logits = model(shard_tokens(galaxy, mesh_config, token_ids), kv_cache=kv, want_logits=True)
    ttnn.synchronize_device(galaxy)
    out = from_mesh_2d(galaxy, logits, dims=(2, 3))

    with torch.no_grad():
        ref_logits = (post_norm.float() @ weights.lm_head.float().T).to(REF_DTYPE)
    assert_pcc("model_logits", ref_logits.unsqueeze(0), out)
